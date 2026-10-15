"""Account security and privacy: forgotten and changed passwords, signing out other devices,
slowing down password guessing, downloading your data, and deleting your account."""

from __future__ import annotations

import html
import json
import logging
import secrets
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from fastapi import APIRouter, BackgroundTasks, Cookie, Depends, HTTPException, Response
from pydantic import BaseModel, EmailStr, Field

from app import auth, db, digest, learning
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
User = Depends(auth.current_user)

RESET_MINUTES = 60
RESETS_PER_HOUR = 3
# Wrong passwords allowed for one email within the window; after that, checking pauses for it.
GUESSES = 10
GUESS_WINDOW_MINUTES = 15


def _ago(minutes: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat(timespec="seconds")


# --------------------------------------------------------------- password guessing


def check_guesses(email: str) -> None:
    """Refuse to check passwords for an email that has had too many wrong ones recently."""
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM login_failures WHERE email = ? AND created_at > ?",
                         (email, _ago(GUESS_WINDOW_MINUTES))).fetchone()[0]
    if n >= GUESSES:
        reset = ", or reset your password" if digest.email_available() else ""
        raise HTTPException(429, f"Too many wrong passwords. Try again in {GUESS_WINDOW_MINUTES} minutes{reset}.")


def wrong_password(email: str) -> None:
    with db.connect() as conn:
        conn.execute("INSERT INTO login_failures (email, created_at) VALUES (?, ?)", (email, db.now()))
        conn.execute("DELETE FROM login_failures WHERE created_at < ?", (_ago(24 * 60),))


def right_password(email: str) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM login_failures WHERE email = ?", (email,))


def _confirm_password(user: dict, password: str) -> None:
    """For changes that need the current password, with the same limit on guessing as signing in."""
    email = user["email"].lower()
    check_guesses(email)
    if not auth.verify_password(password, user["password_hash"]):
        wrong_password(email)
        raise HTTPException(403, "That password isn't right")
    right_password(email)


def _sign_out_elsewhere(conn, user_id: int, keep_session: str | None, keep_device: str | None) -> None:
    """End every other session, stop notifications on every other device, and cancel reset links:
    whoever is securing the account shouldn't leave a way back in."""
    conn.execute("DELETE FROM sessions WHERE user_id = ? AND token IS NOT ?", (user_id, keep_session))
    conn.execute("DELETE FROM push_subscriptions WHERE user_id = ? AND endpoint IS NOT ?", (user_id, keep_device))
    conn.execute("UPDATE password_resets SET used_at = ? WHERE user_id = ? AND used_at IS NULL", (db.now(), user_id))


# -------------------------------------------------------------- forgotten password


class Forgot(BaseModel):
    email: EmailStr


def reset_message(user: dict, token: str) -> EmailMessage:
    link = f"{settings.public_url}/#/reset/{token}"
    rules = f"The link works once, within the next {RESET_MINUTES} minutes, and signs you out everywhere else."
    # The name only once the address is known to be theirs; until then it's whatever was typed at sign-up.
    name = user["display_name"] if user.get("email_confirmed_at") else ""
    msg = EmailMessage()
    msg["Subject"] = "Choose a new password for AI Every Time"
    msg["From"] = settings.mail_from
    msg["To"] = user["email"]
    msg.set_content(
        f"{f'Hi {name}' if name else 'Hello'},\n\nSomeone asked to reset the password for your AI Every Time account. "
        f"If it was you, choose a new one here:\n\n{link}\n\n{rules}\n\n"
        "If it wasn't you, you can ignore this email. Your password hasn't changed.\n"
    )
    msg.add_alternative(
        f"<p>{f'Hi {html.escape(name)}' if name else 'Hello'},</p><p>Someone asked to reset the password for your AI Every Time "
        f'account. If it was you, <a href="{html.escape(link)}">choose a new password</a>.</p><p>{rules}</p>'
        '<p style="color:#777;font-size:12px">If it wasn\'t you, you can ignore this email. Your password hasn\'t changed.</p>',
        subtype="html",
    )
    return msg


@router.post("/auth/forgot", status_code=202)
def forgot(body: Forgot, background: BackgroundTasks):
    """Email a link for choosing a new password. The answer is the same whether or not there's an
    account, and the email is sent after answering, so the timing doesn't tell either."""
    if not digest.email_available():
        raise HTTPException(409, "Password reset by email isn't set up on this server. Ask the site's admin for help.")
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        user = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.lower(),)).fetchone()
        if user is None or not digest.claim_send(conn, user["email"], "reset", RESETS_PER_HOUR):
            return {"ok": True}
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(minutes=RESET_MINUTES)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO password_resets (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                     (auth.hash_token(token), user["id"], db.now(), expires))
    background.add_task(digest.send_quietly, reset_message(dict(user), token), user["id"])
    return {"ok": True}


class Reset(BaseModel):
    token: str = Field(min_length=20, max_length=200)
    password: str = Field(min_length=8, max_length=200)


@router.post("/auth/reset", status_code=204)
def reset(body: Reset, response: Response):
    """Choose a new password with an emailed link. Every other session and device is signed out, and
    any other links for the account stop working; then the learner is signed in here."""
    with db.connect() as conn:
        row = conn.execute("SELECT r.user_id, u.email FROM password_resets r JOIN users u ON u.id = r.user_id "
                           "WHERE r.token_hash = ? AND r.used_at IS NULL AND r.expires_at > ?",
                           (auth.hash_token(body.token), db.now())).fetchone()
        if row is None:
            raise HTTPException(400, "This link has expired or was already used. Ask for a new one.")
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (auth.hash_password(body.password), row["user_id"]))
        _sign_out_elsewhere(conn, row["user_id"], None, None)  # this link and any others stop working too
        conn.execute("DELETE FROM login_failures WHERE email = ?", (row["email"],))
        digest.mark_confirmed(conn, row["user_id"])  # the link reached them, so the address is theirs
    auth.set_cookie(response, auth.create_session(row["user_id"]))


# ------------------------------------------------------------ password and devices


class PasswordChange(BaseModel):
    current_password: str = Field(max_length=200)
    new_password: str = Field(min_length=8, max_length=200)
    # This device's notification endpoint, if it has one, so it keeps getting notifications.
    keep_device: str | None = Field(None, max_length=2000)


@router.post("/me/password", status_code=204)
def change_password(body: PasswordChange, user: dict = User, aiet_session: str | None = Cookie(default=None)):
    """Change the password. Every other phone and browser is signed out and stops getting notifications."""
    _confirm_password(user, body.current_password)
    with db.connect() as conn:
        conn.execute("UPDATE users SET password_hash = ? WHERE id = ?", (auth.hash_password(body.new_password), user["id"]))
        _sign_out_elsewhere(conn, user["id"], aiet_session, body.keep_device)


@router.get("/me/sessions")
def sessions(user: dict = User, aiet_session: str | None = Cookie(default=None)):
    """How many other phones and browsers are signed in."""
    with db.connect() as conn:
        others = conn.execute("SELECT COUNT(*) FROM sessions WHERE user_id = ? AND token IS NOT ? AND expires_at > ?",
                              (user["id"], aiet_session, db.now())).fetchone()[0]
    return {"others": others}


class KeepDevice(BaseModel):
    keep_device: str | None = Field(None, max_length=2000)


@router.post("/me/sign-out-others", status_code=204)
def sign_out_others(body: KeepDevice, user: dict = User, aiet_session: str | None = Cookie(default=None)):
    with db.connect() as conn:
        _sign_out_elsewhere(conn, user["id"], aiet_session, body.keep_device)


# ----------------------------------------------------------------- your data


@router.get("/me/export")
def export(user: dict = User):
    """Everything AI Every Time keeps about the learner, as a JSON download."""
    uid = user["id"]
    with db.connect() as conn:
        def rows(sql: str, *args) -> list[dict]:
            return [dict(r) for r in conn.execute(sql, (uid, *args))]

        designed = [db.row_to_companion(c) | {
            "versions": [{"saved_at": v["created_at"], "persona": json.loads(v["persona"])} for v in conn.execute(
                "SELECT persona, created_at FROM companion_versions WHERE companion_id = ? ORDER BY id", (c["id"],))],
            "has_photo": conn.execute("SELECT 1 FROM companion_photos WHERE companion_id = ?", (c["id"],)).fetchone() is not None,
            "posts": [dict(p) for p in conn.execute(
                "SELECT title, body, topic, created_at FROM posts WHERE companion_id = ? ORDER BY id", (c["id"],))],
        } for c in conn.execute("SELECT * FROM companions WHERE owner_id = ? ORDER BY id", (uid,)).fetchall()]
        decks = [d | {"cards": [dict(x) for x in conn.execute(
            "SELECT front, back, box, due_at, reviews, lapses FROM cards WHERE deck_id = ? ORDER BY id", (d["id"],))]}
            for d in rows("SELECT id, title, source_type, source_ref, created_at FROM decks WHERE user_id = ? ORDER BY id")]
        data = {
            "exported_at": db.now(),
            "account": {k: user[k] for k in ("email", "display_name", "level", "goals", "timezone", "reminder_hour",
                                             "email_confirmed_at", "ai_allowance", "created_at")}
                       | {"role": auth.role_of(user), "email_reminders": bool(user["email_reminders"])},
            "companions_you_designed": designed,
            "companions_you_follow": rows(
                "SELECT f.companion_id, json_extract(c.persona, '$.name') AS name, f.created_at FROM follows f "
                "JOIN companions c ON c.id = f.companion_id WHERE f.user_id = ? ORDER BY f.created_at"),
            "conversations": rows(
                "SELECT m.companion_id, json_extract(c.persona, '$.name') AS companion, m.role, m.content, m.post_id, "
                "m.reaction, m.created_at FROM messages m JOIN companions c ON c.id = m.companion_id WHERE m.user_id = ? ORDER BY m.id"),
            "memories": rows("SELECT companion_id, fact, created_at FROM memories WHERE user_id = ? ORDER BY id"),
            "post_reactions": rows("SELECT post_id, kind, created_at FROM engagements WHERE user_id = ? ORDER BY created_at"),
            "lessons": rows("SELECT lesson_id, status, updated_at FROM lesson_progress WHERE user_id = ? ORDER BY updated_at"),
            # As the quiz page shows them: answers only once submitted.
            "quizzes_and_exams": [learning._public_assessment(r, reveal=r["submitted_at"] is not None)
                                  for r in rows("SELECT * FROM assessments WHERE user_id = ? ORDER BY id")],
            "flashcard_decks": decks,
            "study_plans": [r | {"plan": json.loads(r["plan"])} for r in rows(
                "SELECT weekly_minutes, plan, created_at FROM study_plans WHERE user_id = ? ORDER BY id")],
            "custom_paths": [r | {"data": json.loads(r["data"])} for r in rows(
                "SELECT status, data, review_note, created_at, submitted_at, reviewed_at FROM custom_paths "
                "WHERE user_id = ? ORDER BY id")],
            "reports_you_made": rows("SELECT companion_id, reason, created_at FROM reports WHERE user_id = ? ORDER BY created_at"),
            "appeals": rows("SELECT companion_id, message, status, decision_note, created_at, decided_at FROM appeals "
                            "WHERE user_id = ? ORDER BY id"),
            "notices": rows("SELECT text, created_at, read_at FROM notices WHERE user_id = ? ORDER BY id"),
            "study_days": [r["day"] for r in rows("SELECT day FROM activity WHERE user_id = ? ORDER BY day")],
            "notification_devices": rows("SELECT device, created_at, last_sent_at FROM push_subscriptions WHERE user_id = ?"),
            "ai_requests": rows("SELECT kind, units, failed, input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, "
                                "model, created_at FROM ai_usage WHERE user_id = ? ORDER BY id"),
            "ai_requests_turned_away": rows("SELECT hour, reason, count FROM ai_refusals WHERE user_id = ? ORDER BY hour"),
        }
    return Response(json.dumps(data, ensure_ascii=False, indent=2), media_type="application/json",
                    headers={"Content-Disposition": 'attachment; filename="ai-every-time-data.json"'})


class DeleteAccount(BaseModel):
    password: str = Field(max_length=200)


@router.delete("/me", status_code=204)
def delete_account(body: DeleteAccount, response: Response, user: dict = User):
    """Delete the account and everything in it: the companions they designed and their posts, chats,
    memories, progress and devices. Staff decisions they made stay in the log, and their AI requests
    in the usage totals, without their name."""
    _confirm_password(user, body.password)
    with db.connect() as conn:
        # Lessons companions wrote for the learner's own paths aren't covered by the cascade.
        for r in conn.execute("SELECT id FROM custom_paths WHERE user_id = ?", (user["id"],)).fetchall():
            conn.execute("DELETE FROM lesson_content WHERE lesson_id LIKE ?", (f"custom-{r['id']}-%",))
        conn.execute("DELETE FROM users WHERE id = ?", (user["id"],))
        conn.execute("DELETE FROM login_failures WHERE email = ?", (user["email"].lower(),))
    response.delete_cookie(auth.COOKIE)
