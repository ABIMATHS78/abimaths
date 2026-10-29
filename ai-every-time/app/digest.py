"""Daily reminder email, sent through any SMTP server.

Opt-in per learner, at the hour they chose in their own time zone, and only once they've
confirmed their address (double opt-in), so nobody can sign someone else up for email. The
email lists the same reminders as the in-app "Today" panel and carries a one-click
unsubscribe (RFC 8058) as well as a link to a confirmation page.
"""

from __future__ import annotations

import asyncio
import html
import logging
import secrets
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

from urllib.parse import parse_qs

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse

from app import auth, db, learning, paths, reminders, usage
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/email")
CHECK_EVERY_SECONDS = 15 * 60
CONFIRM_DAYS = 7
CONFIRMATIONS_PER_HOUR = 3
# All the links one address can be sent in a day, whatever accounts ask for them.
ASKED_EMAILS_PER_DAY = 10


def email_available() -> bool:
    return bool(settings.smtp_host and settings.mail_from)


def ensure_token(conn, user_id: int) -> str:
    row = conn.execute("SELECT unsubscribe_token FROM users WHERE id = ?", (user_id,)).fetchone()
    if row["unsubscribe_token"]:
        return row["unsubscribe_token"]
    token = secrets.token_urlsafe(32)
    conn.execute("UPDATE users SET unsubscribe_token = ? WHERE id = ?", (token, user_id))
    return token


def build_message(user: dict, items: list[dict], token: str) -> EmailMessage:
    base = settings.public_url
    unsubscribe = f"{base}/api/email/unsubscribe?token={token}"
    lines = [f"- {r['text']}\n  {paths.absolute(r['link'])}" for r in items]
    msg = EmailMessage()
    msg["Subject"] = f"Today on AI Every Time: {len(items)} thing{'s' if len(items) != 1 else ''} waiting for you"
    msg["From"] = settings.mail_from
    msg["To"] = user["email"]
    msg["List-Unsubscribe"] = f"<{unsubscribe}>"
    msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    msg.set_content(
        f"Hi {user['display_name']},\n\nHere's what's waiting for you today:\n\n" + "\n".join(lines)
        + f"\n\nSee you soon!\n\nYou get this because you turned on email reminders. Stop them: {unsubscribe}\n"
    )
    items_html = "".join(f'<li><a href="{html.escape(paths.absolute(r["link"]))}">{html.escape(r["text"])}</a></li>' for r in items)
    msg.add_alternative(
        f"<p>Hi {html.escape(user['display_name'])},</p><p>Here's what's waiting for you today:</p><ul>{items_html}</ul>"
        f'<p>See you soon!</p><p style="color:#777;font-size:12px">You get this because you turned on email reminders. '
        f'<a href="{html.escape(unsubscribe)}">Stop these emails</a>.</p>',
        subtype="html",
    )
    return msg


def send(msg: EmailMessage) -> None:
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as smtp:
        if settings.smtp_starttls:
            smtp.starttls()
        if settings.smtp_user:
            smtp.login(settings.smtp_user, settings.smtp_password)
        smtp.send_message(msg)


def send_quietly(msg: EmailMessage, user_id: int) -> None:
    """For emails sent after the response: failures are logged, not raised."""
    try:
        send(msg)
    except (smtplib.SMTPException, OSError):
        log.exception("email to user %s failed", user_id)


# ------------------------------------------------------------- confirming addresses


def claim_send(conn, address: str, kind: str, per_hour: int) -> bool:
    """Whether another `kind` email ("confirm" or "reset") may go to this address now, noting it if
    so. Counted by address, not account, so deleting an account and making it again doesn't start
    the count over. Call inside BEGIN IMMEDIATE, so two requests can't both take the last one."""
    now = datetime.now(timezone.utc)
    hour_ago, day_ago = ((now - timedelta(hours=h)).isoformat(timespec="seconds") for h in (1, 24))
    key = auth.hash_token(address.lower())
    conn.execute("DELETE FROM email_sends WHERE created_at < ?", (day_ago,))
    this_hour, today = conn.execute("SELECT COALESCE(SUM(kind = ? AND created_at > ?), 0), COUNT(*) FROM email_sends "
                                    "WHERE address_hash = ?", (kind, hour_ago, key)).fetchone()
    if this_hour >= per_hour or today >= ASKED_EMAILS_PER_DAY:
        return False
    conn.execute("INSERT INTO email_sends (address_hash, kind, created_at) VALUES (?, ?, ?)", (key, kind, db.now()))
    return True


def _what_confirming_does(user: dict | None = None) -> str:
    """Said only to someone confirming would actually help: not, say, a learner with an allowance of their own."""
    text = "It lets AI Every Time send the reminder emails you choose to get."
    raise_to = usage.confirm_would_raise(user) if user else None
    if raise_to:
        text += f" It also raises your AI requests from {usage.allowance(user)} to {raise_to} a day."
    return text


def confirmation_message(user: dict, token: str) -> EmailMessage:
    link = f"{settings.public_url}/api/email/confirm?token={token}"
    why = _what_confirming_does(user)
    msg = EmailMessage()
    msg["Subject"] = "Confirm your email for AI Every Time"
    msg["From"] = settings.mail_from
    msg["To"] = user["email"]
    # No name: the address isn't known to be theirs yet, and the name is whatever was typed at sign-up.
    msg.set_content(
        f"Hello,\n\nPlease confirm that this is your email address on AI Every Time:\n\n{link}\n\n"
        f"{why}\n\nThe link works for {CONFIRM_DAYS} days. "
        "If you didn't make an account on AI Every Time, you can ignore this email.\n"
    )
    msg.add_alternative(
        f"<p>Hello,</p><p>Please confirm that this is your email address on AI Every "
        f'Time.</p><p><a href="{html.escape(link)}">Confirm my email</a></p><p>{html.escape(why)}</p>'
        f"<p>The link works for {CONFIRM_DAYS} days.</p><p style=\"color:#777;font-size:12px\">If you didn't make an "
        "account on AI Every Time, you can ignore this email.</p>",
        subtype="html",
    )
    return msg


def request_confirmation(user: dict, background: BackgroundTasks, per_hour: int = CONFIRMATIONS_PER_HOUR) -> bool:
    """Email a link that confirms the address, unless it's confirmed already or `per_hour` links went
    out in the last hour (links sent without being asked for use 1, so one goes out at a time). The
    email is sent after the response. Returns whether one was sent."""
    if user.get("email_confirmed_at") or not email_available():
        return False
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if not claim_send(conn, user["email"], "confirm", per_hour):
            return False
        token = secrets.token_urlsafe(32)
        expires = (datetime.now(timezone.utc) + timedelta(days=CONFIRM_DAYS)).isoformat(timespec="seconds")
        conn.execute("INSERT INTO email_confirmations (token_hash, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
                     (auth.hash_token(token), user["id"], db.now(), expires))
    background.add_task(send_quietly, confirmation_message(user, token), user["id"])
    return True


def mark_confirmed(conn, user_id: int) -> None:
    """The learner has shown they read mail at their address (a confirmation or password-reset link). If a friend
    invited them, the invite now counts."""
    from app import referrals  # imported here: referrals uses this module

    conn.execute("UPDATE users SET email_confirmed_at = COALESCE(email_confirmed_at, ?) WHERE id = ?", (db.now(), user_id))
    conn.execute("UPDATE email_confirmations SET used_at = ? WHERE user_id = ? AND used_at IS NULL", (db.now(), user_id))
    referrals.qualify(conn, user_id)


def confirm(token: str) -> bool:
    if not token:
        return False
    with db.connect() as conn:
        row = conn.execute("SELECT user_id FROM email_confirmations WHERE token_hash = ? AND used_at IS NULL AND expires_at > ?",
                           (auth.hash_token(token), db.now())).fetchone()
        if row is None:
            return False
        mark_confirmed(conn, row["user_id"])
    return True


@router.post("/confirmation", status_code=202)
def send_confirmation(background: BackgroundTasks, user: dict = Depends(auth.current_user)):
    """Send (again) the link that confirms the learner's email address."""
    if not email_available():
        raise HTTPException(409, "Email isn't set up on this server yet")
    if user["email_confirmed_at"]:
        return {"sent": False, "confirmed": True}
    if not request_confirmation(user, background):
        raise HTTPException(429, "A few links went out in the last hour. Check your inbox and spam folder, or try again later.")
    return {"sent": True, "confirmed": False}


def due_digests(now: datetime | None = None) -> list[tuple[dict, list[dict], str]]:
    """Learners who should get today's email, with their reminders. Each is claimed for their day
    before it's sent, so it goes out at most once a day."""
    now = now or datetime.now(timezone.utc)
    out, users = [], []
    with db.connect() as conn:
        for u in [dict(r) for r in conn.execute(
                "SELECT * FROM users WHERE email_reminders = 1 AND email_confirmed_at IS NOT NULL")]:
            local = learning.local_now(u, now)
            today = local.date().isoformat()
            if local.hour >= reminders.reminder_hour(u) and (u["last_digest_on"] or "") < today:
                if conn.execute("UPDATE users SET last_digest_on = ? WHERE id = ? AND COALESCE(last_digest_on, '') < ?",
                                (today, u["id"], today)).rowcount:
                    users.append(u)
    for u in users:
        try:
            # Notices, streaks and due work matter in an email; new posts alone aren't worth one.
            items = [r for r in reminders.reminders(u) if r["kind"] != "new_posts"]
        except Exception:
            log.exception("reminders for user %s failed", u["id"])
            continue
        if items:
            with db.connect() as conn:
                out.append((u, items, ensure_token(conn, u["id"])))
    return out


async def digest_loop() -> None:
    while True:
        try:
            for user, items, token in due_digests():
                try:
                    await asyncio.to_thread(send, build_message(user, items, token))
                except (smtplib.SMTPException, OSError):
                    log.exception("reminder email to user %s failed", user["id"])
        except Exception:
            log.exception("reminder email run failed")
        await asyncio.sleep(CHECK_EVERY_SECONDS)


# The emails a learner can stop with the link in them: reminders (this module's), and news and offers (app/campaigns.py).
LISTS = {"reminders": ("email_reminders", "reminder emails"), "news": ("marketing_emails", "emails with news and offers")}


def _unsubscribe(token: str, which: str) -> bool:
    if not token:
        return False
    with db.connect() as conn:
        cur = conn.execute(f"UPDATE users SET {LISTS[which][0]} = 0 WHERE unsubscribe_token = ?", (token,))
    return cur.rowcount > 0


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><link rel="icon" href="/static/icons/favicon-32.png" sizes="32x32" type="image/png">
<body style="font:16px system-ui;max-width:480px;margin:48px auto;padding:0 16px">
<p><img src="/static/icons/logo-256.webp" width="56" height="56" alt="AI Every Time"></p>{body}</body></html>"""


def _page(body: str, title: str = "Email reminders", status: int = 200) -> HTMLResponse:
    return HTMLResponse(PAGE.format(title=title, body=body), status_code=status)


@router.get("/confirm", response_class=HTMLResponse)
def confirm_page(token: str = ""):
    # A button rather than confirming on GET, so link scanners in mail systems can't confirm for someone.
    return _page(f"""<h1>Confirm your email?</h1>
<p>This tells AI Every Time the address is yours. {html.escape(_what_confirming_does())}</p>
<form method="post" action="/api/email/confirm"><input type="hidden" name="token" value="{html.escape(token)}">
<button style="font:inherit;padding:8px 16px">Confirm my email</button></form>""", "Confirm your email")


@router.post("/confirm", response_class=HTMLResponse)
async def confirm_submit(request: Request):
    body = parse_qs((await request.body())[:4096].decode("utf-8", "replace"))
    if confirm((body.get("token") or [""])[0]):
        return _page("<h1>Thanks, it's confirmed</h1><p>If you turned on reminder emails, they'll come at the time you "
                     "chose.</p><p><a href='/profile'>Back to AI Every Time</a></p>", "Email confirmed")
    return _page("<h1>Link not recognised</h1><p>It may have expired or been used already. You can ask for a new one "
                 "in your profile.</p>", "Link not recognised", 400)


def _list(name: str) -> str:
    return name if name in LISTS else "reminders"


@router.get("/unsubscribe", response_class=HTMLResponse)
def unsubscribe_page(token: str = "", list: str = "reminders"):
    # A button rather than unsubscribing on GET, so link scanners in mail systems can't do it by accident.
    which = _list(list)
    return _page(f"""<h1>Stop {LISTS[which][1]}?</h1>
<form method="post" action="/api/email/unsubscribe"><input type="hidden" name="token" value="{html.escape(token)}">
<input type="hidden" name="list" value="{which}">
<button style="font:inherit;padding:8px 16px">Stop emails</button></form>
<p>You can turn them back on any time in your profile.</p>""")


@router.post("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe(request: Request, token: str = "", list: str = ""):
    """Handles both the confirmation form (token in the body) and mail clients' one-click POST (token in the URL)."""
    body = parse_qs((await request.body())[:4096].decode("utf-8", "replace"))
    token = token or (body.get("token") or [""])[0]
    which = _list(list or (body.get("list") or [""])[0])
    if _unsubscribe(token, which):
        return _page(f"<h1>Done</h1><p>You won't get {LISTS[which][1]} any more.</p>"
                     "<p><a href='/'>Back to AI Every Time</a></p>")
    return _page("<h1>Link not recognised</h1><p>Turn emails off in your profile instead.</p>", status=400)
