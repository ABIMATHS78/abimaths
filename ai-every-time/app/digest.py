"""Daily reminder email, sent through any SMTP server.

Opt-in per learner, at the hour they chose in their own time zone. The email lists the
same reminders as the in-app "Today" panel and carries a one-click unsubscribe (RFC 8058)
as well as a link to a confirmation page.
"""

from __future__ import annotations

import asyncio
import html
import logging
import secrets
import smtplib
from datetime import datetime, timezone
from email.message import EmailMessage

from urllib.parse import parse_qs

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from app import db, learning, reminders
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/email")
CHECK_EVERY_SECONDS = 15 * 60


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
    lines = [f"- {r['text']}\n  {base}/{r['link']}" for r in items]
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
    items_html = "".join(f'<li><a href="{html.escape(base + "/" + r["link"])}">{html.escape(r["text"])}</a></li>' for r in items)
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


def due_digests(now: datetime | None = None) -> list[tuple[dict, list[dict], str]]:
    """Learners who should get today's email, with their reminders. Each is claimed for their day
    before it's sent, so it goes out at most once a day."""
    now = now or datetime.now(timezone.utc)
    out, users = [], []
    with db.connect() as conn:
        for u in [dict(r) for r in conn.execute("SELECT * FROM users WHERE email_reminders = 1")]:
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


def _unsubscribe(token: str) -> bool:
    if not token:
        return False
    with db.connect() as conn:
        cur = conn.execute("UPDATE users SET email_reminders = 0 WHERE unsubscribe_token = ?", (token,))
    return cur.rowcount > 0


PAGE = """<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Email reminders</title><body style="font:16px system-ui;max-width:480px;margin:48px auto;padding:0 16px">{body}</body></html>"""


@router.get("/unsubscribe", response_class=HTMLResponse)
def unsubscribe_page(token: str = ""):
    # A button rather than unsubscribing on GET, so link scanners in mail systems can't do it by accident.
    return PAGE.format(body=f"""<h1>Stop reminder emails?</h1>
<form method="post" action="/api/email/unsubscribe"><input type="hidden" name="token" value="{html.escape(token)}">
<button style="font:inherit;padding:8px 16px">Stop emails</button></form>
<p>You can turn them back on any time in your profile.</p>""")


@router.post("/unsubscribe", response_class=HTMLResponse)
async def unsubscribe(request: Request, token: str = ""):
    """Handles both the confirmation form (token in the body) and mail clients' one-click POST (token in the URL)."""
    body = parse_qs((await request.body())[:4096].decode("utf-8", "replace"))
    token = token or (body.get("token") or [""])[0]
    if _unsubscribe(token):
        return PAGE.format(body="<h1>Done</h1><p>You won't get reminder emails any more.</p><p><a href='/'>Back to AI Every Time</a></p>")
    return HTMLResponse(PAGE.format(body="<h1>Link not recognised</h1><p>Turn emails off in your profile instead.</p>"), status_code=400)
