"""Confirming email addresses (double opt-in) before any reminder email is sent."""

import re
from datetime import datetime, timezone

import pytest

from tests.test_app import signup

LINKS = {"reset": r"#/reset/([\w-]+)", "confirm": r"/api/email/confirm\?token=([\w-]+)"}


class Mailbox:
    """Stands in for an SMTP server and keeps what was sent."""

    def __init__(self):
        self.sent = []

    def __call__(self, *args, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        pass

    def send_message(self, msg):
        self.sent.append(msg)

    def link_token(self, kind: str, i: int = -1) -> str:
        """The token in the emailed link of that kind ("reset" or "confirm")."""
        return re.search(LINKS[kind], self.sent[i].get_body(("plain",)).get_content()).group(1)

    def reset_link(self, i: int = -1) -> str:
        return self.link_token("reset", i)


@pytest.fixture()
def mailbox(client, monkeypatch):
    from app import digest
    from app.config import settings

    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "mail_from", "AI Every Time <hello@aieverytime.example>")
    monkeypatch.setattr(settings, "public_url", "https://aieverytime.example")
    box = Mailbox()
    monkeypatch.setattr(digest.smtplib, "SMTP", box)
    return box


MORNING = datetime(2026, 1, 1, 9, 0, tzinfo=timezone.utc)


def test_reminder_emails_wait_for_a_confirmed_address(client, mailbox, monkeypatch):
    from app import db, digest
    from app.config import settings

    monkeypatch.setattr(settings, "reminder_hour", 7)
    assert signup(client)["email_confirmed"] is False
    [msg] = mailbox.sent  # sign-up sends the link
    assert msg["To"] == "ada@example.com" and msg["Subject"] == "Confirm your email for AI Every Time"
    assert "raises your AI requests from 30 to 100 a day" in msg.get_body(("plain",)).get_content()
    client.get("/api/lessons/tokens")  # something to remind about
    assert client.patch("/api/me", json={"email_reminders": True}).json()["email_reminders"]
    assert len(mailbox.sent) == 1  # one link at a time unless the learner asks for another
    token = mailbox.link_token("confirm")
    with db.connect() as conn:  # only a hash of the token is kept
        assert conn.execute("SELECT COUNT(*) FROM email_confirmations WHERE token_hash = ?", (token,)).fetchone()[0] == 0
    assert digest.due_digests(MORNING) == []  # nothing until it's confirmed
    assert any(r["kind"] == "confirm_email" for r in client.get("/api/reminders").json())

    # Opening the link shows a button; mail scanners that fetch links don't confirm anything.
    page = client.get(f"/api/email/confirm?token={token}")
    assert page.status_code == 200 and "Confirm my email" in page.text and token in page.text
    assert client.get("/api/me").json()["email_confirmed"] is False
    assert client.post("/api/email/confirm", content="token=nope").status_code == 400
    assert client.post("/api/email/confirm", content=f"token={token}").status_code == 200
    assert client.get("/api/me").json()["email_confirmed"] is True
    assert not any(r["kind"] == "confirm_email" for r in client.get("/api/reminders").json())
    assert client.post("/api/email/confirm", content=f"token={token}").status_code == 400  # used

    [(user, items, _)] = digest.due_digests(MORNING)
    assert user["email"] == "ada@example.com" and any(i["kind"] == "continue_lesson" for i in items)
    assert client.post("/api/email/confirmation").json() == {"sent": False, "confirmed": True}
    # Turning reminders off and on again doesn't ask for another confirmation.
    client.patch("/api/me", json={"email_reminders": False})
    client.patch("/api/me", json={"email_reminders": True})
    assert len(mailbox.sent) == 1


def test_confirmation_links_expire_and_are_limited(client, mailbox):
    from app import db, digest

    signup(client)  # sends the first
    for _ in range(digest.CONFIRMATIONS_PER_HOUR - 1):
        assert client.post("/api/email/confirmation").json() == {"sent": True, "confirmed": False}
    r = client.post("/api/email/confirmation")
    assert r.status_code == 429 and "inbox" in r.json()["detail"]
    assert len(mailbox.sent) == digest.CONFIRMATIONS_PER_HOUR

    with db.connect() as conn:
        conn.execute("UPDATE email_confirmations SET expires_at = '2000-01-01T00:00:00+00:00', "
                     "created_at = '2000-01-01T00:00:00+00:00'")
    assert client.post("/api/email/confirm", content=f"token={mailbox.link_token('confirm')}").status_code == 400
    client.post("/api/email/confirmation")  # the hour has passed, so a new link can go out
    assert client.post("/api/email/confirm", content=f"token={mailbox.link_token('confirm')}").status_code == 200
    # Confirming spends any other links still out there.
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM email_confirmations WHERE used_at IS NULL").fetchone()[0] == 0


def test_resetting_the_password_confirms_the_address(client, mailbox):
    signup(client)
    client.post("/api/auth/logout")
    client.post("/api/auth/forgot", json={"email": "ada@example.com"})
    assert client.post("/api/auth/reset", json={"token": mailbox.reset_link(), "password": "a new password"}).status_code == 204
    assert client.get("/api/me").json()["email_confirmed"] is True


def test_confirmation_needs_email_set_up(client):
    signup(client)
    assert client.post("/api/email/confirmation").status_code == 409


def test_learners_with_reminder_emails_before_confirmation_keep_them(tmp_path, monkeypatch):
    """An existing database: learners who had already turned emails on are treated as confirmed."""
    from app import db
    from app.config import settings

    monkeypatch.setattr(settings, "database_path", str(tmp_path / "old.db"))
    db.init_db()
    with db.connect() as conn:
        conn.execute("ALTER TABLE users DROP COLUMN email_confirmed_at")  # as it was before this release
        for email, on in (("on@example.com", 1), ("off@example.com", 0)):
            conn.execute("INSERT INTO users (email, display_name, password_hash, email_reminders, created_at) "
                         "VALUES (?, 'X', 'x', ?, '2026-01-01T00:00:00+00:00')", (email, on))
    db.init_db()
    with db.connect() as conn:
        confirmed = dict(conn.execute("SELECT email, email_confirmed_at FROM users").fetchall())
    assert confirmed == {"on@example.com": "2026-01-01T00:00:00+00:00", "off@example.com": None}
    db.init_db()  # still idempotent
