from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Cookie, Depends, HTTPException

from app import db
from app.config import settings

COOKIE = "aiet_session"


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(digest.hex(), digest_hex)


def create_session(user_id: int) -> str:
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=settings.session_days)
    with db.connect() as conn:
        conn.execute(
            "INSERT INTO sessions (token, user_id, expires_at) VALUES (?, ?, ?)",
            (token, user_id, expires.isoformat(timespec="seconds")),
        )
    return token


def delete_session(token: str) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM sessions WHERE token = ?", (token,))


def current_user(aiet_session: str | None = Cookie(default=None)) -> dict:
    """FastAPI dependency: the signed-in user, or 401."""
    if not aiet_session:
        raise HTTPException(401, "Not signed in")
    with db.connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id WHERE s.token = ? AND s.expires_at > ?",
            (aiet_session, db.now()),
        ).fetchone()
    if row is None:
        raise HTTPException(401, "Session expired")
    return dict(row)


def optional_user(aiet_session: str | None = Cookie(default=None)) -> dict | None:
    """FastAPI dependency: the signed-in user, or None for anonymous visitors."""
    try:
        return current_user(aiet_session)
    except HTTPException:
        return None


ROLES = ("learner", "reviewer", "moderator", "admin")
# What each staff role may do. Admins can do everything, including assigning roles.
CAN = {
    "moderate": {"moderator", "admin"},  # review reported community companions
    "review": {"reviewer", "admin"},     # expert-review learner-made paths before they're shared
    "admin": {"admin"},
}


def role_of(u: dict) -> str:
    """Emails in AIET_ADMIN_EMAILS are always admins, so a deployment can't lock itself out."""
    return "admin" if u["email"].lower() in settings.admin_emails else (u.get("role") or "learner")


def require(permission: str):
    """FastAPI dependency factory: the signed-in user, if their role allows `permission`."""

    def dep(user: dict = Depends(current_user)) -> dict:
        if role_of(user) not in CAN[permission]:
            raise HTTPException(403, "You don't have permission to do that")
        return user

    return dep


def public_user(u: dict) -> dict:
    return {k: u[k] for k in ("id", "email", "display_name", "level", "goals", "primary_companion_id", "onboarded")} | {
        "onboarded": bool(u["onboarded"]), "role": role_of(u), "email_reminders": bool(u.get("email_reminders")),
    }
