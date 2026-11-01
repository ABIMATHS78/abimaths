"""Staff tokens: keys a tool, such as Claude Code, uses to work the Staff pages' API for an admin, without their
password. An admin makes one on the Staff page's Setup tab and sees it once; only its hash is kept. A token works
only on /api/staff/ (and /api/me, to say whose it is), never to make or cancel tokens, for the admin who made it and
only while they're an admin, until it expires or is cancelled. Changing the admin's password, or signing out their
other devices, cancels it too. Every change made with one is in the decision log, with the token's name.

Tools send it as a header:  Authorization: Bearer aiet_..."""

from __future__ import annotations

import secrets
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth, db, staff

router = APIRouter(prefix="/api/staff/tokens")
Admin = Depends(auth.require("admin"))

# The token on the request in hand, for the decision log (set by Middleware, as the request comes in).
current: ContextVar[str | None] = ContextVar("staff_token", default=None)


class Middleware:
    """Notes the staff token a request carries, so staff.log can say which token made a change. Whether the
    token is any good is auth.current_user's to decide."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        token = None
        if scope["type"] == "http":
            token = auth.bearer(dict(scope["headers"]).get(b"authorization", b"").decode("latin-1"))
        if token is None:
            return await self.app(scope, receive, send)
        reset = current.set(token)
        try:
            await self.app(scope, receive, send)
        finally:
            current.reset(reset)


def in_use(conn) -> str | None:
    """The name of the staff token the request in hand was made with, if any."""
    token = current.get()
    if token is None:
        return None
    row = conn.execute("SELECT name FROM staff_tokens WHERE token_hash = ?", (auth.hash_token(token),)).fetchone()
    return row[0] if row else None


class NewToken(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    days: Literal[7, 30, 90, 365] = 90


@router.get("")
def list_tokens(user: dict = Admin):
    """The tokens that still work, every admin's, so any admin can cancel one."""
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(
            "SELECT t.id, t.name, t.created_at, t.last_used_at, t.expires_at, u.display_name AS admin, "
            "t.user_id = ? AS mine FROM staff_tokens t JOIN users u ON u.id = t.user_id "
            "WHERE t.revoked_at IS NULL AND t.expires_at > ? ORDER BY t.id DESC", (user["id"], db.now()))]


@router.post("", status_code=201)
def make_token(body: NewToken, user: dict = Admin):
    """A new token for the admin making it. The token itself is in this answer only: it can't be shown again."""
    token = f"{auth.TOKEN_PREFIX}{secrets.token_urlsafe(32)}"
    name = body.name.strip()
    if not name:
        raise HTTPException(422, "Give the token a name, such as the tool that will use it")
    expires = (datetime.now(timezone.utc) + timedelta(days=body.days)).isoformat(timespec="seconds")
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO staff_tokens (user_id, name, token_hash, created_at, expires_at) "
                           "VALUES (?, ?, ?, ?, ?)", (user["id"], name, auth.hash_token(token), db.now(), expires))
        # In the decision log against the admin it's for: the log's entries are about people, companions and paths.
        staff.log(conn, user["id"], "user", user["id"], "token:make", f"“{name}”, for {body.days} days")
    return {"id": cur.lastrowid, "name": name, "token": token, "expires_at": expires}


@router.delete("/{token_id}", status_code=204)
def cancel_token(token_id: int, user: dict = Admin):
    with db.connect() as conn:
        row = conn.execute("SELECT name, user_id FROM staff_tokens WHERE id = ? AND revoked_at IS NULL",
                           (token_id,)).fetchone()
        if row is None:
            raise HTTPException(404, "There's no such token, or it was already cancelled")
        conn.execute("UPDATE staff_tokens SET revoked_at = ? WHERE id = ?", (db.now(), token_id))
        staff.log(conn, user["id"], "user", row["user_id"], "token:cancel", f"“{row['name']}”")
