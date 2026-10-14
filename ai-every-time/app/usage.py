"""The daily AI allowance, and a record of every AI request.

Each request a learner makes of Claude (a chat message, a new lesson, a quiz...) costs units
against an allowance over any 24 hours, so the cost of running the site stays predictable when
anyone can sign up. The cost is reserved before Claude is called and refunded if the call to
Claude doesn't go through. An answer that came back but couldn't be used still counts, so
failures can't be turned into free requests. Learners who haven't confirmed their email address
get a smaller allowance while email is set up, admins have none, and an optional ceiling covers
everyone together.

Every request is recorded with the tokens Claude reported, and requests turned away are counted,
for the admin Insights page. Calls to Claude find their record through a context variable, so
code that asks Claude doesn't need to know about allowances.
"""

from __future__ import annotations

import logging
import math
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone

from fastapi import HTTPException

from app import auth, db
from app.config import settings

log = logging.getLogger(__name__)
WINDOW_HOURS = 24
# What each kind of request costs. Mock exams and custom paths ask much more of Claude, so they count double.
COSTS = {"chat": 1, "lesson": 1, "quiz": 1, "exam": 2, "flashcards": 1, "summary": 1, "plan": 1, "path": 2, "post": 1}
NAMES = {"exam": "A mock exam", "path": "A new learning path"}
BUSY = "AI Every Time is very busy right now, so new AI requests are paused. Please try again in an hour or so."

# The usage record that tokens from Claude responses belong to, set around each request.
_current: ContextVar[int | None] = ContextVar("ai_usage_record", default=None)


def _since() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat(timespec="seconds")


def allowance(user: dict) -> int | None:
    """Units the learner may use in any 24 hours; None means no limit."""
    from app import digest  # imported here: digest is further up the chain of modules than this one

    if auth.role_of(user) == "admin" or settings.ai_allowance <= 0:
        return None
    if digest.email_available() and not user.get("email_confirmed_at"):
        return min(settings.ai_allowance_unconfirmed, settings.ai_allowance)
    return settings.ai_allowance


def _spent(conn, user_id: int) -> list:
    """The learner's requests that still count, oldest first."""
    return conn.execute("SELECT units, created_at FROM ai_usage WHERE user_id = ? AND units > 0 AND created_at > ? "
                        "ORDER BY id", (user_id, _since())).fetchall()


def status(user: dict) -> dict:
    """For the profile: the allowance, what's left of it, and what confirming the email address would raise it to."""
    limit = allowance(user)
    with db.connect() as conn:
        used = sum(r["units"] for r in _spent(conn, user["id"]))
    return {"limit": limit, "used": used, "remaining": None if limit is None else max(0, limit - used),
            "confirm_raises_to": settings.ai_allowance if limit is not None and limit < settings.ai_allowance else None}


def _in_words(wait: timedelta) -> str:
    minutes = max(1, math.ceil(wait.total_seconds() / 60))
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours = round(minutes / 60)
    return f"about {hours} hour{'' if hours == 1 else 's'}"


def _refusal(kind: str, limit: int, rows: list) -> tuple[str, timedelta]:
    """What to tell a learner whose allowance doesn't cover this request, and how long until it will."""
    cost, used = COSTS[kind], sum(r["units"] for r in rows)
    freed, wait = 0, None
    for r in rows:  # oldest first: when enough earlier requests stop counting
        freed += r["units"]
        if used - freed + cost <= limit:
            wait = datetime.fromisoformat(r["created_at"]) + timedelta(hours=WINDOW_HOURS) - datetime.now(timezone.utc)
            break
    if wait is None:  # it costs more than the whole allowance
        return f"{NAMES.get(kind, 'This')} needs {cost} AI requests, more than your allowance of {limit}.", timedelta(hours=WINDOW_HOURS)
    if used >= limit:
        text = f"You've reached your limit of {limit} AI requests in 24 hours."
    else:
        text = f"{NAMES.get(kind, 'This request')} counts as {cost} AI requests, and you have {limit - used} left."
    text += f" You can ask again in {_in_words(wait)}."
    if limit < settings.ai_allowance:
        text += f" Confirm your email address to raise your limit to {settings.ai_allowance}."
    return text, wait


def reserve(user: dict, kind: str) -> int:
    """Count one AI request against the allowance and record it, or turn it away with a 429 that
    says when the learner can ask again. Returns the id of the request's record."""
    cost, limit, refusal = COSTS[kind], allowance(user), None
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # check and record in one go, so parallel requests can't both squeeze in
        if limit is not None:
            rows = _spent(conn, user["id"])
            if sum(r["units"] for r in rows) + cost > limit:
                refusal = ("allowance", *_refusal(kind, limit, rows))
        if refusal is None and settings.ai_site_limit > 0 and auth.role_of(user) != "admin":
            total = conn.execute("SELECT COALESCE(SUM(units), 0) FROM ai_usage WHERE created_at > ?", (_since(),)).fetchone()[0]
            if total + cost > settings.ai_site_limit:
                refusal = ("site", BUSY, timedelta(hours=1))
        if refusal:
            # Counted per learner per hour, so a stream of refused requests takes almost no room.
            conn.execute("INSERT INTO ai_refusals (hour, user_id, reason, count) VALUES (?, ?, ?, 1) "
                         "ON CONFLICT (hour, user_id, reason) DO UPDATE SET count = count + 1",
                         (db.now()[:13], user["id"], refusal[0]))
        else:
            row = conn.execute("INSERT INTO ai_usage (user_id, kind, units, created_at) VALUES (?, ?, ?, ?)",
                               (user["id"], kind, cost, db.now())).lastrowid
    if refusal:  # raised once the refusal is counted
        _, text, wait = refusal
        raise HTTPException(429, text, headers={"Retry-After": str(max(60, math.ceil(wait.total_seconds())))})
    return row


def try_reserve(user: dict, kind: str) -> int | None:
    """reserve, for optional work such as a new companion's first post: None instead of a 429."""
    try:
        return reserve(user, kind)
    except HTTPException:
        return None


def refund(row: int) -> None:
    """Stop a request counting against the allowance. Any tokens it used stay on record."""
    with db.connect() as conn:
        conn.execute("UPDATE ai_usage SET units = 0, failed = 1 WHERE id = ?", (row,))


def failed() -> None:
    """The call to Claude for the request in hand didn't go through, so the request doesn't count."""
    row = _current.get()
    if row is not None:
        refund(row)


@contextmanager
def recording(row: int | None):
    """Claude responses inside this block add their tokens to the record `row`."""
    token = _current.set(row)
    try:
        yield
    finally:
        try:
            _current.reset(token)
        except ValueError:  # finished in another context, as a reply stream is when the learner leaves mid-reply
            pass


@contextmanager
def spend(user: dict, kind: str):
    """Count one AI request against the allowance around the code that makes it."""
    with recording(reserve(user, kind)):
        yield


def record_tokens(response) -> None:
    """Add the tokens a Claude response used to the record of the request in hand. With a server-side
    fallback or another server-side step, every attempt is itemised in usage.iterations and counts
    (the top-level usage covers only the attempt that answered). Work no learner asked for, such as
    scheduled posts, gets a record of its own."""
    try:
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        steps = getattr(usage, "iterations", None) or []
        counts = tuple(sum(getattr(s, f, 0) or 0 for s in (steps or [usage])) for f in
                       ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        fallback = int(any(getattr(s, "type", "") == "fallback_message" for s in steps))
        model, row = getattr(response, "model", "") or "", _current.get()
        with db.connect() as conn:
            if row is None:
                conn.execute("INSERT INTO ai_usage (user_id, kind, input_tokens, output_tokens, cache_read_tokens, "
                             "cache_write_tokens, model, fallback, created_at) VALUES (NULL, 'automatic', ?, ?, ?, ?, ?, ?, ?)",
                             (*counts, model, fallback, db.now()))
            else:
                conn.execute("UPDATE ai_usage SET input_tokens = input_tokens + ?, output_tokens = output_tokens + ?, "
                             "cache_read_tokens = cache_read_tokens + ?, cache_write_tokens = cache_write_tokens + ?, "
                             "model = CASE WHEN model = '' THEN ? ELSE model END, fallback = MAX(fallback, ?) WHERE id = ?",
                             (*counts, model, fallback, row))
    except Exception:  # bookkeeping must never break the learner's request
        log.exception("recording AI usage failed")
