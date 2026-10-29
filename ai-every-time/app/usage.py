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
# What each kind of request costs. Mock exams, custom paths and project reviews ask much more of Claude, so they count double.
COSTS = {"chat": 1, "lesson": 1, "quiz": 1, "exam": 2, "flashcards": 1, "summary": 1, "plan": 1, "path": 2, "post": 1,
         # A companion starting a chat: its idea, not the learner's request, so it costs them nothing (once per chat).
         "opening": 0, "lab": 1, "project": 2}
NAMES = {"exam": "A mock exam", "path": "A new learning path", "project": "A project review"}
BUSY = "AI Every Time is very busy right now, so new AI requests are paused. Please try again in an hour or so."
# Nothing a learner writes goes to Claude until they've agreed to it (Apple's guideline 5.1.2(i) asks for this too).
CONSENT = ("Companions are Claude, an AI made by Anthropic, so what you write to them goes to Anthropic. "
           "Agree to that to carry on.")
BUSY_UNCONFIRMED = ("AI Every Time is very busy right now, so AI requests are paused for accounts that haven't confirmed "
                    "their email address. Confirm yours from your profile to carry on, or try again in an hour or so.")

# The usage record that tokens from Claude responses belong to, set around each request.
_current: ContextVar[int | None] = ContextVar("ai_usage_record", default=None)


def _since() -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)).isoformat(timespec="seconds")


def _ceiling_message(user: dict) -> str | None:
    """What the site ceiling tells this learner once it's reached, or None if it doesn't hold them. While
    email is set up it holds only learners who haven't confirmed their address, so a flood of
    throwaway accounts can't pause everyone else. Admins are never held."""
    from app import digest

    if auth.role_of(user) == "admin":
        return None
    if not digest.email_available():
        return BUSY
    return None if user.get("email_confirmed_at") else BUSY_UNCONFIRMED


def unconfirmed_allowance(full: int | None = None) -> int | None:
    """The smaller allowance for learners who haven't confirmed their email address, if there is one:
    only while email is set up, and only when it's below the full allowance (0 turns it off)."""
    from app import digest  # imported here: digest is further up the chain of modules than this one

    lower, full = settings.ai_allowance_unconfirmed, settings.ai_allowance if full is None else full
    return lower if digest.email_available() and 0 < lower < full else None


def _full(user: dict) -> tuple[int, bool]:
    """The learner's allowance as their email address being confirmed or not doesn't change it: their plan's while
    the paywall is on, else the site's (0 for no limit). And whether confirming matters: learners who pay don't
    need to confirm to have their plan's."""
    if settings.paywall:
        from app import billing, plans  # imported here: billing is further up the chain of modules too

        tier = billing.tier(user)
        return plans.plan(tier).daily_allowance(), tier == "free"
    return settings.ai_allowance, True


def allowance(user: dict) -> int | None:
    """Units the learner may use in any 24 hours; None means no limit. An admin can give a learner an
    allowance of their own (0 for no limit), which takes the place of their plan's or the site's."""
    if auth.role_of(user) == "admin":
        return None
    if user.get("ai_allowance") is not None:
        return user["ai_allowance"] or None
    full, confirming_matters = _full(user)
    if full <= 0:
        return None
    lower = unconfirmed_allowance(full) if confirming_matters else None
    return lower if lower is not None and not user.get("email_confirmed_at") else full


def confirm_would_raise(user: dict) -> int | None:
    """The allowance the learner would have after confirming their email address, if that's more."""
    if auth.role_of(user) == "admin" or user.get("ai_allowance") is not None or user.get("email_confirmed_at"):
        return None
    full, confirming_matters = _full(user)
    if full <= 0 or not confirming_matters or unconfirmed_allowance(full) is None:
        return None
    return full


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
            "confirm_raises_to": confirm_would_raise(user)}


def _in_words(wait: timedelta) -> str:
    minutes = max(1, math.ceil(wait.total_seconds() / 60))
    if minutes < 60:
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    hours = round(minutes / 60)
    return f"about {hours} hour{'' if hours == 1 else 's'}"


def _requests(n: int) -> str:
    return f"{n} AI request{'' if n == 1 else 's'}"


def _refusal(kind: str, limit: int, rows: list, raise_to: int | None) -> tuple[str, timedelta]:
    """What to tell a learner whose allowance doesn't cover this request, and how long until it will."""
    cost, used = COSTS[kind], sum(r["units"] for r in rows)
    freed, wait = 0, None
    for r in rows:  # oldest first: when enough earlier requests stop counting
        freed += r["units"]
        if used - freed + cost <= limit:
            wait = datetime.fromisoformat(r["created_at"]) + timedelta(hours=WINDOW_HOURS) - datetime.now(timezone.utc)
            break
    if wait is None:  # it costs more than the whole allowance
        text, wait = f"{NAMES.get(kind, 'This')} needs {_requests(cost)}, more than your limit of {limit}.", timedelta(hours=WINDOW_HOURS)
    else:
        text = (f"You've reached your limit of {_requests(limit)} in 24 hours." if used >= limit else
                f"{NAMES.get(kind, 'This request')} counts as {_requests(cost)}, and you have {limit - used} left.")
        text += f" You can ask again in {_in_words(wait)}."
    if raise_to:
        text += (f" Confirm your email address to raise your limit to {raise_to}: you can send yourself "
                 "a link from your profile.")
    return text, wait


def reserve(user: dict, kind: str) -> int:
    """Count one AI request against the allowance and record it, or turn it away with a 429 that
    says when the learner can ask again. Returns the id of the request's record. A learner who hasn't
    agreed to their words going to Claude is turned away first, with a 403 the page answers by asking."""
    if not user.get("ai_consent_at"):
        raise HTTPException(403, CONSENT, headers={"X-AI-Consent": "needed"})
    cost, limit, refusal = COSTS[kind], allowance(user), None
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")  # check and record in one go, so parallel requests can't both squeeze in
        if limit is not None:
            rows = _spent(conn, user["id"])
            if sum(r["units"] for r in rows) + cost > limit:
                refusal = ("allowance", *_refusal(kind, limit, rows, confirm_would_raise(user)))
        busy = _ceiling_message(user) if settings.ai_site_limit > 0 else None
        if refusal is None and busy:
            total = conn.execute("SELECT COALESCE(SUM(units), 0) FROM ai_usage WHERE created_at > ?", (_since(),)).fetchone()[0]
            if total + cost > settings.ai_site_limit:
                refusal = ("site", busy, timedelta(hours=1))
        if refusal:
            # Counted per learner per hour, so a stream of refused requests takes almost no room.
            conn.execute("INSERT INTO ai_refusals (hour, user_id, reason, count) VALUES (?, ?, ?, 1) "
                         "ON CONFLICT (hour, user_id, reason) DO UPDATE SET count = count + 1",
                         (db.now()[:13], user["id"], refusal[0]))
        else:
            row = conn.execute("INSERT INTO ai_usage (user_id, kind, units, created_at) VALUES (?, ?, ?, ?)",
                               (user["id"], kind, cost, db.now())).lastrowid
    if refusal:  # raised once the refusal is counted
        reason, text, wait = refusal
        headers = {"Retry-After": str(max(60, math.ceil(wait.total_seconds())))}
        if reason == "allowance" and (more := more_with(user)):
            text += f" {more.name} gives you {more.daily_allowance()} a day."
            headers["X-Plan-Needed"] = more.slug
        raise HTTPException(429, text, headers=headers)
    return row


def more_with(user: dict):
    """While the paywall is on, the next plan up that would give the learner more AI requests, if there is one."""
    if not settings.paywall or user.get("ai_allowance") is not None:
        return None
    from app import billing, plans

    mine = plans.plan(billing.tier(user))
    return next((p for p in plans.PLANS.values() if p.slug in plans.SELF_SERVE and p.rank > mine.rank
                 and (p.daily_allowance() == 0 or p.daily_allowance() > mine.daily_allowance() > 0)), None)


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
