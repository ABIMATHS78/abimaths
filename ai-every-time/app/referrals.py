"""Referrals: learners invite friends with a link, and every three friends who join earn them a free month of Plus.

Each learner's link carries a code of their own (/r/<code>), made when they first open the Invite page. A friend who
signs up through it is their invite. It counts once the friend confirms their email address, so made-up addresses earn
nothing (and without email set up, nothing counts); an address that's another invitee's or the learner's own with a
+tag (or, for Gmail, dots) added doesn't count again. Every three that count earn a reward, claimed when the learner
likes: a month of Plus, kept apart from any plan an admin gave them, or for a learner who already pays for Plus or
more, a month of Plus off their next bill. Rewards are capped at twelve a year.
"""

from __future__ import annotations

import logging
import re
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone

import stripe
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import RedirectResponse

from app import auth, billing, db, paths, plans
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
links = APIRouter()  # the short invite links, outside /api
User = Depends(auth.current_user)
PER_REWARD = 3
REWARD_TIER = billing.REWARD_TIER
REWARD_DAYS = 30
MAX_REWARDS_A_YEAR = 12
COOKIE = "aiet_ref"
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I to mix up
CODE = re.compile(r"^[A-Z2-9]{6,12}$")


def code_for(conn, user_id: int) -> str:
    while True:
        found = conn.execute("SELECT referral_code FROM users WHERE id = ?", (user_id,)).fetchone()[0]
        if found:
            return found
        candidate = "".join(secrets.choice(CODE_ALPHABET) for _ in range(8))
        try:  # only where there's none yet, so two pages opened at once end up with the same code
            conn.execute("UPDATE users SET referral_code = ? WHERE id = ? AND referral_code IS NULL", (candidate, user_id))
        except sqlite3.IntegrityError:  # someone else's already: try another
            continue


def link(code: str) -> str:
    return f"{settings.public_url}/r/{code}"


def clean(code: str | None) -> str | None:
    code = (code or "").strip().upper()
    return code if CODE.match(code) else None


def attribute(conn, invitee_id: int, code: str | None) -> bool:
    """At sign-up: note who invited the new learner, if their code is someone else's."""
    code = clean(code)
    if not code:
        return False
    referrer = conn.execute("SELECT id FROM users WHERE referral_code = ?", (code,)).fetchone()
    if referrer is None or referrer["id"] == invitee_id:
        return False
    conn.execute("INSERT OR IGNORE INTO referral_invites (invitee_id, referrer_id, created_at) VALUES (?, ?, ?)",
                 (invitee_id, referrer["id"], db.now()))
    conn.execute("UPDATE users SET signup_source = 'referral' WHERE id = ?", (invitee_id,))
    return True


def _first_name(name: str) -> str:
    return (name or "").strip().split(" ")[0][:30] or "A friend"


def qualify(conn, invitee_id: int) -> None:
    """The invited friend confirmed their address, so the invite counts. Every third earns the inviter a reward."""
    invite = conn.execute("SELECT r.referrer_id, u.display_name, u.email FROM referral_invites r JOIN users u ON u.id = r.invitee_id "
                          "WHERE r.invitee_id = ? AND r.qualified_at IS NULL", (invitee_id,)).fetchone()
    if invite is None:
        return
    referrer = invite["referrer_id"]
    # One mailbox counts once: not the inviter's own, nor one another invite of theirs already counted.
    mailbox = billing.address_key(invite["email"])
    others = conn.execute("SELECT u.email FROM users u WHERE u.id = ? UNION ALL SELECT u.email FROM referral_invites r "
                          "JOIN users u ON u.id = r.invitee_id WHERE r.referrer_id = ? AND r.qualified_at IS NOT NULL",
                          (referrer, referrer)).fetchall()
    if any(billing.address_key(o[0]) == mailbox for o in others):
        return
    conn.execute("UPDATE referral_invites SET qualified_at = ? WHERE invitee_id = ?", (db.now(), invitee_id))
    counted = conn.execute("SELECT COUNT(*) FROM referral_invites WHERE referrer_id = ? AND qualified_at IS NOT NULL",
                           (referrer,)).fetchone()[0]
    earned = conn.execute("SELECT COUNT(*) FROM referral_rewards WHERE user_id = ?", (referrer,)).fetchone()[0]
    year_ago = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat(timespec="seconds")
    this_year = conn.execute("SELECT COUNT(*) FROM referral_rewards WHERE user_id = ? AND created_at > ?",
                             (referrer, year_ago)).fetchone()[0]
    friend = _first_name(invite["display_name"])
    from app import staff

    if counted // PER_REWARD > earned and this_year < MAX_REWARDS_A_YEAR:
        conn.execute("INSERT INTO referral_rewards (user_id, created_at) VALUES (?, ?)", (referrer, db.now()))
        staff.notify(conn, referrer, f"{friend} joined through your invite, so you've earned a free month of Plus. "
                                     "Claim it when you like.", paths.INVITE)
    else:
        left = PER_REWARD - counted % PER_REWARD
        staff.notify(conn, referrer, f"{friend} joined through your invite. {left} more for a free month of Plus.",
                     paths.INVITE)


def subscribed(conn, user_id: int) -> None:
    """The learner started paying for a plan: if a friend invited them, their invite shows it."""
    conn.execute("UPDATE referral_invites SET subscribed_at = COALESCE(subscribed_at, ?) WHERE invitee_id = ?",
                 (db.now(), user_id))


@router.get("/referrals")
def mine(user: dict = User):
    with db.connect() as conn:
        code = code_for(conn, user["id"])
        invites = conn.execute(
            "SELECT u.display_name, r.created_at, r.qualified_at, r.subscribed_at FROM referral_invites r "
            "JOIN users u ON u.id = r.invitee_id WHERE r.referrer_id = ? ORDER BY r.created_at DESC LIMIT 100",
            (user["id"],)).fetchall()
        counted = conn.execute("SELECT COUNT(*) FROM referral_invites WHERE referrer_id = ? AND qualified_at IS NOT NULL",
                               (user["id"],)).fetchone()[0]
        rewards = conn.execute("SELECT COUNT(*) AS earned, COALESCE(SUM(claimed_at IS NULL), 0) AS waiting "
                               "FROM referral_rewards WHERE user_id = ?", (user["id"],)).fetchone()
    return {
        "code": code, "link": link(code), "per_reward": PER_REWARD, "reward": f"A free month of {plans.plan(REWARD_TIER).name}",
        "counted": counted, "toward_next": counted % PER_REWARD, "earned": rewards["earned"], "to_claim": rewards["waiting"],
        "invites": [{"name": _first_name(r["display_name"]), "joined_at": r["created_at"],
                     "status": "subscribed" if r["subscribed_at"] else "joined" if r["qualified_at"] else "unconfirmed"}
                    for r in invites],
    }


def _plus_price() -> int:
    return plans.plan(REWARD_TIER).monthly


@router.post("/referrals/claim")
def claim(user: dict = User):
    """Use a reward: a month of Plus (after any reward months still to run), or for a learner who already pays for
    Plus or more, a month of Plus off their next bill. A plan an admin gave is left as it is either way."""
    billing.throttle(user)
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        reward = conn.execute("SELECT id FROM referral_rewards WHERE user_id = ? AND claimed_at IS NULL ORDER BY id LIMIT 1",
                              (user["id"],)).fetchone()
        if reward is None:
            raise HTTPException(409, "There's no reward to claim yet. Invite friends to earn one.")
        sub = billing.row(conn, user["id"])
        credit = (plans.rank(billing.paid_tier(sub)) >= plans.rank(REWARD_TIER) and sub["stripe_customer_id"]
                  and billing.payments())
        if not credit:
            now = datetime.now(timezone.utc)
            start = max(now, datetime.fromisoformat(sub["reward_until"])) if sub["reward_until"] else now
            before = billing.effective_tier(sub)
            conn.execute("UPDATE subscriptions SET reward_until = ?, updated_at = ? WHERE user_id = ?",
                         ((start + timedelta(days=REWARD_DAYS)).isoformat(timespec="seconds"), db.now(), user["id"]))
            billing._record_change(conn, user["id"], before, billing.effective_tier(billing.row(conn, user["id"])), "referral")
        conn.execute("UPDATE referral_rewards SET claimed_at = ?, applied_as = ? WHERE id = ?",
                     (db.now(), "credit" if credit else "plan", reward["id"]))
    if credit:
        try:
            billing._client().v1.customers.balance_transactions.create(sub["stripe_customer_id"], params={
                "amount": -_plus_price(), "currency": "usd", "description": "Referral reward: a month of Plus"},
                options={"idempotency_key": f"aiet-referral-reward-{reward['id']}"})
        except stripe.StripeError:
            log.exception("referral credit for user %s failed", user["id"])
            with db.connect() as conn:  # the reward waits to be claimed again
                conn.execute("UPDATE referral_rewards SET claimed_at = NULL, applied_as = NULL WHERE id = ?", (reward["id"],))
            raise HTTPException(502, "Your reward couldn't be added to your bill just now. Try again in a minute.")
        return mine(user) | {"message": f"{plans.money(_plus_price())} comes off your next bill."}
    return mine(user) | {"message": "Enjoy a month of Plus! It's on your Billing page."}


@links.get("/r/{code}", include_in_schema=False)
def invite_link(code: str):
    """An invite link: remembers who invited the visitor for when they sign up, then opens the sign-up page."""
    code = clean(code)
    response = RedirectResponse(f"/signup?ref={code}" if code else "/signup", status_code=302)
    if code:
        response.set_cookie(COOKIE, code, max_age=30 * 86400, httponly=True, samesite="lax", secure=settings.secure_cookies)
    return response
