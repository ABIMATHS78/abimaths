"""Plans and payments: what each learner's plan lets them use, and paying for plans with Stripe.

What a learner can use is the higher of two plans: the one they pay for through Stripe (or paid for on the old site)
while its subscription is active, on trial or being retried after a failed payment; and one given to them, by an
admin (Teams members, say) or as a referral reward, for good or until a date. Admins can use everything. The paywall
is on once Stripe is set up, or with AIET_PAYWALL=1; while it's off, everyone can use everything, within the site's
daily AI allowance, as before there were plans.

Free learners have companions, learning paths, quizzes, flashcards and study plans, a few guides a month in full (and
every guide's summary), and the first module of every course. Plans unlock the rest (app/plans.py). Reaching for
something locked gets a 402 that says which plan unlocks it, with the plan in an X-Plan-Needed header.

Stripe keeps the cards: this server never sees one, and checkout and the billing portal are Stripe's own pages.
Stripe reports what happened through signed webhooks, each handled once (it can send one twice). Rather than trust
the order they arrive in, each brings the subscription up to date from Stripe itself. A subscription is also brought
up to date when the learner comes back from checkout, so a slow webhook, or one not set up yet, doesn't keep them
waiting.

    python -m app.billing setup       makes the plans' products and prices in Stripe (once, for each Stripe account)
    python -m app.billing check       shows which Stripe price each plan uses, and any whose amount isn't the plan's
    python -m app.billing reconcile   brings every subscription up to date from Stripe (after an import, say);
                                      --dry-run shows what would change; --end-gone ends plans Stripe doesn't have
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Literal

import stripe
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from app import auth, catalog, db, plans
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
User = Depends(auth.current_user)
Admin = Depends(auth.require("admin"))
# Subscription statuses that keep a paid plan in use. A failed payment is retried for a while (past_due) before
# Stripe gives up and cancels, and the plan carries on meanwhile.
LIVE = ("active", "trialing", "past_due")
STATUSES = ("active", "trialing", "past_due", "canceled", "incomplete", "unpaid", "paused")
FREE_MODULES = 1  # the modules of every course anyone can take
REWARD_TIER = "plus"  # what a referral reward gives
STRIPE_CALLS = (20, 600)
PRICE_CACHE_SECONDS = 3600
NO_PAYMENTS = "Payments aren't set up on this server yet."
# What reaching for something locked says, by feature.
LOCKED = {
    "guides": "You've read your free guides for this month. Read every guide in full with {plan} or any plan above it.",
    "courses": "The rest of this course comes with {plan} and the plans above it: every module, lab and project, and the "
               "certificate.",
    "exams": "Mock exams come with {plan} and the plans above it.",
    "custom_paths": "Learning paths designed for you come with {plan} and the plans above it.",
    "event_pass": "The complimentary GDAI 2026 pass comes with {plan} and Teams.",
}


def paywall() -> bool:
    return settings.paywall


def payments() -> bool:
    return bool(settings.stripe_secret_key)


def _client() -> stripe.StripeClient:
    return stripe.StripeClient(settings.stripe_secret_key, max_network_retries=2)


def _plain(obj) -> dict:
    """Stripe's objects as plain dicts (tests hand in dicts already)."""
    return obj.to_dict() if hasattr(obj, "to_dict") else obj


def _id(ref) -> str | None:
    """The id of a Stripe reference, which is an id or, expanded, the object."""
    return ref.get("id") if isinstance(ref, dict) else ref


def _iso(ts) -> str | None:
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat(timespec="seconds") if ts else None


@contextmanager
def _stripe(what: str):
    """A call to Stripe: a failure is logged and answered with a 502 that says what couldn't be done."""
    try:
        yield
    except stripe.StripeError as e:
        log.warning("Stripe couldn't %s: %s", what, getattr(e, "user_message", None) or type(e).__name__)
        raise HTTPException(502, f"Payments couldn't {what} just now. Try again in a minute.")


# ------------------------------------------------------------------ the plan in use


def row(conn, user_id: int) -> dict:
    """The learner's subscription row, made (on the free plan) the first time it's needed."""
    found = conn.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user_id,)).fetchone()
    if found is None:
        conn.execute("INSERT OR IGNORE INTO subscriptions (user_id, created_at, updated_at) VALUES (?, ?, ?)",
                     (user_id, db.now(), db.now()))
        found = conn.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user_id,)).fetchone()
    return dict(found)


def paid_tier(sub: dict | None) -> str:
    return sub["tier"] if sub and sub["status"] in LIVE else "free"


def granted_tier(sub: dict | None) -> str:
    if sub and sub["grant_tier"] and (sub["grant_until"] is None or sub["grant_until"] > db.now()):
        return sub["grant_tier"]
    return "free"


def reward_tier(sub: dict | None) -> str:
    """Referral rewards are months of Plus of their own, so they neither shorten nor stretch a plan an admin gave."""
    return REWARD_TIER if sub and sub.get("reward_until") and sub["reward_until"] > db.now() else "free"


def effective_tier(sub: dict | None) -> str:
    return plans.best(paid_tier(sub), granted_tier(sub), reward_tier(sub))


def address_key(email: str) -> str:
    """A hash of an address as its mailbox sees it: in lower case, without +tags, and for Gmail without dots, so
    ada+1@gmail.com and a.da@gmail.com are ada@gmail.com. Kept instead of the address where it must outlive an account."""
    local, _, domain = email.strip().lower().partition("@")
    local = local.split("+", 1)[0]
    if domain in ("gmail.com", "googlemail.com"):
        local, domain = local.replace(".", ""), "gmail.com"
    return auth.hash_token(f"{local}@{domain}")


def had_trial(conn, email: str) -> bool:
    return conn.execute("SELECT 1 FROM trials WHERE address_hash = ?", (address_key(email),)).fetchone() is not None


_calls: dict[int, list[float]] = {}


def throttle(user: dict) -> None:
    """At most STRIPE_CALLS[0] requests that reach Stripe per learner in STRIPE_CALLS[1] seconds, so no one can use up
    the site's allowance at Stripe."""
    now, (most, window) = time.time(), STRIPE_CALLS
    recent = [t for t in _calls.get(user["id"], ()) if t > now - window]
    if len(recent) >= most:
        raise HTTPException(429, "That's a lot of tries in a few minutes. Please wait a little and try again.",
                            headers={"Retry-After": str(window // 2)})
    _calls[user["id"]] = recent + [now]


def tier(user: dict) -> str:
    """The plan the learner can use."""
    with db.connect() as conn:
        found = conn.execute("SELECT * FROM subscriptions WHERE user_id = ?", (user["id"],)).fetchone()
    return effective_tier(dict(found) if found else None)


def unlocked(user: dict, feature: str) -> bool:
    if not paywall() or auth.role_of(user) == "admin":
        return True
    return feature in plans.plan(tier(user)).unlocks


def locked(feature: str) -> HTTPException:
    p = plans.lowest_with(feature)
    return HTTPException(402, LOCKED[feature].format(plan=p.name), headers={"X-Plan-Needed": p.slug})


def require(user: dict, feature: str) -> None:
    if not unlocked(user, feature):
        raise locked(feature)


def course_item_open(user: dict, item_id: str) -> bool:
    """Whether the learner can open a course's module, lesson, lab or project, or (by its slug) the whole course.
    Ids that aren't a course's are always open. Everyone can take the first module of every course."""
    p = catalog.place(item_id)
    return p is None or 0 < p.m <= FREE_MODULES or unlocked(user, "courses")


def require_course_item(user: dict, item_id: str) -> None:
    if not course_item_open(user, item_id):
        raise locked("courses")


def guide_access(user: dict, post: dict, record: bool = True) -> dict:
    """Whether the learner can read a guide in full, and how many of this month's free guides they have left (None
    for no limit). A guide opened once is theirs for the rest of the month; opening one more uses one up. Without
    `record` (in chat, say) only guides already opened this month are open. Posts that aren't guides always are."""
    if not post.get("kind") or unlocked(user, "guides"):
        return {"full": True, "left": None, "limit": None}
    month, limit = db.now()[:7], max(0, settings.free_guides)
    with db.connect() as conn:
        if record:
            conn.execute("BEGIN IMMEDIATE")  # count and claim in one go, so two tabs can't both take the last one
        seen = conn.execute("SELECT 1 FROM guide_reads WHERE user_id = ? AND month = ? AND post_id = ?",
                            (user["id"], month, post["id"])).fetchone() is not None
        used = conn.execute("SELECT COUNT(*) FROM guide_reads WHERE user_id = ? AND month = ?", (user["id"], month)).fetchone()[0]
        full = seen or (record and used < limit)
        if full and not seen:
            conn.execute("INSERT INTO guide_reads (user_id, month, post_id) VALUES (?, ?, ?)", (user["id"], month, post["id"]))
            used += 1
    return {"full": full, "left": max(0, limit - used), "limit": limit}


def preview(body: str, share: float = 0.2, most: int = 1500) -> str:
    """The start of a locked guide: about a fifth of it, no more than `most` characters, ending at a paragraph."""
    cut = min(most, max(400, int(len(body) * share)))
    if len(body) <= cut:
        return body
    end = body.rfind("\n\n", 0, cut)
    return body[:end if end > cut // 3 else cut].rstrip()


def _record_change(conn, user_id: int, before: str, after: str, source: str, when: str | None = None) -> None:
    """Note a change in the plan a learner can use, for the analytics."""
    if before == after:
        return
    if after == "free":
        kind = "cancel"
    elif before == "free":
        again = conn.execute("SELECT 1 FROM subscription_events WHERE user_id = ? AND to_tier != 'free'", (user_id,)).fetchone()
        kind = "reactivate" if again else "new"
    else:
        kind = "upgrade" if plans.rank(after) > plans.rank(before) else "downgrade"
    conn.execute("INSERT INTO subscription_events (user_id, from_tier, to_tier, event_type, source, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (user_id, before, after, kind, source, when or db.now()))


def grant(conn, user_id: int, new_tier: str | None, until: str | None, note: str, source: str) -> None:
    """Give a learner a plan, until `until` (an ISO time) or for good, or (with None) take a given plan back."""
    current = row(conn, user_id)
    before = effective_tier(current)
    conn.execute("UPDATE subscriptions SET grant_tier = ?, grant_until = ?, grant_note = ?, updated_at = ? WHERE user_id = ?",
                 (new_tier, until if new_tier else None, note if new_tier else "", db.now(), user_id))
    _record_change(conn, user_id, before, effective_tier(row(conn, user_id)), source)


# ------------------------------------------------------------------- Stripe prices


_price_ids: dict[str, tuple[float, str]] = {}


def lookup_key(tier_slug: str, interval: str) -> str:
    return f"aiet_{tier_slug}_{interval}"


def price_id(tier_slug: str, interval: str) -> str:
    """The Stripe price for a plan and billing period: AIET_STRIPE_PRICES's, or the one with the plan's lookup key."""
    key = f"{tier_slug}_{interval}"
    if configured := settings.stripe_prices.get(key):
        return configured
    cached = _price_ids.get(key)
    if cached and cached[0] > time.time():
        return cached[1]
    with _stripe("find the plan's price"):
        found = _plain(_client().v1.prices.list(params={"lookup_keys": [lookup_key(tier_slug, interval)], "active": True,
                                                        "limit": 1}))
    if not found.get("data"):
        log.error("No Stripe price has the lookup key %s: run `python -m app.billing setup`", lookup_key(tier_slug, interval))
        raise HTTPException(503, "This plan can't be bought just yet. Please try again later.")
    _price_ids[key] = (time.time() + PRICE_CACHE_SECONDS, found["data"][0]["id"])
    return _price_ids[key][1]


def price_plan(price: dict | None, metadata: dict | None = None) -> tuple[str, str] | None:
    """The plan and billing period a Stripe price is for: by its lookup key, AIET_STRIPE_PRICES, the price's metadata,
    its amount (matched against the plans' prices), or failing those the subscription's metadata. None if none fits."""
    price = price if isinstance(price, dict) else {"id": price}
    recurring = price.get("recurring") or {}
    interval = "yearly" if recurring.get("interval") == "year" else "monthly"
    if m := re.fullmatch(r"aiet_([a-z]+)_(monthly|yearly)", price.get("lookup_key") or ""):
        if m[1] in plans.TIERS:
            return m[1], m[2]
    for key, pid in settings.stripe_prices.items():
        name, _, period = key.rpartition("_")
        if pid == price.get("id") and name in plans.TIERS and period in plans.INTERVALS:
            return name, period
    def named(meta: dict) -> tuple[str, str] | None:
        tier_slug = (meta.get("aiet_tier") or meta.get("tier") or "").lower()
        if tier_slug in plans.TIERS and tier_slug != "free":
            period = meta.get("aiet_interval") or meta.get("billingInterval") or meta.get("interval")
            return tier_slug, period if period in plans.INTERVALS else (interval if recurring else "monthly")
        return None

    if found := named(price.get("metadata") or {}):
        return found
    amount = price.get("unit_amount")
    if recurring and amount:
        for p in plans.PLANS.values():
            if p.paid and p.price(interval) == amount:
                return p.slug, interval
    # Last, what the subscription was bought as: it stays the same when the plan changes later.
    return named(metadata or {})


# ------------------------------------------------------------ keeping up with Stripe


def _owner(conn, sub: dict) -> int | None:
    """The learner a Stripe subscription belongs to: already linked to it, named in its metadata (by this site's id,
    or the old site's), or the only learner with its customer."""
    found = conn.execute("SELECT user_id FROM subscriptions WHERE stripe_subscription_id = ?", (sub["id"],)).fetchone()
    if found:
        return found[0]
    meta = sub.get("metadata") or {}
    if str(meta.get("aiet_user_id", "")).isdigit():
        found = conn.execute("SELECT id FROM users WHERE id = ?", (int(meta["aiet_user_id"]),)).fetchone()
        if found:
            return found[0]
    for key in ("userId", "user_id"):
        if meta.get(key):
            found = conn.execute("SELECT id FROM users WHERE legacy_id = ?", (str(meta[key]),)).fetchone()
            if found:
                return found[0]
    customer = _id(sub.get("customer"))
    if customer:
        rows = conn.execute("SELECT user_id FROM subscriptions WHERE stripe_customer_id = ?", (customer,)).fetchall()
        if len(rows) == 1:
            return rows[0][0]
    return None


def apply(sub, user_id: int | None = None) -> int | None:
    """Bring a learner's plan in line with their Stripe subscription, as Stripe has it now. Returns the learner's id,
    or None for a subscription that can't be matched to anyone."""
    sub = _plain(sub)
    status = sub.get("status")
    status = status if status in STATUSES else "canceled"  # incomplete_expired: checkout was never finished
    named = str((sub.get("metadata") or {}).get("aiet_user_id") or "")
    became_paid = False
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        if user_id is not None and named.isdigit() and int(named) != user_id:
            log.warning("Stripe subscription %s was bought for another learner, so it isn't applied", sub.get("id"))
            return None
        uid = user_id or _owner(conn, sub)
        if uid is None:
            if named.isdigit() and status in LIVE and not conn.execute("SELECT 1 FROM users WHERE id = ?", (int(named),)).fetchone():
                # Bought for an account since deleted (ids are never reused): it mustn't go on charging anyone.
                conn.execute("INSERT OR IGNORE INTO stripe_cancellations (stripe_subscription_id, requested_at) VALUES (?, ?)",
                             (sub["id"], db.now()))
                log.warning("Stripe subscription %s belongs to a deleted account: queued for cancelling", sub.get("id"))
            else:
                log.warning("Stripe subscription %s matches no learner", sub.get("id"))
            return None
        current = row(conn, uid)
        same = current["stripe_subscription_id"] == sub["id"]
        if same and current["status"] == "canceled" and status != "canceled":
            return uid  # cancelled is for good at Stripe: this is an older view of it, arriving late
        before = effective_tier(current)
        item = ((sub.get("items") or {}).get("data") or [{}])[0]
        price = item.get("price") or {}
        mapped = price_plan(price, sub.get("metadata"))
        if not same and current["stripe_subscription_id"] and current["status"] in LIVE:
            # Another subscription of theirs is in use. This one ending, or being no better a plan, changes nothing.
            if status not in LIVE or plans.rank(mapped[0] if mapped else "free") <= plans.rank(current["tier"]):
                if status in LIVE:
                    log.warning("learner %s has two subscriptions in use at Stripe, %s and %s: check them", uid,
                                current["stripe_subscription_id"], sub["id"])
                return uid
            log.warning("learner %s has two subscriptions in use at Stripe: following %s, the better plan, rather than %s",
                        uid, sub["id"], current["stripe_subscription_id"])
        if mapped is None:
            log.warning("Stripe price %s of subscription %s isn't one of the plans' prices", _id(price), sub.get("id"))
        # A price that isn't a plan's keeps the plan known for this same subscription; a new one gets none.
        tier_slug, interval = mapped or ((current["tier"], current["billing_interval"]) if same else ("free", "monthly"))
        customer = _id(sub.get("customer"))
        if current["stripe_customer_id"] and customer and customer != current["stripe_customer_id"]:
            log.warning("subscription %s is another Stripe customer's than learner %s's: theirs is kept", sub["id"], uid)
        seats = item.get("quantity") or 1
        amount = price.get("unit_amount") * seats if isinstance(price, dict) and price.get("unit_amount") is not None else None
        trial_end = sub.get("trial_end")
        if trial_end:  # one free trial per address, even across accounts deleted and made again
            email = conn.execute("SELECT email FROM users WHERE id = ?", (uid,)).fetchone()[0]
            conn.execute("INSERT OR IGNORE INTO trials (address_hash, created_at) VALUES (?, ?)", (address_key(email), db.now()))
        conn.execute(
            "UPDATE subscriptions SET tier = ?, billing_interval = ?, status = ?, stripe_customer_id = COALESCE(stripe_customer_id, ?), "
            "stripe_subscription_id = ?, stripe_price_id = ?, seats = ?, amount_cents = COALESCE(?, amount_cents), "
            "current_period_start = ?, current_period_end = ?, trial_end = ?, cancel_at_period_end = ?, canceled_at = ?, "
            "trial_used = MAX(trial_used, ?), intended_tier = CASE WHEN ? THEN NULL ELSE intended_tier END, "
            "intended_interval = CASE WHEN ? THEN NULL ELSE intended_interval END, updated_at = ? WHERE user_id = ?",
            (tier_slug, interval, status, customer, sub["id"],
             _id(price) if price else current["stripe_price_id"], seats, amount,
             # Since Stripe's 2025 API the billing period is on each item; it was on the subscription before.
             _iso(item.get("current_period_start") or sub.get("current_period_start")),
             _iso(item.get("current_period_end") or sub.get("current_period_end")),
             _iso(trial_end), int(bool(sub.get("cancel_at_period_end") or sub.get("cancel_at"))), _iso(sub.get("canceled_at")),
             int(bool(trial_end)), status in LIVE, status in LIVE, db.now(), uid))
        after = effective_tier(row(conn, uid))
        _record_change(conn, uid, before, after, "stripe")
        if status in LIVE and plans.rank(paid_tier(current)) == 0:
            became_paid = True
            from app import referrals

            referrals.subscribed(conn, uid)
    if became_paid:
        log.info("learner %s is now on %s", uid, tier_slug)
    return uid


def sync(sub_id: str, user_id: int | None = None) -> int | None:
    """Fetch a subscription from Stripe and apply it."""
    with _stripe("look up the subscription"):
        sub = _client().v1.subscriptions.retrieve(sub_id)
    return apply(sub, user_id)


def _invoice_subscription(invoice: dict) -> str | None:
    # On the invoice itself before Stripe's 2025 API; in its parent's details since.
    return _id(invoice.get("subscription")) or _id(
        ((invoice.get("parent") or {}).get("subscription_details") or {}).get("subscription"))


def _customer_user(conn, customer: str | None) -> int | None:
    if not customer:
        return None
    found = conn.execute("SELECT user_id FROM subscriptions WHERE stripe_customer_id = ?", (customer,)).fetchone()
    return found[0] if found else None


def _price_by_id(pid: str) -> dict | None:
    """A price named only by its id (as invoice lines name it since Stripe's 2025 API), from Stripe."""
    try:
        return _plain(_client().v1.prices.retrieve(pid))
    except stripe.StripeError:
        return None


def record_payment(invoice: dict) -> None:
    """A paid invoice, as revenue, once."""
    amount = invoice.get("amount_paid") or 0
    if amount <= 0:  # a free trial's first invoice, say
        return
    line = (((invoice.get("lines") or {}).get("data")) or [{}])[0]
    price = line.get("price") or ((line.get("pricing") or {}).get("price_details") or {}).get("price")
    if isinstance(price, str):
        price = _price_by_id(price)
    with db.connect() as conn:
        uid = _customer_user(conn, _id(invoice.get("customer")))
        current = row(conn, uid) if uid else None
        tier_slug, interval = (price_plan(price) if isinstance(price, dict) else None) or (
            (current["tier"], current["billing_interval"]) if current else (None, None))
        paid_at = ((invoice.get("status_transitions") or {}).get("paid_at")) or invoice.get("created")
        conn.execute("INSERT OR IGNORE INTO revenue_events (user_id, kind, amount_cents, currency, tier, billing_interval, "
                     "stripe_ref, created_at) VALUES (?, 'payment', ?, ?, ?, ?, ?, ?)",
                     (uid, amount, invoice.get("currency") or "usd", tier_slug, interval, invoice["id"], _iso(paid_at) or db.now()))


def record_refund(charge: dict) -> None:
    """Money given back on a charge, as negative revenue: whatever wasn't recorded yet (refunds can come in parts)."""
    total = charge.get("amount_refunded") or 0
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        done = -conn.execute("SELECT COALESCE(SUM(amount_cents), 0) FROM revenue_events WHERE kind = 'refund' AND "
                             "stripe_ref LIKE ?", (f"{charge['id']}:%",)).fetchone()[0]
        if total > done:
            uid = _customer_user(conn, _id(charge.get("customer")))
            current = row(conn, uid) if uid else None
            conn.execute("INSERT OR IGNORE INTO revenue_events (user_id, kind, amount_cents, currency, tier, billing_interval, "
                         "stripe_ref, created_at) VALUES (?, 'refund', ?, ?, ?, ?, ?, ?)",
                         (uid, -(total - done), charge.get("currency") or "usd", current["tier"] if current else None,
                          current["billing_interval"] if current else None, f"{charge['id']}:{total}", db.now()))


def _our_session(session: dict) -> int | None:
    """The learner a Checkout Session is for, if this site made it: its checkout puts the learner's id both in
    client_reference_id and in the metadata. Sessions made elsewhere on the same Stripe account (the old site's, or a
    payment link's, whose client_reference_id anyone can set) aren't taken at their word."""
    ref = str(session.get("client_reference_id") or "")
    named = str((session.get("metadata") or {}).get("aiet_user_id") or "")
    return int(ref) if ref.isdigit() and ref == named else None


def handle(event: dict) -> None:
    """One Stripe event, which may already have been handled: everything here can safely happen twice."""
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM stripe_events WHERE id = ?", (event["id"],)).fetchone():
            return
    kind, obj = event["type"], event["data"]["object"]
    from app import campaigns

    if kind == "checkout.session.completed" and obj.get("mode") == "subscription" and obj.get("subscription"):
        uid = None
        if ours := _our_session(obj):  # anything else is matched by its subscription alone
            with db.connect() as conn:
                if conn.execute("SELECT 1 FROM users WHERE id = ?", (ours,)).fetchone():
                    uid = ours
                    row(conn, uid)
                    conn.execute("UPDATE subscriptions SET stripe_customer_id = COALESCE(stripe_customer_id, ?) WHERE user_id = ?",
                                 (_id(obj.get("customer")), uid))
        uid = sync(_id(obj["subscription"]), uid)
        if uid:
            campaigns.billing_email(uid, "started", _id(obj["subscription"]))
    elif kind.startswith("customer.subscription."):
        uid = sync(obj["id"])
        if uid and kind == "customer.subscription.trial_will_end":
            campaigns.billing_email(uid, "trial_ending", f"{obj['id']}:{obj.get('trial_end')}")
        elif uid and kind == "customer.subscription.deleted":
            campaigns.billing_email(uid, "ended", obj["id"])
    elif kind in ("invoice.paid", "invoice.payment_succeeded"):
        record_payment(obj)
    elif kind == "invoice.payment_failed":
        with db.connect() as conn:
            uid = _customer_user(conn, _id(obj.get("customer")))
        if uid:
            campaigns.billing_email(uid, "payment_failed", obj["id"])
    elif kind == "charge.refunded":
        record_refund(obj)
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO stripe_events (id, type, received_at) VALUES (?, ?, ?)",
                     (event["id"], kind, db.now()))


@router.post("/billing/webhook", include_in_schema=False)
async def webhook(request: Request):
    """Where Stripe sends events. Each is checked against the endpoint's signing secret before anything else."""
    if not (payments() and settings.stripe_webhook_secret):
        raise HTTPException(503, "Stripe webhooks aren't set up on this server")
    payload = await request.body()
    try:
        event = stripe.Webhook.construct_event(payload, request.headers.get("stripe-signature"), settings.stripe_webhook_secret)
    except (ValueError, stripe.SignatureVerificationError):
        raise HTTPException(400, "Invalid signature")
    await asyncio.to_thread(handle, _plain(event))
    return {"received": True}


# --------------------------------------------------------------- what learners see


def view(user: dict) -> dict:
    """The learner's plan, as the Billing page and the rest of the app need it."""
    with db.connect() as conn:
        sub = row(conn, user["id"])
    current = effective_tier(sub)
    p = plans.plan(current)
    admin = auth.role_of(user) == "admin"
    unlocks = set(plans.FEATURES) if not paywall() or admin else p.unlocks
    paid = None
    if sub["stripe_subscription_id"] or sub["tier"] != "free":
        paid = {"tier": sub["tier"], "name": plans.plan(sub["tier"]).name, "interval": sub["billing_interval"],
                "status": sub["status"], "live": sub["status"] in LIVE, "seats": sub["seats"],
                "amount": sub["amount_cents"] if sub["amount_cents"] is not None else plans.plan(sub["tier"]).price(sub["billing_interval"]),
                "period_end": sub["current_period_end"], "trial_end": sub["trial_end"] if sub["status"] == "trialing" else None,
                "cancel_at_period_end": bool(sub["cancel_at_period_end"]),
                "manageable": payments() and bool(sub["stripe_subscription_id"] and sub["stripe_customer_id"])}
    given = granted_tier(sub)
    intended = None
    if sub["intended_tier"] and plans.rank(sub["intended_tier"]) > plans.rank(current):
        intended = {"tier": sub["intended_tier"], "name": plans.plan(sub["intended_tier"]).name,
                    "interval": sub["intended_interval"] or "monthly", "at": sub["intended_at"]}
    guides = None
    with db.connect() as conn:
        trial_before = had_trial(conn, user["email"])
        if paywall() and "guides" not in unlocks:
            used = conn.execute("SELECT COUNT(*) FROM guide_reads WHERE user_id = ? AND month = ?",
                                (user["id"], db.now()[:7])).fetchone()[0]
            guides = {"limit": settings.free_guides, "left": max(0, settings.free_guides - used)}
    rewarded = reward_tier(sub) != "free"
    return {
        "tier": current, "name": p.name, "paywall": paywall(), "payments": payments(), "unlocks": sorted(unlocks),
        "subscription": paid,
        "grant": {"tier": given, "name": plans.plan(given).name, "until": sub["grant_until"], "note": sub["grant_note"]}
        if given != "free" else None,
        "reward": {"tier": REWARD_TIER, "name": plans.plan(REWARD_TIER).name, "until": sub["reward_until"]} if rewarded else None,
        "intended": intended, "event_discount": p.event_discount, "guides": guides,
        "trial_available": trial_available(sub, trial_before),
        "has_customer": bool(sub["stripe_customer_id"]),
    }


@router.get("/plans")
def list_plans():
    """The plans, for the pricing page, which anyone can see."""
    return {"plans": plans.public(len(catalog.data().courses)), "currency": "usd", "payments": payments(),
            "paywall": paywall(), "trial_days": settings.trial_days, "teams_email": settings.teams_email,
            "free_guides": settings.free_guides, "free_modules": FREE_MODULES}


@router.get("/billing")
def billing(user: dict = User):
    return view(user)


class PlanChoice(BaseModel):
    tier: Literal["digital", "plus", "pro", "insider"]
    interval: Literal["monthly", "yearly"] = "monthly"


def trial_available(sub: dict, trial_before: bool) -> bool:
    """A free trial is for a first paid plan: not after one here, nor for an address that's had one before."""
    return settings.trial_days > 0 and not sub["trial_used"] and not sub["stripe_subscription_id"] and not trial_before


def _need_payments() -> None:
    if not payments():
        raise HTTPException(503, NO_PAYMENTS)


def _customer(user: dict, sub: dict) -> str:
    """The learner's Stripe customer, made the first time they check out."""
    if sub["stripe_customer_id"]:
        return sub["stripe_customer_id"]
    with _stripe("start checkout"):
        made = _plain(_client().v1.customers.create(
            params={"email": user["email"], "name": user["display_name"], "metadata": {"aiet_user_id": str(user["id"])}},
            # Two quick clicks make one customer, not two.
            options={"idempotency_key": f"aiet-customer-{user['id']}-{user['created_at']}"}))
    with db.connect() as conn:
        conn.execute("UPDATE subscriptions SET stripe_customer_id = COALESCE(stripe_customer_id, ?) WHERE user_id = ?",
                     (made["id"], user["id"]))
        return conn.execute("SELECT stripe_customer_id FROM subscriptions WHERE user_id = ?", (user["id"],)).fetchone()[0]


@router.post("/billing/checkout")
def checkout(body: PlanChoice, user: dict = User):
    """Start paying for a plan: the address of Stripe's checkout page for it. The plan chosen is remembered, so a
    learner who doesn't finish can be reminded. A first paid plan starts with a free trial."""
    _need_payments()
    throttle(user)
    already = HTTPException(409, "You already have a plan, or a payment for one is waiting. See your Billing page.")
    with db.connect() as conn:
        sub = row(conn, user["id"])
        if sub["stripe_subscription_id"] and sub["status"] in LIVE:
            raise already
        trial_before = had_trial(conn, user["email"])
    # Stripe may know of a subscription this server doesn't yet (a checkout finished in another tab, its webhook on
    # the way): one is enough.
    history = []
    if sub["stripe_customer_id"]:
        with _stripe("start checkout"):
            history = _plain(_client().v1.subscriptions.list(params={"customer": sub["stripe_customer_id"], "status": "all",
                                                                     "limit": 20})).get("data") or []
        if going := [x for x in history if x.get("status") in (*LIVE, "incomplete", "unpaid", "paused")]:
            apply(going[0], user["id"])
            raise already
    with db.connect() as conn:
        conn.execute("UPDATE subscriptions SET intended_tier = ?, intended_interval = ?, intended_at = ?, updated_at = ? "
                     "WHERE user_id = ?", (body.tier, body.interval, db.now(), db.now(), user["id"]))
    price = price_id(body.tier, body.interval)
    customer = _customer(user, sub)
    meta = {"aiet_user_id": str(user["id"]), "aiet_tier": body.tier, "aiet_interval": body.interval}
    subscription_data: dict = {"metadata": meta}
    if trial_available(sub, trial_before) and not history:
        subscription_data["trial_period_days"] = settings.trial_days
    base = settings.public_url
    with _stripe("start checkout"):
        session = _plain(_client().v1.checkout.sessions.create(params={
            "mode": "subscription", "customer": customer, "client_reference_id": str(user["id"]),
            "line_items": [{"price": price, "quantity": 1}], "allow_promotion_codes": True, "metadata": meta,
            "subscription_data": subscription_data,
            "success_url": f"{base}/#/billing?checkout={{CHECKOUT_SESSION_ID}}",
            "cancel_url": f"{base}/#/pricing?checkout=cancelled",
        }))
    return {"url": session["url"]}


class Returned(BaseModel):
    session_id: str = Field(pattern=r"^cs_[A-Za-z0-9_]+$", max_length=300)


@router.post("/billing/checkout/confirm")
def checkout_returned(body: Returned, user: dict = User):
    """Back from checkout: bring the plan up to date now rather than wait for Stripe's webhook."""
    _need_payments()
    throttle(user)
    with _stripe("check your payment"):
        session = _plain(_client().v1.checkout.sessions.retrieve(body.session_id))
    if _our_session(session) != user["id"]:
        raise HTTPException(404, "Checkout not found")
    if session.get("status") == "complete" and session.get("subscription"):
        with db.connect() as conn:
            row(conn, user["id"])
            conn.execute("UPDATE subscriptions SET stripe_customer_id = COALESCE(stripe_customer_id, ?) WHERE user_id = ?",
                         (_id(session.get("customer")), user["id"]))
        sync(_id(session["subscription"]), user["id"])
    return view(user)


def _live_subscription(user: dict) -> dict:
    _need_payments()
    with db.connect() as conn:
        sub = row(conn, user["id"])
    if not (sub["stripe_subscription_id"] and sub["status"] in LIVE):
        raise HTTPException(409, "You don't have a paid plan to change. Choose one on the pricing page.")
    return sub


@router.post("/billing/change")
def change_plan(body: PlanChoice, user: dict = User):
    """Move to another plan or billing period. Moving up is charged now, for the rest of the period; moving down
    leaves the difference as credit on the next bill. A move up that the bank wants confirmed waits until it is."""
    sub = _live_subscription(user)
    if (sub["tier"], sub["billing_interval"]) == (body.tier, body.interval):
        return view(user)
    throttle(user)
    price = price_id(body.tier, body.interval)
    up = (plans.rank(body.tier), body.interval == "yearly") > (plans.rank(sub["tier"]), sub["billing_interval"] == "yearly")
    client = _client()
    with _stripe("change your plan"):
        current = _plain(client.v1.subscriptions.retrieve(sub["stripe_subscription_id"]))
        item = current["items"]["data"][0]
        updated = _plain(client.v1.subscriptions.update(sub["stripe_subscription_id"], params={
            "items": [{"id": item["id"], "price": price}],
            "proration_behavior": "always_invoice" if up else "create_prorations",
            "payment_behavior": "pending_if_incomplete",
        }))
    apply(updated, user["id"])
    out = view(user)
    if updated.get("pending_update"):
        out["message"] = "Your bank wants you to confirm the payment. Open Manage payments to finish moving plans."
    return out


def _set_cancel(user: dict, cancel: bool) -> dict:
    sub = _live_subscription(user)
    throttle(user)
    with _stripe("update your plan"):
        client = _client()
        params: dict = {"cancel_at_period_end": True}
        if not cancel:
            now = _plain(client.v1.subscriptions.retrieve(sub["stripe_subscription_id"]))
            # Undone the way it was set: at the end of the period, or (from the billing portal) for a date.
            params = {"cancel_at": ""} if now.get("cancel_at") and not now.get("cancel_at_period_end") else {"cancel_at_period_end": False}
        updated = client.v1.subscriptions.update(sub["stripe_subscription_id"], params=params)
    apply(updated, user["id"])
    return view(user)


@router.post("/billing/cancel")
def cancel(user: dict = User):
    """Stop the plan renewing. It carries on until the end of the period already paid for."""
    return _set_cancel(user, True)


@router.post("/billing/resume")
def resume(user: dict = User):
    """Keep a plan that was set to stop at the end of its period."""
    return _set_cancel(user, False)


@router.post("/billing/portal")
def portal(user: dict = User):
    """Stripe's billing portal, for cards, invoices and receipts."""
    _need_payments()
    throttle(user)
    with db.connect() as conn:
        sub = row(conn, user["id"])
    if not sub["stripe_customer_id"]:
        raise HTTPException(409, "There's nothing to manage yet: you haven't paid for a plan.")
    with _stripe("open the billing portal"):
        session = _plain(_client().v1.billing_portal.sessions.create(params={
            "customer": sub["stripe_customer_id"], "return_url": f"{settings.public_url}/#/billing"}))
    return {"url": session["url"]}


# ----------------------------------------------------------------- deleted accounts


def cancel_for_deleted(sub: dict | None) -> None:
    """A deleted account's subscriptions stop at once: the one known here, and any other its Stripe customer has. If
    Stripe can't be reached (or isn't set up yet), they're queued and cancelled as soon as they can be: no one goes on
    paying for an account they deleted."""
    refs = [ref for ref in ((sub or {}).get("stripe_subscription_id") if (sub or {}).get("status") in LIVE else None,
                            (sub or {}).get("stripe_customer_id")) if ref]
    if not refs:
        return
    with db.connect() as conn:
        conn.executemany("INSERT OR IGNORE INTO stripe_cancellations (stripe_subscription_id, requested_at) VALUES (?, ?)",
                         [(ref, db.now()) for ref in refs])
    cancel_queued()


def cancel_queued() -> int:
    """Cancel the subscriptions of deleted accounts still waiting (a subscription, or every one of a customer's).
    Returns how many waiting were dealt with."""
    if not payments():
        return 0
    with db.connect() as conn:
        waiting = [r[0] for r in conn.execute("SELECT stripe_subscription_id FROM stripe_cancellations ORDER BY requested_at")]
    done, client = 0, _client()
    for sub_id in waiting:
        try:
            if sub_id.startswith("cus_"):
                listed = _plain(client.v1.subscriptions.list(params={"customer": sub_id, "status": "all", "limit": 100}))
                for x in listed.get("data") or []:
                    if x.get("status") in (*LIVE, "incomplete", "unpaid", "paused"):
                        client.v1.subscriptions.cancel(x["id"], params={"prorate": False})
            else:
                client.v1.subscriptions.cancel(sub_id, params={"prorate": False})
        except stripe.InvalidRequestError as e:  # already cancelled, or gone
            log.info("subscription %s of a deleted account: %s", sub_id, type(e).__name__)
        except stripe.StripeError as e:
            with db.connect() as conn:
                conn.execute("UPDATE stripe_cancellations SET attempts = attempts + 1, last_error = ? "
                             "WHERE stripe_subscription_id = ?", (type(e).__name__, sub_id))
            continue
        with db.connect() as conn:
            conn.execute("DELETE FROM stripe_cancellations WHERE stripe_subscription_id = ?", (sub_id,))
        done += 1
    return done


# -------------------------------------------------------------------------- admins


class PlanGrant(BaseModel):
    tier: Literal["digital", "plus", "pro", "insider", "teams"] | None = Field(...)
    until: str | None = Field(None, pattern=r"^\d{4}-\d{2}-\d{2}$")  # the last day, or None for good
    note: str = Field("", max_length=200)


@router.post("/staff/users/{user_id}/plan")
def give_plan(user_id: int, body: PlanGrant, user: dict = Admin):
    """Give a learner a plan (each member of a Teams account, say), until a day or for good, or take it back. A plan
    they pay for carries on either way: they can use the higher of the two."""
    from app import staff

    until = None
    if body.tier and body.until:
        try:
            end = datetime.fromisoformat(body.until).replace(tzinfo=timezone.utc) + timedelta(days=1)
        except ValueError:
            raise HTTPException(422, "until: a date like 2027-01-31")
        if end <= datetime.now(timezone.utc):
            raise HTTPException(422, "Choose a day that hasn't passed")
        until = end.isoformat(timespec="seconds")
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM users WHERE id = ?", (user_id,)).fetchone() is None:
            raise HTTPException(404, "User not found")
        grant(conn, user_id, body.tier, until, body.note.strip(), "admin")
        staff.log(conn, user["id"], "user", user_id, f"plan:{body.tier or 'none'}" + (f" until {body.until}" if until else ""),
                  body.note.strip())
        if body.tier:
            text = f"You've been given the {plans.plan(body.tier).name} plan" + (f", until {body.until}." if until else ".")
        else:
            text = "The plan you were given has ended. Any plan you pay for carries on."
        staff.notify(conn, user_id, text, "#/billing")
    with db.connect() as conn:
        target = dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())
    return view(target)


# --------------------------------------------------------------------- command line


# The events the webhook endpoint in Stripe needs to send.
WEBHOOK_EVENTS = ("checkout.session.completed", "customer.subscription.created", "customer.subscription.updated",
                  "customer.subscription.deleted", "customer.subscription.trial_will_end", "invoice.paid",
                  "invoice.payment_failed", "charge.refunded")


def setup() -> list[str]:
    """Make a product for each paid plan and a price for each billing period, with lookup keys this app finds them by.
    Prices that exist already are left alone, so it can run again. Returns the lookup keys of the prices it made."""
    client, made = _client(), []
    keys = [lookup_key(p.slug, i) for p in plans.PLANS.values() if p.paid for i in plans.INTERVALS]
    existing = {pr["lookup_key"] for pr in _plain(client.v1.prices.list(params={"lookup_keys": keys, "limit": 100}))["data"]}
    for p in plans.PLANS.values():
        if not p.paid or all(lookup_key(p.slug, i) in existing for i in plans.INTERVALS):
            continue
        product = _plain(client.v1.products.create(params={
            "name": f"AI Every Time {p.name}", "description": p.tagline, "metadata": {"aiet_tier": p.slug}}))
        for interval in plans.INTERVALS:
            if lookup_key(p.slug, interval) in existing:
                continue
            client.v1.prices.create(params={
                "product": product["id"], "currency": "usd", "unit_amount": p.price(interval),
                "recurring": {"interval": "year" if interval == "yearly" else "month"},
                "lookup_key": lookup_key(p.slug, interval), "nickname": f"{p.name} {interval}",
                "metadata": {"aiet_tier": p.slug, "aiet_interval": interval}})
            made.append(lookup_key(p.slug, interval))
    _price_ids.clear()
    return made


def price_rows() -> list[dict]:
    """The Stripe price behind each paid plan and period, and whether it costs what the plan says."""
    rows = []
    for p in plans.PLANS.values():
        for interval in plans.INTERVALS if p.paid else ():
            row = {"tier": p.slug, "name": p.name, "interval": interval, "expected": p.price(interval)}
            try:
                pid = price_id(p.slug, interval)
                price = _plain(_client().v1.prices.retrieve(pid))
            except HTTPException as e:
                if e.status_code != 503:  # Stripe couldn't be reached, or refused the key: not a missing price
                    raise
                rows.append(row | {"price": None, "amount": None, "ok": False})
                continue
            rows.append(row | {"price": pid, "amount": price.get("unit_amount"), "ok": price.get("unit_amount") == p.price(interval)})
    return rows


def check() -> int:
    """Print the Stripe price behind each plan; return how many are missing or don't cost what the plan says."""
    rows = price_rows()
    for r in rows:
        print(f"{r['tier']} {r['interval']}: " + ("no price" if r["price"] is None else
              f"{r['price']} {plans.money(r['amount'] or 0)}" + ("" if r["ok"] else f" (the plan says {plans.money(r['expected'])})")))
    return sum(not r["ok"] for r in rows)


@router.get("/staff/stripe")
def stripe_status(user: dict = Admin):
    """For admins: whether payments are set up, the webhook Stripe needs, and the price behind each plan."""
    with db.connect() as conn:
        one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
        out = {"payments": payments(), "webhook_secret": bool(settings.stripe_webhook_secret), "paywall": paywall(),
               "webhook_url": f"{settings.public_url}/api/billing/webhook", "events": WEBHOOK_EVENTS,
               "linked": one("SELECT COUNT(*) FROM subscriptions WHERE stripe_subscription_id IS NOT NULL"),
               "queued_cancellations": one("SELECT COUNT(*) FROM stripe_cancellations"),
               "last_event": one("SELECT MAX(received_at) FROM stripe_events"), "prices": [], "prices_error": None}
    if payments():
        try:
            out["prices"] = price_rows()
        except (HTTPException, stripe.StripeError):
            out["prices_error"] = "Stripe couldn't be reached with the secret key set on the server. Check STRIPE_SECRET_KEY."
    return out


@router.post("/staff/stripe/setup")
def stripe_setup(user: dict = Admin):
    """For admins: make the plans' products and prices in Stripe (what `python -m app.billing setup` does)."""
    _need_payments()
    with _stripe("make the plans"):
        made = setup()
        return {"made": made, "prices": price_rows()}


class ReconcileIn(BaseModel):
    dry_run: bool = True
    end_gone: bool = False  # end the plans of subscriptions Stripe doesn't have


@router.post("/staff/stripe/reconcile")
def stripe_reconcile(body: ReconcileIn, user: dict = Admin):
    """For admins: bring every subscription up to date from Stripe (what `python -m app.billing reconcile` does)."""
    _need_payments()
    with _stripe("check the subscriptions"):
        report = reconcile(body.dry_run, body.end_gone)
    if not body.dry_run:
        from app import staff

        with db.connect() as conn:
            staff.log(conn, user["id"], "user", user["id"], "reconcile", f"{report['updated']} subscriptions brought up to date")
    return report


def reconcile(dry_run: bool = False, end_gone: bool = False) -> dict:
    """Bring every subscription with a Stripe subscription up to date from Stripe, and cancel any queued for
    deleted accounts. A subscription Stripe doesn't have is left as it is unless `end_gone`, which ends the plan (for
    once a person has looked into them). Reports counts only."""
    report = {"checked": 0, "updated": 0, "gone": 0, "ended": 0, "failed": 0, "changes": {}}
    with db.connect() as conn:
        subs = [dict(r) for r in conn.execute("SELECT * FROM subscriptions WHERE stripe_subscription_id IS NOT NULL")]
    client = _client()
    for sub in subs:
        report["checked"] += 1
        try:
            found = _plain(client.v1.subscriptions.retrieve(sub["stripe_subscription_id"]))
        except stripe.InvalidRequestError:
            report["gone"] += 1  # not in this Stripe account
            if end_gone and not dry_run and sub["status"] in LIVE:
                with db.connect() as conn:
                    before = effective_tier(row(conn, sub["user_id"]))
                    conn.execute("UPDATE subscriptions SET status = 'canceled', updated_at = ? WHERE user_id = ?",
                                 (db.now(), sub["user_id"]))
                    _record_change(conn, sub["user_id"], before, effective_tier(row(conn, sub["user_id"])), "admin")
                report["ended"] += 1
            continue
        except stripe.StripeError:
            report["failed"] += 1
            continue
        item = ((found.get("items") or {}).get("data") or [{}])[0]
        new = (price_plan(item.get("price"), found.get("metadata")) or (sub["tier"], sub["billing_interval"]))[0]
        status = found.get("status") if found.get("status") in STATUSES else "canceled"
        if (new, status) != (sub["tier"], sub["status"]):
            key = f"{sub['tier']}/{sub['status']} -> {new}/{status}"
            report["changes"][key] = report["changes"].get(key, 0) + 1
        if not dry_run:
            apply(found, sub["user_id"])
            report["updated"] += 1
    if not dry_run:
        report["deleted_accounts_cancelled"] = cancel_queued()
    return report


def main(argv: list[str]) -> int:
    import json

    if not payments():
        print("Set STRIPE_SECRET_KEY first.", file=sys.stderr)
        return 2
    db.init_db()
    command = argv[0] if argv else ""
    if command == "setup":
        for key in setup():
            print(f"made {key}")
        print("Stripe has a price for every plan.")
        return 0
    if command == "check":
        return 1 if check() else 0
    if command == "reconcile":
        print(json.dumps(reconcile(dry_run="--dry-run" in argv, end_gone="--end-gone" in argv), indent=2))
        return 0
    print(__doc__.split("\n\n")[-1], file=sys.stderr)
    return 2


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main(sys.argv[1:]))
