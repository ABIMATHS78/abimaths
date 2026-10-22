"""Emails about billing, and emails with news and offers.

Billing emails go to whoever they concern: a plan started, a trial about to end, a payment that didn't go through,
a plan that ended. News and offers go only to learners who said yes to them (at sign-up, or later in their profile),
whose address is confirmed, and every one carries a one-click unsubscribe:

- a reminder a day or two after a learner chose a plan on the pricing page without finishing checkout;
- a short weekly series for learners on the free plan, from their fourth day, on what the plans add (seven at most);
- emails an admin writes, to everyone on the free plan or every learner, sent a batch at a time.

Plan emails go out only while the paywall is on and payments are set up. Links are tagged (utm_source=email,
utm_campaign=...) so it's clear which emails bring people back. Every email is noted in email_log before it goes,
so none goes twice, even across restarts.
"""

from __future__ import annotations

import asyncio
import html
import logging
import smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth, billing, catalog, db, digest, plans
from app.config import settings

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/staff/campaigns")
Admin = Depends(auth.require("admin"))
CHECK_EVERY_SECONDS = 15 * 60
BATCH = 300  # emails from one campaign per run, so a big send is spread out
NUDGE_FROM_DAYS = 3
NUDGE_EVERY_DAYS = 7
FOLLOW_UP_AFTER_HOURS, FOLLOW_UP_BEFORE_HOURS = 24, 48
# Who news and offers may go to.
AGREED = "u.marketing_emails = 1 AND u.email_confirmed_at IS NOT NULL"


def link(route: str, campaign: str) -> str:
    """A link into the app, tagged with the email it came from."""
    return f"{settings.public_url}/?utm_source=email&utm_medium=email&utm_campaign={campaign}#/{route}"


def _date(iso: str | None) -> str:
    when = datetime.fromisoformat(iso) if iso else None
    return f"{when.day} {when:%B %Y}" if when else ""


def compose(user: dict, subject: str, paragraphs: list[str], button: tuple[str, str] | None = None,
            news_token: str | None = None) -> EmailMessage:
    """An email in the site's layout. News and offers (given the learner's unsubscribe token) can be stopped in
    one click, from the mail program or the link at the bottom."""
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = settings.mail_from
    msg["To"] = user["email"]
    stop = f"{settings.public_url}/api/email/unsubscribe?token={news_token}&list=news" if news_token else None
    if stop:
        msg["List-Unsubscribe"] = f"<{stop}>"
        msg["List-Unsubscribe-Post"] = "List-Unsubscribe=One-Click"
    hello = f"Hi {user['display_name']},"
    paragraphs = [p for p in paragraphs if p]
    why = ("You get this because you said yes to news and offers from AI Every Time."
           if stop else "You get this because it's about your AI Every Time account.")
    text = [hello, *paragraphs]
    if button:
        text.append(f"{button[0]}: {button[1]}")
    text.append(why + (f" Stop these emails: {stop}" if stop else ""))
    msg.set_content("\n\n".join(text) + "\n")
    cta = (f'<p style="margin:24px 0"><a href="{html.escape(button[1])}" style="background:#5b4bdb;color:#fff;'
           f'padding:12px 20px;border-radius:999px;text-decoration:none;font-weight:600">{html.escape(button[0])}</a></p>'
           if button else "")
    body = "".join(f"<p>{html.escape(p)}</p>" for p in paragraphs)
    footer = html.escape(why) + (f' <a href="{html.escape(stop)}" style="color:#777">Stop these emails</a>.' if stop else "")
    msg.add_alternative(
        '<div style="font:16px/1.5 system-ui,sans-serif;color:#1d1b2c;max-width:560px;margin:0 auto;padding:24px">'
        f'<p><img src="{settings.public_url}/static/icons/logo-256.webp" width="48" height="48" alt="AI Every Time"></p>'
        f"<p>{html.escape(hello)}</p>{body}{cta}"
        f'<p style="color:#777;font-size:12px;margin-top:32px">{footer}</p></div>', subtype="html")
    return msg


def _claim(conn, user_id: int, kind: str, key: str) -> int | None:
    """Note that an email is going, unless it went already: the email_log row's id, or None."""
    cur = conn.execute("INSERT OR IGNORE INTO email_log (user_id, kind, key, sent_at) VALUES (?, ?, ?, ?)",
                       (user_id, kind, key, db.now()))
    return cur.lastrowid if cur.rowcount else None


def _send(msg: EmailMessage, log_id: int, user_id: int) -> bool:
    try:
        digest.send(msg)
        return True
    except (smtplib.SMTPException, OSError):
        log.exception("email to user %s failed", user_id)
        with db.connect() as conn:
            conn.execute("UPDATE email_log SET failed = 1 WHERE id = ?", (log_id,))
        return False


# ----------------------------------------------------------------- billing emails


def _billing_message(user: dict, kind: str, sub: dict) -> EmailMessage | None:
    p = plans.plan(sub["tier"])
    billing_link = f"{settings.public_url}/#/billing"
    if kind == "started":
        if sub["status"] == "trialing":
            first = (f"Your {p.name} plan has started, with a free trial until {_date(sub['trial_end'])}. You won't be "
                     "charged before then, and you can cancel any time from your Billing page.")
        else:
            first = f"Your {p.name} plan has started. Thank you for learning with AI Every Time."
        adds = [name for f, name in plans.FEATURES.items() if f in p.unlocks]
        return compose(user, f"Welcome to {p.name}", [first, "It includes: " + "; ".join(adds) + "." if adds else ""],
                       ("Start learning", f"{settings.public_url}/#/courses"))
    if kind == "trial_ending":
        price = plans.money(sub["amount_cents"] if sub["amount_cents"] is not None else p.price(sub["billing_interval"]))
        per = "year" if sub["billing_interval"] == "yearly" else "month"
        return compose(user, f"Your free trial ends on {_date(sub['trial_end'])}", [
            f"Your {p.name} plan begins when your trial ends, at {price} a {per}. To keep it, there's nothing to do.",
            "If it isn't for you, cancel from your Billing page before then and you won't be charged."],
            ("Manage billing", billing_link))
    if kind == "payment_failed":
        return compose(user, "Your payment didn't go through", [
            f"Your bank declined the payment for your {p.name} plan. It will be tried again over the next few days.",
            "To keep your plan, check or update your card from your Billing page."], ("Update your card", billing_link))
    if kind == "ended":
        return compose(user, f"Your {p.name} plan has ended", [
            "You're on the free plan now. Your progress, certificates and companions are all still here.",
            "You can pick a plan again whenever you like."], ("See the plans", f"{settings.public_url}/#/pricing"))
    return None


def billing_email(user_id: int, kind: str, key: str) -> bool:
    """An email about the learner's plan, at most once per `key` (a subscription or invoice id)."""
    if not digest.email_available():
        return False
    with db.connect() as conn:
        user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if user is None:
            return False
        msg = _billing_message(dict(user), kind, billing.row(conn, user_id))
        log_id = _claim(conn, user_id, kind, key) if msg else None
    return _send(msg, log_id, user_id) if log_id else False


# ------------------------------------------------------------------ news and offers


# The weekly series for learners on the free plan: (subject, what it says, button, where it goes).
NUDGES = (
    ("Make the most of your first week", [
        "Your companions are ready whenever you are: ask them anything, from your first prompt to building an agent.",
        "When you want structure, every course's first module is free, and Plus opens the rest: labs, projects and "
        "certificates."], "Browse courses", "courses"),
    ("Finish a course, earn the certificate", [
        "Each course ends with a capstone project your instructor reviews, and a certificate you can share.",
        "With Plus you can take every module of every course, and earn Professional Certificates across a series."],
     "See Plus", "pricing"),
    ("Read every guide", [
        "Our education guides walk through tools and ideas step by step, with an infographic for each.",
        "Free accounts read {free_guides} a month in full. Digital and up read them all."], "See the plans", "pricing"),
    ("Practise like it's exam day", [
        "Mock exams cover a whole course under time, then show what to review.",
        "They come with Plus and up, along with every course in full."], "See the plans", "pricing"),
    ("A learning path made just for you", [
        "Tell your companion what you want to learn, and with Pro it designs a path on that topic, with lessons and "
        "quizzes."], "See Pro", "pricing"),
    ("Invite three friends, get a month of Plus", [
        "Share your invite link. When three friends join and confirm their email address, a free month of Plus is yours."],
     "Get your link", "referral"),
    ("Try any plan free for {trial_days} days", [
        "Every paid plan starts with a free trial. Cancel before it ends and you won't be charged."],
     "Start your free trial", "pricing"),
)


def _news_token(conn, user_id: int) -> str:
    return digest.ensure_token(conn, user_id)


def _news(conn, user: dict, kind: str, key: str, subject: str, paragraphs: list[str], button: tuple[str, str]) -> bool:
    log_id = _claim(conn, user["id"], kind, key)
    if log_id is None:
        return False
    msg = compose(user, subject, paragraphs, button, _news_token(conn, user["id"]))
    conn.commit()  # noted before it goes, so a crash mid-send doesn't send it twice
    return _send(msg, log_id, user["id"])


def _selling() -> bool:
    return billing.paywall() and billing.payments()


def follow_ups(now: datetime | None = None) -> int:
    """Remind learners who chose a plan a day or two ago and didn't finish checkout."""
    if not _selling():
        return 0
    now = now or datetime.now(timezone.utc)
    early, late = ((now - timedelta(hours=h)).isoformat(timespec="seconds") for h in (FOLLOW_UP_BEFORE_HOURS, FOLLOW_UP_AFTER_HOURS))
    sent = 0
    with db.connect() as conn:
        rows = conn.execute(f"SELECT u.*, s.intended_tier, s.intended_interval, s.intended_at FROM users u "
                            f"JOIN subscriptions s ON s.user_id = u.id WHERE {AGREED} AND s.intended_tier IS NOT NULL "
                            "AND s.intended_at BETWEEN ? AND ?", (early, late)).fetchall()
        for r in rows:
            sub = billing.row(conn, r["id"])
            if plans.rank(billing.effective_tier(sub)) >= plans.rank(r["intended_tier"]):
                continue
            p = plans.plan(r["intended_tier"])
            trial = (f" It starts with {settings.trial_days} days free, and you can cancel any time."
                     if not sub["trial_used"] and not sub["stripe_subscription_id"] and settings.trial_days else "")
            sent += _news(conn, dict(r), "follow_up", r["intended_at"], f"Still thinking about {p.name}?", [
                f"You chose {p.name} on AI Every Time but didn't get to the end of checkout.{trial}",
                f"{p.name} adds: " + "; ".join(h.format(free_guides=settings.free_guides, courses=len(catalog.data().courses))
                                              for h in p.highlights[1:3]) + "."],
                ("Finish upgrading", link("billing", "checkout-follow-up")))
    return sent


def nudges(now: datetime | None = None) -> int:
    """The weekly series for learners on the free plan."""
    if not _selling():
        return 0
    now = now or datetime.now(timezone.utc)
    joined_before = (now - timedelta(days=NUDGE_FROM_DAYS)).isoformat(timespec="seconds")
    last_before = (now - timedelta(days=NUDGE_EVERY_DAYS)).isoformat(timespec="seconds")
    sent = 0
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT u.*, (SELECT COUNT(*) FROM email_log e WHERE e.user_id = u.id AND e.kind = 'nudge') AS step, "
            f"(SELECT MAX(sent_at) FROM email_log e WHERE e.user_id = u.id AND e.kind = 'nudge') AS last FROM users u "
            f"WHERE {AGREED} AND u.created_at < ?", (joined_before,)).fetchall()
        for r in rows:
            if r["step"] >= len(NUDGES) or (r["last"] and r["last"] > last_before) or sent >= BATCH:
                continue
            sub = billing.row(conn, r["id"])
            if billing.effective_tier(sub) != "free" or auth.role_of(dict(r)) == "admin":
                continue
            subject, paragraphs, label, route = NUDGES[r["step"]]
            fill = {"free_guides": settings.free_guides, "trial_days": settings.trial_days}
            sent += _news(conn, dict(r), "nudge", str(r["step"] + 1), subject.format(**fill),
                          [p.format(**fill) for p in paragraphs], (label, link(route, f"nudge-{r['step'] + 1}")))
    return sent


def _audience_sql(audience: str) -> str:
    return AGREED if audience == "everyone" else (
        f"{AGREED} AND NOT EXISTS (SELECT 1 FROM subscriptions s WHERE s.user_id = u.id AND "
        "((s.tier != 'free' AND s.status IN ('active', 'trialing', 'past_due')) OR "
        "(s.grant_tier IS NOT NULL AND (s.grant_until IS NULL OR s.grant_until > :now))))")


def campaign_batch() -> int:
    """Send the next batch of the oldest unfinished campaign."""
    sent = 0
    with db.connect() as conn:
        c = conn.execute("SELECT * FROM campaigns WHERE finished_at IS NULL ORDER BY id LIMIT 1").fetchone()
        if c is None:
            return 0
        rows = conn.execute(
            f"SELECT u.* FROM users u WHERE {_audience_sql(c['audience'])} AND NOT EXISTS (SELECT 1 FROM email_log e "
            "WHERE e.user_id = u.id AND e.kind = 'campaign' AND e.key = :key) ORDER BY u.id LIMIT :batch",
            {"now": db.now(), "key": str(c["id"]), "batch": BATCH}).fetchall()
        for r in rows:
            sent += _news(conn, dict(r), "campaign", str(c["id"]), c["subject"], c["body"].split("\n\n"),
                          ("Open AI Every Time", link("", f"campaign-{c['id']}")))
        if len(rows) < BATCH:
            conn.execute("UPDATE campaigns SET finished_at = ? WHERE id = ?", (db.now(), c["id"]))
    return sent


def run(now: datetime | None = None) -> dict:
    if not digest.email_available():
        return {}
    return {"follow_up": follow_ups(now), "nudge": nudges(now), "campaign": campaign_batch()}


async def loop() -> None:
    """Every quarter of an hour: due emails, and cancelling deleted accounts' subscriptions that Stripe couldn't
    take at the time."""
    while True:
        try:
            await asyncio.to_thread(run)
            if billing.payments():
                await asyncio.to_thread(billing.cancel_queued)
        except Exception:
            log.exception("campaign run failed")
        await asyncio.sleep(CHECK_EVERY_SECONDS)


# -------------------------------------------------------------------------- admins


class CampaignIn(BaseModel):
    subject: str = Field(min_length=3, max_length=150)
    body: str = Field(min_length=10, max_length=5000)  # paragraphs separated by blank lines
    audience: Literal["free", "everyone"] = "free"
    test: bool = False  # send it to yourself only, to see it first


def _audience_size(audience: str) -> int:
    with db.connect() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM users u WHERE {_audience_sql(audience)}", {"now": db.now()}).fetchone()[0]


@router.post("", status_code=201)
def create(body: CampaignIn, user: dict = Admin):
    """Send an email to everyone on the free plan (or every learner) who said yes to news and offers, a batch every
    quarter of an hour. With test, it goes only to you."""
    if not digest.email_available():
        raise HTTPException(409, "Email isn't set up on this server")
    paragraphs = [p.strip() for p in body.body.strip().split("\n\n") if p.strip()]
    if body.test:
        with db.connect() as conn:
            token = _news_token(conn, user["id"])
        msg = compose(user, f"[Test] {body.subject}", paragraphs, ("Open AI Every Time", link("", "campaign-test")), token)
        try:
            digest.send(msg)
        except (smtplib.SMTPException, OSError):
            raise HTTPException(502, "The test email couldn't be sent. Check the SMTP settings.")
        return {"test": True, "sent": 1}
    with db.connect() as conn:
        cid = conn.execute("INSERT INTO campaigns (subject, body, audience, created_by, created_at) VALUES (?, ?, ?, ?, ?)",
                           (body.subject.strip(), "\n\n".join(paragraphs), body.audience, user["id"], db.now())).lastrowid
    return {"id": cid, "audience": _audience_size(body.audience)}


@router.get("")
def campaigns(user: dict = Admin):
    with db.connect() as conn:
        rows = conn.execute("SELECT c.id, c.subject, c.audience, c.created_at, c.finished_at, u.display_name AS created_by, "
                            "(SELECT COUNT(*) FROM email_log e WHERE e.kind = 'campaign' AND e.key = CAST(c.id AS TEXT) "
                            "AND e.failed = 0) AS sent FROM campaigns c LEFT JOIN users u ON u.id = c.created_by "
                            "ORDER BY c.id DESC LIMIT 20").fetchall()
    return {"campaigns": [dict(r) for r in rows],
            "audiences": {"free": _audience_size("free"), "everyone": _audience_size("everyone")},
            "email_available": digest.email_available()}
