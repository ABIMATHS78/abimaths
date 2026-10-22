"""The admin Analytics page: growth, revenue, plans, conversations and referrals, worked out from the database each
time it's opened, over the last 7, 30 or 90 days.

It shows counts and money, never what anyone wrote: conversations are counted, not read. Learners appear by name,
with their email address shortened, and Stripe's ids aren't shown at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

from app import auth, billing, catalog, db, plans

router = APIRouter(prefix="/api/staff")


def _masked(email: str) -> str:
    name, _, domain = email.partition("@")
    return f"{name[:2]}•••@{domain}"


def _monthly(sub: dict) -> float:
    """What a subscription brings in a month, in cents: a yearly one's price over twelve months."""
    amount = sub["amount_cents"] if sub["amount_cents"] is not None else plans.plan(sub["tier"]).price(sub["billing_interval"])
    return amount / 12 if sub["billing_interval"] == "yearly" else amount


@router.get("/analytics")
def analytics(days: int = Query(30, ge=7, le=90), user: dict = Depends(auth.require("admin"))):
    now = datetime.now(timezone.utc)
    day_list = [(now - timedelta(days=i)).date().isoformat() for i in range(days - 1, -1, -1)]  # oldest first, UTC
    since = day_list[0] + "T00:00:00+00:00"
    with db.connect() as conn:
        def one(sql: str, *args):
            return conn.execute(sql, args).fetchone()[0]

        def by_day(sql: str, *args) -> dict:
            return {r[0]: r[1] for r in conn.execute(sql, args)}

        users_total = one("SELECT COUNT(*) FROM users")
        before = one("SELECT COUNT(*) FROM users WHERE created_at < ?", since)
        joined = by_day("SELECT substr(created_at, 1, 10), COUNT(*) FROM users WHERE created_at >= ? GROUP BY 1", since)
        growth, running = [], before
        for d in day_list:
            running += joined.get(d, 0)
            growth.append({"day": d, "new": joined.get(d, 0), "total": running})

        subs = [dict(r) for r in conn.execute("SELECT * FROM subscriptions")]
        with_rows = {s["user_id"] for s in subs}
        tiers = {t: 0 for t in plans.TIERS}
        tiers["free"] = users_total - len(with_rows)
        statuses, intervals, mrr, paying, trialing, given = {}, {"monthly": 0, "yearly": 0}, 0.0, 0, 0, 0
        for s in subs:
            current = billing.effective_tier(s)
            tiers[current] += 1
            if billing.paid_tier(s) != "free":
                statuses[s["status"]] = statuses.get(s["status"], 0) + 1
                intervals[s["billing_interval"]] += 1
                if s["status"] == "trialing":
                    trialing += 1
                else:
                    paying += 1
                    mrr += _monthly(s) if s["stripe_subscription_id"] else 0
            elif billing.granted_tier(s) != "free" or billing.reward_tier(s) != "free":
                given += 1
        open_intents = sum(1 for s in subs if s["intended_tier"]
                           and plans.rank(billing.effective_tier(s)) < plans.rank(s["intended_tier"]))

        revenue_days = {r["day"]: (r["payments"], r["refunds"]) for r in conn.execute(
            "SELECT substr(created_at, 1, 10) AS day, SUM(CASE WHEN kind = 'payment' THEN amount_cents ELSE 0 END) AS payments, "
            "SUM(CASE WHEN kind = 'refund' THEN amount_cents ELSE 0 END) AS refunds FROM revenue_events "
            "WHERE created_at >= ? AND currency = 'usd' GROUP BY day", (since,))}
        by_tier = [dict(r) for r in conn.execute(
            "SELECT COALESCE(tier, 'unknown') AS tier, SUM(amount_cents) AS amount, SUM(kind = 'payment') AS payments "
            "FROM revenue_events WHERE created_at >= ? AND currency = 'usd' GROUP BY 1 ORDER BY amount DESC", (since,))]
        recent = [{"when": r["created_at"], "name": r["display_name"] or "Deleted account",
                   "email": _masked(r["email"]) if r["email"] else "", "kind": r["kind"], "amount": r["amount_cents"],
                   "currency": r["currency"], "tier": r["tier"], "interval": r["billing_interval"]}
                  for r in conn.execute("SELECT e.*, u.display_name, u.email FROM revenue_events e "
                                        "LEFT JOIN users u ON u.id = e.user_id ORDER BY e.created_at DESC, e.id DESC LIMIT 20")]
        revenue_period = one("SELECT COALESCE(SUM(amount_cents), 0) FROM revenue_events WHERE created_at >= ? AND currency = 'usd'",
                             since)
        other_currencies = one("SELECT COUNT(*) FROM revenue_events WHERE created_at >= ? AND currency != 'usd'", since)

        flows = [dict(r) for r in conn.execute(
            "SELECT from_tier, to_tier, event_type, COUNT(*) AS count FROM subscription_events WHERE created_at >= ? "
            "GROUP BY 1, 2, 3 ORDER BY count DESC LIMIT 30", (since,))]
        kinds = ("new", "upgrade", "downgrade", "cancel", "reactivate")
        per_day: dict[str, dict] = {d: {k: 0 for k in kinds} for d in day_list}
        for r in conn.execute("SELECT substr(created_at, 1, 10) AS day, event_type, COUNT(*) AS n FROM subscription_events "
                              "WHERE created_at >= ? GROUP BY 1, 2", (since,)):
            if r["day"] in per_day:
                per_day[r["day"]][r["event_type"]] = r["n"]
        events = [{"when": r["created_at"], "name": r["display_name"] or "Deleted account", "from": r["from_tier"],
                   "to": r["to_tier"], "type": r["event_type"], "source": r["source"]}
                  for r in conn.execute("SELECT e.*, u.display_name FROM subscription_events e LEFT JOIN users u "
                                        "ON u.id = e.user_id ORDER BY e.created_at DESC, e.id DESC LIMIT 25")]
        period_counts = {k: sum(per_day[d][k] for d in day_list) for k in kinds}

        messages = by_day("SELECT substr(created_at, 1, 10), COUNT(*) FROM messages WHERE role = 'user' AND created_at >= ? "
                          "GROUP BY 1", since)
        conversations = {
            "threads": one("SELECT COUNT(*) FROM (SELECT 1 FROM messages GROUP BY user_id, companion_id)"),
            "messages": one("SELECT COUNT(*) FROM messages"),
            "messages_in_period": sum(messages.values()),
            "learners_in_period": one("SELECT COUNT(DISTINCT user_id) FROM messages WHERE role = 'user' AND created_at >= ?",
                                      since),
            "days": [{"day": d, "messages": messages.get(d, 0)} for d in day_list],
            "top_companions": [dict(r) for r in conn.execute(
                "SELECT c.id, json_extract(c.persona, '$.name') AS name, json_extract(c.persona, '$.flag') AS flag, "
                "COUNT(*) AS messages, COUNT(DISTINCT m.user_id) AS learners FROM messages m "
                "JOIN companions c ON c.id = m.companion_id WHERE m.role = 'user' AND m.created_at >= ? "
                "GROUP BY c.id ORDER BY messages DESC LIMIT 8", (since,))],
        }
        referrals = {
            "invites": one("SELECT COUNT(*) FROM referral_invites WHERE created_at >= ?", since),
            "counted": one("SELECT COUNT(*) FROM referral_invites WHERE qualified_at >= ?", since),
            "subscribed": one("SELECT COUNT(*) FROM referral_invites WHERE subscribed_at >= ?", since),
            "rewards_earned": one("SELECT COUNT(*) FROM referral_rewards WHERE created_at >= ?", since),
            "rewards_claimed": one("SELECT COUNT(*) FROM referral_rewards WHERE claimed_at >= ?", since),
            "top": [dict(r) for r in conn.execute(
                "SELECT u.display_name AS name, COUNT(*) AS counted FROM referral_invites r JOIN users u ON u.id = r.referrer_id "
                "WHERE r.qualified_at IS NOT NULL GROUP BY r.referrer_id ORDER BY counted DESC LIMIT 5")],
        }
        email = {
            "agreed_to_news": one("SELECT COUNT(*) FROM users WHERE marketing_emails = 1"),
            "sent": {r[0]: r[1] for r in conn.execute("SELECT kind, COUNT(*) FROM email_log WHERE sent_at >= ? AND failed = 0 "
                                                     "GROUP BY kind", (since,))},
            "signups_by_source": {r[0] or "direct": r[1] for r in conn.execute(
                "SELECT signup_source, COUNT(*) FROM users WHERE created_at >= ? GROUP BY 1 ORDER BY 2 DESC LIMIT 10", (since,))},
        }
        imported = {
            "accounts": one("SELECT COUNT(*) FROM users WHERE legacy_id IS NOT NULL"),
            "signed_in_here": one("SELECT COUNT(*) FROM users WHERE legacy_id IS NOT NULL AND password_hash LIKE 'scrypt$%'"),
            "without_password": one("SELECT COUNT(*) FROM users WHERE legacy_id IS NOT NULL AND password_hash = ''"),
            "folded_in": one("SELECT COUNT(*) FROM legacy_accounts"),
        }
        overview = {
            "users": users_total, "new_users": sum(joined.values()),
            "active_learners": one("SELECT COUNT(DISTINCT user_id) FROM activity WHERE day >= ?", day_list[0]),
            "paying": paying, "trialing": trialing, "given_plans": given, "mrr": round(mrr),
            "revenue": revenue_period,
            "revenue_all_time": one("SELECT COALESCE(SUM(amount_cents), 0) FROM revenue_events WHERE currency = 'usd'"),
            "payments_in_other_currencies": other_currencies,
            "checkouts_unfinished": open_intents, "guides": one("SELECT COUNT(*) FROM posts WHERE kind IS NOT NULL"),
            "courses": len(catalog.data().courses), "enrolments": one("SELECT COUNT(*) FROM course_enrollments"),
            "certificates": one("SELECT COUNT(*) FROM certificates"),
        }
    return {
        "days": days, "generated_at": db.now(), "paywall": billing.paywall(), "payments": billing.payments(),
        "overview": overview, "growth": growth,
        "revenue": {"days": [{"day": d, "payments": revenue_days.get(d, (0, 0))[0] or 0,
                              "refunds": revenue_days.get(d, (0, 0))[1] or 0} for d in day_list],
                    "by_tier": by_tier, "recent": recent},
        "subscriptions": {"tiers": [{"tier": t, "name": plans.plan(t).name, "users": n} for t, n in tiers.items()],
                          "statuses": statuses, "intervals": intervals, "flows": flows,
                          "days": [{"day": d, **per_day[d]} for d in day_list], "period": period_counts, "recent": events},
        "conversations": conversations, "referrals": referrals, "email": email, "imported": imported,
    }
