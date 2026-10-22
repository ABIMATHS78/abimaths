"""Bring the old site's accounts and subscriptions over, keeping everyone's plan and how they sign in.

Run it on the server, with the two exports from the old site's database (they hold people's details, so they never go
in the repository, in a chat or anywhere else but the server):

    python scripts/import_legacy.py --users users.csv --subscriptions subscriptions.csv --dry-run
    python scripts/import_legacy.py --users users.csv --subscriptions subscriptions.csv --report import-report.json

--dry-run does everything and then undoes it, to see the report first. What it prints, and the report it writes,
hold counts and the old site's ids only: never an email address, a name, a password hash or a whole Stripe id.

For each account it keeps the old site's id, when it was made and last signed in, how it signed in, and its role, and
its password: the old site's bcrypt hashes work here as they are (and are stored this site's way at the next
sign-in). Accounts that signed in with Google, Apple or Microsoft have no password; they choose one with a reset
link (the address is taken as confirmed, as those services confirmed it). Sessions and reset links aren't brought
over, so everyone signs in again. Nobody is signed up for news and offers: that needs a yes here.

Each account keeps its plan, from its subscription: the plan, monthly or yearly, its status, its Stripe customer and
subscription, the period it's paid up to, and a plan chosen at checkout but not paid for. Paid plans carry on as they
are until Stripe says otherwise: run `python -m app.billing reconcile` once Stripe is set up, to bring each up to date
from Stripe. Accounts with no subscription start on the free plan. An account with two subscription rows keeps the
better one, and the report lists the other.

Where several accounts share an email address (the old site told case apart: Ada@ and ada@), the one with the best
plan becomes the account here (then a password, then the latest sign-in); the others are kept in legacy_accounts,
linked to it, and listed in the report. They are never merged silently. An address already used by an account made
here is left alone and reported, unless --link-existing links the two (keeping the password chosen here).

Running it again adds accounts that weren't there before and leaves the ones already brought over as they are;
--refresh-subscriptions updates their plans from a newer export instead (for the final export, before switching over).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from email_validator import EmailNotValidError, validate_email  # noqa: E402

from app import billing, companions, db, plans, reminders  # noqa: E402

USER_COLUMNS = {"id", "name", "email", "loginMethod", "role", "createdAt", "lastSignedIn", "passwordHash"}
SUB_COLUMNS = {"id", "userId", "tier", "billingInterval", "stripeCustomerId", "stripeSubscriptionId", "status",
               "currentPeriodStart", "currentPeriodEnd", "cancelAtPeriodEnd", "createdAt", "updatedAt", "intendedTier",
               "intendedBillingInterval", "intendedAt"}
BCRYPT = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")
CONFIRMED_BY = {"google", "apple", "microsoft"}  # sign-in services that confirmed the address
STATUS = {"active": "active", "trialing": "trialing", "past_due": "past_due", "canceled": "canceled",
          "incomplete": "incomplete", "unpaid": "unpaid", "paused": "paused"}
MAX_PROBLEMS = 500


def _blank(v: str | None) -> str:
    v = (v or "").strip()
    return "" if v.upper() == "NULL" else v


def _when(v: str | None) -> str | None:
    """The old site's times, "2026-01-31 09:30:00" (UTC) or milliseconds since 1970, as this site's."""
    v = _blank(v)
    if not v:
        return None
    if v.isdigit():
        n = int(v)
        return datetime.fromtimestamp(n / 1000 if n > 10**11 else n, timezone.utc).isoformat(timespec="seconds")
    try:
        when = datetime.fromisoformat(v.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (when if when.tzinfo else when.replace(tzinfo=timezone.utc)).astimezone(timezone.utc).isoformat(timespec="seconds")


def _stripe_id(v: str | None, prefix: str) -> str | None:
    v = _blank(v)
    return v if v.startswith(prefix) and re.fullmatch(r"[A-Za-z0-9_]{6,255}", v) else None


def masked(stripe_id: str | None) -> str | None:
    return f"{stripe_id[:4]}…{stripe_id[-4:]}" if stripe_id else None


def read(path: str, columns: set[str]) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh)
        missing = columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path} is missing the columns {', '.join(sorted(missing))}")
        return list(reader)


def _live_rank(sub: dict | None) -> tuple:
    """How good a subscription row is: a plan in use first, then the better plan, then the latest."""
    if sub is None:
        return (0, 0, "")
    live = sub["status"] in billing.LIVE
    return (int(live), plans.rank(sub["tier"]) if live else 0, sub["updated_at"] or "")


def clean_subscription(s: dict, problems: list) -> dict:
    tier = _blank(s["tier"]).lower()
    if tier not in plans.TIERS:
        problems.append({"subscription": s["id"], "issue": "unknown plan, so free"})
        tier = "free"
    status = STATUS.get(_blank(s["status"]).lower())
    if status is None:
        problems.append({"subscription": s["id"], "issue": "unknown status, so canceled"})
        status = "canceled"
    intended = _blank(s["intendedTier"]).lower()
    intended_interval = _blank(s["intendedBillingInterval"]).lower()
    customer, subscription = _stripe_id(s["stripeCustomerId"], "cus_"), _stripe_id(s["stripeSubscriptionId"], "sub_")
    if tier != "free" and status in billing.LIVE and not subscription:
        problems.append({"subscription": s["id"], "issue": "a paid plan without a Stripe subscription"})
    return {
        "legacy_id": _blank(s["id"]), "user": _blank(s["userId"]), "tier": tier,
        "billing_interval": "yearly" if _blank(s["billingInterval"]).lower() == "yearly" else "monthly",
        "status": status, "stripe_customer_id": customer, "stripe_subscription_id": subscription,
        "current_period_start": _when(s["currentPeriodStart"]), "current_period_end": _when(s["currentPeriodEnd"]),
        "cancel_at_period_end": int(_blank(s["cancelAtPeriodEnd"]) in ("1", "true", "TRUE")),
        "intended_tier": intended if intended in plans.TIERS and intended != "free" else None,
        "intended_interval": intended_interval if intended_interval in plans.INTERVALS else None,
        "intended_at": _when(s["intendedAt"]), "created_at": _when(s["createdAt"]) or db.now(),
        "updated_at": _when(s["updatedAt"]) or _when(s["createdAt"]) or db.now(),
        "conversations_used": int(_blank(s.get("companionConversationsUsed")) or 0)
        if _blank(s.get("companionConversationsUsed")).isdigit() else 0,
    }


def plan_import(users: list[dict], subs: list[dict]) -> tuple[list[dict], dict]:
    """Work out, without touching the database, what each account becomes. Returns the accounts to bring over (each
    with its subscription and the accounts folded into it) and the report so far."""
    problems: list = []
    report: dict = {"source": {"accounts": len(users), "subscriptions": len(subs)}}
    by_user: dict[str, list[dict]] = {}
    for s in subs:
        clean = clean_subscription(s, problems)
        by_user.setdefault(clean["user"], []).append(clean)
    extra = []
    best_sub: dict[str, dict] = {}
    for uid, rows in by_user.items():
        rows.sort(key=_live_rank, reverse=True)
        best_sub[uid] = rows[0]
        for other in rows[1:]:
            extra.append({"account": uid, "subscription": other["legacy_id"], "plan": other["tier"], "status": other["status"],
                          "stripe_subscription": masked(other["stripe_subscription_id"])})
    known = {_blank(u["id"]) for u in users}
    orphans = [uid for uid in by_user if uid not in known]

    accounts: dict[str, list[dict]] = {}
    invalid = 0
    for u in users:
        legacy_id = _blank(u["id"])
        try:
            email = validate_email(_blank(u["email"]), check_deliverability=False).normalized.lower()
        except EmailNotValidError:
            invalid += 1
            problems.append({"account": legacy_id, "issue": "no valid email address, so not brought over"})
            continue
        hash_ = _blank(u["passwordHash"])
        if hash_ and not BCRYPT.match(hash_):
            problems.append({"account": legacy_id, "issue": "password hash isn't bcrypt, so they reset it"})
            hash_ = ""
        method = _blank(u["loginMethod"]).lower() or "email"
        accounts.setdefault(email, []).append({
            "legacy_id": legacy_id, "email": email, "name": re.sub(r"\s+", " ", _blank(u["name"]))[:60],
            "password_hash": hash_, "login_method": method, "role": "admin" if _blank(u["role"]).lower() == "admin" else "learner",
            "created_at": _when(u["createdAt"]) or db.now(), "last_signed_in": _when(u["lastSignedIn"]),
            "subscription": best_sub.get(legacy_id),
        })

    chosen, folded, conflicts = [], 0, []
    for email, group in accounts.items():
        group.sort(key=lambda a: (_live_rank(a["subscription"]), bool(a["subscription"] and a["subscription"]["stripe_subscription_id"]),
                                  bool(a["password_hash"]), a["last_signed_in"] or ""), reverse=True)
        primary, others = group[0], group[1:]
        primary["folded"] = others
        folded += len(others)
        paying = [a for a in group if a["subscription"] and a["subscription"]["status"] in billing.LIVE
                  and a["subscription"]["tier"] != "free"]
        if len(paying) > 1:
            conflicts.append({"kept": primary["legacy_id"], "also_paying": [a["legacy_id"] for a in paying[1:]],
                              "note": "two paid plans for one address: check them in Stripe"})
        if others:
            problems.append({"account": primary["legacy_id"], "issue": "shares its address with "
                             + ", ".join(a["legacy_id"] for a in others) + ", kept in legacy_accounts"})
        chosen.append(primary)
    report |= {
        "invalid_email": invalid, "folded_duplicates": folded, "duplicate_addresses": sum(1 for g in accounts.values() if len(g) > 1),
        "extra_subscription_rows": extra, "paid_conflicts": conflicts,
        "subscriptions_without_account": len(orphans), "problems": problems,
    }
    return chosen, report


def _insert_subscription(conn, user_id: int, sub: dict | None, refresh: bool) -> None:
    if sub is None:
        billing.row(conn, user_id)
        return
    paid_before = sub["stripe_subscription_id"] is not None or sub["tier"] != "free"
    values = (sub["tier"], sub["billing_interval"], sub["status"], sub["stripe_customer_id"], sub["stripe_subscription_id"],
              sub["current_period_start"], sub["current_period_end"], sub["cancel_at_period_end"], int(paid_before),
              sub["intended_tier"], sub["intended_interval"], sub["intended_at"], sub["legacy_id"], sub["created_at"],
              sub["updated_at"])
    columns = ("tier, billing_interval, status, stripe_customer_id, stripe_subscription_id, current_period_start, "
               "current_period_end, cancel_at_period_end, trial_used, intended_tier, intended_interval, intended_at, legacy_id, "
               "created_at, updated_at")
    update = ", ".join(f"{c.strip()} = excluded.{c.strip()}" for c in columns.split(",") if c.strip() != "created_at")
    conn.execute(f"INSERT INTO subscriptions (user_id, {columns}) VALUES (?, {', '.join('?' * len(values))}) "
                 f"ON CONFLICT (user_id) DO {'UPDATE SET ' + update if refresh else 'NOTHING'}", (user_id, *values))
    # The plan's history, for the analytics: when it started, and when it ended if it has.
    if sub["tier"] != "free" and not conn.execute("SELECT 1 FROM subscription_events WHERE user_id = ? AND source = 'import'",
                                                  (user_id,)).fetchone():
        billing._record_change(conn, user_id, "free", sub["tier"], "import", sub["created_at"])
        if sub["status"] not in billing.LIVE:
            billing._record_change(conn, user_id, sub["tier"], "free", "import", sub["updated_at"])


def run(users_csv: str, subs_csv: str, dry_run: bool, link_existing: bool, refresh: bool) -> dict:
    db.init_db()
    companions.seed_starters()  # so new accounts can follow the featured companions, as at sign-up
    chosen, report = plan_import(read(users_csv, USER_COLUMNS), read(subs_csv, SUB_COLUMNS))
    counts = Counter()
    passwords, methods, roles = Counter(), Counter(), Counter()
    with db.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        for a in chosen:
            sub = a["subscription"]
            found = conn.execute("SELECT id FROM users WHERE legacy_id = ?", (a["legacy_id"],)).fetchone()
            if found:
                counts["already_imported"] += 1
                if refresh:
                    _insert_subscription(conn, found["id"], sub, refresh=True)
                    counts["subscriptions_refreshed"] += 1
                continue
            taken = conn.execute("SELECT id, legacy_id FROM users WHERE email = ?", (a["email"],)).fetchone()
            if taken:
                if not link_existing or taken["legacy_id"]:
                    counts["address_taken_here"] += 1
                    report["problems"].append({"account": a["legacy_id"], "issue": "an account here already has its address"
                                               + ("" if link_existing else "; --link-existing links them")})
                    continue
                conn.execute("UPDATE users SET legacy_id = ?, login_method = ? WHERE id = ?",
                             (a["legacy_id"], a["login_method"], taken["id"]))
                current = billing.row(conn, taken["id"])
                if sub and _live_rank(sub) > _live_rank(current):
                    _insert_subscription(conn, taken["id"], sub, refresh=True)
                counts["linked_to_existing"] += 1
                user_id = taken["id"]
            else:
                confirmed = a["created_at"] if a["login_method"] in CONFIRMED_BY else None
                user_id = conn.execute(
                    "INSERT INTO users (email, display_name, password_hash, created_at, role, legacy_id, login_method, "
                    "last_signed_in, email_confirmed_at, signup_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'old-site')",
                    (a["email"], a["name"] or a["email"].split("@")[0][:60], a["password_hash"], a["created_at"], a["role"],
                     a["legacy_id"], a["login_method"], a["last_signed_in"], confirmed)).lastrowid
                conn.execute("INSERT INTO follows (user_id, companion_id, created_at) SELECT ?, id, ? FROM companions "
                             "WHERE is_starter = 1 AND featured = 1 AND retired = 0", (user_id, db.now()))
                reminders.mark_feed_seen(conn, user_id)
                try:
                    _insert_subscription(conn, user_id, sub, refresh=False)
                except sqlite3.IntegrityError:  # its Stripe subscription is someone else's here already
                    report["problems"].append({"account": a["legacy_id"], "issue": "its Stripe subscription belongs to "
                                               "another account, so it starts on the free plan"})
                    billing.row(conn, user_id)
                counts["created"] += 1
                passwords["bcrypt kept" if a["password_hash"] else "none: reset link needed"] += 1
                methods[a["login_method"]] += 1
                roles[a["role"]] += 1
            for other in a["folded"]:
                osub = other["subscription"] or {}
                conn.execute("INSERT OR IGNORE INTO legacy_accounts (legacy_id, user_id, login_method, tier, status, "
                             "stripe_customer_id, stripe_subscription_id, created_at, last_signed_in, imported_at) "
                             "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (other["legacy_id"], user_id, other["login_method"], osub.get("tier"), osub.get("status"),
                              osub.get("stripe_customer_id"), osub.get("stripe_subscription_id"), other["created_at"],
                              other["last_signed_in"], db.now()))
        plan_counts, statuses = Counter(), Counter()
        for r in conn.execute("SELECT s.* FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.legacy_id IS NOT NULL"):
            plan_counts[billing.effective_tier(dict(r))] += 1
            if r["tier"] != "free":
                statuses[f"{r['tier']} {r['status']}"] += 1
        stripe_subs = conn.execute("SELECT COUNT(*) FROM subscriptions s JOIN users u ON u.id = s.user_id "
                                   "WHERE u.legacy_id IS NOT NULL AND s.stripe_subscription_id IS NOT NULL").fetchone()[0]
        if dry_run:
            conn.rollback()
    report["problems"] = report["problems"][:MAX_PROBLEMS]
    return {"dry_run": dry_run, **report, "accounts": dict(counts), "passwords": dict(passwords),
            "sign_in_methods": dict(methods), "roles": dict(roles), "plans_in_use": dict(plan_counts),
            "paid_plans_by_status": dict(statuses), "stripe_subscriptions_to_reconcile": stripe_subs,
            "not_brought_over": ["sessions", "password reset links", "companion conversation counts (allowances start afresh)"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--users", required=True, help="the old site's users export (CSV)")
    parser.add_argument("--subscriptions", required=True, help="the old site's subscriptions export (CSV)")
    parser.add_argument("--dry-run", action="store_true", help="do everything, then undo it, and report")
    parser.add_argument("--link-existing", action="store_true", help="link old accounts to accounts made here with the same address")
    parser.add_argument("--refresh-subscriptions", action="store_true", help="update plans of accounts already brought over")
    parser.add_argument("--report", help="write the full report (counts and the old site's ids only) to this JSON file")
    args = parser.parse_args(argv)
    result = run(args.users, args.subscriptions, args.dry_run, args.link_existing, args.refresh_subscriptions)
    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    summary = {k: v for k, v in result.items() if k not in ("problems", "extra_subscription_rows", "paid_conflicts")}
    summary["problems"] = len(result["problems"])
    summary["extra_subscription_rows"] = len(result["extra_subscription_rows"])
    summary["paid_conflicts"] = len(result["paid_conflicts"])
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print("Dry run: nothing was saved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
