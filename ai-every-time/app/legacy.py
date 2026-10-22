"""Bringing the old site's accounts and subscriptions over, keeping everyone's plan and how they sign in. Run on
the server from the command line (scripts/import_legacy.py) or by an admin on the Staff page's Setup tab, with the
two exports from the old site's database: they hold people's details, so they go to the server and nowhere else.

Everything it reports is counts and the old site's ids: never an email address, a name, a password hash or a whole
Stripe id. See scripts/import_legacy.py for what it keeps, and how duplicates, missing passwords and running it again
are handled.
"""

from __future__ import annotations

import csv
import io
import re
import sqlite3
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from email_validator import EmailNotValidError, validate_email
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth, billing, companions, db, plans, reminders
from app.config import settings

router = APIRouter(prefix="/api/staff")

USER_COLUMNS = {"id", "name", "email", "loginMethod", "role", "createdAt", "lastSignedIn", "passwordHash"}
SUB_COLUMNS = {"id", "userId", "tier", "billingInterval", "stripeCustomerId", "stripeSubscriptionId", "status",
               "currentPeriodStart", "currentPeriodEnd", "cancelAtPeriodEnd", "createdAt", "updatedAt", "intendedTier",
               "intendedBillingInterval", "intendedAt"}
BCRYPT = re.compile(r"^\$2[aby]\$\d{2}\$[./A-Za-z0-9]{53}$")
OLD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # the old site's ids, the only cell values the report repeats
CONFIRMED_BY = {"google", "apple", "microsoft"}  # sign-in services that confirmed the address
LOGIN_METHODS = {"email", "google", "apple", "microsoft", "github", "facebook"}
STATUS = {"active": "active", "trialing": "trialing", "past_due": "past_due", "canceled": "canceled",
          "incomplete": "incomplete", "unpaid": "unpaid", "paused": "paused"}
MAX_PROBLEMS = 500
BATCH = 250  # accounts saved together; between batches other requests get their turn
RESET_SHARED = "reset link needed: the address had more than one account"


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


class BadExport(ValueError):
    """An export that isn't the old site's table: the columns it's missing."""


def read(text: str, columns: set[str], name: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(text.lstrip("\ufeff"), newline=""))
    missing = columns - set(reader.fieldnames or [])
    if missing:
        raise BadExport(f"The {name} export is missing the columns {', '.join(sorted(missing))}")
    return list(reader)


def _live_rank(sub: dict | None) -> tuple:
    """How good a subscription row is: a plan in use first, then the better plan, then the latest."""
    if sub is None:
        return (0, 0, "")
    live = sub["status"] in billing.LIVE
    return (int(live), plans.rank(sub["tier"]) if live else 0, sub["updated_at"] or "")


def clean_subscription(s: dict, problems: list) -> dict | None:
    sid, owner = _blank(s["id"]), _blank(s["userId"])
    if not (OLD_ID.match(sid) and OLD_ID.match(owner)):
        problems.append({"subscription": "(not an id)", "issue": "a subscription row without proper ids, left out"})
        return None
    tier = _blank(s["tier"]).lower()
    if tier not in plans.TIERS:
        problems.append({"subscription": sid, "issue": "unknown plan, so free"})
        tier = "free"
    status = STATUS.get(_blank(s["status"]).lower())
    if status is None:
        problems.append({"subscription": sid, "issue": "unknown status, so canceled"})
        status = "canceled"
    intended = _blank(s["intendedTier"]).lower()
    intended_interval = _blank(s["intendedBillingInterval"]).lower()
    customer, subscription = _stripe_id(s["stripeCustomerId"], "cus_"), _stripe_id(s["stripeSubscriptionId"], "sub_")
    period_end = _when(s["currentPeriodEnd"])
    grant = None
    if tier != "free" and status in billing.LIVE and not subscription:
        # Nothing at Stripe would ever end it, so it's given until the end of the period it was paid to, if there's one.
        if period_end and period_end <= db.now():
            problems.append({"subscription": sid, "issue": "a paid plan without a Stripe subscription, ended already, so free"})
        else:
            grant = (tier, period_end)
            problems.append({"subscription": sid, "issue": "a paid plan without a Stripe subscription, given "
                             + ("until the end of its period" if period_end else "for good: check it")})
        status = "canceled"
    return {
        "legacy_id": sid, "user": owner, "tier": tier,
        "billing_interval": "yearly" if _blank(s["billingInterval"]).lower() == "yearly" else "monthly",
        "status": status, "stripe_customer_id": customer, "stripe_subscription_id": subscription,
        "current_period_start": _when(s["currentPeriodStart"]), "current_period_end": period_end,
        "cancel_at_period_end": int(_blank(s["cancelAtPeriodEnd"]) in ("1", "true", "TRUE")),
        "intended_tier": intended if intended in plans.TIERS and intended != "free" else None,
        "intended_interval": intended_interval if intended_interval in plans.INTERVALS else None,
        "intended_at": _when(s["intendedAt"]), "created_at": _when(s["createdAt"]) or db.now(),
        "updated_at": _when(s["updatedAt"]) or _when(s["createdAt"]) or db.now(), "grant": grant,
    }


def plan_import(users: list[dict], subs: list[dict]) -> tuple[list[dict], dict]:
    """Work out, without touching the database, what each account becomes. Returns the accounts to bring over (each
    with its subscription and the accounts folded into it) and the report so far."""
    problems: list = []
    report: dict = {"source": {"accounts": len(users), "subscriptions": len(subs)}}
    by_user: dict[str, list[dict]] = {}
    for s in subs:
        if clean := clean_subscription(s, problems):
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
        if not OLD_ID.match(legacy_id):
            invalid += 1
            problems.append({"account": "(not an id)", "issue": "an account row without a proper id, left out"})
            continue
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
            "password_hash": hash_, "login_method": method if method in LOGIN_METHODS else "other",
            "role": "admin" if _blank(u["role"]).lower() == "admin" else "learner",
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
                             + ", ".join(a["legacy_id"] for a in others) + ", kept in legacy_accounts; a new password is needed"})
        chosen.append(primary)
    report |= {
        "invalid_email": invalid, "folded_duplicates": folded, "duplicate_addresses": sum(1 for g in accounts.values() if len(g) > 1),
        "extra_subscription_rows": extra, "paid_conflicts": conflicts,
        "subscriptions_without_account": len(orphans), "problems": problems,
    }
    return chosen, report


SUB_FIELDS = ("tier", "billing_interval", "status", "stripe_customer_id", "stripe_subscription_id", "current_period_start",
              "current_period_end", "cancel_at_period_end", "trial_used", "intended_tier", "intended_interval", "intended_at",
              "legacy_id", "created_at", "updated_at")


def _insert_subscription(conn, user_id: int, sub: dict | None, refresh: bool) -> None:
    if sub is None:
        billing.row(conn, user_id)
        return
    paid_before = sub["stripe_subscription_id"] is not None or sub["tier"] != "free"
    values = {f: sub.get(f) for f in SUB_FIELDS} | {"trial_used": int(paid_before)}
    update = ", ".join(f"{f} = excluded.{f}" for f in SUB_FIELDS if f != "created_at")
    conn.execute(f"INSERT INTO subscriptions (user_id, {', '.join(SUB_FIELDS)}) VALUES (?, {', '.join('?' * len(SUB_FIELDS))}) "
                 f"ON CONFLICT (user_id) DO {'UPDATE SET ' + update if refresh else 'NOTHING'}",
                 (user_id, *values.values()))
    if sub["grant"]:  # a paid plan nothing at Stripe would end: given until its period ends, if it has one
        conn.execute("UPDATE subscriptions SET grant_tier = ?, grant_until = ?, grant_note = 'From the old site' "
                     "WHERE user_id = ? AND grant_tier IS NULL", (*sub["grant"], user_id))
    # The plan's history, for the analytics: when it started, and when it ended if it has.
    if sub["tier"] != "free" and not conn.execute("SELECT 1 FROM subscription_events WHERE user_id = ? AND source = 'import'",
                                                  (user_id,)).fetchone():
        billing._record_change(conn, user_id, "free", sub["tier"], "import", sub["created_at"])
        if sub["status"] not in billing.LIVE and not sub["grant"]:
            billing._record_change(conn, user_id, sub["tier"], "free", "import", sub["updated_at"])


def _fold(conn, user_id: int, others: list[dict]) -> None:
    for other in others:
        if conn.execute("SELECT 1 FROM users WHERE legacy_id = ?", (other["legacy_id"],)).fetchone():
            continue  # it's an account here already
        osub = other["subscription"] or {}
        conn.execute("INSERT OR IGNORE INTO legacy_accounts (legacy_id, user_id, login_method, tier, status, "
                     "stripe_customer_id, stripe_subscription_id, created_at, last_signed_in, imported_at) "
                     "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                     (other["legacy_id"], user_id, other["login_method"], osub.get("tier"), osub.get("status"),
                      osub.get("stripe_customer_id"), osub.get("stripe_subscription_id"), other["created_at"],
                      other["last_signed_in"], db.now()))


def _bring_over(conn, a: dict, t: dict, problems: list, link_existing: bool, refresh: bool) -> None:
    """One account. Its changes are undone together if any part of it can't be saved."""
    sub = a["subscription"]
    note = lambda issue: problems.append({"account": a["legacy_id"], "issue": issue})  # noqa: E731
    found = conn.execute("SELECT id FROM users WHERE legacy_id = ?", (a["legacy_id"],)).fetchone()
    if found:
        t["accounts"]["already_imported"] += 1
        if refresh and sub:
            current = billing.row(conn, found["id"])
            # Only a plan left as it was brought over: one they've since changed here stays as it is.
            if (current["stripe_subscription_id"], current["legacy_id"]) == (sub["stripe_subscription_id"], sub["legacy_id"]):
                _insert_subscription(conn, found["id"], sub, refresh=True)
                t["accounts"]["subscriptions_refreshed"] += 1
            else:
                note("its plan has changed here since it was brought over, so the export's isn't copied over it")
        _fold(conn, found["id"], a["folded"])
        return
    taken = conn.execute("SELECT id, legacy_id, email_confirmed_at FROM users WHERE email = ?", (a["email"],)).fetchone()
    if taken and taken["legacy_id"]:  # an account of the old site's with this address came over before
        t["accounts"]["folded_into_earlier_import"] += 1
        note("shares its address with an account brought over before, so it's kept with that one: check its plan")
        _fold(conn, taken["id"], [a, *a["folded"]])
        return
    if taken:
        if not link_existing:
            t["accounts"]["address_taken_here"] += 1
            note("an account here already has its address; --link-existing links them")
            return
        if not taken["email_confirmed_at"]:
            # Anyone can sign up with an address: only its owner, having confirmed it, gets the old site's plan.
            t["accounts"]["address_taken_here"] += 1
            note("an account here has its address but hasn't confirmed it, so it isn't linked")
            return
        conn.execute("UPDATE users SET legacy_id = ?, login_method = ? WHERE id = ?", (a["legacy_id"], a["login_method"], taken["id"]))
        if sub and _live_rank(sub) > _live_rank(billing.row(conn, taken["id"])):
            _insert_subscription(conn, taken["id"], sub, refresh=True)
        t["accounts"]["linked_to_existing"] += 1
        _fold(conn, taken["id"], a["folded"])
        return
    # Accounts that shared an address on the old site may not all have been its owner's: one of them comes over, and
    # chooses a new password with a link sent to the address.
    shared = bool(a["folded"])
    confirmed = a["created_at"] if a["login_method"] in CONFIRMED_BY and not shared else None
    user_id = conn.execute(
        "INSERT INTO users (email, display_name, password_hash, created_at, role, legacy_id, login_method, "
        "last_signed_in, email_confirmed_at, signup_source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'old-site')",
        (a["email"], a["name"] or a["email"].split("@")[0][:60], "" if shared else a["password_hash"], a["created_at"],
         a["role"], a["legacy_id"], a["login_method"], a["last_signed_in"], confirmed)).lastrowid
    conn.execute("INSERT INTO follows (user_id, companion_id, created_at) SELECT ?, id, ? FROM companions "
                 "WHERE is_starter = 1 AND featured = 1 AND retired = 0", (user_id, db.now()))
    reminders.mark_feed_seen(conn, user_id)
    _insert_subscription(conn, user_id, sub, refresh=False)
    _fold(conn, user_id, a["folded"])
    t["accounts"]["created"] += 1
    t["passwords"][RESET_SHARED if shared else "bcrypt kept" if a["password_hash"] else "none: reset link needed"] += 1
    t["methods"][a["login_method"]] += 1
    t["roles"][a["role"]] += 1


def _bring_all(conn, accounts: list[dict], t: dict, problems: list, link_existing: bool, refresh: bool) -> None:
    for a in accounts:
        conn.execute("SAVEPOINT account")
        try:
            _bring_over(conn, a, t, problems, link_existing, refresh)
        except sqlite3.IntegrityError:  # its Stripe subscription is another account's here, say
            conn.execute("ROLLBACK TO account")
            t["accounts"]["left_out"] += 1
            problems.append({"account": a["legacy_id"], "issue": "clashes with an account here (its Stripe subscription "
                             "is another's), so it was left out"})
        conn.execute("RELEASE account")


def _totals(conn) -> dict:
    plan_counts, statuses = Counter(), Counter()
    for r in conn.execute("SELECT s.* FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.legacy_id IS NOT NULL"):
        plan_counts[billing.effective_tier(dict(r))] += 1
        if r["tier"] != "free":
            statuses[f"{r['tier']} {r['status']}"] += 1
    stripe_subs = conn.execute("SELECT COUNT(*) FROM subscriptions s JOIN users u ON u.id = s.user_id "
                               "WHERE u.legacy_id IS NOT NULL AND s.stripe_subscription_id IS NOT NULL").fetchone()[0]
    return {"plans_in_use": dict(plan_counts), "paid_plans_by_status": dict(statuses),
            "stripe_subscriptions_to_reconcile": stripe_subs}


def run(users_csv: str, subs_csv: str, dry_run: bool, link_existing: bool, refresh: bool) -> dict:
    """Import from the two export files."""
    with open(users_csv, encoding="utf-8-sig") as u, open(subs_csv, encoding="utf-8-sig") as s:
        return run_text(u.read(), s.read(), dry_run, link_existing, refresh)


def run_text(users: str, subscriptions: str, dry_run: bool, link_existing: bool = False, refresh: bool = False) -> dict:
    """Import from the two exports' text. Nothing is kept of the text itself. A dry run works on a copy of the
    database, so the site carries on meanwhile and nothing of it stays; a real one saves a batch of accounts at a time."""
    users_rows, subs_rows = read(users, USER_COLUMNS, "users"), read(subscriptions, SUB_COLUMNS, "subscriptions")
    db.init_db()
    companions.seed_starters()  # so new accounts can follow the featured companions, as at sign-up
    chosen, report = plan_import(users_rows, subs_rows)
    t = {"accounts": Counter(), "passwords": Counter(), "methods": Counter(), "roles": Counter()}
    if dry_run:
        with tempfile.TemporaryDirectory() as tmp:
            copy = str(Path(tmp) / "dry-run.db")
            src, dst = sqlite3.connect(settings.database_path), sqlite3.connect(copy)
            try:
                src.backup(dst)
            finally:
                src.close()
                dst.close()
            conn = sqlite3.connect(copy, isolation_level=None)
            conn.row_factory = sqlite3.Row
            try:
                conn.execute("PRAGMA foreign_keys = ON")
                conn.execute("BEGIN")
                _bring_all(conn, chosen, t, report["problems"], link_existing, refresh)
                totals = _totals(conn)
            finally:
                conn.close()
    else:
        for start in range(0, len(chosen), BATCH):
            with db.connect() as conn:
                conn.execute("BEGIN IMMEDIATE")
                _bring_all(conn, chosen[start:start + BATCH], t, report["problems"], link_existing, refresh)
        with db.connect() as conn:
            totals = _totals(conn)
    report["problems"] = report["problems"][:MAX_PROBLEMS]
    return {"dry_run": dry_run, **report, "accounts": dict(t["accounts"]), "passwords": dict(t["passwords"]),
            "sign_in_methods": dict(t["methods"]), "roles": dict(t["roles"]), **totals,
            "not_brought_over": ["sessions", "password reset links", "companion conversation counts (allowances start afresh)"]}


def summary(report: dict) -> dict:
    """The report as printed: the lists as counts."""
    out = {k: v for k, v in report.items() if k not in ("problems", "extra_subscription_rows", "paid_conflicts")}
    for k in ("problems", "extra_subscription_rows", "paid_conflicts"):
        out[k] = len(report[k])
    return out


class ImportIn(BaseModel):
    users: str = Field(min_length=10, max_length=10_000_000)  # the users export, as CSV text
    subscriptions: str = Field(min_length=10, max_length=10_000_000)
    dry_run: bool = True
    link_existing: bool = False
    refresh_subscriptions: bool = False


@router.post("/import")
def import_exports(body: ImportIn, user: dict = Depends(auth.require("admin"))):
    """For admins: bring the old site's members over from its two exports, sent from the browser. A dry run (the
    default) does everything and undoes it. The exports aren't kept; the answer is the report."""
    from app import staff

    try:
        report = run_text(body.users, body.subscriptions, body.dry_run, body.link_existing, body.refresh_subscriptions)
    except BadExport as e:
        raise HTTPException(422, str(e))
    if not body.dry_run:
        with db.connect() as conn:
            staff.log(conn, user["id"], "user", user["id"], "import",
                      f"{report['accounts'].get('created', 0)} accounts brought over from the old site")
    return report
