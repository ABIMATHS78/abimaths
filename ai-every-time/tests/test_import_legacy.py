"""Bringing the old site's accounts and subscriptions over (scripts/import_legacy.py), with made-up people."""

import csv
import json

import bcrypt
import pytest

from tests.test_billing import sql

USERS = ["id", "openId", "name", "email", "loginMethod", "role", "createdAt", "updatedAt", "lastSignedIn", "passwordHash"]
SUBS = ["id", "userId", "tier", "billingInterval", "stripeCustomerId", "stripeSubscriptionId", "status", "currentPeriodStart",
        "currentPeriodEnd", "cancelAtPeriodEnd", "companionConversationsUsed", "companionConversationsResetAt", "createdAt",
        "updatedAt", "intendedTier", "intendedBillingInterval", "intendedAt"]


def hashed(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(4)).decode()


def user(uid, email, name, method="email", role="user", password=None, last="2026-09-01 10:00:00"):
    return {"id": uid, "openId": f"open-{uid}", "name": name, "email": email, "loginMethod": method, "role": role,
            "createdAt": "2025-03-04 05:06:07", "updatedAt": "2026-09-01 10:00:00", "lastSignedIn": last,
            "passwordHash": hashed(password) if password else ""}


def sub(sid, uid, tier="free", status="active", stripe=None, interval="monthly", intended="", updated="2026-09-01 10:00:00"):
    return {"id": sid, "userId": uid, "tier": tier, "billingInterval": interval,
            "stripeCustomerId": f"cus_{stripe}" if stripe else "", "stripeSubscriptionId": f"sub_{stripe}" if stripe else "",
            "status": status, "currentPeriodStart": "1790000000000" if stripe else "", "currentPeriodEnd": "",
            "cancelAtPeriodEnd": "0", "companionConversationsUsed": "3", "companionConversationsResetAt": "",
            "createdAt": "2025-06-01 00:00:00", "updatedAt": updated, "intendedTier": intended,
            "intendedBillingInterval": "monthly" if intended else "", "intendedAt": "1790000000000" if intended else ""}


def write(path, columns, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, columns)
        w.writeheader()
        w.writerows(rows)
    return str(path)


@pytest.fixture()
def exports(tmp_path):
    users = [
        user("1001", "ada@example.com", "Ada Old", password="free pass", last="2025-01-01 00:00:00"),
        user("1002", "Ada@Example.com", "Ada Lovelace", password="plus pass"),  # the same address, told apart by case
        user("1003", "kofi@example.com", "Kofi Mensah", method="google"),
        user("1004", "boss@example.com", "The Boss", role="admin", password="boss pass"),
        user("1005", "not an address", "Nobody", password="x pass"),
        user("1006", "zara@example.com", "Zara", password="zara pass"),
        user("1007", "ama@example.com", "Ama", password="ama pass"),
    ]
    subs = [
        sub("1", "1001"), sub("2", "1002", "plus", stripe="ada"), sub("3", "1003", "insider", "past_due", stripe="kofi"),
        sub("4", "1006", "pro", "canceled", stripe="zold", updated="2026-01-01 00:00:00"),
        sub("5", "1006", "digital", stripe="znew"), sub("6", "1007", intended="plus"),
    ]
    return write(tmp_path / "users.csv", USERS, users), write(tmp_path / "subscriptions.csv", SUBS, subs)


def run_import(exports, **kw):
    import scripts.import_legacy as importer

    return importer.run(*exports, dry_run=kw.get("dry_run", False), link_existing=kw.get("link_existing", False),
                        refresh=kw.get("refresh", False))


def test_a_dry_run_reports_and_changes_nothing(client, exports):
    report = run_import(exports, dry_run=True)
    assert report["accounts"] == {"created": 5} and report["folded_duplicates"] == 1 and report["invalid_email"] == 1
    assert report["plans_in_use"] == {"plus": 1, "insider": 1, "digital": 1, "free": 2}
    assert report["stripe_subscriptions_to_reconcile"] == 3 and len(report["extra_subscription_rows"]) == 1
    text = json.dumps(report)
    assert "@example.com" not in text and "$2b$" not in text and "Lovelace" not in text and "cus_ada" not in text
    assert sql("SELECT COUNT(*) AS n FROM users WHERE legacy_id IS NOT NULL")[0]["n"] == 0


def test_accounts_keep_their_plans_passwords_and_roles(client, exports):
    report = run_import(exports)
    assert report["accounts"] == {"created": 5} and report["roles"] == {"learner": 4, "admin": 1}
    assert report["passwords"] == {"bcrypt kept": 3, "none: reset link needed": 1,
                                   "reset link needed: the address had more than one account": 1}
    # The duplicate address became one account, the one with the paid plan. Not knowing which of the two was the
    # address's owner, neither password works: a reset link sent to the address decides.
    ada = sql("SELECT * FROM users WHERE email = 'ada@example.com'")[0]
    assert ada["legacy_id"] == "1002" and ada["display_name"] == "Ada Lovelace" and ada["created_at"] == "2025-03-04T05:06:07+00:00"
    assert ada["password_hash"] == "" and ada["email_confirmed_at"] is None
    assert sql("SELECT legacy_id, user_id FROM legacy_accounts") == [{"legacy_id": "1001", "user_id": ada["id"]}]
    for password in ("free pass", "plus pass"):
        assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": password}).status_code == 401
    ada_plan = sql("SELECT * FROM subscriptions WHERE user_id = ?", ada["id"])[0]
    assert (ada_plan["tier"], ada_plan["status"], ada_plan["stripe_subscription_id"], ada_plan["trial_used"]) == ("plus", "active", "sub_ada", 1)
    me = client.post("/api/auth/login", json={"email": "zara@example.com", "password": "zara pass"}).json()
    assert me["plan"]["tier"] == "digital" and me["marketing_emails"] is False
    client.post("/api/auth/logout")
    # Google sign-in: no password, so a reset link; the address is taken as confirmed.
    kofi = sql("SELECT * FROM users WHERE email = 'kofi@example.com'")[0]
    assert kofi["password_hash"] == "" and kofi["email_confirmed_at"] and kofi["login_method"] == "google"
    # The old admin is an admin here; the account with two subscription rows keeps the one in use.
    client.post("/api/auth/logout")
    assert client.post("/api/auth/login", json={"email": "boss@example.com", "password": "boss pass"}).json()["role"] == "admin"
    zara = sql("SELECT s.tier, s.status FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.email = 'zara@example.com'")
    assert zara == [{"tier": "digital", "status": "active"}]
    ama = sql("SELECT s.* FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.email = 'ama@example.com'")[0]
    assert ama["tier"] == "free" and ama["intended_tier"] == "plus" and ama["trial_used"] == 0
    # Paid plans' history, for the analytics.
    kinds = [r["event_type"] for r in sql("SELECT event_type FROM subscription_events WHERE source = 'import' ORDER BY id")]
    assert sorted(kinds) == ["new", "new", "new"]
    # Everyone follows the featured companions, as at sign-up.
    assert sql("SELECT COUNT(*) AS n FROM follows WHERE user_id = ?", ada["id"])[0]["n"] > 0


def test_running_it_again_changes_nothing_unless_asked(client, exports, tmp_path):
    run_import(exports)
    again = run_import(exports)
    assert again["accounts"] == {"already_imported": 5}
    assert sql("SELECT COUNT(*) AS n FROM users")[0]["n"] == 5
    assert sql("SELECT COUNT(*) AS n FROM subscription_events")[0]["n"] == 3
    # A newer export: Kofi's payment went through. --refresh-subscriptions brings it in.
    rows = list(csv.DictReader(open(exports[1], encoding="utf-8")))
    rows[2]["status"] = "active"
    newer = (exports[0], write(tmp_path / "newer.csv", SUBS, rows))
    assert run_import(newer)["accounts"] == {"already_imported": 5}
    assert sql("SELECT s.status FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.email = 'kofi@example.com'")[0]["status"] == "past_due"
    assert run_import(newer, refresh=True)["accounts"]["subscriptions_refreshed"] == 4  # the admin has no subscription
    assert sql("SELECT s.status FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.email = 'kofi@example.com'")[0]["status"] == "active"


def test_an_address_already_used_here_is_reported_or_linked(client, exports):
    from tests.test_app import signup

    signup(client, email="zara@example.com", name="Zara Here")
    report = run_import(exports)
    assert report["accounts"]["address_taken_here"] == 1
    assert any("already has its address" in p["issue"] for p in report["problems"])
    assert sql("SELECT legacy_id FROM users WHERE email = 'zara@example.com'")[0]["legacy_id"] is None
    # Anyone could have signed up with the address: until it's confirmed, the old site's plan isn't handed over.
    unconfirmed = run_import(exports, link_existing=True)
    assert unconfirmed["accounts"]["address_taken_here"] == 1 and "linked_to_existing" not in unconfirmed["accounts"]
    assert any("hasn't confirmed" in p["issue"] for p in unconfirmed["problems"])
    from tests.test_billing import run as sql_run

    sql_run("UPDATE users SET email_confirmed_at = created_at WHERE email = 'zara@example.com'")
    linked = run_import(exports, link_existing=True)
    assert linked["accounts"]["linked_to_existing"] == 1
    zara = sql("SELECT u.legacy_id, u.display_name, s.tier FROM users u JOIN subscriptions s ON s.user_id = u.id "
               "WHERE u.email = 'zara@example.com'")[0]
    assert zara == {"legacy_id": "1006", "display_name": "Zara Here", "tier": "digital"}  # their password and name stay
    assert client.post("/api/auth/login", json={"email": "zara@example.com", "password": "correct horse"}).status_code == 200


def test_the_command_line_writes_a_report_without_personal_details(client, exports, tmp_path, capsys):
    import scripts.import_legacy as importer

    out = tmp_path / "report.json"
    assert importer.main(["--users", exports[0], "--subscriptions", exports[1], "--dry-run", "--report", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Dry run: nothing was saved." in printed and "@example.com" not in printed
    report = json.loads(out.read_text())
    assert report["dry_run"] is True and "@example.com" not in out.read_text()


def test_admins_import_from_the_setup_page(client, exports, monkeypatch):
    from app.config import settings
    from tests.test_app import signup

    monkeypatch.setattr(settings, "admin_emails", frozenset({"admin@example.com"}))  # not one of the old site's

    signup(client, email="kofi.here@example.com", name="Kofi")
    users, subs = (open(path, encoding="utf-8").read() for path in exports)
    assert client.post("/api/staff/import", json={"users": users, "subscriptions": subs}).status_code == 403
    client.post("/api/auth/logout")
    signup(client, email="admin@example.com", name="Admin")
    checked = client.post("/api/staff/import", json={"users": users, "subscriptions": subs})
    assert checked.status_code == 200 and checked.json()["dry_run"] is True and checked.json()["accounts"] == {"created": 5}
    assert "@example.com" not in checked.text and "$2b$" not in checked.text
    assert sql("SELECT COUNT(*) AS n FROM users WHERE legacy_id IS NOT NULL")[0]["n"] == 0
    done = client.post("/api/staff/import", json={"users": users, "subscriptions": subs, "dry_run": False}).json()
    assert done["accounts"] == {"created": 5}
    assert sql("SELECT COUNT(*) AS n FROM users WHERE legacy_id IS NOT NULL")[0]["n"] == 5
    assert any(r["action"] == "import" for r in client.get("/api/staff/log").json())
    bad = client.post("/api/staff/import", json={"users": "id,name\n1,Ada\n", "subscriptions": subs})
    assert bad.status_code == 422 and "missing the columns" in bad.json()["detail"]


def test_a_plan_changed_here_isnt_overwritten_by_a_newer_export(client, exports):
    run_import(exports)
    from tests.test_billing import run as sql_run

    # Zara started a Pro plan here after coming over.
    sql_run("UPDATE subscriptions SET tier = 'pro', status = 'trialing', stripe_subscription_id = 'sub_here' "
            "WHERE user_id = (SELECT id FROM users WHERE email = 'zara@example.com')")
    report = run_import(exports, refresh=True)
    assert any("changed here" in p["issue"] for p in report["problems"])
    assert sql("SELECT s.tier, s.stripe_subscription_id FROM subscriptions s JOIN users u ON u.id = s.user_id "
               "WHERE u.email = 'zara@example.com'") == [{"tier": "pro", "stripe_subscription_id": "sub_here"}]


def test_paid_plans_stripe_doesnt_hold_end_when_their_period_does(client, tmp_path):
    users = [user("3001", "kemi@example.com", "Kemi", password="kemi pass"), user("3002", "tolu@example.com", "Tolu", password="t pass"),
             user("3003", "bola@example.com", "Bola", password="b pass")]
    subs = [sub("1", "3001", "insider"), sub("2", "3002", "pro"), sub("3", "3003", "plus")]
    subs[0]["currentPeriodEnd"] = "2099-01-31 00:00:00"  # paid to a day still to come
    subs[1]["currentPeriodEnd"] = "2024-01-31 00:00:00"  # and to one that's passed
    report = run_import((write(tmp_path / "u.csv", USERS, users), write(tmp_path / "s.csv", SUBS, subs)))
    assert report["plans_in_use"] == {"insider": 1, "free": 1, "plus": 1}
    given = {r["email"]: (r["grant_tier"], r["grant_until"]) for r in sql(
        "SELECT u.email, s.grant_tier, s.grant_until FROM subscriptions s JOIN users u ON u.id = s.user_id")}
    assert given["kemi@example.com"] == ("insider", "2099-01-31T00:00:00+00:00")
    assert given["tolu@example.com"] == (None, None)
    assert given["bola@example.com"] == ("plus", None)  # no end known: kept, and flagged for a person to check
    assert any("for good: check it" in p["issue"] for p in report["problems"])


def test_the_report_never_repeats_what_isnt_an_id(client, tmp_path):
    odd = user("4001", "ok@example.com", "Ok", password="ok pass")
    odd["loginMethod"] = "someone@example.com"
    bad_id = user("ada@example.com", "fine@example.com", "Bad id", password="x pass")
    report = run_import((write(tmp_path / "u.csv", USERS, [odd, bad_id]), write(tmp_path / "s.csv", SUBS, [])))
    text = json.dumps(report)
    assert "@example.com" not in text and report["sign_in_methods"] == {"other": 1} and report["invalid_email"] == 1


def test_a_newer_duplicate_is_kept_with_the_account_brought_over_before(client, exports, tmp_path):
    run_import(exports)
    # A newer export has another account with Ama's address, on a better plan than hers.
    rows = list(csv.DictReader(open(exports[0], encoding="utf-8")))
    rows.append(user("1009", "AMA@example.com", "Ama Again", password="again"))
    subs = list(csv.DictReader(open(exports[1], encoding="utf-8"))) + [sub("7", "1009", "pro", stripe="ama2")]
    report = run_import((write(tmp_path / "users2.csv", USERS, rows), write(tmp_path / "subs2.csv", SUBS, subs)))
    assert report["accounts"] == {"already_imported": 4, "folded_into_earlier_import": 1}
    assert any("brought over before" in p["issue"] for p in report["problems"])
    ama = sql("SELECT id FROM users WHERE email = 'ama@example.com'")[0]["id"]
    assert sql("SELECT legacy_id, user_id, tier FROM legacy_accounts WHERE legacy_id = '1009'") == [
        {"legacy_id": "1009", "user_id": ama, "tier": "pro"}]  # kept with her account, for a person to check
    assert sql("SELECT tier FROM subscriptions WHERE user_id = ?", ama) == [{"tier": "free"}]
