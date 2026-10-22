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
    assert report["passwords"] == {"bcrypt kept": 4, "none: reset link needed": 1}
    # The duplicate address became one account: the one with the paid plan, and its password.
    ada = sql("SELECT * FROM users WHERE email = 'ada@example.com'")[0]
    assert ada["legacy_id"] == "1002" and ada["display_name"] == "Ada Lovelace" and ada["created_at"] == "2025-03-04T05:06:07+00:00"
    assert sql("SELECT legacy_id, user_id FROM legacy_accounts") == [{"legacy_id": "1001", "user_id": ada["id"]}]
    assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": "free pass"}).status_code == 401
    me = client.post("/api/auth/login", json={"email": "ada@example.com", "password": "plus pass"}).json()
    assert me["plan"]["tier"] == "plus" and me["plan"]["subscription"]["status"] == "active"
    assert me["plan"]["trial_available"] is False and me["marketing_emails"] is False
    assert sql("SELECT stripe_subscription_id FROM subscriptions WHERE user_id = ?", ada["id"])[0]["stripe_subscription_id"] == "sub_ada"
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
    assert run_import(newer, refresh=True)["accounts"]["subscriptions_refreshed"] == 5
    assert sql("SELECT s.status FROM subscriptions s JOIN users u ON u.id = s.user_id WHERE u.email = 'kofi@example.com'")[0]["status"] == "active"


def test_an_address_already_used_here_is_reported_or_linked(client, exports):
    from tests.test_app import signup

    signup(client, email="zara@example.com", name="Zara Here")
    report = run_import(exports)
    assert report["accounts"]["address_taken_here"] == 1
    assert any("already has its address" in p["issue"] for p in report["problems"])
    assert sql("SELECT legacy_id FROM users WHERE email = 'zara@example.com'")[0]["legacy_id"] is None
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
