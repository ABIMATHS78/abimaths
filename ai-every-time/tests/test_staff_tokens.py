"""Staff tokens (app/tokens.py): a tool, such as Claude Code, uses one to work the Staff pages' API for an admin,
without their password. It's shown once and kept as a hash; it works only on the Staff pages' API, and only while its
admin is one, until it expires, is cancelled, or the admin changes their password; the decision log names it."""

import hashlib
from datetime import date, timedelta

import pytest

from tests.test_app import signup
from tests.test_staff import admin_email  # noqa: F401  (a fixture)


def make(client, name="Claude Code", days=90):
    r = client.post("/api/staff/tokens", json={"name": name, "days": days})
    assert r.status_code == 201, r.text
    return r.json()


def tool(token):
    """The headers a tool sends: the token alone, with no session."""
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def ada_and_boss(client, admin_email):  # noqa: F811
    """Ada, a learner, and Boss, an admin, who is signed in."""
    ada = signup(client)
    client.post("/api/auth/logout")
    signup(client, email=admin_email, name="Boss")
    return ada


def test_an_admin_makes_a_token_shown_once_and_kept_as_a_hash(client, ada_and_boss):
    from app import db

    made = make(client)
    assert made["token"].startswith("aiet_") and len(made["token"]) > 40 and made["name"] == "Claude Code"
    with db.connect() as conn:
        stored = conn.execute("SELECT token_hash FROM staff_tokens").fetchone()[0]
    assert stored == hashlib.sha256(made["token"].encode()).hexdigest()
    listed = client.get("/api/staff/tokens").json()
    assert [t["name"] for t in listed] == ["Claude Code"] and "token" not in listed[0] and listed[0]["admin"] == "Boss"
    assert client.get("/api/staff/log").json()[0]["action"] == "token:make"


def test_a_tool_gives_a_plan_with_it_and_the_log_says_so(client, ada_and_boss):
    token = make(client)["token"]
    client.cookies.clear()
    assert client.get("/api/me", headers=tool(token)).json()["email"] == "boss@example.com"
    found = client.get("/api/staff/users", params={"q": "ada@example.com"}, headers=tool(token))
    assert found.status_code == 200 and [u["id"] for u in found.json()] == [ada_and_boss["id"]]
    until = (date.today() + timedelta(days=30)).isoformat()
    given = client.post(f"/api/staff/users/{ada_and_boss['id']}/plan", headers=tool(token),
                        json={"tier": "insider", "until": until, "note": "Complimentary"})
    assert given.status_code == 200, given.text
    entry = client.get("/api/staff/log", headers=tool(token)).json()[0]
    assert entry["action"].startswith("plan:insider") and entry["note"] == "Complimentary · via token “Claude Code”"
    # Ada sees it as given, until the day chosen.
    client.post("/api/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    assert client.get("/api/me").json()["plan"]["tier"] == "insider"


def test_it_works_nowhere_else(client, ada_and_boss):
    token = make(client)["token"]
    client.cookies.clear()
    for method, path in [("GET", "/api/me/export"), ("PATCH", "/api/me"), ("DELETE", "/api/me"),
                         ("POST", "/api/companions"), ("GET", "/api/staff/tokens"),
                         ("POST", "/api/staff/tokens"), ("DELETE", "/api/staff/tokens/1")]:
        r = client.request(method, path, headers=tool(token), json={"name": "More"} if method != "GET" else None)
        assert r.status_code == 403 and "Staff tokens" in r.json()["detail"], (method, path, r.text)
    assert client.get("/api/staff/users", headers=tool("aiet_not-a-token")).status_code == 401
    assert client.get("/api/staff/users", headers={"Authorization": "Bearer something-else"}).status_code == 401


def test_cancelled_and_expired_tokens_stop_working(client, ada_and_boss):
    from app import db

    first, second = make(client, "First")["token"], make(client, "Second", days=7)["token"]
    cancel = next(t["id"] for t in client.get("/api/staff/tokens").json() if t["name"] == "First")
    assert client.delete(f"/api/staff/tokens/{cancel}").status_code == 204
    with db.connect() as conn:
        conn.execute("UPDATE staff_tokens SET expires_at = '2000-01-01T00:00:00+00:00' WHERE name = 'Second'")
    assert client.get("/api/staff/tokens").json() == []
    session = dict(client.cookies)
    client.cookies.clear()
    for token in (first, second):
        assert client.get("/api/staff/users", headers=tool(token)).status_code == 401
    client.cookies.update(session)
    assert client.delete(f"/api/staff/tokens/{cancel}").status_code == 404  # once is enough


def test_only_admins_make_them_and_one_stops_when_its_admin_isnt_one(client, ada_and_boss, monkeypatch):
    from app.config import settings

    token = make(client)["token"]
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    assert client.post("/api/staff/tokens", json={"name": "Mine"}).status_code == 403
    client.cookies.clear()
    monkeypatch.setattr(settings, "admin_emails", frozenset())  # Boss isn't an admin any more
    r = client.get("/api/staff/users", headers=tool(token))
    assert r.status_code == 403 and "isn't an admin" in r.json()["detail"]


def test_changing_the_password_cancels_them(client, ada_and_boss):
    token = make(client)["token"]
    assert client.post("/api/me/password", json={"current_password": "correct horse",
                                                  "new_password": "battery staple"}).status_code == 204
    client.cookies.clear()
    assert client.get("/api/staff/users", headers=tool(token)).status_code == 401


def test_a_session_comes_first(client, ada_and_boss):
    """A learner's browser that also sends an admin's token is still the learner."""
    token = make(client)["token"]
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": "ada@example.com", "password": "correct horse"})
    assert client.get("/api/staff/users", headers=tool(token)).status_code == 403
