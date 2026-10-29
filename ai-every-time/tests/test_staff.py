"""Staff roles, moderation of reported companions, expert review of shared paths, email reminders."""

from datetime import datetime, timezone

import pytest

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)


@pytest.fixture()
def admin_email(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    return "boss@example.com"


def reported_zara(client):
    """Ada publishes Zara and three learners report her, which hides her."""
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    for i in range(3):
        as_user(client, f"r{i}@example.com", f"R{i}")
        client.post(f"/api/companions/{zara['id']}/report", json={"reason": "misleading" if i else "spam"})
    return zara


def notices_for(client):
    return [r for r in client.get("/api/reminders").json() if r["kind"] == "notice"]


def test_roles_and_permissions(client, admin_email):
    signup(client, email="ada@example.com")
    assert client.get("/api/me").json()["role"] == "learner"
    for url in ("/api/staff/companions", "/api/staff/paths", "/api/staff/users", "/api/staff/log"):
        assert client.get(url).status_code == 403, url

    as_user(client, admin_email, "Boss")
    assert client.get("/api/me").json()["role"] == "admin"
    users = {u["email"]: u for u in client.get("/api/staff/users").json()}
    ada = users["ada@example.com"]
    assert users[admin_email]["fixed"] is True
    assert client.post(f"/api/staff/users/{users[admin_email]['id']}/role", json={"role": "learner"}).status_code == 409
    assert client.post(f"/api/staff/users/{ada['id']}/role", json={"role": "moderator"}).json()["role"] == "moderator"

    as_user(client, "ada@example.com", "Ada")
    assert client.get("/api/me").json()["role"] == "moderator"
    assert client.get("/api/staff/companions").status_code == 200
    assert client.get("/api/staff/paths").status_code == 403  # moderators don't review paths
    assert client.get("/api/staff/users").status_code == 403
    assert client.get("/api/staff/log").status_code == 200


def test_moderator_restores_hidden_companion(client, admin_email):
    zara = reported_zara(client)
    as_user(client, admin_email, "Boss")
    queue = client.get("/api/staff/companions").json()
    item = next(c for c in queue if c["id"] == zara["id"])
    assert client.get(f"/api/companions/{zara['id']}/avatar").status_code == 200  # moderators can see hidden faces
    assert item["hidden"] and item["reports"] == 3 and item["reasons"] == {"spam": 1, "misleading": 2}
    assert item["designer"] == "Ada"

    assert client.post(f"/api/staff/companions/{zara['id']}", json={"action": "restore", "note": "Checked, fine"}).json()["ok"]
    assert zara["id"] not in [c["id"] for c in client.get("/api/staff/companions").json()]
    as_user(client, "late@example.com", "Late")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 200  # visible again

    as_user(client, "ada@example.com", "Ada")
    notes = notices_for(client)
    assert notes and "back in the community" in notes[0]["text"]
    assert client.post(f"/api/reminders/notices/{notes[0]['id']}/read").status_code == 204
    assert not notices_for(client)

    as_user(client, admin_email, "Boss")
    log = client.get("/api/staff/log").json()
    assert log[0]["action"] == "restore" and log[0]["staff_name"] == "Boss" and log[0]["target_id"] == zara["id"]


def test_moderator_removes_companion_for_good(client, admin_email):
    zara = reported_zara(client)
    as_user(client, admin_email, "Boss")
    url = f"/api/staff/companions/{zara['id']}"
    assert client.post(url, json={"action": "remove"}).status_code == 422  # must explain
    assert client.post(url, json={"action": "remove", "note": "It taught false facts about vaccines."}).status_code == 200
    assert client.post(url, json={"action": "restore"}).status_code == 404  # already decided

    as_user(client, "ada@example.com", "Ada")
    mine = client.get(f"/api/companions/{zara['id']}").json()  # designer keeps it privately
    assert mine["removed"] and not mine["is_public"]
    assert client.post(f"/api/companions/{zara['id']}/publish").status_code == 409
    assert "false facts about vaccines" in notices_for(client)[0]["text"]
    as_user(client, "late@example.com", "Late")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 404


def test_dismissing_reports_keeps_companion_visible(client, admin_email):
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    as_user(client, "r0@example.com", "R0")
    client.post(f"/api/companions/{zara['id']}/report", json={"reason": "other"})
    as_user(client, admin_email, "Boss")
    assert not client.get("/api/staff/companions").json()[0]["hidden"]
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "dismiss"})
    assert client.get("/api/staff/companions").json() == []
    as_user(client, "ada@example.com", "Ada")
    assert "No action was needed" in notices_for(client)[0]["text"]


def make_path(client):
    return client.post("/api/custom-paths", json={"topic": "AI in agriculture", "lessons": 3}).json()


def test_path_review_approve_with_corrections_shares_it(client, admin_email):
    signup(client)
    path = make_path(client)
    assert path["status"] == "private" and path["mine"]
    lesson_id = path["lessons"][0]["id"]
    client.post(f"/api/lessons/{lesson_id}/complete")
    client.get(f"/api/lessons/{lesson_id}")  # caches the lesson text
    assert client.post(f"/api/custom-paths/{path['id']}/submit").json()["status"] == "pending"
    assert client.post(f"/api/custom-paths/{path['id']}/submit").status_code == 409

    # Not shared while pending
    as_user(client, "bob@example.com", "Bob")
    assert path["id"] not in [p["id"] for p in client.get("/api/paths").json()]

    as_user(client, admin_email, "Boss")
    queue = client.get("/api/staff/paths").json()
    item = next(p for p in queue if p["id"] == path["id"])
    assert item["owner_name"] == "Ada" and len(item["lessons"]) == 3
    fixed = [dict(l, key_points=["Corrected fact one.", "Corrected fact two."]) for l in item["lessons"]]
    bad = client.post(f"/api/staff/paths/{path['id']}", json={"decision": "approve", "lessons": fixed[:2]})
    assert bad.status_code == 422  # lesson count must stay the same
    ok = client.post(f"/api/staff/paths/{path['id']}", json={"decision": "approve", "lessons": fixed, "title": "AI on the Farm"})
    assert ok.json()["status"] == "approved"

    as_user(client, "bob@example.com", "Bob")
    shared = next(p for p in client.get("/api/paths").json() if p["id"] == path["id"])
    assert shared["status"] == "approved" and not shared["mine"] and shared["suggested_by"] == "Ada"
    assert shared["title"] == "AI on the Farm"
    lesson = client.get(f"/api/lessons/{lesson_id}").json()
    assert "Corrected fact one." in lesson["body"]  # re-taught from the corrected key points
    assert client.delete(f"/api/custom-paths/{path['id']}").status_code == 404  # not Bob's
    # Bob has opened one lesson, so the shared path now counts in his progress.
    assert path["id"] in [p["id"] for p in client.get("/api/progress").json()["paths"]]

    as_user(client, "ada@example.com", "Ada")
    assert client.delete(f"/api/custom-paths/{path['id']}").status_code == 409  # shared: unshare first
    me_view = next(p for p in client.get("/api/paths").json() if p["id"] == path["id"])
    assert me_view["lessons"][0]["status"] == "completed"  # Ada's progress survived the review
    assert "passed expert review" in notices_for(client)[0]["text"]


def test_path_review_reject_unshare_and_self_approval(client, admin_email, monkeypatch):
    signup(client)
    path = make_path(client)
    client.post(f"/api/custom-paths/{path['id']}/submit")
    as_user(client, admin_email, "Boss")
    url = f"/api/staff/paths/{path['id']}"
    assert client.post(url, json={"decision": "reject"}).status_code == 422  # must explain
    assert client.post(url, json={"decision": "reject", "note": "Lesson 2 overstates yields."}).json()["status"] == "rejected"

    as_user(client, "ada@example.com", "Ada")
    mine = next(p for p in client.get("/api/paths").json() if p["id"] == path["id"])
    assert mine["status"] == "rejected" and "overstates" in mine["review_note"]
    assert client.post(f"/api/custom-paths/{path['id']}/submit").json()["status"] == "pending"  # resubmit
    assert client.post(f"/api/custom-paths/{path['id']}/withdraw").json()["status"] == "private"
    client.post(f"/api/custom-paths/{path['id']}/submit")

    as_user(client, admin_email, "Boss")
    client.post(url, json={"decision": "approve"})
    assert client.post(url, json={"decision": "unshare"}).status_code == 422
    assert client.post(url, json={"decision": "unshare", "note": "Outdated."}).json()["status"] == "private"
    as_user(client, "bob@example.com", "Bob")
    assert path["id"] not in [p["id"] for p in client.get("/api/paths").json()]

    # Reviewers can't approve their own paths.
    as_user(client, admin_email, "Boss")
    own = make_path(client)
    client.post(f"/api/custom-paths/{own['id']}/submit")
    assert client.post(f"/api/staff/paths/{own['id']}", json={"decision": "approve"}).status_code == 403


def test_email_reminders_opt_in_digest_and_unsubscribe(client, monkeypatch):
    from app import digest
    from app.config import settings

    signup(client)
    assert client.patch("/api/me", json={"email_reminders": True}).status_code == 409  # not configured
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "mail_from", "AI Every Time <hello@example.com>")
    monkeypatch.setattr(settings, "public_url", "https://aieverytime.example")
    from tests.test_email import Mailbox

    box = Mailbox()
    monkeypatch.setattr(digest.smtplib, "SMTP", box)
    me = client.patch("/api/me", json={"email_reminders": True}).json()
    assert me["email_reminders"] and me["email_available"] and not me["email_confirmed"]

    client.get("/api/lessons/tokens")  # something to remind about
    # Ada hasn't picked a reminder time and has no known time zone, so it's the default hour in UTC.
    monkeypatch.setattr(settings, "reminder_hour", 7)
    assert digest.due_digests(datetime(2026, 1, 1, 7, 5, tzinfo=timezone.utc)) == []  # not until it's confirmed
    assert client.post("/api/email/confirm", content=f"token={box.link_token('confirm')}").status_code == 200
    assert digest.due_digests(datetime(2026, 1, 1, 6, 59, tzinfo=timezone.utc)) == []
    due = digest.due_digests(datetime(2026, 1, 1, 7, 5, tzinfo=timezone.utc))
    assert len(due) == 1
    user, items, token = due[0]
    assert any(i["kind"] == "continue_lesson" for i in items)
    assert digest.due_digests(datetime(2026, 1, 1, 21, 0, tzinfo=timezone.utc)) == []  # once a day

    msg = digest.build_message(user, items, token)
    assert msg["To"] == "ada@example.com" and "List-Unsubscribe=One-Click" == msg["List-Unsubscribe-Post"]
    text = msg.get_body(("plain",)).get_content()
    assert "https://aieverytime.example/lessons/tokens" in text and token in text

    sent = []
    monkeypatch.setattr(digest.smtplib, "SMTP", lambda *a, **k: FakeSMTP(sent))
    digest.send(msg)
    assert sent == ["starttls", "send"]

    # Unsubscribe page doesn't act on GET; the POST (form or one-click) does.
    assert "Stop emails" in client.get(f"/api/email/unsubscribe?token={token}").text
    assert client.get("/api/me").json()["email_reminders"]
    assert client.post(f"/api/email/unsubscribe?token={token}", content="List-Unsubscribe=One-Click").status_code == 200
    assert not client.get("/api/me").json()["email_reminders"]
    assert client.post("/api/email/unsubscribe", content="token=nope").status_code == 400


class FakeSMTP:
    def __init__(self, sent):
        self.sent = sent

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def starttls(self):
        self.sent.append("starttls")

    def send_message(self, msg):
        self.sent.append("send")


def test_admin_emails_need_a_confirmed_address_while_email_is_on(client, admin_email, mailbox):
    """Otherwise whoever signs up with the address before its owner would be an admin."""
    assert signup(client, email=admin_email, name="Boss")["role"] == "learner"
    assert client.get("/api/staff/insights").status_code == 403
    assert client.post("/api/email/confirm", content=f"token={mailbox.link_token('confirm')}").status_code == 200
    assert client.get("/api/me").json()["role"] == "admin"
    assert client.get("/api/staff/insights").status_code == 200
