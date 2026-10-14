"""Appeals against removals from the community, and staff queues in the reminders list."""

import pytest

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_staff import make_path, reported_zara

MESSAGE = "Zara teaches with made-up characters on purpose; she doesn't claim to be a real teacher."


@pytest.fixture()
def staff(monkeypatch):
    """Boss is an admin by configuration; Mo is made a moderator by Boss."""
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))

    def setup(client):
        as_user(client, "mo@example.com", "Mo")
        mo = client.get("/api/me").json()
        as_user(client, "boss@example.com", "Boss")
        assert client.post(f"/api/staff/users/{mo['id']}/role", json={"role": "moderator"}).status_code == 200

    return setup


def removed_zara(client, staff):
    """Zara is reported, hidden, and then removed by Boss."""
    zara = reported_zara(client)
    staff(client)
    r = client.post(f"/api/staff/companions/{zara['id']}", json={"action": "remove", "note": "Pretends to be a real teacher."})
    assert r.status_code == 200
    return zara


def test_designer_appeals_and_another_moderator_restores(client, staff):
    zara = removed_zara(client, staff)
    as_user(client, "ada@example.com", "Ada")
    mod = client.get(f"/api/companions/{zara['id']}").json()["moderation"]
    assert mod["note"] == "Pretends to be a real teacher." and mod["can_appeal"] and mod["appeal"] is None
    assert any("second look" in r["text"] for r in client.get("/api/reminders").json() if r["kind"] == "notice")

    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": "Please"}).status_code == 422
    r = client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE})
    assert r.status_code == 201 and r.json()["appeal"]["status"] == "open" and not r.json()["can_appeal"]
    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE}).status_code == 409
    assert client.get("/api/staff/appeals").status_code == 403  # learners can't see the queue

    # The moderator who removed Zara can't give the second look.
    as_user(client, "boss@example.com", "Boss")
    [item] = client.get("/api/staff/appeals").json()
    assert item["message"] == MESSAGE and item["removal"]["by"] == "Boss" and item["can_decide"] is False
    assert item["companion"]["id"] == zara["id"] and item["designer"] == "Ada"
    assert client.post(f"/api/staff/appeals/{item['id']}", json={"decision": "overturn"}).status_code == 403

    as_user(client, "mo@example.com", "Mo")
    assert client.get("/api/staff/appeals").json()[0]["can_decide"] is True
    assert client.post(f"/api/staff/appeals/{item['id']}", json={"decision": "overturn", "note": "Fair point."}).json() == {"ok": True}
    assert client.post(f"/api/staff/appeals/{item['id']}", json={"decision": "uphold", "note": "x"}).status_code == 409
    assert client.get("/api/staff/appeals").json() == []
    assert client.get("/api/staff/appeals?status=overturned").json()[0]["decided_by"] == "Mo"

    # Back in the community for everyone, and Ada was told.
    as_user(client, "r0@example.com", "R0")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 200
    as_user(client, "ada@example.com", "Ada")
    c = client.get(f"/api/companions/{zara['id']}").json()
    assert c["is_public"] and not c["hidden"] and not c["removed"] and c["moderation"] is None
    assert any("appeal was accepted" in r["text"] and "Fair point." in r["text"]
               for r in client.get("/api/reminders").json() if r["kind"] == "notice")
    as_user(client, "boss@example.com", "Boss")
    assert any(e["action"] == "appeal:overturn" and e["staff_name"] == "Mo" for e in client.get("/api/staff/log").json())

    # A later removal is a new decision, so it can be appealed too.
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "remove", "note": "Now it does name a real school."})
    as_user(client, "ada@example.com", "Ada")
    assert client.get(f"/api/companions/{zara['id']}").json()["moderation"]["can_appeal"]
    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE}).status_code == 201


def test_upheld_appeal_is_final_for_that_removal(client, staff):
    zara = removed_zara(client, staff)
    as_user(client, "ada@example.com", "Ada")
    client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE})

    as_user(client, "mo@example.com", "Mo")
    appeal_id = client.get("/api/staff/appeals").json()[0]["id"]
    assert client.post(f"/api/staff/appeals/{appeal_id}", json={"decision": "uphold"}).status_code == 422  # needs a reason
    assert client.post(f"/api/staff/appeals/{appeal_id}", json={"decision": "uphold", "note": "The bio names a real school."}).status_code == 200

    as_user(client, "ada@example.com", "Ada")
    c = client.get(f"/api/companions/{zara['id']}").json()
    assert c["removed"] and c["moderation"]["appeal"]["status"] == "upheld" and not c["moderation"]["can_appeal"]
    assert c["moderation"]["appeal"]["decision_note"] == "The bio names a real school."
    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE}).status_code == 409
    assert client.post(f"/api/companions/{zara['id']}/publish").status_code == 409  # still can't be republished
    assert any("removal stands" in r["text"] for r in client.get("/api/reminders").json() if r["kind"] == "notice")


def test_moderators_cant_decide_their_own_appeals(client, staff):
    staff(client)
    # Mo, a moderator, designs a companion that Boss then removes.
    as_user(client, "mo@example.com", "Mo")
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{mine['id']}/publish")
    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/companions/{mine['id']}", json={"action": "remove", "note": "Pretends to be a real teacher."})
    as_user(client, "mo@example.com", "Mo")
    assert client.post(f"/api/companions/{mine['id']}/appeal", json={"message": MESSAGE}).status_code == 201

    [item] = client.get("/api/staff/appeals").json()
    assert item["can_decide"] is False
    assert client.post(f"/api/staff/appeals/{item['id']}", json={"decision": "overturn"}).status_code == 403
    assert not [r for r in client.get("/api/reminders").json() if r["link"] == "#/staff/appeals"]
    assert client.get(f"/api/companions/{mine['id']}").json()["removed"]

    # A third moderator gives it the second look.
    as_user(client, "kai@example.com", "Kai")
    kai = client.get("/api/me").json()
    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/users/{kai['id']}/role", json={"role": "moderator"})
    as_user(client, "kai@example.com", "Kai")
    assert client.post(f"/api/staff/appeals/{item['id']}", json={"decision": "overturn"}).status_code == 200


def test_only_your_own_removed_companions_can_be_appealed(client, staff):
    signup(client)
    zara = client.post("/api/companions", json={"persona": {"name": "Zara", "country": "Ghana"}}).json()
    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE}).status_code == 409  # not removed
    starter = next(c for c in client.get("/api/companions").json() if c["is_starter"])
    assert client.post(f"/api/companions/{starter['id']}/appeal", json={"message": MESSAGE}).status_code == 404
    as_user(client, "bo@example.com", "Bo")
    assert client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE}).status_code == 404
    assert client.get(f"/api/companions/{starter['id']}").json()["moderation"] is None


def test_staff_see_waiting_work_in_their_reminders(client, staff):
    zara = reported_zara(client)  # hidden after three reports, waiting for a moderator
    path = make_path(client)  # made by R2, the last reporter
    client.post(f"/api/custom-paths/{path['id']}/submit")
    staff(client)

    def work(email, name):
        as_user(client, email, name)
        return {r["link"]: r["text"] for r in client.get("/api/reminders").json() if r["kind"] == "staff"}

    assert work("boss@example.com", "Boss") == {
        "#/staff/reports": "1 reported companion waiting for a moderator.",
        "#/staff/paths": "1 learner path waiting for expert review.",
    }
    assert work("mo@example.com", "Mo") == {"#/staff/reports": "1 reported companion waiting for a moderator."}
    assert work("ada@example.com", "Ada") == {}

    as_user(client, "boss@example.com", "Boss")
    client.post(f"/api/staff/companions/{zara['id']}", json={"action": "remove", "note": "Pretends to be a real teacher."})
    as_user(client, "ada@example.com", "Ada")
    client.post(f"/api/companions/{zara['id']}/appeal", json={"message": MESSAGE})
    # Mo can take the second look; Boss made the removal, so it isn't waiting on Boss.
    assert work("mo@example.com", "Mo") == {"#/staff/appeals": "1 appeal waiting for a second look."}
    assert "#/staff/appeals" not in work("boss@example.com", "Boss")
