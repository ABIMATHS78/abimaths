"""Companions are Claude, so what a learner writes to them goes to Anthropic. Sign-up says so, and nobody is asked
first: companions reply from the start. A learner can turn Claude off in their profile, and then nothing they write
goes to it until they turn it back on."""

import time

from tests.test_app import persona
from tests.test_usage import chat, records, starter


def join(client):
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "password": "correct horse",
                                              "display_name": "Ada"})
    assert r.status_code == 201, r.text
    return r.json()


def turn(client, on):
    r = client.post("/api/me/ai-consent", json={"agree": on})
    assert r.status_code == 200, r.text
    return r.json()


def test_companions_reply_without_asking_first(client):
    assert join(client)["ai_consent"] is True and client.get("/api/me").json()["ai_consent"] is True
    assert chat(client, starter(client)).status_code == 200
    assert [r["kind"] for r in records()] == ["chat"]


def test_learners_who_never_agreed_get_replies_too(client):
    from app import db

    me = join(client)
    with db.connect() as conn:  # as for everyone who joined while the site asked first, and didn't agree
        row = conn.execute("SELECT ai_consent_at, ai_off_at FROM users WHERE id = ?", (me["id"],)).fetchone()
    assert tuple(row) == (None, None)
    assert chat(client, starter(client)).status_code == 200


def test_turned_off_in_the_profile_nothing_goes_to_claude(client):
    join(client)
    companion = starter(client)
    assert turn(client, on=False)["ai_consent"] is False and client.get("/api/me").json()["ai_consent"] is False
    r = chat(client, companion)
    assert r.status_code == 403 and "profile" in r.json()["detail"] and "Anthropic" in r.json()["detail"]
    for path, body in [("/api/assessments", {"kind": "quiz", "source_type": "lesson", "source_ref": "tokens"}),
                       ("/api/summaries", {"source_type": "lesson", "source_ref": "tokens"})]:
        r = client.post(path, json=body)
        assert r.status_code == 403, (path, r.text)
    assert client.get("/api/lessons/tokens").status_code == 403
    # A companion still says hello, from a template rather than from Claude.
    opening = client.post(f"/api/companions/{companion}/opening", json={})
    assert opening.status_code == 200 and opening.text
    # A new companion starts without the first post Claude would write for it.
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()
    time.sleep(0.2)
    assert client.get(f"/api/companions/{zara['id']}/posts").json() == []
    assert records() == []


def test_turned_back_on_replies_carry_on(client):
    join(client)
    companion = starter(client)
    turn(client, on=False)
    assert turn(client, on=True)["ai_consent"] is True
    assert chat(client, companion).status_code == 200
    assert [r["kind"] for r in records()] == ["chat"]


def test_turning_it_off_and_on_is_in_the_data_download(client):
    join(client)
    turn(client, on=False)
    off = client.get("/api/me/export").json()["account"]
    assert off["ai_off_at"][:4].isdigit() and off["ai_consent_at"] is None
    turn(client, on=True)
    on = client.get("/api/me/export").json()["account"]
    assert on["ai_off_at"] is None and on["ai_consent_at"][:4].isdigit()


def test_only_a_signed_in_learner_can_turn_it_off(client):
    assert client.post("/api/me/ai-consent", json={"agree": False}).status_code == 401
