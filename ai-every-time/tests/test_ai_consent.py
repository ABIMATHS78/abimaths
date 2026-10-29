"""Nothing a learner writes goes to Claude until they've agreed to it: at sign-up, or later from the page, and they can
take it back. Apple's App Review guideline 5.1.2(i) asks for this before personal data reaches a third-party AI."""

import time

from tests.test_app import persona
from tests.test_usage import chat, records, starter


def join(client, agree):
    r = client.post("/api/auth/signup", json={"email": "ada@example.com", "password": "correct horse",
                                              "display_name": "Ada", "ai_consent": agree})
    assert r.status_code == 201, r.text
    return r.json()


def test_without_agreeing_nothing_goes_to_claude(client):
    assert join(client, agree=False)["ai_consent"] is False
    companion = starter(client)
    r = chat(client, companion)
    assert r.status_code == 403 and r.headers["X-AI-Consent"] == "needed"
    assert "Anthropic" in r.json()["detail"]
    for path, body in [("/api/assessments", {"kind": "quiz", "source_type": "lesson", "source_ref": "tokens"}),
                       ("/api/summaries", {"source_type": "lesson", "source_ref": "tokens"})]:
        r = client.post(path, json=body)
        assert r.status_code == 403 and r.headers["X-AI-Consent"] == "needed", (path, r.text)
    assert client.get("/api/lessons/tokens").status_code == 403
    # A companion still says hello, from a template rather than from Claude.
    opening = client.post(f"/api/companions/{companion}/opening", json={})
    assert opening.status_code == 200 and opening.text
    # A new companion starts without the first post Claude would write for it.
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()
    time.sleep(0.2)
    assert client.get(f"/api/companions/{zara['id']}/posts").json() == []
    assert records() == []


def test_agreeing_later_lets_ai_requests_through_and_it_can_be_taken_back(client):
    join(client, agree=False)
    companion = starter(client)
    me = client.post("/api/me/ai-consent", json={"agree": True}).json()
    assert me["ai_consent"] is True and client.get("/api/me").json()["ai_consent"] is True
    assert chat(client, companion).status_code == 200
    assert [r["kind"] for r in records()] == ["chat"]

    assert client.post("/api/me/ai-consent", json={"agree": False}).json()["ai_consent"] is False
    assert chat(client, companion).status_code == 403
    assert len(records()) == 1


def test_agreeing_at_sign_up_is_recorded_with_the_time(client):
    from app import db

    me = join(client, agree=True)
    assert me["ai_consent"] is True
    with db.connect() as conn:
        when = conn.execute("SELECT ai_consent_at FROM users WHERE id = ?", (me["id"],)).fetchone()[0]
    assert when and when[:4].isdigit()
    assert chat(client, starter(client)).status_code == 200
    # It's in the learner's data download too.
    assert client.get("/api/me/export").json()["account"]["ai_consent_at"] == when


def test_only_a_signed_in_learner_can_agree(client):
    assert client.post("/api/me/ai-consent", json={"agree": True}).status_code == 401
