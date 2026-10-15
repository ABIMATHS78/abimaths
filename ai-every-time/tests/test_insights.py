"""The admin Insights page: learners, learning, community and AI use."""

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_usage import chat, starter


def test_insights_are_for_admins(client, monkeypatch):
    from app import db
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    me = signup(client)
    assert client.get("/api/staff/insights").status_code == 403
    with db.connect() as conn:
        conn.execute("UPDATE users SET role = 'moderator' WHERE id = ?", (me["id"],))
    assert client.get("/api/staff/insights").status_code == 403
    as_user(client, "boss@example.com", "Boss")
    assert client.get("/api/staff/insights").status_code == 200


def test_insights_count_learners_learning_and_ai_use(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    monkeypatch.setattr(settings, "ai_allowance", 2)
    signup(client)
    cid = starter(client)
    chat(client, cid)
    chat(client, cid)
    assert chat(client, cid, "Turned away").status_code == 429
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()  # no post: none left
    client.post(f"/api/companions/{zara['id']}/publish")

    as_user(client, "bo@example.com", "Bo")
    client.post(f"/api/companions/{zara['id']}/follow")
    liked = client.get("/api/feed").json()[0]
    client.post(f"/api/posts/{liked['id']}/engage", json={"kind": "like"})
    client.get("/api/lessons/tokens")
    client.post("/api/lessons/tokens/complete")
    quiz = client.post("/api/assessments", json={"source_type": "lesson", "source_ref": "tokens"}).json()
    result = client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": [0] * quiz["total"]}).json()

    as_user(client, "boss@example.com", "Boss")
    d = client.get("/api/staff/insights").json()
    assert d["learners"] == {"total": 3, "new_this_week": 3, "active_this_week": 2, "confirmed": 0, "with_notifications": 0}
    assert d["learning"] == {"lessons_completed": 1, "quizzes": 1, "quiz_average": result["percent"], "chat_messages": 2}
    assert d["community"]["designed"] == 1 and d["community"]["published"] == 1
    assert [(c["name"], c["designer"], c["followers"]) for c in d["community"]["most_followed"]] == [("Zara", "Ada", 2)]
    assert [t["topic"] for t in d["community"]["topics"]] == [liked["topic"]]  # topics nobody engaged with are left out

    ai = d["ai"]
    assert len(ai["days"]) == 14 and ai["days"][-1]["requests"] == 4 and sum(x["requests"] for x in ai["days"]) == 4
    assert {k["kind"]: k["requests"] for k in ai["kinds"]} == {"chat": 2, "lesson": 1, "quiz": 1}
    # Ada's third message and Zara's first post were both turned away.
    assert ai["turned_away"] == {"allowance": {"requests": 2, "learners": 1}, "site": {"requests": 0, "learners": 0}}
    assert ai["units_last_24h"] == 4
    assert ai["limits"] == {"allowance": 2, "unconfirmed": None, "site": 0}


def test_insights_count_tokens_failures_and_old_requests_apart(client, monkeypatch):
    from app import db, main, usage
    from app.config import settings
    from tests.test_usage import message, tokens

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    user = main._load_user(signup(client, email="boss@example.com", name="Boss")["id"])
    with usage.recording(usage.reserve(user, "summary")):
        usage.record_tokens(message("{}", usage=tokens(1000, 200, cache_read=500, cache_write=100)))
    with usage.recording(usage.reserve(user, "plan")):
        usage.failed()
    usage.record_tokens(message("{}", usage=tokens(300, 400)))  # a scheduled post
    old = usage.reserve(user, "chat")
    with db.connect() as conn:  # from before the 14 days
        conn.execute("UPDATE ai_usage SET created_at = '2000-01-01T00:00:00+00:00' WHERE id = ?", (old,))

    ai = client.get("/api/staff/insights").json()["ai"]
    kinds = {k["kind"]: k for k in ai["kinds"]}
    assert set(kinds) == {"summary", "plan", "automatic"}
    assert (kinds["summary"]["input_tokens"], kinds["summary"]["output_tokens"], kinds["summary"]["cache_read_tokens"],
            kinds["summary"]["cache_write_tokens"]) == (1000, 200, 500, 100)
    assert (kinds["plan"]["failed"], kinds["plan"]["units"]) == (1, 0)
    # The chart counts learners' requests that went through: not the failed one or the scheduled post.
    assert ai["days"][-1] == {"day": ai["days"][-1]["day"], "requests": 1, "tokens": 1800}


def test_insights_estimate_the_cost_at_list_prices(client, monkeypatch):
    from app import insights, main, usage
    from app.config import settings
    from tests.test_usage import message, tokens

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    user = main._load_user(signup(client, email="boss@example.com", name="Boss")["id"])
    # $4 per million input tokens, $20 output, $0.20 cache reads, and writes at 1.25 times input.
    with usage.recording(usage.reserve(user, "summary")):
        usage.record_tokens(message("{}", usage=tokens(1000, 200, cache_read=500, cache_write=100)))
    # A dated model name has its model's prices; a model without a price is left out and named.
    usage.record_tokens(message("{}", usage=tokens(300, 400), model="claude-opus-4-8-20260101"))
    usage.record_tokens(message("{}", usage=tokens(1_000_000, 0), model="someone-elses-model"))

    ai = client.get("/api/staff/insights").json()["ai"]
    costs = {k["kind"]: k["cost"] for k in ai["kinds"]}
    assert costs == {"summary": 0.0086, "automatic": 0.0115}
    assert ai["cost"] == 0.0201 and ai["unpriced_models"] == ["someone-elses-model"]

    monkeypatch.setattr(settings, "prices", {"someone-elses-model": (1.0, 2.0, 0.1)})
    ai = client.get("/api/staff/insights").json()["ai"]
    assert ai["cost"] == 1.0201 and ai["unpriced_models"] == []
    assert insights.price("claude-opus-5-5") == (4.0, 20.0, 0.2)
    assert insights.price("claude-opus-5-50") is None  # only a date after the name, not another model


def test_prices_can_be_set_in_the_environment(monkeypatch):
    from app import config

    monkeypatch.setenv("AIET_PRICES", '{"my-model": [1, 2, 0.1], "bad": [1]}')
    assert config._prices() == {"my-model": (1.0, 2.0, 0.1)}
    monkeypatch.setenv("AIET_PRICES", "not json")
    assert config._prices() == {}


def test_bad_prices_are_skipped_one_by_one(monkeypatch):
    import json

    from app import config

    monkeypatch.setenv("AIET_PRICES", json.dumps({"good": [1, 2, 0.1], "typo": 5, "short": [1, 2], "null": [1, 2, None],
                                                  "word": "123", "yes": [True, 1, 1], "negative": [-1, 2, 0.1]}))
    assert config._prices() == {"good": (1.0, 2.0, 0.1)}
    monkeypatch.setenv("AIET_PRICES", '{"nan": [NaN, 1, 1], "inf": [Infinity, 1, 1], "huge": [1e999, 1, 1], "big": [%s, 1, 1]}' % ("9" * 400))
    assert config._prices() == {}  # and nothing crashes on the way
    monkeypatch.setenv("AIET_PRICES", "[1, 2, 3]")
    assert config._prices() == {}
