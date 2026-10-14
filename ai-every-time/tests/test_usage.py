"""The daily AI allowance, and the record of AI requests and the tokens they used."""

import threading
import time
from types import SimpleNamespace

import anthropic
import pytest
from anthropic.types.beta import BetaUsage
from pydantic import ValidationError

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_email import mailbox  # noqa: F401  (fixture)


def chat(client, companion_id, text="What is a token?"):
    return client.post(f"/api/companions/{companion_id}/chat", json={"message": text})


def starter(client) -> int:
    return next(c["id"] for c in client.get("/api/companions").json() if c["is_starter"])


def used(client) -> int:
    return client.get("/api/me").json()["ai_allowance"]["used"]


def records() -> list[dict]:
    from app import db

    with db.connect() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM ai_usage ORDER BY id")]


def refusals() -> list[tuple]:
    from app import db

    with db.connect() as conn:
        return [tuple(r) for r in conn.execute("SELECT reason, count FROM ai_refusals ORDER BY hour, reason")]


def wait_for(check, seconds=3.0):
    """Background work (a companion's first post, memories from a chat) finishes soon after the response."""
    end = time.time() + seconds
    while not check() and time.time() < end:
        time.sleep(0.02)
    assert check()


def tokens(i=100, o=50, cache_read=0, cache_write=0, iterations=None) -> BetaUsage:
    return BetaUsage.model_validate({"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": cache_read,
                                     "cache_creation_input_tokens": cache_write, "iterations": iterations})


def answer(parsed=None, stop_reason="end_turn", usage=None, model="claude-opus-5-5"):
    """A response from Claude as the code reads it."""
    return SimpleNamespace(parsed_output=parsed, stop_reason=stop_reason, usage=usage or tokens(), model=model)


class FakeStream:
    def __init__(self, outcome):
        self.outcome = outcome

    async def __aenter__(self):
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self

    async def __aexit__(self, *a):
        return False

    @property
    async def text_stream(self):
        for text in self.outcome["text"]:
            yield text

    async def get_final_message(self):
        return self.outcome["final"]


class FakeClaude:
    """Stands in for the Anthropic client: answers each kind of request with what the test set up,
    a response or an exception to raise."""

    def __init__(self, **answers):
        self.answers = answers
        self.beta = SimpleNamespace(messages=self)

    async def parse(self, *, output_format, **kwargs):
        outcome = self.answers[output_format.__name__]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def stream(self, **kwargs):
        return FakeStream(self.answers["stream"])


@pytest.fixture()
def claude(client, monkeypatch):
    """Talk to a stand-in for Claude instead of using the offline demo replies."""
    from app import llm
    from app.config import settings

    from app import db

    def starters_have_posts():
        with db.connect() as conn:
            return not conn.execute("SELECT 1 FROM companions c WHERE is_starter = 1 AND NOT EXISTS "
                                    "(SELECT 1 FROM posts p WHERE p.companion_id = c.id)").fetchone()

    wait_for(starters_have_posts)  # starter companions write their first posts in demo mode at start-up
    fake = FakeClaude()
    monkeypatch.setattr(settings, "offline", False)
    monkeypatch.setattr(llm, "client", lambda: fake)
    return fake


def test_the_allowance_turns_requests_away_and_says_when(client, monkeypatch):
    from app import db
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance", 3)
    assert signup(client)["ai_allowance"] == {"limit": 3, "used": 0, "remaining": 3, "confirm_raises_to": None}
    cid = starter(client)
    for _ in range(3):
        assert chat(client, cid).status_code == 200
    r = chat(client, cid, "One more?")
    assert r.status_code == 429
    assert r.json()["detail"] == "You've reached your limit of 3 AI requests in 24 hours. You can ask again in about 24 hours."
    assert 86000 < int(r.headers["Retry-After"]) <= 86400
    # The message turned away isn't kept, and refusals are counted, together within the hour.
    assert "One more?" not in [m["content"] for m in client.get(f"/api/companions/{cid}/messages").json()]
    chat(client, cid)
    assert refusals() == [("allowance", 2)]
    assert client.get("/api/me").json()["ai_allowance"] == {"limit": 3, "used": 3, "remaining": 0, "confirm_raises_to": None}

    # A request stops counting 24 hours after it was made; the wait is until enough of them have.
    def ago(hours):
        from datetime import datetime, timedelta, timezone

        return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat(timespec="seconds")

    with db.connect() as conn:
        for row, hours in zip([r["id"] for r in records()], (23, 22, 1)):
            conn.execute("UPDATE ai_usage SET created_at = ? WHERE id = ?", (ago(hours), row))
    assert chat(client, cid).json()["detail"].endswith("You can ask again in about 1 hour.")
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": "ai-foundations"})
    assert exam.status_code == 429 and exam.json()["detail"].endswith("You can ask again in about 2 hours.")
    with db.connect() as conn:
        conn.execute("UPDATE ai_usage SET created_at = ? WHERE id = ?", (ago(25), records()[0]["id"]))
    assert chat(client, cid).status_code == 200


def test_mock_exams_and_new_paths_count_double(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance", 3)
    signup(client)
    chat(client, starter(client))
    chat(client, starter(client))
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": "ai-foundations"})
    assert exam.status_code == 429
    assert exam.json()["detail"].startswith("A mock exam counts as 2 AI requests, and you have 1 left. You can ask again in")
    path = client.post("/api/custom-paths", json={"topic": "AI for farming"})
    assert path.status_code == 429 and path.json()["detail"].startswith("A new learning path counts as 2 AI requests")
    quiz = client.post("/api/assessments", json={"source_type": "path", "source_ref": "ai-foundations"})
    assert quiz.status_code == 201 and used(client) == 3

    monkeypatch.setattr(settings, "ai_allowance", 1)
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": "ai-foundations"})
    assert exam.json()["detail"].startswith("A mock exam needs 2 AI requests, more than your allowance of 1.")


def test_every_kind_of_ai_request_counts(client):
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": True}).json()
    wait_for(lambda: client.get(f"/api/companions/{zara['id']}/posts").json())  # its first post
    chat(client, zara["id"])
    client.get("/api/lessons/tokens")
    client.post("/api/assessments", json={"source_type": "lesson", "source_ref": "tokens"})
    client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": "ai-foundations"})
    client.post("/api/decks", json={"source_type": "lesson", "source_ref": "tokens"})
    client.post("/api/summaries", json={"source_type": "lesson", "source_ref": "tokens"})
    client.post("/api/study-plan", json={})
    client.post("/api/custom-paths", json={"topic": "AI for farming"})
    client.post(f"/api/companions/{zara['id']}/posts/generate")
    assert [(r["kind"], r["units"]) for r in records()] == [
        ("post", 1), ("chat", 1), ("lesson", 1), ("quiz", 1), ("exam", 2), ("flashcards", 1), ("summary", 1), ("plan", 1),
        ("path", 2), ("post", 1)]
    assert used(client) == 12


def test_lessons_cost_only_the_first_time_they_are_taught(client):
    signup(client)
    client.get("/api/lessons/tokens")
    client.get("/api/lessons/tokens")
    assert used(client) == 1
    as_user(client, "bo@example.com", "Bo")  # taught by the same companion, so it's ready
    client.get("/api/lessons/tokens")
    assert used(client) == 0


def test_failed_calls_are_refunded_and_unusable_answers_count(client, claude):
    from app.tutor import Summary

    signup(client)

    def summary():
        return client.post("/api/summaries", json={"source_type": "lesson", "source_ref": "tokens"})

    claude.answers["Summary"] = anthropic.APIConnectionError(request=None)
    assert summary().status_code == 502
    assert used(client) == 0 and (records()[-1]["units"], records()[-1]["failed"]) == (0, 1)

    # Claude answered, so these count even though the answer couldn't be used.
    claude.answers["Summary"] = answer(stop_reason="refusal", usage=tokens(900, 12))
    assert summary().status_code == 502
    try:
        Summary.model_validate_json('{"summary": "cut sh')
    except ValidationError as e:
        claude.answers["Summary"] = e  # a reply cut short doesn't fit the schema
    assert summary().status_code == 502
    claude.answers["Summary"] = answer(Summary(summary="Short.", key_takeaways=["a"], next_step="Quiz."), usage=tokens(1000, 80))
    assert summary().status_code == 200
    assert used(client) == 3
    assert [(r["units"], r["input_tokens"], r["output_tokens"]) for r in records()[-3:]] == [(1, 900, 12), (1, 0, 0), (1, 1000, 80)]


def test_chat_records_the_reply_and_the_memories_taken_from_it(client, claude):
    from app.llm import MemoryUpdate

    signup(client)
    cid = starter(client)
    # The first model declined and the fallback model answered: both attempts count.
    final = SimpleNamespace(stop_reason="end_turn", model="claude-opus-4-8", usage=tokens(1100, 200, iterations=[
        {"type": "message", "input_tokens": 1100, "output_tokens": 0, "cache_read_input_tokens": 800,
         "cache_creation_input_tokens": 0},
        {"type": "fallback_message", "model": "claude-opus-4-8", "input_tokens": 1100, "output_tokens": 200,
         "cache_read_input_tokens": 0, "cache_creation_input_tokens": 300},
    ]))
    claude.answers |= {"stream": {"text": ["Tokens are ", "chunks of text."], "final": final},
                       "MemoryUpdate": answer(MemoryUpdate(new_facts=["Is curious about tokens"]), usage=tokens(400, 20))}
    r = chat(client, cid)
    assert r.status_code == 200 and r.text == "Tokens are chunks of text."
    wait_for(lambda: records()[-1]["input_tokens"] == 2600)  # memories are taken after the reply
    rec = records()[-1]
    assert (rec["kind"], rec["units"], rec["output_tokens"], rec["cache_read_tokens"], rec["cache_write_tokens"],
            rec["model"], rec["fallback"]) == ("chat", 1, 220, 800, 300, "claude-opus-4-8", 1)
    assert [m["fact"] for m in client.get(f"/api/companions/{cid}/memories").json()] == ["Is curious about tokens"]

    # When Claude can't be reached the learner gets an apology, and the message doesn't count.
    claude.answers["stream"] = anthropic.APIConnectionError(request=None)
    assert "couldn't connect" in chat(client, cid).text
    assert (records()[-1]["units"], records()[-1]["failed"]) == (0, 1)
    assert used(client) == 1


def test_confirming_the_email_address_raises_the_allowance(client, mailbox, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance", 4)
    monkeypatch.setattr(settings, "ai_allowance_unconfirmed", 2)
    assert signup(client)["ai_allowance"] == {"limit": 2, "used": 0, "remaining": 2, "confirm_raises_to": 4}
    cid = starter(client)
    chat(client, cid)
    chat(client, cid)
    r = chat(client, cid)
    assert r.status_code == 429 and r.json()["detail"].endswith("Confirm your email address to raise your limit to 4.")
    assert client.post("/api/email/confirm", content=f"token={mailbox.link_token('confirm')}").status_code == 200
    assert chat(client, cid).status_code == 200
    assert client.get("/api/me").json()["ai_allowance"] == {"limit": 4, "used": 3, "remaining": 1, "confirm_raises_to": None}


def test_admins_have_no_limit_and_the_site_ceiling_pauses_everyone_else(client, monkeypatch):
    from app import usage
    from app.config import settings

    monkeypatch.setattr(settings, "admin_emails", frozenset({"boss@example.com"}))
    monkeypatch.setattr(settings, "ai_allowance", 2)
    monkeypatch.setattr(settings, "ai_site_limit", 4)
    assert signup(client, email="boss@example.com", name="Boss")["ai_allowance"]["limit"] is None
    cid = starter(client)
    for _ in range(3):
        assert chat(client, cid).status_code == 200
    as_user(client, "ada@example.com", "Ada")
    assert chat(client, cid).status_code == 200  # four in 24 hours, everyone together
    r = chat(client, cid)
    assert r.status_code == 429 and r.json()["detail"] == usage.BUSY and r.headers["Retry-After"] == "3600"
    as_user(client, "boss@example.com", "Boss")
    assert chat(client, cid).status_code == 200  # admins aren't paused
    assert refusals() == [("site", 1)]


def test_parallel_requests_cannot_overspend(client, monkeypatch):
    from fastapi import HTTPException

    from app import main, usage
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance", 5)
    user = main._load_user(signup(client)["id"])
    results, start = [], threading.Barrier(20)

    def ask():
        start.wait()
        try:
            results.append(usage.reserve(user, "chat"))
        except HTTPException:
            results.append(None)

    threads = [threading.Thread(target=ask) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(r is not None for r in results) == 5 and used(client) == 5


def test_a_new_companions_first_post_counts(client, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "ai_allowance", 1)
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": True}).json()
    wait_for(lambda: client.get(f"/api/companions/{zara['id']}/posts").json())
    assert used(client) == 1
    # With nothing left a new companion starts without a post, and asking for one says why.
    zed = client.post("/api/companions", json={"persona": persona(name="Zed"), "make_primary": False})
    assert zed.status_code == 201
    time.sleep(0.2)
    assert client.get(f"/api/companions/{zed.json()['id']}/posts").json() == []
    r = client.post(f"/api/companions/{zed.json()['id']}/posts/generate")
    assert r.status_code == 429 and "limit of 1" in r.json()["detail"]


def test_tokens_are_recorded_for_work_no_learner_asked_for(client, monkeypatch):
    from app import db, usage

    signup(client)
    usage.record_tokens(answer(usage=tokens(300, 600, cache_read=200)))  # e.g. a starter companion's scheduled post
    rec = records()[-1]
    assert (rec["user_id"], rec["kind"], rec["units"], rec["input_tokens"], rec["output_tokens"], rec["cache_read_tokens"]) == (
        None, "automatic", 0, 300, 600, 200)

    def broken():
        raise OSError("disk full")

    with monkeypatch.context() as m:
        m.setattr(db, "connect", broken)
        usage.record_tokens(answer())  # bookkeeping never breaks a request


def test_ai_requests_are_in_the_data_download_and_outlive_the_account(client):
    signup(client)
    chat(client, starter(client))
    [req] = client.get("/api/me/export").json()["ai_requests"]
    assert req["kind"] == "chat" and req["units"] == 1
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 204
    assert [(r["user_id"], r["kind"]) for r in records()] == [(None, "chat")]  # in the totals, without the account
