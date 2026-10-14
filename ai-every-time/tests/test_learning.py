import json
from datetime import datetime, timedelta, timezone

from tests.test_app import persona, signup


def onboard(client, email="ada@example.com", name="Ada"):
    signup(client, email=email, name=name)
    starter = client.get("/api/companions").json()[0]
    client.post(f"/api/companions/{starter['id']}/primary")
    return starter


def test_paths_and_lesson_taught_by_primary_companion(client):
    starter = onboard(client)
    paths = client.get("/api/paths").json()
    assert len(paths) >= 5 and all(p["total"] == len(p["lessons"]) for p in paths)
    lesson_id = paths[0]["lessons"][0]["id"]

    lesson = client.get(f"/api/lessons/{lesson_id}").json()
    assert lesson["teacher"]["id"] == starter["id"]
    assert lesson["status"] == "started" and lesson["body"] and lesson["next"]

    assert client.post(f"/api/lessons/{lesson_id}/complete").status_code == 204
    paths = client.get("/api/paths").json()
    assert paths[0]["completed"] == 1 and paths[0]["lessons"][0]["status"] == "completed"
    assert client.get("/api/lessons/not-a-lesson").status_code == 404


def test_lesson_cached_per_companion_and_custom_teacher(client):
    onboard(client)
    mine = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()
    a = client.get("/api/lessons/tokens").json()
    b = client.get(f"/api/lessons/tokens?companion_id={mine['id']}").json()
    assert b["teacher"]["name"] == "Zara" and a["teacher"]["id"] != mine["id"]
    from app import db

    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM lesson_content WHERE lesson_id = 'tokens'").fetchone()[0] == 2


def test_quiz_hides_answers_until_submitted_and_grades(client):
    onboard(client)
    quiz = client.post("/api/assessments", json={"source_type": "lesson", "source_ref": "tokens"}).json()
    n = quiz["total"]
    assert 4 <= n <= 5 and quiz["kind"] == "quiz"
    assert all("answer_index" not in q for q in quiz["questions"])
    assert "answer_index" not in json.dumps(client.get(f"/api/assessments/{quiz['id']}").json())

    from app import db

    with db.connect() as conn:
        truth = [q["answer_index"] for q in json.loads(
            conn.execute("SELECT questions FROM assessments WHERE id = ?", (quiz["id"],)).fetchone()[0])]
    answers = truth[:-2] + [(truth[-2] + 1) % 4, None]
    result = client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": answers}).json()
    pct = round(100 * (n - 2) / n)
    assert result["score"] == n - 2 and result["percent"] == pct and result["passed"] is False
    assert [q["correct"] for q in result["questions"]] == [True] * (n - 2) + [False, False]
    assert all(q["explanation"] for q in result["questions"])
    assert client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": answers}).status_code == 409

    paths = client.get("/api/paths").json()
    lesson = next(l for p in paths for l in p["lessons"] if l["id"] == "tokens")
    assert lesson["best_score"] == pct
    prog = client.get("/api/progress").json()
    assert prog["quizzes_taken"] == 1 and prog["quiz_average"] == pct
    assert {"lesson_id": "tokens", "title": "Tokens: how models read text", "best_score": pct} in prog["needs_review"]


def test_mock_exam_covers_path_with_breakdown(client):
    onboard(client)
    assert client.post("/api/assessments", json={"kind": "exam", "source_type": "lesson", "source_ref": "tokens"}).status_code == 422
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": "llms"}).json()
    assert exam["total"] == 12 and exam["time_limit_minutes"] == 20
    result = client.post(f"/api/assessments/{exam['id']}/submit", json={"answers": []}).json()
    assert result["score"] == 0 and result["over_time"] is False
    assert sum(b["total"] for b in result["by_lesson"]) == 12
    assert len(result["by_lesson"]) >= 3


def test_quiz_from_post_and_ownership(client):
    onboard(client)
    post_id = client.get("/api/feed").json()[0]["id"]
    quiz = client.post("/api/assessments", json={"source_type": "post", "source_ref": str(post_id)})
    assert quiz.status_code == 201, quiz.text
    qid = quiz.json()["id"]
    client.post("/api/auth/logout")
    onboard(client, email="bob@example.com", name="Bob")
    assert client.get(f"/api/assessments/{qid}").status_code == 404
    assert client.post(f"/api/assessments/{qid}/submit", json={"answers": [0]}).status_code == 404


def test_flashcards_spaced_repetition(client):
    onboard(client)
    deck = client.post("/api/decks", json={"source_type": "lesson", "source_ref": "agent-loop", "count": 4}).json()
    assert deck["total"] == 4 and deck["due"] == 4
    due = client.get(f"/api/review?deck_id={deck['id']}").json()
    assert len(due) == 4

    good = client.post(f"/api/cards/{due[0]['id']}/review", json={"grade": "good"}).json()
    assert good["box"] == 2 and good["reviews"] == 1
    easy = client.post(f"/api/cards/{due[1]['id']}/review", json={"grade": "easy"}).json()
    assert easy["box"] == 3
    again = client.post(f"/api/cards/{due[2]['id']}/review", json={"grade": "again"}).json()
    assert again["box"] == 1 and again["lapses"] == 1
    assert datetime.fromisoformat(again["due_at"]) < datetime.now(timezone.utc) + timedelta(minutes=11)
    assert datetime.fromisoformat(easy["due_at"]) > datetime.now(timezone.utc) + timedelta(days=2)

    assert len(client.get(f"/api/review?deck_id={deck['id']}").json()) == 1
    prog = client.get("/api/progress").json()
    assert prog["cards_total"] == 4 and prog["cards_due"] == 1

    client.post("/api/auth/logout")
    onboard(client, email="bob@example.com", name="Bob")
    assert client.post(f"/api/cards/{due[3]['id']}/review", json={"grade": "good"}).status_code == 404
    assert client.get(f"/api/decks/{deck['id']}").status_code == 404


def test_summaries_and_conversation_flashcards(client):
    starter = onboard(client)
    assert client.post("/api/summaries", json={"source_type": "conversation", "source_ref": str(starter["id"])}).status_code == 400
    client.post(f"/api/companions/{starter['id']}/chat", json={"message": "Explain overfitting with a football example please"})
    s = client.post("/api/summaries", json={"source_type": "conversation", "source_ref": str(starter["id"])}).json()
    assert s["key_takeaways"] and s["next_step"]
    s = client.post("/api/summaries", json={"source_type": "lesson", "source_ref": "bias-fairness"}).json()
    assert s["title"] == "Bias and fairness"
    deck = client.post("/api/decks", json={"source_type": "conversation", "source_ref": str(starter["id"]), "count": 3})
    assert deck.status_code == 201, deck.text


def test_lesson_tutoring_chat_uses_lesson_context(client, monkeypatch):
    starter = onboard(client)
    seen = {}
    from app import llm

    real = llm.build_system

    def spy(persona, learner, memories, post):
        seen["context"] = post
        return real(persona, learner, memories, post)

    monkeypatch.setattr(llm, "build_system", spy)
    from app.llm import learner_prompt

    ctx = __import__("app.learning", fromlist=["lesson_context"]).lesson_context(starter["id"], "tool-use")
    assert "Tools: letting models take actions" in ctx["title"]
    assert "tool use" in learner_prompt({"display_name": "A", "level": "beginner", "goals": ""}, [], ctx).lower()
    r = client.post(f"/api/companions/{starter['id']}/chat", json={"message": "Why do tool descriptions matter?", "lesson_id": "tool-use"})
    assert r.status_code == 200
    r = client.post(f"/api/companions/{starter['id']}/chat", json={"message": "hi", "lesson_id": "nope"})
    assert r.status_code == 404


def test_study_plan_follows_progress(client):
    onboard(client)
    client.post("/api/lessons/what-is-ai/complete")
    assert client.post("/api/study-plan", json={"paths": ["bogus"]}).status_code == 422
    plan = client.post("/api/study-plan", json={"weekly_minutes": 60, "weeks": 2, "paths": ["ai-foundations"]}).json()
    items = [i for w in plan["weeks"] for i in w["items"]]
    assert items and all(i["path_id"] == "ai-foundations" for i in items)
    assert "what-is-ai" not in {i["lesson_id"] for i in items}
    assert plan["done"] == 0 and plan["total"] == len(items)

    first = items[0]
    client.post(f"/api/lessons/{first['lesson_id']}/complete")
    plan = client.get("/api/study-plan").json()
    assert plan["done"] == 1


def test_progress_streak_and_next_lesson(client):
    onboard(client)
    prog = client.get("/api/progress").json()
    assert prog["streak_days"] == 0 and prog["next_lesson"]["lesson_id"] == "what-is-ai"
    client.post("/api/lessons/what-is-ai/complete")
    from app import db

    me = client.get("/api/me").json()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    with db.connect() as conn:
        conn.execute("INSERT INTO activity (user_id, day) VALUES (?, ?)", (me["id"], yesterday))
    prog = client.get("/api/progress").json()
    assert prog["streak_days"] == 2 and prog["lessons_completed"] == 1
    assert prog["next_lesson"]["lesson_id"] == "rules-vs-learning"
    assert prog["last_7_days"][-1]["active"] and prog["last_7_days"][-2]["active"]
