"""Community companions, photo avatars, voice, custom learning paths and reminders."""

import struct
import zlib

from tests.test_app import persona, signup

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00\x00\x00\rIHDR" + struct.pack(">II", 1, 1) + b"\x08\x06\x00\x00\x00" + \
    struct.pack(">I", zlib.crc32(b"IHDR" + struct.pack(">II", 1, 1) + b"\x08\x06\x00\x00\x00"))


def as_user(client, email, name):
    client.post("/api/auth/logout")
    client.post("/api/auth/login", json={"email": email, "password": "correct horse"})
    me = client.get("/api/me")
    if me.status_code == 401:
        signup(client, email=email, name=name)


def publish_zara(client):
    """Ada designs and publishes Zara; returns the companion."""
    signup(client)
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": True}).json()
    r = client.post(f"/api/companions/{zara['id']}/publish")
    assert r.status_code == 200 and r.json()["is_public"]
    return zara


def test_published_companion_is_discoverable_without_private_instructions(client):
    zara = publish_zara(client)
    as_user(client, "bob@example.com", "Bob")
    listed = {c["id"]: c for c in client.get("/api/companions").json()}
    assert zara["id"] in listed
    assert listed[zara["id"]]["designer"] == "Ada"
    assert listed[zara["id"]]["persona"]["custom_instructions"] == ""
    seen = client.get(f"/api/companions/{zara['id']}").json()
    assert seen["persona"]["custom_instructions"] == "" and not seen["is_mine"]

    # Bob can follow, chat with and remix Zara, but can't edit her or see her instructions.
    assert client.post(f"/api/companions/{zara['id']}/follow").status_code == 204
    assert client.post(f"/api/companions/{zara['id']}/chat", json={"message": "hi"}).status_code == 200
    remix = client.post(f"/api/companions/{zara['id']}/remix", json={"overrides": {"teaching_style": "storytelling"}}).json()
    assert remix["persona"]["custom_instructions"] == ""
    assert client.put(f"/api/companions/{zara['id']}", json=persona()).status_code == 403
    assert client.post(f"/api/companions/{zara['id']}/publish").status_code == 403

    # The designer still sees their own instructions.
    as_user(client, "ada@example.com", "Ada")
    assert client.get(f"/api/companions/{zara['id']}").json()["persona"]["custom_instructions"]


def test_custom_instructions_only_shape_the_designers_chats(client):
    zara = publish_zara(client)
    as_user(client, "bob@example.com", "Bob")
    from app.companions import get, persona_for

    bob = client.get("/api/me").json()
    assert persona_for(get(zara["id"]), bob["id"]).custom_instructions == ""
    assert persona_for(get(zara["id"]), zara["owner_id"]).custom_instructions.startswith("Quiz me")
    assert persona_for(get(zara["id"]), None).custom_instructions == ""  # shared content, e.g. cached lessons


def test_unpublish_hides_companion_and_its_posts(client):
    zara = publish_zara(client)
    post = client.post(f"/api/companions/{zara['id']}/posts/generate").json()
    as_user(client, "bob@example.com", "Bob")
    client.post(f"/api/companions/{zara['id']}/follow")
    client.post(f"/api/companions/{zara['id']}/primary")
    assert post["id"] in [p["id"] for p in client.get("/api/feed").json()]

    as_user(client, "ada@example.com", "Ada")
    assert client.delete(f"/api/companions/{zara['id']}/publish").json()["is_public"] is False
    as_user(client, "bob@example.com", "Bob")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 404
    assert post["id"] not in [p["id"] for p in client.get("/api/feed").json()]
    assert client.get(f"/api/companions/{zara['id']}/avatar").status_code == 404
    # Bob's primary is gone, so lessons fall back to a starter companion instead of failing.
    lesson = client.get("/api/lessons/tokens")
    assert lesson.status_code == 200 and lesson.json()["teacher"]["id"] != zara["id"]


def test_reports_hide_companion_for_review(client):
    zara = publish_zara(client)
    assert client.post(f"/api/companions/{zara['id']}/report", json={"reason": "spam"}).status_code == 400
    for i in range(3):
        as_user(client, f"r{i}@example.com", f"R{i}")
        assert client.post(f"/api/companions/{zara['id']}/report", json={"reason": "misleading"}).status_code == 204
    as_user(client, "late@example.com", "Late")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 404
    as_user(client, "ada@example.com", "Ada")
    me_view = client.get(f"/api/companions/{zara['id']}").json()
    assert me_view["hidden"] is True
    assert client.post(f"/api/companions/{zara['id']}/publish").status_code == 409
    starter = client.get("/api/companions").json()[-1]
    assert starter["is_starter"]
    assert client.post(f"/api/companions/{starter['id']}/report", json={"reason": "spam"}).status_code == 400


def test_photo_avatar_upload_validation_and_privacy(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    url = f"/api/companions/{mine['id']}/photo"
    assert client.put(url, content=b"<svg onload=alert(1)>", headers={"Content-Type": "image/png"}).status_code == 415
    assert client.put(url, content=PNG[:4] + b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
    assert client.put(url, content=PNG, headers={"Content-Type": "text/html"}).status_code == 204

    r = client.get(f"/api/companions/{mine['id']}/avatar")
    assert r.content == PNG and r.headers["content-type"] == "image/png"
    assert r.headers["x-content-type-options"] == "nosniff"
    assert "default-src 'none'" in r.headers["content-security-policy"]

    # Private companion photos are not visible to others or to anonymous visitors.
    client.post("/api/auth/logout")
    assert client.get(f"/api/companions/{mine['id']}/avatar").status_code == 404
    assert client.get("/api/companions/1/avatar", follow_redirects=False).status_code == 307  # starters stay public
    assert client.get("/api/companions/1/avatar?face=1").status_code == 200
    signup(client, email="eve@example.com", name="Eve")
    assert client.get(f"/api/companions/{mine['id']}/avatar").status_code == 404
    assert client.put(url, content=PNG).status_code == 403

    as_user(client, "ada@example.com", "Ada")
    assert client.delete(url).status_code == 204
    assert client.get(f"/api/companions/{mine['id']}/avatar").headers["content-type"].startswith("image/svg+xml")


def test_voice_settings_validated_and_defaulted(client):
    signup(client)
    c = client.post("/api/companions", json={"persona": persona(voice={"lang": "en-GH", "rate": 1.2, "pitch": 0.8})}).json()
    assert c["persona"]["voice"] == {"enabled": True, "lang": "en-GH", "rate": 1.2, "pitch": 0.8}
    assert client.post("/api/companions", json={"persona": persona(voice={"rate": 9})}).status_code == 422
    assert client.post("/api/companions", json={"persona": persona(voice={"lang": "<script>"})}).status_code == 422
    starter = client.get("/api/companions").json()[-1]
    assert starter["persona"]["voice"]["lang"].startswith("en-")


def test_custom_path_is_private_and_fully_usable(client):
    signup(client)
    path = client.post("/api/custom-paths", json={"topic": "AI in agriculture", "lessons": 4}).json()
    assert path["custom"] and path["total"] == 4 and path["id"].startswith("custom-")
    lesson_id = path["lessons"][0]["id"]
    assert path["id"] in [p["id"] for p in client.get("/api/paths").json()]

    assert client.get(f"/api/lessons/{lesson_id}").json()["path"]["id"] == path["id"]
    assert client.post(f"/api/lessons/{lesson_id}/complete").status_code == 204
    quiz = client.post("/api/assessments", json={"source_type": "lesson", "source_ref": lesson_id})
    assert quiz.status_code == 201, quiz.text
    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "path", "source_ref": path["id"]})
    assert exam.status_code == 201, exam.text
    assert client.post("/api/decks", json={"source_type": "path", "source_ref": path["id"], "count": 3}).status_code == 201
    plan = client.post("/api/study-plan", json={"paths": [path["id"]], "weekly_minutes": 60, "weeks": 2}).json()
    assert {i["path_id"] for w in plan["weeks"] for i in w["items"]} == {path["id"]}
    starter = client.get("/api/companions").json()[0]
    assert client.post(f"/api/companions/{starter['id']}/chat", json={"message": "?", "lesson_id": lesson_id}).status_code == 200

    prog = client.get("/api/progress").json()
    assert any(p["id"] == path["id"] and p["completed"] == 1 for p in prog["paths"])

    as_user(client, "bob@example.com", "Bob")
    assert client.get(f"/api/lessons/{lesson_id}").status_code == 404
    assert path["id"] not in [p["id"] for p in client.get("/api/paths").json()]
    assert client.delete(f"/api/custom-paths/{path['id']}").status_code == 404

    as_user(client, "ada@example.com", "Ada")
    assert client.delete(f"/api/custom-paths/{path['id']}").status_code == 204
    assert path["id"] not in [p["id"] for p in client.get("/api/paths").json()]
    assert client.get("/api/study-plan").json()["total"] == 0  # plan items for the deleted path drop out
    assert client.delete("/api/custom-paths/custom-nope").status_code == 404
    assert client.post("/api/custom-paths", json={"topic": "x"}).status_code == 422


def test_reminders(client):
    signup(client)
    starter = client.get("/api/companions").json()[0]
    client.post(f"/api/companions/{starter['id']}/primary")
    assert "new_posts" not in {r["kind"] for r in client.get("/api/reminders").json()}  # welcome posts aren't new
    import asyncio

    from app import companions, content

    asyncio.run(content.generate_for(companions.get(starter["id"])))
    new = [r for r in client.get("/api/reminders").json() if r["kind"] == "new_posts"]
    assert new and new[0]["count"] == 1
    assert client.post("/api/reminders/feed-seen").status_code == 204
    assert "new_posts" not in {r["kind"] for r in client.get("/api/reminders").json()}

    client.get("/api/lessons/tokens")
    client.post("/api/decks", json={"source_type": "lesson", "source_ref": "tokens", "count": 3})
    client.post("/api/assessments", json={"source_type": "lesson", "source_ref": "agent-loop"})
    client.post("/api/study-plan", json={"paths": ["llms"], "weekly_minutes": 60, "weeks": 2})
    rem = client.get("/api/reminders").json()
    by_kind = {r["kind"]: r for r in rem}
    assert by_kind["cards_due"]["count"] == 3 and by_kind["cards_due"]["link"] == "/flashcards"
    assert by_kind["continue_lesson"]["link"] == "/lessons/tokens"
    assert by_kind["unfinished_quiz"]["link"].startswith("/quizzes/")
    assert by_kind["plan"]["text"].startswith("This week")
    assert [r["priority"] for r in rem] == sorted(r["priority"] for r in rem)


def test_reminders_streak_at_risk(client):
    signup(client)
    from datetime import datetime, timedelta, timezone

    from app import db

    me = client.get("/api/me").json()
    yesterday = (datetime.now(timezone.utc).date() - timedelta(days=1)).isoformat()
    with db.connect() as conn:
        conn.execute("INSERT INTO activity (user_id, day) VALUES (?, ?)", (me["id"], yesterday))
    streak = [r for r in client.get("/api/reminders").json() if r["kind"] == "streak"]
    assert streak and "1-day streak" in streak[0]["text"]


def test_migration_adds_columns_to_old_database(tmp_path, monkeypatch):
    import sqlite3

    from app import db
    from app.config import settings

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    old_companions = db.SCHEMA.split("CREATE TABLE IF NOT EXISTS companions (")[1].split(");")[0]
    conn.execute(f"CREATE TABLE companions ({old_companions})")
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, email TEXT, display_name TEXT, password_hash TEXT, "
                 "level TEXT, goals TEXT, primary_companion_id INTEGER, onboarded INTEGER, created_at TEXT)")
    conn.commit()
    conn.close()
    monkeypatch.setattr(settings, "database_path", str(path))
    db.init_db()
    with db.connect() as c:
        cols = {r["name"] for r in c.execute("PRAGMA table_info(companions)")}
        ucols = {r["name"] for r in c.execute("PRAGMA table_info(users)")}
    assert {"is_public", "hidden", "published_at"} <= cols and "feed_seen_post_id" in ucols
    db.init_db()  # idempotent
