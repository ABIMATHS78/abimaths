"""Blocking a community companion's designer: their companions and posts disappear for the learner who blocked them,
who can unblock them again. Apple's App Review guideline 1.2 asks for this where people make things others see."""

from tests.test_community import as_user, publish_zara


def test_blocking_a_designer_hides_their_companions_and_posts(client):
    zara = publish_zara(client)
    post = client.post(f"/api/companions/{zara['id']}/posts/generate").json()
    as_user(client, "bob@example.com", "Bob")
    client.post(f"/api/companions/{zara['id']}/follow")
    client.post(f"/api/companions/{zara['id']}/primary")
    assert post["id"] in [p["id"] for p in client.get("/api/feed").json()]

    assert client.post(f"/api/companions/{zara['id']}/block").status_code == 204
    assert zara["id"] not in [c["id"] for c in client.get("/api/companions").json()]
    assert post["id"] not in [p["id"] for p in client.get("/api/feed").json()]
    for path in (f"/api/companions/{zara['id']}", f"/api/companions/{zara['id']}/posts", f"/api/posts/{post['id']}",
                 f"/api/companions/{zara['id']}/avatar"):
        assert client.get(path).status_code == 404, path
    assert client.post(f"/api/companions/{zara['id']}/chat", json={"message": "hi"}).status_code == 404
    assert client.get("/api/me").json()["primary_companion_id"] is None
    blocks = client.get("/api/blocks").json()
    assert [b["name"] for b in blocks] == ["Ada"]

    # Everyone else still sees Zara, Ada included.
    as_user(client, "cleo@example.com", "Cleo")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 200
    as_user(client, "ada@example.com", "Ada")
    assert client.get(f"/api/companions/{zara['id']}").status_code == 200

    # Unblocked, Zara is back for Bob, who'd follow her again himself if he wanted to.
    as_user(client, "bob@example.com", "Bob")
    assert client.delete(f"/api/blocks/{blocks[0]['id']}").status_code == 204
    assert client.get("/api/blocks").json() == []
    back = client.get(f"/api/companions/{zara['id']}")
    assert back.status_code == 200 and back.json()["following"] is False


def test_only_other_learners_designers_can_be_blocked(client):
    zara = publish_zara(client)
    assert client.post(f"/api/companions/{zara['id']}/block").status_code == 400  # her own designer
    starter = next(c for c in client.get("/api/companions").json() if c["is_starter"])
    assert client.post(f"/api/companions/{starter['id']}/block").status_code == 400
    assert client.post("/api/companions/999999/block").status_code == 404


def test_blocks_are_in_the_data_download_and_go_with_the_account(client):
    from app import db

    zara = publish_zara(client)
    as_user(client, "bob@example.com", "Bob")
    client.post(f"/api/companions/{zara['id']}/block")
    assert [b["designer"] for b in client.get("/api/me/export").json()["designers_you_blocked"]] == ["Ada"]
    assert client.request("DELETE", "/api/me", json={"password": "correct horse"}).status_code == 204
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM blocks").fetchone()[0] == 0
