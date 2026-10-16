"""Companions start each new chat themselves, from what the learner is into."""

import anthropic

from tests.test_app import persona
from tests.test_community import as_user
from tests.test_usage import claude, message, records, tokens, used  # noqa: F401  (claude: fixture)


def join(client, name="Ada", email=None, **profile):
    r = client.post("/api/auth/signup", json={"email": email or f"{name.lower()}@example.com", "password": "correct horse",
                                              "display_name": name, **profile})
    assert r.status_code == 201, r.text
    return r.json()


def by_slug(client, slug) -> dict:
    return next(c for c in client.get("/api/companions").json() if c["slug"] == slug)


def opening(client, companion_id, post_id=None):
    return client.post(f"/api/companions/{companion_id}/opening", json={"post_id": post_id})


def conversation(client, companion_id) -> list[tuple[str, str]]:
    return [(m["role"], m["content"]) for m in client.get(f"/api/companions/{companion_id}/messages").json()]


def test_a_companion_starts_a_new_chat_from_the_learner_s_interests(client):
    assert join(client, interests="  football, cooking ")["interests"] == "football, cooking"
    amahle = by_slug(client, "amahle-ndlovu")
    r = opening(client, amahle["id"])
    assert r.status_code == 200
    first = r.text
    assert first.startswith("Hi Ada! I'm Amahlé Ndlovù from South Africa.") and "you're into football" in first
    assert conversation(client, amahle["id"]) == [("assistant", first)]  # kept as the chat's first message
    assert opening(client, amahle["id"]).status_code == 204  # only once
    assert client.post(f"/api/companions/{amahle['id']}/chat", json={"message": "Yes please!"}).status_code == 200
    assert [role for role, _ in conversation(client, amahle["id"])] == ["assistant", "user", "assistant"]
    assert used(client) == 1  # the reply; starting the chat cost the learner nothing


def test_without_interests_it_starts_from_the_goals_or_asks_what_they_re_into(client):
    join(client, goals="Build my first AI app")
    amahle = by_slug(client, "amahle-ndlovu")
    assert 'You told me what you\'d like to learn: "Build my first AI app"' in opening(client, amahle["id"]).text
    # Interests added to the profile later are what the next companion starts from.
    assert client.patch("/api/me", json={"interests": " chess, jazz "}).json()["interests"] == "chess, jazz"
    assert "you're into chess" in opening(client, by_slug(client, "kwame-asante")["id"]).text
    assert client.get("/api/me/export").json()["account"]["interests"] == "chess, jazz"

    client.post("/api/auth/logout")
    join(client, "Bo")  # no goals or interests
    assert "what are you into outside AI?" in opening(client, amahle["id"]).text
    mojeed = by_slug(client, "mojeed-abisiga")
    assert "I'm an AI learning companion modelled on Mojeed Abisiga." in opening(client, mojeed["id"]).text
    # Arriving from one of its posts, it starts from that.
    zara = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()
    post = client.post(f"/api/companions/{zara['id']}/posts/generate").json()
    assert f'You\'ve been reading "{post["title"]}"' in opening(client, zara["id"], post["id"]).text


def test_a_chat_the_learner_began_gets_no_opening(client):
    join(client, interests="football")
    mine = client.post("/api/companions", json={"persona": persona(), "make_primary": False}).json()
    client.post(f"/api/companions/{mine['id']}/chat", json={"message": "Hi Zara"})
    assert opening(client, mine["id"]).status_code == 204
    assert conversation(client, mine["id"])[0] == ("user", "Hi Zara")
    as_user(client, "eve@example.com", "Eve")
    assert opening(client, mine["id"]).status_code == 404  # someone else's private companion


def test_claude_writes_the_opening_and_it_stays_in_the_conversation(claude, client):
    from app import llm

    join(client, interests="football, cooking", goals="Build my first AI app")
    amahle = by_slug(client, "amahle-ndlovu")
    claude.answers["stream"] = {"text": ["Hi Ada! Football fan? ", "Scouts use models to spot talent. Which club is yours?"],
                                "final": message(usage=tokens(1200, 80, cache_read=900))}
    r = opening(client, amahle["id"])
    first = "Hi Ada! Football fan? Scouts use models to spot talent. Which club is yours?"
    assert r.text == first and conversation(client, amahle["id"]) == [("assistant", first)]
    asked = claude.streamed
    assert asked["messages"] == [{"role": "user", "content": "(Ada has opened a chat with you.)"}]
    stable, learner = asked["system"]
    assert stable["cache_control"] == {"type": "ephemeral"} and "Amahlé" in stable["text"]  # the prefix chat caches
    assert "Interests (draw examples from them when it helps): football, cooking" in learner["text"]
    assert "# Start the conversation" in learner["text"] and "Start from their interests" in learner["text"]
    (rec,) = [r for r in records() if r["kind"] == "opening"]
    assert (rec["units"], rec["failed"], rec["input_tokens"], rec["cache_read_tokens"]) == (0, 0, 1200, 900)
    assert used(client) == 0

    # The learner's reply: Claude sees the chat as it happened, the opening included.
    claude.answers["stream"] = {"text": ["Arsenal, of course."], "final": message()}
    client.post(f"/api/companions/{amahle['id']}/chat", json={"message": "Arsenal! How do the models work?"})
    assert claude.streamed["messages"] == [
        {"role": "user", "content": llm.opened({"display_name": "Ada"})},
        {"role": "assistant", "content": first},
        {"role": "user", "content": "Arsenal! How do the models work?"},
    ]
    assert claude.streamed["system"][0] == stable

    # When Claude can't be reached, the companion still starts the chat, from the template.
    claude.answers["stream"] = anthropic.APIConnectionError(request=None)
    kwame = by_slug(client, "kwame-asante")
    assert "you're into football" in opening(client, kwame["id"]).text
    assert [r["failed"] for r in records() if r["kind"] == "opening"] == [0, 1]
