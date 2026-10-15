import time

def signup(client, email="ada@example.com", name="Ada"):
    r = client.post("/api/auth/signup", json={"email": email, "password": "correct horse", "display_name": name,
                                              "goals": "Build my first AI app"})
    assert r.status_code == 201, r.text
    return r.json()


def persona(**kw):
    base = {
        "name": "Zara", "country": "Ghana", "flag": "🇬🇭", "tagline": "Agents, the Accra way",
        "expertise": ["AI Agents", "Prompt Engineering"], "teaching_style": "socratic",
        "personality": {"warmth": 90, "humor": 80, "formality": 10, "directness": 50, "enthusiasm": 90, "patience": 70},
        "writing_style": {"length": "concise", "level": "beginner", "emoji": "expressive", "signature_phrase": "Ayekoo!"},
        "avatar": {"skin": "#5C3A21", "hair_style": "afro", "hair_color": "#0D0D0D", "background": "#FFB800",
                   "accessory": "headwrap", "accessory_color": "#009C3B", "expression": "grin"},
        "interests": ["highlife", "football"],
        "custom_instructions": "Quiz me at the end of each explanation.",
    }
    base.update(kw)
    return base


def test_starters_seeded_with_content(client):
    signup(client)
    companions = client.get("/api/companions").json()
    starters = [c for c in companions if c["is_starter"]]
    assert len(starters) == 8
    assert {c["persona"]["country"] for c in starters} >= {"Nigeria", "Japan", "Brazil", "India"}
    for c in starters:
        assert client.get(f"/api/companions/{c['id']}/avatar.svg").text.startswith("<svg")
    # Initial posts are generated in the background on startup.
    for _ in range(50):
        if all(client.get(f"/api/companions/{c['id']}/posts").json() for c in starters):
            break
        time.sleep(0.05)
    assert all(client.get(f"/api/companions/{c['id']}/posts").json() for c in starters)


def test_signup_requires_onboarding_then_choose_starter(client):
    me = signup(client)
    assert me["onboarded"] is False and me["primary_companion_id"] is None
    assert all(c["following"] for c in client.get("/api/companions").json() if c["is_starter"])
    starter = client.get("/api/companions").json()[0]
    assert client.post(f"/api/companions/{starter['id']}/primary").status_code == 204
    me = client.get("/api/me").json()
    assert me["onboarded"] and me["primary_companion_id"] == starter["id"]
    assert client.get(f"/api/companions/{starter['id']}").json()["following"]


def test_design_own_companion_edit_and_restore(client):
    signup(client)
    r = client.post("/api/companions", json={"persona": persona(), "make_primary": True})
    assert r.status_code == 201, r.text
    mine = r.json()
    assert mine["persona"]["name"] == "Zara" and not mine["is_starter"]
    assert client.get("/api/me").json()["primary_companion_id"] == mine["id"]

    updated = client.put(f"/api/companions/{mine['id']}", json=persona(name="Zara 2.0", teaching_style="storytelling"))
    assert updated.status_code == 200 and updated.json()["persona"]["name"] == "Zara 2.0"

    versions = client.get(f"/api/companions/{mine['id']}/versions").json()
    assert [v["persona"]["name"] for v in versions] == ["Zara 2.0", "Zara"]
    restored = client.post(f"/api/companions/{mine['id']}/versions/{versions[1]['id']}/restore").json()
    assert restored["persona"]["name"] == "Zara"


def test_remix_starter_keeps_original(client):
    signup(client)
    starter = client.get("/api/companions").json()[0]
    r = client.post(f"/api/companions/{starter['id']}/remix", json={"overrides": {"teaching_style": "socratic"}})
    assert r.status_code == 201
    remix = r.json()
    assert remix["remixed_from"] == starter["id"]
    assert remix["persona"]["name"] == f"My {starter['persona']['name']}"
    assert remix["persona"]["teaching_style"] == "socratic"
    assert client.get(f"/api/companions/{starter['id']}").json()["persona"] == starter["persona"]


def test_cannot_edit_starter_or_see_others_companions(client):
    signup(client)
    starter = client.get("/api/companions").json()[0]
    assert client.put(f"/api/companions/{starter['id']}", json=persona()).status_code == 403
    mine = client.post("/api/companions", json={"persona": persona()}).json()

    client.post("/api/auth/logout")
    signup(client, email="bob@example.com", name="Bob")
    assert client.get(f"/api/companions/{mine['id']}").status_code == 404
    assert client.put(f"/api/companions/{mine['id']}", json=persona()).status_code == 403
    assert client.post(f"/api/companions/{mine['id']}/chat", json={"message": "hi"}).status_code == 404
    assert mine["id"] not in [c["id"] for c in client.get("/api/companions").json()]


def test_invalid_persona_rejected(client):
    signup(client)
    bad_colour = persona(avatar={"skin": "red"})
    assert client.post("/api/companions", json={"persona": bad_colour}).status_code == 422
    assert client.post("/api/companions", json={"persona": persona(expertise=["Astrology"])}).status_code == 422
    assert client.post("/api/companions", json={"persona": persona(name="")}).status_code == 422


def test_chat_streams_saves_history_and_remembers(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    r = client.post(f"/api/companions/{mine['id']}/chat", json={"message": "My name is Ada and I want to learn reinforcement learning."})
    assert r.status_code == 200
    assert "Zara" in r.text
    msgs = client.get(f"/api/companions/{mine['id']}/messages").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]

    for _ in range(50):
        facts = [m["fact"] for m in client.get(f"/api/companions/{mine['id']}/memories").json()]
        if facts:
            break
        time.sleep(0.05)
    assert "Prefers to be called Ada" in facts
    assert "Wants to learn reinforcement learning" in facts

    mem = client.get(f"/api/companions/{mine['id']}/memories").json()[0]
    assert client.delete(f"/api/memories/{mem['id']}").status_code == 204
    assert mem["id"] not in [m["id"] for m in client.get(f"/api/companions/{mine['id']}/memories").json()]


def test_content_engagement_drives_topic_scores(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    for _ in range(50):  # wait for the new companion's first post, written in the background
        if client.get(f"/api/companions/{mine['id']}/posts").json():
            break
        time.sleep(0.05)
    post = client.post(f"/api/companions/{mine['id']}/posts/generate").json()
    assert post["topic"] in ["AI Agents", "Prompt Engineering"]
    assert post["id"] in [p["id"] for p in client.get("/api/feed").json()]

    for kind in ["view", "like", "helpful"]:
        assert client.post(f"/api/posts/{post['id']}/engage", json={"kind": kind}).status_code == 204
    r = client.post(f"/api/companions/{mine['id']}/chat", json={"message": "Tell me more", "post_id": post["id"]})
    assert r.status_code == 200
    scores = client.get("/api/insights/topics").json()
    # Every post exists in a followed feed, so the feed gives the per-topic post count.
    same_topic = [p for p in client.get("/api/feed").json() if p["topic"] == post["topic"]]
    assert scores[post["topic"]] == round((1 + 3 + 4 + 6) / len(same_topic), 2)

    feed_post = next(p for p in client.get("/api/feed").json() if p["id"] == post["id"])
    assert feed_post["liked"] and feed_post["likes"] == 1


def test_auth_errors(client):
    assert client.get("/api/me").status_code == 401
    signup(client)
    assert client.post("/api/auth/signup", json={"email": "ada@example.com", "password": "another one", "display_name": "A"}).status_code == 409
    client.post("/api/auth/logout")
    assert client.get("/api/me").status_code == 401
    assert client.post("/api/auth/login", json={"email": "ada@example.com", "password": "wrong pass"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "ADA@example.com", "password": "correct horse"}).status_code == 200


def test_system_prompt_reflects_persona():
    from app.llm import build_system
    from app.persona import Persona

    system = build_system(Persona.model_validate(persona()), {"display_name": "Ada", "level": "beginner", "goals": "x"},
                          ["Works in finance"], None)
    stable, learner = system[0]["text"], system[1]["text"]
    assert "Zara" in stable and "Ghana" in stable and "Ayekoo!" in stable
    assert "guiding questions" in stable and "Quiz me" in stable
    assert "very warm" in stable and "very casual" in stable
    assert "Works in finance" in learner and "Ada" in learner


def test_health_check(client):
    assert client.get("/healthz").json() == {"ok": True}  # no sign-in needed
