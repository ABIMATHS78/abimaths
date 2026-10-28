"""Emoji reactions in chat: the learner's to the companion's messages, and the companion's to the learner's."""

import asyncio

from tests.test_app import signup
from tests.test_community import as_user
from tests.test_usage import FakeStream, chat, claude, message, starter  # noqa: F401  (claude: fixture)


def thread(client, cid) -> list[tuple]:
    return [(m["role"], m["content"], m["reaction"]) for m in client.get(f"/api/companions/{cid}/messages").json()]


def test_the_learner_reacts_to_the_companions_messages(client):
    signup(client)
    cid = starter(client)
    r = chat(client, cid)
    sent = int(r.headers["X-Message-Id"])
    mine, reply = client.get(f"/api/companions/{cid}/messages", params={"since": sent}).json()
    assert (mine["id"], mine["role"], reply["role"]) == (sent, "user", "assistant")

    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "👍"}).status_code == 204
    # Another emoji takes the first one's place; however it's written, it's stored the one way.
    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "❤"}).status_code == 204
    assert thread(client, cid)[-1][2] == "❤️"
    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "👍🏾"}).status_code == 204
    assert thread(client, cid)[-1][2] == "👍"
    assert client.delete(f"/api/messages/{reply['id']}/reaction").status_code == 204
    assert thread(client, cid)[-1][2] is None

    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "🦄"}).status_code == 422
    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "👍" * 20}).status_code == 422
    # Not their own messages: the companion reacts to those.
    assert client.put(f"/api/messages/{sent}/reaction", json={"emoji": "👍"}).status_code == 400
    assert client.delete(f"/api/messages/{sent}/reaction").status_code == 400

    # Nor anyone else's conversation.
    as_user(client, "bob@example.com", "Bob")
    assert client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "👍"}).status_code == 404
    assert client.delete(f"/api/messages/{reply['id']}/reaction").status_code == 404
    assert client.put("/api/messages/999999/reaction", json={"emoji": "👍"}).status_code == 404


def test_in_demo_mode_the_companion_reacts_to_thanks_and_good_news(client):
    signup(client)
    cid = starter(client)
    chat(client, cid, "What is a token?")
    chat(client, cid, "Thank you, that finally makes sense!")
    chat(client, cid, "haha, good one")
    reactions = [(content, reaction) for role, content, reaction in thread(client, cid) if role == "user"]
    assert reactions == [("What is a token?", None), ("Thank you, that finally makes sense!", "❤️"), ("haha, good one", "😂")]
    # The companion's own messages wait for the learner's reaction.
    assert all(reaction is None for role, _, reaction in thread(client, cid) if role == "assistant")


def test_the_companions_reaction_is_taken_off_its_reply(client, claude):
    signup(client)
    cid = starter(client)
    claude.answers["stream"] = {"text": ["<reac", "tion>🎉</reac", "tion>\n\nWell done, ", "that's it!"],
                                "final": message("<reaction>🎉</reaction>\n\nWell done, that's it!")}
    r = chat(client, cid, "I passed my exam!")
    assert r.text == "Well done, that's it!"
    assert thread(client, cid)[-2:] == [("user", "I passed my exam!", "🎉"), ("assistant", "Well done, that's it!", None)]

    # Most replies don't react, and one that only looks like a tag at first is left as it is.
    claude.answers["stream"] = {"text": ["<", "b> is bold in HTML."], "final": message("<b> is bold in HTML.")}
    assert chat(client, cid, "What's <b>?").text == "<b> is bold in HTML."
    assert thread(client, cid)[-2][2] is None

    # A reaction it isn't allowed is dropped, and so is its tag.
    claude.answers["stream"] = {"text": ["<reaction>🦄</reaction>Sure."], "final": message("Sure.")}
    assert chat(client, cid, "Can I ask something?").text == "Sure."
    assert thread(client, cid)[-2][2] is None


def test_a_reply_that_is_only_a_reaction_isnt_kept(client, claude):
    signup(client)
    cid = starter(client)
    claude.answers["stream"] = {"text": ["<reaction>👍</reaction>"], "final": message("<reaction>👍</reaction>")}
    assert chat(client, cid, "Ok, thanks").text == ""
    assert thread(client, cid)[-1] == ("user", "Ok, thanks", "👍")
    # The next message still makes a valid conversation: the API needs non-empty, alternating turns.
    seen = []
    claude.stream = lambda **kw: (seen.append(kw), FakeStream({"text": ["Hi!"], "final": message("Hi!")}))[1]
    chat(client, cid, "Hello again")
    assert all(m["content"].strip() for m in seen[-1]["messages"])
    assert [m["role"] for m in seen[-1]["messages"]][-1] == "user"


def test_the_companion_sees_the_learners_reaction(client, claude):
    signup(client)
    cid = starter(client)
    claude.answers["stream"] = {"text": ["Tokens are chunks of text."], "final": message("Tokens are chunks of text.")}
    chat(client, cid)
    reply = client.get(f"/api/companions/{cid}/messages").json()[-1]
    client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "💡"})
    seen = []
    claude.stream = lambda **kw: (seen.append(kw), FakeStream({"text": ["Glad!"], "final": message("Glad!")}))[1]
    chat(client, cid, "Tell me more")
    _, rules, about = seen[-1]["system"]
    assert "<reaction>" in rules["text"] and "cache_control" in rules  # how to react: part of the cached prefix
    assert "The learner reacted 💡 to your last message" in about["text"]


def test_reactions_are_in_the_data_export(client):
    signup(client)
    cid = starter(client)
    chat(client, cid, "Thanks!")
    reply = client.get(f"/api/companions/{cid}/messages").json()[-1]
    client.put(f"/api/messages/{reply['id']}/reaction", json={"emoji": "🙏"})
    conversation = client.get("/api/me/export").json()["conversations"]
    assert [(m["role"], m["reaction"]) for m in conversation] == [("user", "❤️"), ("assistant", "🙏")]


def test_splitting_the_reaction_off_the_start():
    from app.reactions import split

    async def run(chunks):
        got = []

        async def source():
            for c in chunks:
                yield c

        text = "".join([c async for c in split(source(), got.append)])
        return text, got

    def go(chunks):
        return asyncio.run(run(chunks))

    assert go(["Hello", " there"]) == ("Hello there", [])
    assert go(["  <reaction>❤</reaction>  Hi"]) == ("Hi", ["❤️"])  # spaces around it, no variation selector
    assert go(["<", "r", "e", "a", "c", "t", "i", "o", "n", ">", "😂", "<", "/reaction>", "Ha!"]) == ("Ha!", ["😂"])
    assert go(["<rea"]) == ("<rea", [])  # ended before it could be a tag
    assert go(["<reaction>🎉"]) == ("<reaction>🎉", [])  # never closed
    assert go(["<reaction>" + "x" * 40 + " and on"]) == ("<reaction>" + "x" * 40 + " and on", [])
    assert go(["<reaction>👍</reaction>"]) == ("", ["👍"])
    assert go(["Answer first <reaction>👍</reaction>"]) == ("Answer first <reaction>👍</reaction>", [])  # only at the start
    assert go([]) == ("", [])
    assert go(["  ", "Hi"]) == ("  Hi", [])  # a plain reply comes through exactly as written
    # A "tag" with a sentence in it isn't a reaction: it's left as written, whatever the chunks.
    long = "<reaction>🎉 Congratulations on passing, that is huge news</reaction> Well done!"
    assert go([long]) == go([long[i:i + 4] for i in range(0, len(long), 4)]) == (long, [])
    assert go(["<reaction>🎉</reaction>"[i:i + 3] for i in range(0, 23, 3)] + ["Yes!"]) == ("Yes!", ["🎉"])


def test_tags_elsewhere_are_kept_out_of_the_saved_reply():
    from app.reactions import canonical, strip

    assert strip("Answer first <reaction>👍</reaction> then more") == "Answer first then more"
    assert canonical("👍🏽") == "👍" and canonical("❤") == "❤️" and canonical("🦄") is None and canonical("") is None
