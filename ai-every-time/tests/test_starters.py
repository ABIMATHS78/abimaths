"""The starter companions from starter_companions.csv: their personas, follower counts, photos and first posts."""

import asyncio
import csv
import io
import time

import httpx
import pytest
from PIL import Image

from tests.test_app import persona, signup
from tests.test_community import as_user
from tests.test_usage import wait_for


def starters(client) -> list[dict]:
    return [c for c in client.get("/api/companions").json() if c["is_starter"]]


def by_slug(client, slug) -> dict:
    return next(c for c in client.get("/api/companions").json() if c["slug"] == slug)


def picture(w=600, h=900, fmt="PNG") -> bytes:
    out = io.BytesIO()
    Image.new("RGB", (w, h), (200, 120, 40)).save(out, fmt)
    return out.getvalue()


def test_every_csv_row_has_a_persona_and_a_flag():
    from app.persona import EXPERTISE_AREAS
    from app.starter_companions import CSV_PATH, PERSONAS, REGIONS, STARTERS

    with CSV_PATH.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == len(STARTERS) == 103
    assert set(PERSONAS) == {r["slug"] for r in rows}  # none written for a row that's gone, none missing
    for row in rows:
        s = STARTERS[row["slug"]]
        p = s.persona
        assert (p.name, p.country, s.region, s.followers) == (row["name"], row["country"], row["region"], int(row["followerCount"]))
        assert p.flag and s.region in REGIONS and s.photo_url.startswith("https://")
        assert p.tagline and p.bio and p.expertise and set(p.expertise) <= set(EXPERTISE_AREAS)
        assert p.writing_style.signature_phrase and p.writing_style.sample_line and p.interests
        assert "{" not in p.bio  # names filled in
    assert {s.slug for s in STARTERS.values() if s.real_person} == {"mojeed-abisiga", "gbenga-akingbulere", "ogweda-haruna"}
    # Modelled on real people, and saying so, without a guess at anyone's pronouns.
    for slug in ("mojeed-abisiga", "gbenga-akingbulere", "ogweda-haruna"):
        bio = STARTERS[slug].persona.bio
        assert bio.startswith("An AI learning companion modelled on ") and "doesn't speak for" in bio
        assert not {" he ", " she ", " his ", " her "} & {f" {w} " for w in bio.lower().split()}
    assert STARTERS["aleksandra-sorokina"].persona.flag == "🇦🇶" and STARTERS["amahle-ndlovu"].persona.flag == "🇿🇦"


def test_the_most_followed_in_each_region_is_featured():
    from app.starter_companions import STARTERS

    featured = {s.region: s for s in STARTERS.values() if s.featured}
    assert len(featured) == 7
    for region, s in featured.items():
        assert s.followers == max(x.followers for x in STARTERS.values() if x.region == region)
    assert featured["Africa"].slug == "mojeed-abisiga"


def test_names_and_counts_come_from_the_csv(tmp_path):
    from app.starter_companions import CSV_PATH, load

    text = CSV_PATH.read_text(encoding="utf-8")
    renamed = tmp_path / "renamed.csv"
    renamed.write_text(text.replace("Amahlé Ndlovù,", "Amahle Ndlovu,").replace(",31498\n", ',"40,000"\n'), encoding="utf-8")
    s = load(renamed)["amahle-ndlovu"]
    assert s.persona.name == "Amahle Ndlovu" and s.persona.bio.startswith("Amahle is a data scientist")
    assert s.followers == 40000  # thousands separators are fine

    # A row the site can't read stops it from starting, rather than its companion being removed.
    rows = text.splitlines()
    for broken, problem in [(rows + [rows[1]], "used twice"), ([rows[0], rows[1].rsplit(",", 1)[0] + ",lots"], "line 2"),
                            ([rows[0].replace("followerCount", "followers"), rows[1]], "followerCount"), ([rows[0]], "no companions")]:
        bad = tmp_path / "bad.csv"
        bad.write_text("\n".join(broken) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=problem):
            load(bad)


def test_followers_show_on_starters_only(client):
    signup(client)
    mojeed = by_slug(client, "mojeed-abisiga")
    # The CSV's count, plus everyone following here: Ada starts out following the featured starters.
    assert mojeed["followers"] == 111170 + 1 and mojeed["following"]
    amahle = by_slug(client, "amahle-ndlovu")
    assert amahle["followers"] == 31498 and not amahle["following"]
    client.post(f"/api/companions/{amahle['id']}/follow")
    assert client.get(f"/api/companions/{amahle['id']}").json()["followers"] == 31499
    client.delete(f"/api/companions/{amahle['id']}/follow")
    assert by_slug(client, "amahle-ndlovu")["followers"] == 31498

    # Companions learners design don't show a follower count, even when they have followers.
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{zara['id']}/publish")
    as_user(client, "bob@example.com", "Bob")
    client.post(f"/api/companions/{zara['id']}/follow")
    assert client.get(f"/api/companions/{zara['id']}").json()["followers"] is None
    assert next(c for c in client.get("/api/companions").json() if c["id"] == zara["id"])["followers"] is None


def test_starters_are_listed_most_followed_first(client):
    from app import db

    signup(client, email="bo@example.com", name="Bo")
    bobby = client.post("/api/companions", json={"persona": persona(name="Bobby")}).json()
    client.post(f"/api/companions/{bobby['id']}/publish")
    as_user(client, "ada@example.com", "Ada")
    client.post(f"/api/companions/{bobby['id']}/follow")
    mine = [client.post("/api/companions", json={"persona": persona(name=n), "make_primary": False}).json() for n in ("A", "B")]
    listed = client.get("/api/companions").json()
    # The learner's own first, in the order they made them, though Bobby has more followers; then others'.
    assert [c["id"] for c in listed[:3]] == [mine[0]["id"], mine[1]["id"], bobby["id"]]
    counts = [c["followers"] for c in listed if c["is_starter"]]
    assert counts == sorted(counts, reverse=True) and counts[0] == 111172
    with db.connect() as conn:  # follower counts are looked up by companion for every page that lists them
        assert "idx_follows_companion" in {r["name"] for r in conn.execute("PRAGMA index_list(follows)")}


def test_welcome_faces_are_public(client):
    faces = client.get("/api/starters/faces").json()  # signed out
    assert len(faces) == 8 and faces[0]["name"] == "Mojeed Abisiga"
    assert {f["name"] for f in faces[:7]} >= {"Agnieszka Kowalskà", "Arohà Ngàta", "Aleksandrà Sorokinà"}


def test_upgrading_retires_the_old_starters(client, monkeypatch):
    """A site that had the eight earlier starters: they leave the catalogue, but learners who talked to one keep
    it, with their chats, and nobody is sent back to choose a companion."""
    from dataclasses import replace

    from app import companions, db
    from app.starter_companions import STARTERS

    ada = signup(client)  # Adaeze was Ada's companion, and they talked
    as_user(client, "bob@example.com", "Bob")  # Bob never talked to the old starters
    bob = client.get("/api/me").json()
    with db.connect() as conn:
        adaeze, haruto = (conn.execute(
            "INSERT INTO companions (slug, owner_id, is_starter, persona, created_at, updated_at) "
            "SELECT ?, NULL, 1, persona, created_at, updated_at FROM companions WHERE slug = 'amahle-ndlovu'", (slug,)).lastrowid
            for slug in ("adaeze-nigeria", "haruto-japan"))
        for uid in (ada["id"], bob["id"]):  # everyone followed every starter at sign-up back then
            conn.execute("DELETE FROM follows WHERE user_id = ?", (uid,))
            for cid in (adaeze, haruto):
                conn.execute("INSERT INTO follows (user_id, companion_id, created_at) VALUES (?, ?, ?)", (uid, cid, db.now()))
        conn.execute("UPDATE users SET primary_companion_id = ?, onboarded = 1 WHERE id = ?", (adaeze, ada["id"]))
        conn.execute("INSERT INTO messages (user_id, companion_id, role, content, created_at) VALUES (?, ?, 'user', 'hi', ?)",
                     (ada["id"], adaeze, db.now()))
        # And a starter whose follower count and photo address have changed in the CSV since.
        conn.execute("UPDATE companions SET base_followers = 5, photo_url = 'https://old.example/x.png' WHERE slug = 'kwame-asante'")
        kwame = conn.execute("SELECT id FROM companions WHERE slug = 'kwame-asante'").fetchone()["id"]
        conn.execute("INSERT INTO companion_photos (companion_id, mime, data, updated_at) VALUES (?, 'image/webp', x'00', ?)",
                     (kwame, db.now()))

    companions.seed_starters()
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM companions WHERE is_starter = 1 AND retired = 0").fetchone()[0] == 103
        assert [tuple(r) for r in conn.execute("SELECT slug, retired, featured FROM companions WHERE id IN (?, ?) ORDER BY id",
                                               (adaeze, haruto))] == [("adaeze-nigeria", 1, 0), ("haruto-japan", 1, 0)]
        assert conn.execute("SELECT COUNT(*) FROM messages WHERE companion_id = ?", (adaeze,)).fetchone()[0] == 1  # kept
        user = conn.execute("SELECT * FROM users WHERE id = ?", (ada["id"],)).fetchone()
        assert user["primary_companion_id"] == adaeze and user["onboarded"]  # nobody sent back to onboarding

        def follows(uid):
            return {r[0] for r in conn.execute("SELECT c.slug FROM follows f JOIN companions c ON c.id = f.companion_id "
                                               "WHERE f.user_id = ? AND c.featured = 0", (uid,))}, \
                conn.execute("SELECT COUNT(*) FROM follows f JOIN companions c ON c.id = f.companion_id "
                             "WHERE f.user_id = ? AND c.featured = 1", (uid,)).fetchone()[0]
        # Ada keeps the one she talked to; follows from sign-up end; both now follow the featured starters.
        assert follows(ada["id"]) == ({"adaeze-nigeria"}, 7) and follows(bob["id"]) == (set(), 7)
        row = conn.execute("SELECT base_followers, photo_url FROM companions WHERE id = ?", (kwame,)).fetchone()
        assert row["base_followers"] == 29701 and "Kwam" in row["photo_url"]
        assert not conn.execute("SELECT 1 FROM companion_photos WHERE companion_id = ?", (kwame,)).fetchone()  # fetched anew

    # Bob doesn't see the old ones anywhere; Ada still has Adaeze, marked retired.
    assert not [c for c in client.get("/api/companions").json() if c["retired"]]
    assert all(f["id"] not in (adaeze, haruto) for f in client.get("/api/starters/faces").json())
    as_user(client, "ada@example.com", "Ada")
    assert [c["slug"] for c in client.get("/api/companions").json() if c["retired"]] == ["adaeze-nigeria"]
    assert client.get(f"/api/companions/{adaeze}/messages").json()[0]["content"] == "hi"
    assert client.get("/api/me").json()["onboarded"]

    # Starting again changes nothing; listing a retired one in the CSV again brings it back.
    with db.connect() as conn:
        before = conn.execute("SELECT updated_at FROM companions WHERE id = ?", (adaeze,)).fetchone()[0]
    companions.seed_starters()
    with db.connect() as conn:
        assert conn.execute("SELECT updated_at FROM companions WHERE id = ?", (adaeze,)).fetchone()[0] == before
    monkeypatch.setattr(companions, "STARTERS", STARTERS | {"adaeze-nigeria": replace(STARTERS["amahle-ndlovu"], slug="adaeze-nigeria")})
    companions.seed_starters()
    with db.connect() as conn:
        assert conn.execute("SELECT retired FROM companions WHERE id = ?", (adaeze,)).fetchone()[0] == 0


def test_real_people_cant_be_remixed_and_show_initials(client):
    signup(client)
    mojeed = by_slug(client, "mojeed-abisiga")
    assert mojeed["real_person"]
    r = client.post(f"/api/companions/{mojeed['id']}/remix", json={})
    assert r.status_code == 403 and "real people" in r.json()["detail"]
    face = client.get(f"/api/companions/{mojeed['id']}/avatar?face=1").text
    assert face.startswith("<svg") and ">MA</text>" in face  # initials, not a drawn guess at a face
    assert "<path" in client.get(f"/api/companions/{by_slug(client, 'amahle-ndlovu')['id']}/avatar?face=1").text


def test_remixing_a_starter_keeps_its_photo(client):
    from app import db

    signup(client)
    amahle = by_slug(client, "amahle-ndlovu")
    with db.connect() as conn:
        conn.execute("INSERT INTO companion_photos (companion_id, mime, data, updated_at) VALUES (?, 'image/webp', ?, ?)",
                     (amahle["id"], b"RIFF....WEBPVP8 ", db.now()))
    assert client.get(f"/api/companions/{amahle['id']}").json()["has_photo"]
    kept = client.post(f"/api/companions/{amahle['id']}/remix", json={"make_primary": False}).json()
    r = client.get(f"/api/companions/{kept['id']}/avatar")
    assert r.headers["content-type"] == "image/webp" and r.content == b"RIFF....WEBPVP8 "
    drawn = client.post(f"/api/companions/{amahle['id']}/remix", json={"keep_photo": False, "make_primary": False}).json()
    assert client.get(f"/api/companions/{drawn['id']}/avatar").headers["content-type"] == "image/svg+xml"
    # Another learner's photo stays theirs.
    again = client.post(f"/api/companions/{kept['id']}/remix", json={"make_primary": False}).json()
    assert client.get(f"/api/companions/{again['id']}/avatar").headers["content-type"] == "image/svg+xml"


def test_starter_photos_are_fetched_once_and_kept_small(client, monkeypatch):
    from app import db, photos
    from app.config import settings

    asked = []
    original = picture()

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(str(request.url))
        if request.url.host == "drive.google.com":
            return httpx.Response(200, text="<html>Sign in</html>", headers={"content-type": "text/html"})
        if "Kwam" in request.url.path:
            return httpx.Response(404)
        return httpx.Response(200, content=original, headers={"content-type": "image/png"})

    monkeypatch.setattr(settings, "fetch_photos", True)
    monkeypatch.setattr(photos, "transport", httpx.MockTransport(handler))
    assert asyncio.run(photos.ensure_starter_photos()) == 103 - 4  # three Drive links aren't shared, and one 404

    signup(client)
    amahle = by_slug(client, "amahle-ndlovu")
    r = client.get(f"/api/companions/{amahle['id']}/avatar", params={"v": amahle["updated_at"]})
    assert r.headers["content-type"] == "image/webp" and r.headers["cache-control"] == "private, max-age=86400"
    with Image.open(io.BytesIO(r.content)) as im:
        assert im.size == (320, 320)
    assert len(r.content) < len(original) / 10
    # Drive sharing links are asked for as the picture itself.
    assert any(u.startswith("https://drive.google.com/thumbnail?id=1m8eYmxRLwk-kR3C8iomLAtmdVSY2jC53") for u in asked)

    # One that couldn't be fetched sends the browser to its own address meanwhile, and is tried again later.
    kwame = by_slug(client, "kwame-asante")
    r = client.get(f"/api/companions/{kwame['id']}/avatar", follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("https://d2xsxph8kpxj0f.cloudfront.net/")
    assert r.headers["cache-control"] == "no-cache"
    gbenga = by_slug(client, "gbenga-akingbulere")
    r = client.get(f"/api/companions/{gbenga['id']}/avatar", follow_redirects=False)
    assert r.headers["location"] == "https://drive.google.com/thumbnail?id=1m8eYmxRLwk-kR3C8iomLAtmdVSY2jC53&sz=w1000"

    photos._tried.clear()
    monkeypatch.setattr(photos, "transport", httpx.MockTransport(
        lambda request: httpx.Response(200, content=picture(900, 600, "JPEG"))))
    client.get(f"/api/companions/{kwame['id']}/avatar", follow_redirects=False)  # asks for it in the background
    wait_for(lambda: client.get(f"/api/companions/{kwame['id']}/avatar").headers["content-type"] == "image/webp")
    with db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM companion_photos").fetchone()[0] == 103 - 3
    # The next start fetches only what's still missing.
    assert asyncio.run(photos.ensure_starter_photos()) == 3
    assert asyncio.run(photos.ensure_starter_photos()) == 0


def test_a_photo_is_fetched_once_at_a_time(client, monkeypatch):
    """Faces asked for while the start-up fetch runs, or by many pages at once, don't fetch a photo twice, and no
    more than four photos are fetched at the same time."""
    from app import photos
    from app.config import settings

    asked, now, most = [], [0], [0]
    original = picture(60, 60)

    async def handler(request):
        asked.append(str(request.url))
        now[0] += 1
        most[0] = max(most[0], now[0])
        await asyncio.sleep(0.01)
        now[0] -= 1
        return httpx.Response(200, content=original)

    monkeypatch.setattr(settings, "fetch_photos", True)
    monkeypatch.setattr(photos, "transport", httpx.MockTransport(handler))

    async def everything_at_once():
        ids = [r["id"] for r in photos._missing()]
        await asyncio.gather(photos.ensure_starter_photos(), *(photos.fetch_later(i) for i in ids),
                             *(photos.fetch_later(i) for i in ids))

    asyncio.run(everything_at_once())
    assert len(asked) == len(set(asked)) == 103 and most[0] <= photos.AT_ONCE
    assert not photos._missing()


def test_learners_photos_arent_kept_by_browsers(client):
    from tests.test_community import PNG

    signup(client)
    zara = client.post("/api/companions", json={"persona": persona()}).json()
    client.put(f"/api/companions/{zara['id']}/photo", content=PNG)
    zara = client.get(f"/api/companions/{zara['id']}").json()
    r = client.get(f"/api/companions/{zara['id']}/avatar", params={"v": zara["updated_at"]})
    assert r.headers["cache-control"] == "no-cache"  # unpublishing it hides it at once


def test_photos_are_never_fetched_when_turned_off(client, monkeypatch):
    from app import photos

    monkeypatch.setattr(photos, "transport", httpx.MockTransport(lambda request: pytest.fail("fetched")))
    assert asyncio.run(photos.ensure_starter_photos()) == 0  # AIET_FETCH_PHOTOS=0 in the tests
    signup(client)
    r = client.get(f"/api/companions/{by_slug(client, 'amahle-ndlovu')['id']}/avatar", follow_redirects=False)
    assert r.status_code == 307


def test_drive_links_become_direct_and_others_stay():
    from app.photos import direct_url

    assert direct_url("https://drive.google.com/file/d/1Ap0TEAfihdXsIDlJjKxVZtJABXscGOQI/view?usp=sharing") == \
        "https://drive.google.com/thumbnail?id=1Ap0TEAfihdXsIDlJjKxVZtJABXscGOQI&sz=w1000"
    assert direct_url("https://drive.google.com/open?id=1Ap0TEAfihdXsIDlJjKxVZtJABXscGOQI").startswith(
        "https://drive.google.com/thumbnail?id=1Ap0TEAfihdXsIDlJjKxVZtJABXscGOQI")
    plain = "https://d2xsxph8kpxj0f.cloudfront.net/a/b/Amahl%C3%A9_1.png"
    assert direct_url(plain) == plain


def test_thumbnails_keep_transparency_and_turn_upright():
    from app.photos import thumbnail

    out = io.BytesIO()
    Image.new("RGBA", (400, 300), (0, 0, 0, 0)).save(out, "PNG")
    with Image.open(io.BytesIO(thumbnail(out.getvalue()))) as im:
        assert im.format == "WEBP" and im.size == (320, 320) and im.mode == "RGBA"


def test_a_starter_writes_its_first_post_when_first_followed(client):
    signup(client)
    amahle = by_slug(client, "amahle-ndlovu")
    assert client.get(f"/api/companions/{amahle['id']}/posts").json() == []  # not featured: no post at start-up
    client.post(f"/api/companions/{amahle['id']}/follow")
    wait_for(lambda: len(client.get(f"/api/companions/{amahle['id']}/posts").json()) == 1)
    client.delete(f"/api/companions/{amahle['id']}/follow")
    client.post(f"/api/companions/{amahle['id']}/follow")
    client.post(f"/api/companions/{amahle['id']}/primary")
    time.sleep(0.2)
    assert len(client.get(f"/api/companions/{amahle['id']}/posts").json()) == 1  # only the first time


def test_first_posts_are_capped(client, monkeypatch):
    """The platform pays for starters' first posts, so one that didn't work out isn't tried again for an hour,
    there are at most 20 an hour, and retired starters don't post."""
    from app import content, db, llm
    from app.config import settings

    signup(client)
    wait_for(lambda: len(content._first_post_tries) >= 7)  # the featured starters' first posts at start-up
    asked = []

    async def unusable(persona, topics, recent):
        asked.append(persona.name)
        return None  # e.g. cut short or refused

    monkeypatch.setattr(llm, "generate_post", unusable)
    amahle = by_slug(client, "amahle-ndlovu")
    for _ in range(3):
        client.delete(f"/api/companions/{amahle['id']}/follow")
        client.post(f"/api/companions/{amahle['id']}/follow")
    time.sleep(0.2)
    assert asked == ["Amahlé Ndlovù"]  # once, not once per follow

    others = [c for c in starters(client) if not c["following"] and c["id"] != amahle["id"]][:30]
    for c in others:
        client.post(f"/api/companions/{c['id']}/follow")
    time.sleep(0.3)
    assert len(asked) + 7 == content.FIRST_POSTS_PER_HOUR  # 7 featured, then 13 more this hour

    content._first_post_tries.clear()
    with db.connect() as conn:
        conn.execute("UPDATE companions SET retired = 1 WHERE id = ?", (amahle["id"],))
    assert asyncio.run(content.first_post(content.companions.get(amahle["id"]))) is None
    assert asked.count("Amahlé Ndlovù") == 1


def test_scheduled_posts_go_to_followed_starters_a_few_at_a_time(client):
    from app import content, db

    signup(client)

    def featured_have_posts():
        with db.connect() as conn:
            return not conn.execute("SELECT 1 FROM companions c WHERE featured = 1 AND NOT EXISTS "
                                    "(SELECT 1 FROM posts p WHERE p.companion_id = c.id)").fetchone()

    wait_for(featured_have_posts)  # the start-up posts are done
    with db.connect() as conn:
        followed = {r[0] for r in conn.execute("SELECT DISTINCT companion_id FROM follows")}
        before = conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0]
    assert asyncio.run(content.generate_for_starters(limit=3)) == 3
    with db.connect() as conn:
        new = [r[0] for r in conn.execute("SELECT companion_id FROM posts ORDER BY id DESC LIMIT 3")]
        assert conn.execute("SELECT COUNT(*) FROM posts").fetchone()[0] == before + 3
    assert set(new) <= followed
