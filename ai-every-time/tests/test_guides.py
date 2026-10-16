"""The education guides from aieverytime.org: loaded as posts by the companions who wrote them, kept up to date
when the file changes, and browsed in the feed, the library and on a page of their own."""

import gzip
import json
import unicodedata

import pytest

from tests.test_app import signup


def two_starters():
    """Two starters from different regions that aren't featured, so they haven't posted on their own."""
    from app.starter_companions import STARTERS

    plain = [s for s in STARTERS.values() if not s.featured and not s.real_person]
    a = plain[0]
    b = next(s for s in plain if s.region != a.region)
    return a, b


def guide(gid, slug, published, *, kind="tutorial", level="advanced", topic="AI Development", title=None,
          summary="A summary.", body=None):
    title = title or f"Guide {gid}"
    return {
        "id": gid, "companion": slug, "published": published, "kind": kind, "level": level, "topic": topic,
        "title": title, "summary": summary, "minutes": 12,
        "body": body or f"## Part One\n\nText of guide {gid}.\n\n## Part Two\n\n- A point\n",
        "image": f"https://cdn.example/generated/{gid}.png",
    }


def write(path, guides):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"since": "2026-09-01", "source": "test.csv", "guides": guides}, f)


def load(monkeypatch, path, images=None):
    from app import guides
    from app.config import settings

    monkeypatch.setattr(settings, "guides", str(path))
    monkeypatch.setattr(guides, "IMAGES", images or path.parent / "no-corrected-graphics")
    return guides.seed()


def companion_id(client, slug) -> int:
    return next(c["id"] for c in client.get("/api/companions").json() if c["slug"] == slug)


def titles(client, url="/api/guides", **params) -> list[str]:
    r = client.get(url, params=params)
    assert r.status_code == 200, r.text
    return [g["title"] for g in r.json()]


def test_guides_become_posts_by_the_companions_who_wrote_them(client, tmp_path, monkeypatch, caplog):
    a, b = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [guide("11", a.slug, "2026-09-02T08:00:00Z"),
                 guide("12", b.slug, "2026-10-01T09:30:00Z", kind="cheat_sheet", level="beginner"),
                 guide("13", "not-a-companion", "2026-09-03T00:00:00Z")])
    assert load(monkeypatch, path) == 2
    assert "not-a-companion" in caplog.text  # skipped, and said so

    signup(client)
    listed = client.get("/api/guides").json()
    assert [g["title"] for g in listed] == ["Guide 12", "Guide 11"]  # newest first
    newest = listed[0]
    # Dated when the site published it, written as the app writes times, so it sorts with other posts.
    assert newest["created_at"] == "2026-10-01T09:30:00+00:00"
    assert (newest["kind"], newest["level"], newest["minutes"], newest["summary"]) == ("cheat_sheet", "beginner", 12, "A summary.")
    assert newest["graphic"] == "https://cdn.example/generated/12.png"  # the site's copy: none corrected here
    assert newest["body"] == ""  # lists leave the long text out; the guide's own page has it
    whole = client.get(f"/api/posts/{newest['id']}").json()
    assert whole["body"].startswith("## Part One") and whole["companion_id"] == companion_id(client, b.slug)
    assert client.get("/api/posts/999999").status_code == 404


def test_loading_again_updates_guides_in_place(client, tmp_path, monkeypatch):
    a, b = two_starters()
    path = tmp_path / "guides.json.gz"
    first, second = guide("21", a.slug, "2026-09-05T10:00:00Z"), guide("22", b.slug, "2026-09-06T10:00:00Z")
    write(path, [first, second])
    assert load(monkeypatch, path) == 2
    signup(client)
    by_title = {g["title"]: g for g in client.get("/api/guides").json()}
    kept_id, gone_id = by_title["Guide 21"]["id"], by_title["Guide 22"]["id"]
    assert client.post(f"/api/posts/{kept_id}/engage", json={"kind": "like"}).status_code == 204

    assert load(monkeypatch, path) is None  # the same file isn't read again

    write(path, [first | {"title": "Guide 21, revised"}, guide("23", b.slug, "2026-09-07T10:00:00Z")])
    assert load(monkeypatch, path) == 2
    listed = client.get("/api/guides").json()
    assert [g["title"] for g in listed] == ["Guide 23", "Guide 21, revised"]
    revised = next(g for g in listed if g["title"] == "Guide 21, revised")
    assert (revised["id"], revised["likes"], revised["liked"]) == (kept_id, 1, True)  # the same post, likes and all
    assert client.get(f"/api/posts/{gone_id}").status_code == 404  # no longer in the file


def test_a_guide_shows_its_corrected_graphic_once_there_is_one(client, tmp_path, monkeypatch):
    a, b = two_starters()
    path, images = tmp_path / "guides.json.gz", tmp_path / "graphics"
    images.mkdir()
    (images / "61.webp").write_bytes(b"RIFF")
    write(path, [guide("61", a.slug, "2026-09-02T08:00:00Z"), guide("62", b.slug, "2026-09-03T08:00:00Z"),
                 guide("63", b.slug, "2026-09-04T08:00:00Z") | {"image": None}])
    assert load(monkeypatch, path, images) == 3
    signup(client)
    graphics = {g["title"]: g["graphic"] for g in client.get("/api/guides").json()}
    assert graphics == {"Guide 61": "/static/guides/61.webp",  # corrected: served by the app
                        "Guide 62": "https://cdn.example/generated/62.png",  # not yet: the site's copy
                        "Guide 63": None}  # the site gave it none
    # A graphic corrected later is picked up the next time the app starts, though the guides file is the same.
    (images / "62.webp").write_bytes(b"RIFF")
    assert load(monkeypatch, path, images) == 3
    assert client.get("/api/guides").json()[1]["graphic"] == "/static/guides/62.webp"
    assert load(monkeypatch, path, images) is None  # and nothing to do when nothing changed


def test_guides_dont_arrive_as_new_posts(client, tmp_path, monkeypatch):
    from app import db
    from app.starter_companions import STARTERS

    signup(client)  # following the featured starters
    featured = next(s for s in STARTERS.values() if s.featured and not s.real_person)
    path = tmp_path / "guides.json.gz"
    write(path, [guide(str(30 + i), featured.slug, f"2026-09-{10 + i}T10:00:00Z") for i in range(5)])
    assert load(monkeypatch, path) == 5
    assert "new_posts" not in [r["kind"] for r in client.get("/api/reminders").json()]
    # A post the companion writes afterwards is news.
    with db.connect() as conn:
        conn.execute("INSERT INTO posts (companion_id, title, body, topic, created_at) VALUES (?, 'Fresh', 'Hi', 'AI', ?)",
                     (companion_id(client, featured.slug), db.now()))
    new = [r for r in client.get("/api/reminders").json() if r["kind"] == "new_posts"]
    assert new and new[0]["count"] == 1


def test_the_library_narrows_down_by_region_kind_level_topic_and_words(client, tmp_path, monkeypatch):
    a, b = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [
        guide("41", a.slug, "2026-09-01T10:00:00Z", title="Vector Search with DuckDB", topic="AI Development"),
        guide("42", b.slug, "2026-09-02T10:00:00Z", title="Prompts for Teachers", kind="cheat_sheet", level="beginner",
              topic="Prompt Engineering"),
        guide("43", a.slug, "2026-09-03T10:00:00Z", title="Pricing with AI", kind="cheat_sheet", topic="Business & AI"),
    ])
    load(monkeypatch, path)
    signup(client)
    assert titles(client) == ["Pricing with AI", "Prompts for Teachers", "Vector Search with DuckDB"]
    assert titles(client, region=a.region) == ["Pricing with AI", "Vector Search with DuckDB"]
    assert titles(client, kind="cheat_sheet") == ["Pricing with AI", "Prompts for Teachers"]
    assert titles(client, level="beginner") == ["Prompts for Teachers"]
    assert titles(client, topic="Business & AI") == ["Pricing with AI"]
    assert titles(client, companion=companion_id(client, b.slug)) == ["Prompts for Teachers"]
    assert titles(client, q="duckdb vector") == ["Vector Search with DuckDB"]  # every word, any case
    assert titles(client, q="100%") == []  # wildcards are just characters
    # A companion's name finds their guides with or without its accents.
    plain_name = "".join(ch for ch in unicodedata.normalize("NFKD", b.persona.name) if not unicodedata.combining(ch))
    assert titles(client, q=plain_name.split()[0].lower()) == ["Prompts for Teachers"]

    topics = client.get("/api/guides/topics").json()
    assert {t["topic"]: t["guides"] for t in topics} == {"AI Development": 1, "Prompt Engineering": 1, "Business & AI": 1}

    assert client.get("/api/guides", params={"kind": "essay"}).status_code == 422
    assert client.get("/api/guides", params={"before": "not a cursor"}).status_code == 422


def test_lists_come_newest_first_a_page_at_a_time(client, tmp_path, monkeypatch):
    a, _ = two_starters()
    path = tmp_path / "guides.json.gz"
    # Out of order in the file: the dates decide.
    days = [(i * 7) % 25 + 1 for i in range(25)]
    write(path, [guide(str(100 + d), a.slug, f"2026-09-{d:02d}T10:00:00Z") for d in days])
    load(monkeypatch, path)
    signup(client)
    cid = companion_id(client, a.slug)
    client.post(f"/api/companions/{cid}/follow")

    page = client.get(f"/api/companions/{cid}/posts").json()
    assert len(page) == 20
    assert [p["created_at"][:10] for p in page] == [f"2026-09-{d:02d}" for d in range(25, 5, -1)]
    rest = client.get(f"/api/companions/{cid}/posts", params={"before": f"{page[-1]['created_at']}|{page[-1]['id']}"}).json()
    assert [p["created_at"][:10] for p in rest] == [f"2026-09-{d:02d}" for d in range(5, 0, -1)]
    # The feed pages the same way, and the library too.
    feed = client.get("/api/feed").json()
    later = client.get("/api/feed", params={"before": f"{feed[-1]['created_at']}|{feed[-1]['id']}"}).json()
    assert not {p["id"] for p in feed} & {p["id"] for p in later}
    first = client.get("/api/guides").json()
    assert len(first) == 24
    assert len(client.get("/api/guides", params={"before": f"{first[-1]['created_at']}|{first[-1]['id']}"}).json()) == 1


def test_asking_a_companion_about_a_guide(client, tmp_path, monkeypatch):
    a, _ = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [guide("51", a.slug, "2026-09-01T10:00:00Z", body="## Setup\n\nInstall the thing.\n")])
    load(monkeypatch, path)
    signup(client)
    gid = client.get("/api/guides").json()[0]["id"]
    cid = companion_id(client, a.slug)
    r = client.post(f"/api/companions/{cid}/chat", json={"message": "What do I install?", "post_id": gid})
    assert r.status_code == 200 and r.text
    # Study tools work from the whole guide.
    r = client.post("/api/summaries", json={"source_type": "post", "source_ref": str(gid)})
    assert r.status_code in (200, 201), r.text


@pytest.mark.parametrize("value", ["", "0"])
def test_where_the_guides_come_from(monkeypatch, value):
    from app import guides
    from app.config import settings

    monkeypatch.setattr(settings, "guides", value)
    assert guides.path() == (None if value == "0" else guides.GUIDES_PATH)
