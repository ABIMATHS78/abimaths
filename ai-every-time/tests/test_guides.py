"""The education guides from aieverytime.org: loaded as posts by the companions who wrote them, kept up to date
when the file changes, and browsed in the feed, the library and on a page of their own."""

import gzip
import json
import unicodedata
from pathlib import Path

import pytest

from tests.test_app import signup
from tests.test_billing import paywall, pays_for  # noqa: F401  (paywall is a fixture)


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


def load(monkeypatch, path, images=None, written=None):
    """Loads the guides in the file at `path`, with only the corrected graphics in `images` and the guides written
    here in `written` (none unless given)."""
    from app import guides
    from app.config import settings

    monkeypatch.setattr(settings, "guides", str(path))
    monkeypatch.setattr(guides, "IMAGES", images or path.parent / "no-corrected-graphics")
    monkeypatch.setattr(guides, "WRITTEN", written or path.parent / "no-written-guides")
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


# ------------------------------------------------------------------ guides written here, and their downloads

APP_DIR = Path(__file__).resolve().parent.parent / "app"
KIT = "claude-skills-starter-kit"  # Mojeed Abisiga's guide to Claude skills, with skills to download


def load_the_kit(client, tmp_path, monkeypatch) -> tuple[int, int]:
    """Loads the guides written here, the starter kit among them, beside one from the site, and signs a learner up:
    the kit's id and the other guide's."""
    from app import guides

    a, _ = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [guide("61", a.slug, "2026-09-01T10:00:00Z")])
    load(monkeypatch, path, images=guides.IMAGES, written=APP_DIR / "guide_data")
    signup(client)
    listed = {g["title"]: g["id"] for g in client.get("/api/guides").json()}
    kit = next(i for t, i in listed.items() if t.startswith("The Claude Skills Starter Kit"))
    return kit, listed["Guide 61"]


def written_guide(folder, name="a-written-guide", **meta):
    """A small guide written here, in folder/name, with a file to download."""
    d = folder / name
    (d / "downloads").mkdir(parents=True)
    (d / "downloads" / "notes.txt").write_text("Some notes.")
    (d / "guide.md").write_text("## Start\n\nThe text.\n")
    (d / "guide.json").write_text(json.dumps({
        "companion": "nobody", "title": "A written guide", "summary": "A summary.", "kind": "cheat_sheet",
        "level": "intermediate", "topic": "Prompt Engineering", "published": "2026-10-01T08:00:00Z",
        "downloads": [{"file": "notes.txt", "title": "Notes", "about": "Some notes."}]} | meta))
    return d


def test_guides_written_here_load_with_the_others(client, tmp_path, monkeypatch):
    kit, other = load_the_kit(client, tmp_path, monkeypatch)
    post = client.get(f"/api/posts/{kit}").json()
    assert post["companion_id"] == companion_id(client, "mojeed-abisiga")
    assert (post["kind"], post["level"], post["topic"]) == ("tutorial", "beginner", "AI Tools & Tutorials")
    assert post["created_at"] == "2026-10-28T09:00:00+00:00" and post["minutes"] >= 10
    assert post["graphic"] == f"/static/guides/{KIT}.webp" and client.get(post["graphic"]).status_code == 200
    assert post["body"].startswith("## Teach Claude a job once")
    # What it offers to download, in its own order, each with its size: listed on its page.
    names = [d["name"] for d in post["downloads"]]
    assert names[0] == f"{KIT}.zip" and "clarify-first.zip" in names and names[-1] == "claude-skills-cheat-sheet.pdf"
    assert len(names) == 13 and all(d["title"] and d["about"] and d["size"] > 0 for d in post["downloads"])
    # The guides from the site offer none.
    assert "downloads" not in client.get(f"/api/posts/{other}").json()
    assert client.get(f"/api/posts/{other}/downloads/{KIT}.zip").status_code == 404


def test_a_guide_sends_its_downloads(client, tmp_path, monkeypatch):
    import io
    import stat
    import zipfile

    kit, _ = load_the_kit(client, tmp_path, monkeypatch)
    url = f"/api/posts/{kit}/downloads"
    skills = APP_DIR / "guide_data" / KIT / "downloads" / "skills"
    # A skill is a zip with its folder at the top, the way Claude takes skills.
    r = client.get(f"{url}/clarify-first.zip")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    assert r.headers["content-disposition"] == 'attachment; filename="clarify-first.zip"'
    z = zipfile.ZipFile(io.BytesIO(r.content))
    assert z.namelist() == ["clarify-first/SKILL.md"]
    assert z.read("clarify-first/SKILL.md") == (skills / "clarify-first" / "SKILL.md").read_bytes()
    # All ten: each skill's folder at the top, ready to unzip into ~/.claude/skills/, with scripts still runnable.
    z = zipfile.ZipFile(io.BytesIO(client.get(f"{url}/{KIT}.zip").content))
    tops = {n.split("/")[0] for n in z.namelist()}
    assert tops == {p.name for p in skills.iterdir() if p.is_dir()} and len(tops) == 10
    assert all(f"{t}/SKILL.md" in z.namelist() for t in tops)
    assert z.getinfo("plain-technical-english/scripts/check_ste.py").external_attr >> 16 & stat.S_IXUSR
    assert not any("__pycache__" in n or n.endswith(".pyc") for n in z.namelist())
    # A file goes as it is.
    r = client.get(f"{url}/SKILL-template.md")
    assert r.headers["content-type"].startswith("text/markdown") and r.text.startswith("---\nname: your-skill-name")
    r = client.get(f"{url}/claude-skills-cheat-sheet.pdf")
    assert r.headers["content-type"] == "application/pdf" and r.content.startswith(b"%PDF")
    # Only what the guide offers: not the guide's own files, nor what's inside a zip.
    for name in ("guide.json", "guide.md", "SKILL.md", "skills.zip", "..%2Fguide.json"):
        assert client.get(f"{url}/{name}").status_code == 404, name


def test_downloads_come_with_reading_the_guide_in_full(client, paywall, tmp_path, monkeypatch):  # noqa: F811
    from app.config import settings

    monkeypatch.setattr(settings, "free_guides", 1)
    kit, other = load_the_kit(client, tmp_path, monkeypatch)
    client.get(f"/api/posts/{other}")  # this month's free guide goes on another one
    locked = client.get(f"/api/posts/{kit}").json()
    assert locked["locked"] is True and len(locked["downloads"]) == 13  # listed, to show what reading it brings
    r = client.get(f"/api/posts/{kit}/downloads/clarify-first.zip")
    assert r.status_code == 402 and r.headers["X-Plan-Needed"] == "digital"
    pays_for(client, "digital")
    assert client.get(f"/api/posts/{kit}/downloads/clarify-first.zip").status_code == 200


def test_a_link_on_another_site_cant_spend_a_free_guide_on_a_download(client, paywall, tmp_path, monkeypatch):  # noqa: F811
    from app.config import settings

    monkeypatch.setattr(settings, "free_guides", 1)
    a, _ = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [])
    written_guide(tmp_path / "written", companion=a.slug)
    load(monkeypatch, path, written=tmp_path / "written")
    signup(client)
    gid = client.get("/api/guides").json()[0]["id"]
    away = client.get(f"/api/posts/{gid}/downloads/notes.txt", headers={"Sec-Fetch-Site": "cross-site"})
    assert away.status_code == 402
    assert client.get("/api/billing").json()["guides"] == {"limit": 1, "left": 1}
    r = client.get(f"/api/posts/{gid}/downloads/notes.txt")  # from the app itself, it opens the guide
    assert r.status_code == 200 and r.text == "Some notes." and r.headers["content-type"].startswith("text/plain")
    assert client.get("/api/billing").json()["guides"] == {"limit": 1, "left": 0}


def test_changing_a_written_guide_updates_it_in_place(client, tmp_path, monkeypatch):
    from app import guides

    a, _ = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [])
    folder = written_guide(tmp_path / "written", companion=a.slug)
    assert load(monkeypatch, path, written=tmp_path / "written") == 1
    signup(client)
    gid = client.get("/api/guides").json()[0]["id"]
    assert guides.seed() is None  # nothing changed
    (folder / "guide.md").write_text("## New text\n\nChanged.\n")
    assert guides.seed() == 1
    assert client.get(f"/api/posts/{gid}").json()["body"] == "## New text\n\nChanged."  # the same post, new text


@pytest.mark.parametrize("change, reason", [
    ({"kind": "essay"}, "kind must be one of"),
    ({"published": "2026-10-01T08:00:00"}, "needs a time zone"),
    ({"downloads": [{"file": "missing.txt", "title": "Missing", "about": "Isn't there."}]}, "nothing at downloads/missing.txt"),
    ({"downloads": [{"file": "../guide.json", "title": "Escape", "about": "Above downloads/."}]}, "should match pattern"),
    ({"downloads": [{"file": "notes.txt", "title": "A", "about": "A."}, {"file": "notes.txt", "title": "B", "about": "B."}]},
     "two downloads have the same name"),
    ({"surprise": True}, "Extra inputs are not permitted"),
])
def test_a_written_guide_that_doesnt_fit_says_why(tmp_path, change, reason):
    from app import guides

    written_guide(tmp_path, **change)
    with pytest.raises(ValueError) as e:
        guides.written(tmp_path)
    assert "guide_data/a-written-guide" in str(e.value) and reason in str(e.value)


def test_a_written_guide_is_named_with_letters(tmp_path):
    from app import guides

    written_guide(tmp_path, name="12345")  # the site's guides have numbers
    with pytest.raises(ValueError, match="name the folder"):
        guides.written(tmp_path)


def test_every_skill_in_the_kit_follows_the_rules(tmp_path):
    import importlib.util
    import shutil

    skills = APP_DIR / "guide_data" / KIT / "downloads" / "skills"
    spec = importlib.util.spec_from_file_location("check_skill", skills / "skill-maker" / "scripts" / "check_skill.py")
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    folders = sorted(p for p in skills.iterdir() if p.is_dir())
    assert len(folders) == 10
    for f in folders:
        assert checker.check(f) == [], f.name
    # And the checker that skill-maker brings catches the usual mistakes.
    bad = tmp_path / "Weekly_Report"
    shutil.copytree(skills / "clarify-first", bad)
    (bad / "SKILL.md").write_text("---\nname: claude-helper\ndescription: Does this: and that " + "x" * 200 + "\n---\n\nSee "
                                  "`scripts/missing.py`.\n")
    found = " | ".join(checker.check(bad))
    for problem in ('must not contain "claude"', "make them match", "colon followed by a space", "the Claude apps take 200",
                    "scripts/missing.py"):
        assert problem in found, problem
