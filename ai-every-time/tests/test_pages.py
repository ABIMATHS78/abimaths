"""Every page at an address in words (app/paths.py, app/pages.py): guides by their title, companions by their name,
courses by theirs. The server answers each with the app and the page's own title, description and picture for link
previews; addresses with an id, from before, move to the words; and what's shared opens for visitors who aren't
signed in, as the start of it, with a way to sign up and read on."""

import re

import pytest

from tests.test_app import persona, signup
from tests.test_billing import paywall  # noqa: F401  (a fixture)
from tests.test_guides import companion_id, guide, load, two_starters, write

LONG = "\n\n".join(f"## Part {n}\n\nThe text of part {n}, which goes on for a while to make the guide long." * 3
                   for n in range(1, 30))


def head(page) -> dict[str, str]:
    """The page's title, and its link-preview tags by name."""
    tags = dict(re.findall(r'<meta (?:property|name)="([^"]+)" content="([^"]*)">', page.text))
    tags["title"] = re.search(r"<title>(.*?)</title>", page.text).group(1)
    tags["canonical"] = (re.search(r'<link rel="canonical" href="([^"]+)">', page.text) or [None, None])[1]
    return tags


@pytest.fixture()
def guides(client, tmp_path, monkeypatch):
    """Two guides by two starters: a long one, and one with the same title as it."""
    a, b = two_starters()
    path = tmp_path / "guides.json.gz"
    write(path, [guide("11", a.slug, "2026-09-02T08:00:00Z", title="What's an LLM? A 5-minute guide", body=LONG,
                       summary="How large language models work, in five minutes."),
                 guide("12", b.slug, "2026-10-01T09:30:00Z", title="What's an LLM? A 5-minute guide")])
    load(monkeypatch, path)
    signup(client)
    found = {g["id"]: g for g in client.get("/api/guides").json()}
    client.post("/api/auth/logout")
    first, second = sorted(found.values(), key=lambda g: g["id"])
    return first, second, a, b


def test_a_guide_is_known_by_its_title(guides):
    from app.starter_companions import STARTERS

    first, second, a, b = guides
    # The older of two with the same title keeps the plain words; the newer has its author's name after them.
    author = STARTERS[b.slug].persona.name.split()[0]
    assert first["slug"] == "whats-an-llm-a-5-minute-guide"
    assert second["slug"] == f"whats-an-llm-a-5-minute-guide-by-{paths_slug(author)}"


def paths_slug(text):
    from app import paths

    return paths.slugify(text)


def test_words_for_an_address():
    from app import paths

    assert paths.slugify("Ünïcödé & Ácçents: Claude’s skills") == "unicode-accents-claudes-skills"
    assert paths.slugify("2024") == "post-2024" and paths.slugify("—") == "post"  # never only digits, never empty
    assert len(paths.slugify("word " * 40)) <= 80 and not paths.slugify("word " * 40).endswith("-")


def test_posts_from_before_slugs_get_theirs_when_the_site_starts(client):
    from app import db, paths

    with db.connect() as conn:
        cid, name = conn.execute("SELECT id, json_extract(persona, '$.name') FROM companions WHERE is_starter = 1 "
                                 "LIMIT 1").fetchone()
        for _ in range(3):  # three posts with the same title, from before posts had slugs
            conn.execute("INSERT INTO posts (companion_id, title, body, topic, created_at) "
                         "VALUES (?, 'A post from before', 'Hi.', 'AI', ?)", (cid, db.now()))
        assert paths.fill_post_slugs(conn) == 3
        slugs = [r[0] for r in conn.execute("SELECT slug FROM posts WHERE title = 'A post from before' ORDER BY id")]
    by = paths.slugify(name.split()[0])
    assert slugs == ["a-post-from-before", f"a-post-from-before-by-{by}", f"a-post-from-before-by-{by}-2"]


def test_a_companion_writes_its_posts_with_their_own_address(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    post = client.post(f"/api/companions/{mine['id']}/posts/generate").json()
    assert post["slug"] and client.get(f"/api/posts/{post['slug']}").json()["id"] == post["id"]


def test_a_shared_guide_shows_what_it_is_in_its_link_preview(client, guides):
    from app.config import settings

    first, *_ = guides
    page = client.get(f"/guides/{first['slug']}")
    assert page.status_code == 200 and '<script src="/static/app.js"></script>' in page.text
    tags = head(page)
    assert tags["title"] == "What&#x27;s an LLM? A 5-minute guide · AI Every Time"
    assert tags["og:title"] == "What&#x27;s an LLM? A 5-minute guide" and tags["og:type"] == "article"
    assert tags["description"] == tags["og:description"] == "How large language models work, in five minutes."
    assert tags["og:image"] == "https://cdn.example/generated/11.png" and tags["twitter:card"] == "summary_large_image"
    assert tags["canonical"] == tags["og:url"] == f"{settings.public_url}/guides/{first['slug']}"
    assert "robots" not in tags  # for search engines too


def test_links_from_before_move_to_the_words(client, guides):
    first, *_ = guides
    for old in (f"/guides/{first['id']}", f"/posts/{first['slug']}", f"/posts/{first['id']}"):
        moved = client.get(old, follow_redirects=False)
        assert moved.status_code == 301 and moved.headers["location"] == f"/guides/{first['slug']}", old
    moved = client.get(f"/guides/{first['id']}?utm_source=x", follow_redirects=False)
    assert moved.headers["location"] == f"/guides/{first['slug']}?utm_source=x"  # the link's tags come along
    assert client.get("/guides/", follow_redirects=False).headers["location"] == "/guides"


@pytest.mark.parametrize("address", ["/guides/nothing-here", "/guides/99999", "/nope", "/posts", "/tracks",
                                     "/courses/nope", "/courses/python-programming-pycharm/junk",
                                     "/companions/nobody-here", "/certificates/NOPE", "/quizzes/abc"])
def test_an_address_thats_not_a_page_says_so(client, address):
    page = client.get(address)
    assert page.status_code == 404 and head(page)["robots"] == "noindex"
    assert '<script src="/static/app.js"></script>' in page.text  # the app says so, in the site's frame
    assert client.get("/api/nothing-here").json() == {"detail": "Not Found"}  # the API's own stay JSON


def test_pages_for_learners_stay_out_of_search_engines(client):
    for address in ("/feed", "/profile", "/flashcards", "/flashcards/3", "/lessons/what-is-ai", "/staff/reports"):
        page = client.get(address)
        assert page.status_code == 200 and head(page)["robots"] == "noindex", address
    assert "robots" not in head(client.get("/courses"))


def test_a_visitor_reads_the_start_of_a_guide(client, guides):
    first, _, a, _ = guides
    for ref in (first["slug"], str(first["id"])):
        post = client.get(f"/api/posts/{ref}").json()
        assert post["id"] == first["id"] and post["locked"] and post["signed_out"]
        assert post["body"].startswith("## Part 1") and len(post["body"]) < len(LONG) // 2
        assert "guides" not in post and post["liked"] is False
    # Its author, and their other guides, for the rest of the page.
    c = client.get(f"/api/companions/{post['companion_id']}").json()
    assert c["slug"] == a.slug and c["owner_id"] is None and c["persona"]["custom_instructions"] == ""
    assert [g["id"] for g in client.get("/api/guides", params={"companion": a.slug}).json()] == [first["id"]]
    assert client.get("/api/guides/topics").status_code == 200
    # Liking it, or counting it as read, is for learners.
    assert client.post(f"/api/posts/{first['id']}/engage", json={"kind": "like"}).status_code == 401
    signup(client, email="grace@example.com", name="Grace")
    assert client.get(f"/api/posts/{first['slug']}").json()["body"] == LONG  # signed up, they read it all


def test_a_preview_doesnt_end_on_a_heading():
    from app import billing

    body = "Intro.\n\n" + "x" * 400 + "\n\n## What comes next\n\n" + "y" * 3000
    assert billing.preview(body) == "Intro.\n\n" + "x" * 400


def test_whats_private_stays_private(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    post = client.post(f"/api/companions/{mine['id']}/posts/generate").json()
    assert client.get(f"/companions/{mine['slug']}").status_code == 200  # its designer can open its page
    assert "og:description" not in head(client.get(f"/posts/{post['slug']}"))  # with nothing about it in the head
    client.post("/api/auth/logout")
    for address in (f"/api/companions/{mine['slug']}", f"/api/companions/{mine['id']}", f"/api/posts/{post['slug']}",
                    f"/api/companions/{mine['id']}/posts", f"/companions/{mine['slug']}", f"/posts/{post['slug']}"):
        assert client.get(address).status_code == 404, address
    assert mine["id"] not in [c["id"] for c in client.get("/api/companions").json()]


def test_a_published_companion_is_open_to_everyone(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona()}).json()
    client.post(f"/api/companions/{mine['id']}/publish")
    client.post("/api/auth/logout")
    seen = client.get(f"/api/companions/{mine['slug']}").json()
    assert seen["owner_id"] is None and seen["persona"]["custom_instructions"] == "" and seen["designer"] == "Ada"
    listed = {c["id"]: c for c in client.get("/api/companions").json()}
    assert mine["id"] in listed and all(c["owner_id"] is None for c in listed.values())
    assert all(c["persona"]["custom_instructions"] == "" for c in listed.values())
    assert all(p["body"] == "" for p in client.get(f"/api/companions/{mine['id']}/posts").json())  # titles only


def test_a_starter_companions_page_and_its_preview(client):
    from app.config import settings

    signup(client)
    starter = client.get("/api/companions").json()[0]
    client.post("/api/auth/logout")
    moved = client.get(f"/companions/{starter['id']}", follow_redirects=False)
    assert moved.status_code == 301 and moved.headers["location"] == f"/companions/{starter['slug']}"
    tags = head(client.get(f"/companions/{starter['slug']}"))
    assert tags["og:title"].startswith(starter["persona"]["name"]) and "AI Learning Companion" in tags["og:title"]
    assert tags["og:image"] == f"{settings.public_url}/api/companions/{starter['id']}/avatar"
    assert client.get(f"/api/companions/{starter['slug']}").json()["id"] == starter["id"]
    for address in (f"/companions/{starter['slug']}/edit", f"/companions/{starter['slug']}/customise", "/companions/new"):
        assert client.get(address).status_code == 200, address


def test_courses_and_tracks_are_open_to_everyone(client, paywall):  # noqa: F811
    from app import catalog

    course, track = catalog.data().courses[0], catalog.data().tracks[0]
    listed = client.get("/api/courses").json()
    assert listed["full_access"] is False and not any(c["enrolled"] for c in listed["courses"])
    page = client.get(f"/api/courses/{course.slug}").json()
    assert page["enrolled"] is False and page["access"]["full"] is False and page["my_rating"] is None
    assert page["syllabus"][1]["locked"] and not page["syllabus"][0]["locked"]  # as for anyone on the free plan
    assert client.get(f"/api/tracks/{track.slug}").json()["done"] == 0
    assert head(client.get(f"/courses/{course.slug}"))["og:title"] == course.title
    assert client.get(f"/courses/{course.slug}/module-1/lesson-1").status_code == 200
    assert head(client.get(f"/tracks/{track.slug}"))["og:title"] == f"{track.title} {track.kind}"
    assert client.post(f"/api/courses/{course.slug}/enroll").status_code == 401  # taking one is for learners


def test_the_head_is_the_pages_own_and_safe(client):
    signup(client)
    mine = client.post("/api/companions", json={"persona": persona(name='Zara "<b>"', tagline='A <script>alert(1)</script>')}).json()
    client.post(f"/api/companions/{mine['id']}/publish")
    page = client.get(f"/companions/{mine['slug']}")
    assert "<script>alert" not in page.text and "<b>" not in page.text
    assert head(page)["og:title"].startswith("Zara &quot;&lt;b&gt;&quot;")


@pytest.mark.parametrize("address", ["//evil.example/", "///evil.example/", "/%2F%2Fevil.example/", "/%5Cevil.example/",
                                     "/%5C%5Cevil.example/", "/%09/evil.example/"])
def test_a_trailing_slash_only_ever_moves_within_the_site(client, address):
    moved = client.get(f"http://testserver{address}", follow_redirects=False)
    # One slash, then something a browser can't read as another site's address (as //evil.example or /\evil.example).
    assert moved.status_code == 301 and re.fullmatch(r"/[^/\\\s]\S*", moved.headers["location"]), moved.headers["location"]
