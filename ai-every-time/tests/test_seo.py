"""What search engines and AI assistants read (app/seo.py): robots.txt, the sitemap, llms.txt, each public page's
structured data and its content as plain HTML, and one address for each page."""

import json
import re
from xml.etree import ElementTree

from app import seo
from app.config import settings
from tests.test_app import persona, signup
from tests.test_pages import LONG, guides  # noqa: F401  (a fixture)

NS = {"s": "http://www.sitemaps.org/schemas/sitemap/0.9"}


def structured(page) -> list[dict]:
    found = re.search(r'<script type="application/ld\+json">(.*?)</script>', page.text, re.S)
    return json.loads(found[1])["@graph"] if found else []


def body(page) -> str:
    return re.search(r'<div id="app" class="app">(.*?)</div>\s*<div id="toast"', page.text, re.S)[1]


def test_robots_welcomes_search_engines_and_ai_assistants(client):
    r = client.get("/robots.txt")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
    for bot in ("*", "Googlebot", "Bingbot", "GPTBot", "OAI-SearchBot", "ClaudeBot", "Google-Extended", "PerplexityBot"):
        assert f"User-agent: {bot}\nAllow: /" in r.text
    # Nothing blocked: a blocked page Google finds a link to is "indexed, though blocked by robots.txt".
    assert "Disallow" not in r.text
    assert f"Sitemap: {settings.public_url}/sitemap.xml" in r.text


def test_every_page_in_the_sitemap_can_be_indexed(client, guides):
    first, second, a, b = guides
    # A learner's own companion, not published, and its post: neither belongs in search results.
    signup(client, email="grace@example.com", name="Grace")
    mine = client.post("/api/companions", json={"persona": persona(name="Private Pat")})
    assert mine.status_code == 201, mine.text
    mine = mine.json()
    client.post("/api/auth/logout")

    r = client.get("/sitemap.xml")
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/xml")
    locs = [e.text for e in ElementTree.fromstring(r.text).findall("s:url/s:loc", NS)]
    paths = [loc.removeprefix(settings.public_url) for loc in locs]
    assert len(set(paths)) == len(paths)
    for must in ("/", "/courses", "/guides", "/companions", "/pricing", "/courses/java-programming-intellij",
                 "/tracks/network-engineer", f"/guides/{first['slug']}", f"/guides/{second['slug']}", f"/companions/{a.slug}"):
        assert must in paths, must
    assert f"/companions/{mine['slug']}" not in paths and "/feed" not in paths and "/signin" not in paths

    # Each answers 200 itself (no redirect), isn't marked noindex, names itself as its canonical address, and says
    # what it is in plain HTML with valid structured data: so none is left out of search results.
    for path, loc in zip(paths, locs):
        page = client.get(path, follow_redirects=False)
        assert page.status_code == 200, path
        robots = re.search(r'<meta name="robots" content="([^"]*)">', page.text)[1]
        assert "noindex" not in robots and "max-image-preview:large" in robots, path
        assert re.search(r'<link rel="canonical" href="([^"]+)">', page.text)[1] == loc, path
        assert "<h1>" in body(page) and structured(page), path


def test_a_course_says_what_it_is_to_search_engines(client):
    page = client.get("/courses/java-programming-intellij")
    html = body(page)
    assert "<h1>Java Programming with IntelliJ IDEA</h1>" in html
    assert "Object-oriented programming" in html and "Model a smart home" in html  # a module, and its lab
    things = {t["@type"]: t for t in structured(page)}
    assert {"Organization", "WebSite", "BreadcrumbList", "Course"} <= things.keys()
    course = things["Course"]
    assert course["name"] == "Java Programming with IntelliJ IDEA" and course["provider"] == {"@id": f"{settings.public_url}/#organization"}
    assert course["offers"][0]["category"] == "Partially Free"  # the first module of every course is free
    assert course["hasCourseInstance"][0]["courseMode"] == "Online"
    assert [i["name"] for i in course["hasCourseInstance"][0]["instructor"]][0].startswith("Agnieszka")
    assert [c["name"] for c in things["BreadcrumbList"]["itemListElement"]] == ["Home", "Courses", course["name"]]


def test_crawlers_read_what_visitors_read_of_a_guide(client, guides):
    first, *_ = guides
    page = client.get(f"/guides/{first['slug']}")
    html = body(page)
    # The start of it, as a visitor who isn't signed in sees, never more.
    assert "<h1>What&#x27;s an LLM? A 5-minute guide</h1>" in html and "<h3>Part 1</h3>" in html  # the guide's own headings go under the page's title
    assert "Part 29" not in html and "Sign up free to read the whole guide" in html
    article = next(t for t in structured(page) if t["@type"] == "Article")
    assert article["headline"] == "What's an LLM? A 5-minute guide" and article["author"]["name"]
    assert article["description"] == "How large language models work, in five minutes."


def test_pages_for_learners_have_no_content_for_crawlers(client):
    for address in ("/feed", "/signin", "/progress"):
        page = client.get(address)
        assert 'content="noindex"' in page.text and '<div id="app" class="app"></div>' in page.text
        assert not structured(page)


def test_llms_txt_sums_up_the_site_for_ai_assistants(client, guides):
    first, *_ = guides
    short, full = client.get("/llms.txt"), client.get("/llms-full.txt")
    assert short.status_code == full.status_code == 200 and short.text.startswith("# AI Every Time\n\n> ")
    for text in (short.text, full.text):
        assert f"[Java Programming with IntelliJ IDEA]({settings.public_url}/courses/java-programming-intellij)" in text
        assert "Network Engineer Professional Certificate" in text
    assert f"/guides/{first['slug']}" not in short.text and "/llms-full.txt" in short.text
    assert f"[What's an LLM? A 5-minute guide]({settings.public_url}/guides/{first['slug']})" in full.text


def test_the_apps_data_stays_out_of_search_results(client):
    assert client.get("/api/courses").headers["x-robots-tag"] == "noindex"
    assert "x-robots-tag" not in client.get("/courses").headers


def test_pages_at_renders_address_move_to_the_sites_own(client, monkeypatch):
    monkeypatch.setattr(settings, "public_url", "https://www.aieverytime.org")
    render = {"host": "ai-every-time.onrender.com"}
    r = client.get("/courses/java-programming-intellij?ref=x", headers=render, follow_redirects=False)
    assert r.status_code == 301 and r.headers["location"] == "https://www.aieverytime.org/courses/java-programming-intellij?ref=x"
    assert client.get("/sitemap.xml", headers=render, follow_redirects=False).status_code == 301
    # The app's data and the health check stay put, for the iPhone app and Render; and so does anything but reading.
    assert client.get("/api/courses", headers=render, follow_redirects=False).status_code == 200
    assert client.get("/healthz", headers=render, follow_redirects=False).status_code in (200, 503)
    assert client.post("/api/auth/logout", headers=render, follow_redirects=False).status_code != 301
    # At the site's own address, and while the site has no address of its own, nothing moves.
    assert client.get("/courses", follow_redirects=False).status_code == 200
    monkeypatch.setattr(settings, "public_url", "https://ai-every-time.onrender.com")
    assert client.get("/courses", headers=render, follow_redirects=False).status_code == 200


def test_guides_markdown_is_made_safe():
    out = seo.markdown("## Hi\n\n<script>alert(1)</script> **bold** [ok](https://example.com) [bad](javascript:alert(1))\n\n"
                       "- one\n- two\n\n```\n<b>code</b>\n```\n\n| a | b |\n|---|---|\n| 1 | 2 |")
    assert "<script" not in out and "javascript:" not in out
    assert "<h3>Hi</h3>" in out and "<strong>bold</strong>" in out and "<ul><li>one</li><li>two</li></ul>" in out
    assert '<a href="https://example.com" rel="nofollow noopener">ok</a>' in out and "&lt;b&gt;code&lt;/b&gt;" in out
    assert "<td>1</td>" in out and "---" not in out
