"""Builds the sample site and checks it the way a reader, a search engine and the main site's own checks would."""

import json
import re
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from nle.build import Builder
from nle.config import load_site
from nle.guardrails import brand_mentions

ENGINE = Path(__file__).resolve().parent.parent
MAIN_SITE = ENGINE.parent / "mojeedabisiga.com"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("dist")
    report = Builder(drafts=False, out_root=out, og=False).build()
    return report, out


def test_build_is_clean_and_complete(built):
    report, out = built
    assert report.ok, report.problems
    for rel in ("index.html", "archive/index.html", "guides/index.html", "start/index.html", "library/index.html",
                "about/index.html", "subscribe/index.html", "feed.xml", "sitemap.xml", "search.json",
                "p/make-your-first-skill/index.html", "guides/claude-skills-from-zero/index.html", "static/css/site.css"):
        assert (report.out_dir / rel).exists(), rel
    assert (out / "email" / "make-your-first-skill.html").exists()


def test_every_internal_link_resolves(built):
    report, _ = built
    base = load_site().base_path
    for page in report.out_dir.rglob("*.html"):
        for target in re.findall(r'(?:href|src)="([^"#?]+)', page.read_text(encoding="utf-8")):
            if target.startswith(("http", "mailto:", "data:")):
                continue
            assert target.startswith(base), f"{page}: {target}"
            path = report.out_dir / target[len(base):]
            assert path.exists() or (path / "index.html").exists(), f"{page.relative_to(report.out_dir)} links to {target}"


def test_feeds_sitemap_search_and_structured_data(built):
    report, _ = built
    rss = ET.parse(report.out_dir / "feed.xml").getroot()
    assert len(rss.findall("./channel/item")) >= 4
    urls = [loc.text for loc in ET.parse(report.out_dir / "sitemap.xml").getroot().iter("{http://www.sitemaps.org/schemas/sitemap/0.9}loc")]
    assert all(u.startswith("https://mojeedabisiga.com/learn/") for u in urls) and any("check-a-skill" in u for u in urls)
    index = json.loads((report.out_dir / "search.json").read_text(encoding="utf-8"))
    assert {"issue:make-your-first-skill", "issue:ten-skills-worth-writing", "issue:check-a-skill-before-you-trust-it",
            "issue:skill-description", "guide:claude-skills-from-zero", "guide:skill-or-something-else"} <= {e["ref"] for e in index}
    page = (report.out_dir / "p/make-your-first-skill/index.html").read_text(encoding="utf-8")
    data = json.loads(re.search(r'<script type="application/ld\+json">(.*?)</script>', page, re.S).group(1))
    assert data["@type"] == "BlogPosting" and data["author"]["@id"] == "https://mojeedabisiga.com/#person"
    for tag in ('rel="canonical" href="https://mojeedabisiga.com/learn/p/make-your-first-skill/"', 'name="description"', 'property="og:image"'):
        assert tag in page


def test_tagline_accent_becomes_italic_on_the_page_and_plain_in_the_title(built):
    from nle.build import accent_markup, plain_accent
    assert str(accent_markup("Data & AI, *made easy* <for> all")) == "Data &amp; AI, <em>made easy</em> &lt;for&gt; all"
    assert plain_accent("Data & AI, *made easy* for everyone.") == "Data & AI, made easy for everyone."
    report, _ = built
    home = (report.out_dir / "index.html").read_text(encoding="utf-8")
    title = re.search(r"<title>(.*?)</title>", home, re.S).group(1)
    assert "*" not in title and "<em>" in re.search(r"<h1>(.*?)</h1>", home, re.S).group(1)


def test_home_page_shows_the_photo_set_in_site_json(built):
    report, _ = built
    hero = load_site()["hero_image"]
    home = (report.out_dir / "index.html").read_text(encoding="utf-8")
    assert f'class="hero-photo" src="{load_site().link(hero["src"])}"' in home
    assert f'width="{hero["width"]}" height="{hero["height"]}"' in home


def test_nothing_published_mentions_the_old_brand(built):
    report, out = built
    names = load_site()["never_mention"]
    for path in list(report.out_dir.rglob("*")) + list((out / "email").glob("*")):
        if path.suffix in (".html", ".xml", ".json"):
            text = path.read_text(encoding="utf-8")
            errors = [m for _, m, level in brand_mentions(text, names) if level == "error"]
            assert not errors, f"{path}: {errors}"


def test_email_copy_works_in_email_programs(built):
    _, out = built
    mail = (out / "email" / "make-your-first-skill.html").read_text(encoding="utf-8")
    assert "<button" not in mail and "<script" not in mail and 'class="prompt"' not in mail
    assert 'href="https://mojeedabisiga.com/learn/' in mail and "style=" in mail
    assert "subscribe-box" not in mail and "test it with five requests" in mail     # no subscribe box, nothing cut


@pytest.mark.skipif(not (MAIN_SITE / "tools" / "check_site.py").exists(), reason="the main site isn't next to the engine")
def test_main_site_checks_still_pass_with_the_publication_inside(built, tmp_path):
    report, _ = built
    site_copy = tmp_path / "site"
    shutil.copytree(MAIN_SITE / "site", site_copy)
    shutil.copytree(report.out_dir, site_copy / load_site().base_path.strip("/"), dirs_exist_ok=True)
    result = subprocess.run([sys.executable, str(MAIN_SITE / "tools" / "check_site.py"), str(site_copy)],
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
