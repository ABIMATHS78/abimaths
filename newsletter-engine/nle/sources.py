"""Downloads the training material: example issues from a Substack publication (through its public API), the
pages that example posts link to (lnkd.in short links are followed), and a few site pages. Everything is saved in
training/data/, which is never committed: the material belongs to its creator and is only measured.

training/sources.local.json (also never committed) says what to fetch:
    {"substack": "example.substack.com",
     "link_texts": "../linkedin-engine/training/data",     # optional: text files whose links to follow
     "links": ["https://lnkd.in/xxxx"],                    # optional: more links
     "pages": {"home": "https://example.substack.com/"}}   # optional: pages to keep for reference
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

from .paths import DATA_DIR, ENGINE_DIR, SOURCES_PATH

UA = {"User-Agent": "Mozilla/5.0 (newsletter-engine training)"}
URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s)\]\"'<>]*)?", re.I)
LNKD_TARGET_RE = re.compile(r'data-tracking-control-name="external_url_click"[^>]*href="([^"]+)"')


def get(url: str, timeout: int = 40) -> str:
    request = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "replace")


def resolve_short_link(url: str) -> str | None:
    """lnkd.in shows a 'you're leaving LinkedIn' page; the real address is in its link."""
    page = get(url)
    match = LNKD_TARGET_RE.search(page) or re.search(r'<a[^>]+href="(https?://(?!www\.linkedin|lnkd|static\.licdn)[^"]+)"', page)
    return match.group(1).replace("&amp;", "&") if match else None


def links_in_texts(folder: Path) -> list[str]:
    found = []
    for path in sorted(Path(folder).glob("*.txt")):
        for raw in URL_RE.findall(path.read_text(encoding="utf-8", errors="replace")):
            url = raw.rstrip(".,;:!?)")
            url = url if url.startswith("http") else "https://" + url
            if url not in found:
                found.append(url)
    return found


def substack_archive(publication: str, limit: int = 300) -> list[dict]:
    posts, cursor = [], None
    while len(posts) < limit:
        query = "archive?sort=new&limit=12" + (f"&cursor={urllib.parse.quote(cursor)}" if cursor else "")
        batch = json.loads(get(f"https://{publication}/api/v1/{query}"))
        items = batch.get("posts", []) if isinstance(batch, dict) else batch
        posts += items
        cursor = batch.get("nextCursor") if isinstance(batch, dict) else None
        if not items or not cursor:
            break
        time.sleep(0.3)
    return posts


def substack_post(publication: str, slug: str) -> dict:
    return json.loads(get(f"https://{publication}/api/v1/posts/{slug}"))


def load_sources(path: Path | None = None) -> dict:
    path = Path(path) if path else SOURCES_PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def fetch(sources: dict, data_dir: Path | None = None, log=print) -> dict:
    data = Path(data_dir) if data_dir else DATA_DIR
    (data / "posts").mkdir(parents=True, exist_ok=True)
    (data / "pages").mkdir(parents=True, exist_ok=True)
    publication = sources.get("substack", "").removeprefix("https://").strip("/")

    links = list(sources.get("links", []))
    if sources.get("link_texts"):
        folder = (ENGINE_DIR / sources["link_texts"]).resolve()
        if folder.exists():
            links += links_in_texts(folder)
    resolved = {}
    for url in dict.fromkeys(links):
        if "lnkd.in/" in url:
            try:
                resolved[url] = resolve_short_link(url)
            except Exception as error:
                log(f"  couldn't follow {url}: {error}")
                resolved[url] = None
        else:
            resolved[url] = url
    (data / "links.json").write_text(json.dumps(resolved, indent=1), encoding="utf-8")

    slugs = []
    for target in resolved.values():
        if target and publication and publication in target and "/p/" in target:
            slugs.append(target.split("/p/")[1].split("?")[0].strip("/"))
    if publication:
        archive = substack_archive(publication)
        (data / "archive.json").write_text(json.dumps(archive), encoding="utf-8")
        log(f"  archive: {len(archive)} issues")
        for slug in dict.fromkeys(slugs):
            target = data / "posts" / f"{slug}.json"
            if not target.exists():
                target.write_text(json.dumps(substack_post(publication, slug)), encoding="utf-8")
                time.sleep(0.2)
        log(f"  issues linked from the example posts: {len(set(slugs))}")
    for name, url in sources.get("pages", {}).items():
        try:
            (data / "pages" / f"{name}.html").write_text(get(url), encoding="utf-8")
        except Exception as error:
            log(f"  couldn't fetch {url}: {error}")
    return {"links": len(resolved), "issues": len(set(slugs))}
