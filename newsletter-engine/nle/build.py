"""Builds the website from the content folder: every page, the search index, the RSS feed, the sitemap, social
images (when Chromium is available) and an email-ready copy of each issue.

The output goes to dist/<base_path>/ (for example dist/learn/), so `python -m nle serve` shows it at the same
address it will have online, and `python -m nle publish` copies that folder into the main site.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import html
import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .config import Site, load_site
from .content import Item, Library, load_library
from .markup import REF_RE, inline
from .md import Rendered, Renderer
from .paths import CONTENT_DIR, DIST_DIR, MEDIA_DIR, STATIC_DIR, TEMPLATES_DIR

PAGES = {"subscribe": "subscribe/", "archive": "archive/", "guides": "guides/", "start": "start/",
         "library": "library/", "about": "about/", "home": ""}


@dataclass
class Entry:
    """An issue or guide, ready for the templates."""

    item: Item
    url: str
    abs_url: str
    date_label: str = ""
    updated_label: str = ""
    cover_url: str = ""
    html: str = ""
    toc: list = field(default_factory=list)
    minutes: int = 1
    rendered: Rendered | None = None
    prev: "Entry | None" = None
    next: "Entry | None" = None
    in_path: bool = False
    number: int = 0
    ends_with_subscribe: bool = False

    def __getattr__(self, name):          # title, subtitle, slug, tags... come from the item
        return getattr(self.item, name)

    @property
    def kind_label(self) -> str:
        return "Guide" if self.item.kind == "guide" else "Newsletter"

    @property
    def description_text(self) -> str:
        if self.item.description:
            return self.item.description
        text = self.item.subtitle or (self.rendered.text if self.rendered else "")
        return _clip(text, 158)


@dataclass
class BuildReport:
    out_dir: Path
    pages: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)       # block publishing
    notes: list[str] = field(default_factory=list)          # worth a look
    og_images: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


def _clip(text: str, limit: int) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def _date_label(value: dt.date | None) -> str:
    return value.strftime("%b %-d, %Y") if value else ""


class Builder:
    def __init__(self, site: Site | None = None, library: Library | None = None, *, drafts: bool = False,
                 out_root: Path | None = None, content_dir: Path | None = None, og: bool | None = None):
        self.site = site or load_site()
        self.content_dir = Path(content_dir) if content_dir else CONTENT_DIR
        self.media_dir = self.content_dir / "media" if content_dir else MEDIA_DIR
        self.full = library or load_library(self.content_dir)
        self.library = self.full.visible(drafts)
        self.drafts = drafts
        self.out_root = Path(out_root) if out_root else DIST_DIR
        self.out = self.out_root / self.site.base_path.strip("/") if self.site.base_path.strip("/") else self.out_root
        self.og = og
        self._uid = 0
        self.env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)), autoescape=select_autoescape(["html", "xml"]),
                               trim_blocks=True, lstrip_blocks=True)
        self.env.globals.update(site=self.site, link=self.site.link, year=dt.date.today().year,
                                version=self._asset_version(), uid=self._next_uid, last_uid=lambda: self._uid)
        self.entries: dict[str, Entry] = {}

    # --- helpers ---------------------------------------------------------------------------------------------
    def _next_uid(self) -> int:
        self._uid += 1
        return self._uid

    def _asset_version(self) -> str:
        digest = hashlib.sha1()
        for path in sorted(STATIC_DIR.rglob("*")):
            if path.is_file() and path.suffix in (".css", ".js"):
                digest.update(path.read_bytes())
        return digest.hexdigest()[:8]

    def url_for(self, ref: str) -> str | None:
        kind, _, slug = ref.partition(":")
        if kind == "page":
            return self.site.link(PAGES[slug]) if slug in PAGES else None
        entry = self.entries.get(ref)
        return entry.url if entry else None

    def media_url(self, src: str) -> str | None:
        clean = src.lstrip("./")
        if ".." in Path(clean).parts or not (self.media_dir / clean).is_file():
            return None
        return self.site.link("media/" + clean)

    def _subscribe_html(self, label: str = "") -> str:
        macros = self.env.get_template("_macros.html").module
        return str(macros.subscribe_box(title=label or ""))

    def _card_html(self, ref: str) -> str | None:
        entry = self.entries.get(ref)
        if not entry:
            return None
        macros = self.env.get_template("_macros.html").module
        sub = f'<span class="ec-sub">{html.escape(entry.subtitle)}</span>' if entry.subtitle else ""
        return (f'<a class="embed-card" href="{html.escape(entry.url)}">{macros.titlecard(entry, small=True)}'
                f'<span><span class="ec-kind">{entry.kind_label}</span><span class="ec-title">{html.escape(entry.title)}</span>{sub}</span></a>')

    def _page(self, key: str, path: str, title: str, description: str, section: str = "", **extra) -> dict:
        page = {"key": key, "title": title, "description": _clip(description, 158), "canonical": self.site.absolute(path),
                "og_image": self.site.absolute("static/img/og-default.jpg"), "section": section}
        page.update(extra)
        return page

    def _write(self, rel: str, text: str, report: BuildReport) -> None:
        target = self.out / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        report.pages.append(rel)

    def _render(self, template: str, rel: str, report: BuildReport, **context) -> None:
        self._write(rel, self.env.get_template(template).render(nav=self.nav, **context), report)

    # --- build -----------------------------------------------------------------------------------------------
    def build(self) -> BuildReport:
        report = BuildReport(out_dir=self.out)
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        self._prepare_entries()
        self._render_bodies(report)
        self._link_neighbours()
        self.nav = self._nav()

        issues = [self.entries[i.ref] for i in self.library.issues]
        guides = [self.entries[g.ref] for g in self.library.guides]
        modules, lesson_count = self._modules(report)
        start_items = [e for m in modules for e in m["entries"]][:3] or [g for g in guides if g.featured][:3]

        site = self.site
        self.og_done = set()
        if self.og is not False:
            report.og_images = self._og_images(issues + guides, report, required=bool(self.og))
        self._render("home.html", "index.html", report, issues=issues, guides=guides, start_items=start_items,
                     page=self._page("home", "", f"{site['name']}: {site['tagline']}".rstrip(": "), site["description"],
                                     section="home", jsonld=self._blog_jsonld()))
        tags = [t for t, _ in sorted(_count(t for e in issues for t in e.tags).items(), key=lambda kv: (-kv[1], kv[0]))]
        self._render("archive.html", "archive/index.html", report, issues=issues, tags=tags[:12],
                     page=self._page("archive", "archive/", f"Archive · {site['name']}",
                                     f"Every issue of {site['name']}'s newsletter.", section="newsletter"))
        levels = [lvl for lvl in ("beginner", "intermediate", "advanced") if any(g.level == lvl for g in guides)]
        self._render("guides.html", "guides/index.html", report, guides=guides, levels=levels,
                     page=self._page("guides", "guides/", f"Free guides · {site['name']}",
                                     f"Step-by-step guides from {site['author']['name']}.", section="guides"))
        for entry in issues + guides:
            self._render_entry(entry, report)
        if modules:
            path = self.library.path
            self._render("start.html", "start/index.html", report, path=path, modules=modules, lesson_count=lesson_count,
                         page=self._page("start", "start/", f"{path.get('title') or 'Start here'} · {site['name']}",
                                         path.get("intro") or "Where to begin, in order.", section="start"))
        self._render_library(report)
        self._render_about(report)
        self._render("subscribe.html", "subscribe/index.html", report, issues=issues,
                     page=self._page("subscribe", "subscribe/", f"Subscribe · {site['name']}",
                                     site["newsletter"].get("promise") or site["description"], section="subscribe"))
        self._write("search.json", json.dumps(self._search_index(issues + guides), ensure_ascii=False), report)
        self._write("feed.xml", self._feed(issues + guides), report)
        self._write("sitemap.xml", self._sitemap(report.pages, issues + guides), report)
        self._copy_assets()
        self._email_copies(issues)
        return report

    def _prepare_entries(self) -> None:
        for item in self.library.all():
            path = f"p/{item.slug}/" if item.kind == "issue" else f"guides/{item.slug}/"
            entry = Entry(item=item, url=self.site.link(path), abs_url=self.site.absolute(path),
                          date_label=_date_label(item.date), updated_label=_date_label(item.updated))
            if item.cover:
                entry.cover_url = item.cover if item.cover.startswith(("http://", "https://")) else (self.media_url(item.cover) or "")
            self.entries[item.ref] = entry

    def _render_bodies(self, report: BuildReport) -> None:
        renderer = Renderer(resolve=self.url_for, card=self._card_html, subscribe=self._subscribe_html,
                            media=self.media_url, draft=self.drafts)
        for entry in self.entries.values():
            item = entry.item
            rendered = renderer.render(item.body)
            entry.rendered, entry.html, entry.toc, entry.minutes = rendered, rendered.html, rendered.toc, rendered.minutes
            tail = [line.strip() for line in item.body.strip().splitlines() if line.strip()][-3:]
            entry.ends_with_subscribe = any(line.startswith("::subscribe") for line in tail)
            where = f"{item.kind} '{item.slug}'"
            for problem in rendered.problems:
                report.problems.append(f"{where}: {problem}")
            if item.cover and not entry.cover_url:
                report.problems.append(f"{where}: the cover {item.cover} isn't in content/media/")
            for shot in rendered.screenshots:
                (report.notes if not item.published else report.problems).append(f"{where}: screenshot still to add: {shot}")

    def _link_neighbours(self) -> None:
        issues = [self.entries[i.ref] for i in self.library.issues]
        for index, entry in enumerate(issues):
            entry.prev = issues[index + 1] if index + 1 < len(issues) else None      # older
            entry.next = issues[index - 1] if index > 0 else None                    # newer

    def _modules(self, report: BuildReport) -> tuple[list[dict], int]:
        modules, number = [], 0
        for module in self.library.path.get("modules", []):
            entries = []
            for ref in module.get("items", []):
                entry = self.entries.get(ref)
                if not entry:
                    if REF_RE.match(ref) and ref in self.full.by_ref():
                        report.notes.append(f"Start here: {ref} is a draft, so it's left out")
                    else:
                        report.problems.append(f"Start here: {ref} doesn't exist")
                    continue
                number += 1
                entry.number, entry.in_path = number, True
                entries.append(entry)
            if entries:
                modules.append({"title": module.get("title", ""), "text": module.get("text", ""), "entries": entries})
        # inside the path, guides link to the next lesson
        ordered = [e for m in modules for e in m["entries"]]
        for index, entry in enumerate(ordered):
            if entry.item.kind == "guide":
                entry.prev = ordered[index - 1] if index > 0 else None
                entry.next = ordered[index + 1] if index + 1 < len(ordered) else None
        return modules, number

    def _nav(self) -> list[dict]:
        labels = self.site["pages"]
        nav = [{"key": "newsletter", "label": labels["newsletter_label"], "href": self.site.link("archive/")},
               {"key": "guides", "label": labels["guides_label"], "href": self.site.link("guides/")}]
        if self.library.path.get("modules"):
            nav.append({"key": "start", "label": labels["path_label"], "href": self.site.link("start/")})
        if self.library.library.get("items"):
            nav.append({"key": "library", "label": labels["library_label"], "href": self.site.link("library/")})
        nav.append({"key": "about", "label": "About", "href": self.site.link("about/")})
        return nav

    def _render_entry(self, entry: Entry, report: BuildReport) -> None:
        item = entry.item
        rel = ("p/" if item.kind == "issue" else "guides/") + item.slug + "/index.html"
        og = self.site.absolute(f"og/{item.slug}.png") if item.ref in self.og_done else ""
        if not og and entry.cover_url:
            og = self.site.base_url + entry.cover_url if entry.cover_url.startswith("/") else entry.cover_url
        page = self._page(item.ref, rel.removesuffix("index.html"), f"{item.title} · {self.site['name']}",
                          entry.description_text, section="newsletter" if item.kind == "issue" else "guides",
                          og_type="article", og_title=item.title, progress=True, jsonld=self._article_jsonld(entry),
                          published=item.date.isoformat() if item.date else "",
                          draft="" if item.published else "This page isn't in the sitemap or the feed.")
        if og:
            page["og_image"] = og
        if not item.published:
            page["noindex"] = True
        self._render("article.html", rel, report, e=entry, page=page)

    def _render_library(self, report: BuildReport) -> None:
        library = dict(self.library.library)
        raw = library.get("items", [])
        if not raw:
            return
        items, categories = [], {}
        for spec in raw:
            category = spec.get("category", "Other")
            key = re.sub(r"[^a-z0-9]+", "-", category.lower()).strip("-")
            categories.setdefault(key, category)
            item = dict(spec, category=category, category_key=key, text_html=inline(spec.get("text", ""), self.url_for))
            if spec.get("download"):
                download = spec["download"]
                item["download_url"] = download if download.startswith(("http://", "https://")) else self.media_url(download)
                if not item["download_url"]:
                    report.problems.append(f"library: the download {download} isn't in content/media/")
            if spec.get("read"):
                item["read_url"] = self.url_for(spec["read"]) if REF_RE.match(spec["read"]) else spec["read"]
                if not item["read_url"]:
                    report.problems.append(f"library: {spec['read']} doesn't exist")
            items.append(item)
        library["faq"] = [dict(qa, a_html=inline(qa.get("a", ""), self.url_for)) for qa in library.get("faq", [])]
        self._render("library.html", "library/index.html", report, library=library, items=items,
                     categories=[{"key": k, "label": v} for k, v in categories.items()],
                     page=self._page("library", "library/", f"{library.get('title') or 'Library'} · {self.site['name']}",
                                     library.get("intro") or "Prompts, tools and files to use today.", section="library"))

    def _render_about(self, report: BuildReport) -> None:
        about = self.library.about.strip() or self.site["author"]["bio"]
        rendered = Renderer(resolve=self.url_for, card=self._card_html, subscribe=lambda label: "",
                            media=self.media_url).render(about)
        for problem in rendered.problems:
            report.problems.append(f"about page: {problem}")
        author = self.site["author"]
        self._render("about.html", "about/index.html", report, about_html=rendered.html,
                     about_title=f"About {self.site['name']}" if self.site["name"] != author["name"] else f"Hi, I'm {author['name'].split()[0]}",
                     page=self._page("about", "about/", f"About · {self.site['name']}", author.get("bio") or self.site["description"],
                                     section="about"))

    # --- data files ------------------------------------------------------------------------------------------
    def _search_index(self, entries: list[Entry]) -> list[dict]:
        return [{"ref": e.ref, "url": e.url, "title": e.title, "subtitle": e.subtitle, "tags": list(e.tags),
                 "kind": e.item.kind, "text": _clip(e.rendered.text if e.rendered else "", 6000)} for e in entries]

    def _feed(self, entries: list[Entry]) -> str:
        site = self.site
        items = sorted((e for e in entries if e.item.published), key=lambda e: e.item.sort_date, reverse=True)[:30]
        def absolutize(text: str) -> str:
            return re.sub(r'(href|src)="(/[^"]*)"', lambda m: f'{m.group(1)}="{site.base_url}{m.group(2)}"', text)
        parts = []
        for e in items:
            when = dt.datetime.combine(e.item.sort_date, dt.time(9, 0), tzinfo=dt.timezone.utc)
            parts.append(
                "<item>"
                f"<title>{xml_escape(e.title)}</title><link>{xml_escape(e.abs_url)}</link><guid isPermaLink=\"true\">{xml_escape(e.abs_url)}</guid>"
                f"<pubDate>{when.strftime('%a, %d %b %Y %H:%M:%S +0000')}</pubDate>"
                f"<description>{xml_escape(e.description_text)}</description>"
                f"<content:encoded><![CDATA[{absolutize(e.html).replace(']]>', ']]&gt;')}]]></content:encoded>"
                "</item>")
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/" '
                'xmlns:atom="http://www.w3.org/2005/Atom"><channel>'
                f"<title>{xml_escape(site['name'])}</title><link>{xml_escape(site.absolute(''))}</link>"
                f"<description>{xml_escape(site['description'])}</description><language>{site['language']}</language>"
                f'<atom:link href="{xml_escape(site.absolute("feed.xml"))}" rel="self" type="application/rss+xml"/>'
                + "".join(parts) + "</channel></rss>\n")

    def _sitemap(self, pages: list[str], entries: list[Entry]) -> str:
        lastmod = {("p/" if e.item.kind == "issue" else "guides/") + e.slug + "/index.html": (e.item.updated or e.item.date)
                   for e in entries}
        drafts = {("p/" if e.item.kind == "issue" else "guides/") + e.slug + "/index.html" for e in entries if not e.item.published}
        urls = []
        for rel in pages:
            if not rel.endswith("index.html") or rel in drafts:
                continue
            loc = self.site.absolute(rel.removesuffix("index.html"))
            mod = lastmod.get(rel)
            urls.append(f"<url><loc>{xml_escape(loc)}</loc>" + (f"<lastmod>{mod.isoformat()}</lastmod>" if mod else "") + "</url>")
        return ('<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
                + "".join(urls) + "</urlset>\n")

    def _blog_jsonld(self) -> dict:
        author = self.site["author"]
        return {"@context": "https://schema.org", "@type": "Blog", "name": self.site["name"], "url": self.site.absolute(""),
                "description": self.site["description"], "inLanguage": self.site["language"],
                "author": _person_ref(author)}

    def _article_jsonld(self, entry: Entry) -> dict:
        item = entry.item
        data = {"@context": "https://schema.org", "@type": "BlogPosting" if item.kind == "issue" else "TechArticle",
                "headline": item.title, "description": entry.description_text, "url": entry.abs_url,
                "mainEntityOfPage": entry.abs_url, "inLanguage": self.site["language"], "author": _person_ref(self.site["author"]),
                "wordCount": entry.rendered.words if entry.rendered else 0}
        if item.date:
            data["datePublished"] = item.date.isoformat()
        if item.updated or item.date:
            data["dateModified"] = (item.updated or item.date).isoformat()
        if item.tags:
            data["keywords"] = ", ".join(item.tags)
        return data

    # --- files -----------------------------------------------------------------------------------------------
    def _copy_assets(self) -> None:
        shutil.copytree(STATIC_DIR, self.out / "static", dirs_exist_ok=True)
        if self.media_dir.exists():
            shutil.copytree(self.media_dir, self.out / "media", dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns(".*", "README*"))

    def _email_copies(self, issues: list[Entry]) -> None:
        """An email-ready copy of each issue (inline styles, absolute links) for pasting into a newsletter tool."""
        from .email import email_html

        folder = self.out_root / "email"
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        renderer = Renderer(resolve=self.url_for, card=self._card_html, subscribe=lambda label: "", media=self.media_url)
        for entry in issues:
            body = renderer.render(entry.item.body).html
            (folder / f"{entry.slug}.html").write_text(email_html(entry, self.site, body), encoding="utf-8")

    _og_checked: bool | None = None

    def _can_render_og(self) -> bool:
        if self._og_checked is None:
            try:
                import playwright.sync_api  # noqa: F401
                self._og_checked = True
            except ImportError:
                self._og_checked = False
        return self._og_checked

    def _og_images(self, entries: list[Entry], report: BuildReport, required: bool) -> int:
        """1200x630 social images (the title card) so LinkedIn and others show a picture when the page is shared."""
        if not entries:
            return 0
        if not self._can_render_og():
            msg = "Chromium (Playwright) isn't installed, so pages use the default social image"
            (report.problems if required else report.notes).append(msg)
            return 0
        from .og import render_og_images

        try:
            made = render_og_images(entries, self.site, self.out / "og", self.env)
        except Exception as error:  # a missing browser shouldn't stop the build
            msg = f"social images weren't made ({type(error).__name__}: {error}); pages use the default image"
            (report.problems if required else report.notes).append(msg)
            return 0
        self.og_done = set(made)
        return len(made)


def _person_ref(author: dict) -> dict:
    ref = {"@type": "Person", "name": author.get("name", "")}
    if author.get("person_id"):
        ref["@id"] = author["person_id"]
    if author.get("url"):
        ref["url"] = author["url"]
    return ref


def _count(values) -> dict:
    counts: dict = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def build(drafts: bool = False, out_root: Path | None = None, og: bool | None = None) -> BuildReport:
    return Builder(drafts=drafts, out_root=out_root, og=og).build()
