"""The app's pages, each at its own address (app/paths.py). Every page is the same file, static/index.html, which
static/app.js fills in; its head is the page's own, with a title, description and picture, so a shared link shows what
it's about, and search engines know too. An address that isn't a page, or a guide, companion or course that isn't
there, answers 404 (the app says so). Addresses with an id where a slug goes, as before slugs, move to the slug."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException

from app import catalog, companions, content, courses, db, paths
from app.config import settings

INDEX = Path(__file__).resolve().parent.parent / "static" / "index.html"
SITE = "AI Every Time"
DEFAULT_IMAGE = "/static/icons/icon-512.png"

# Pages that are only their name: their title, and whether anyone may open them (the rest are for learners signed in,
# and stay out of search engines).
NAMED = {
    "": (None, True), "signin": ("Sign in", True), "signup": ("Create your free account", True),
    "get-started": ("Get started", False), "feed": ("Your feed", False), "guides": ("Guides", True),
    "companions": ("AI Learning Companions", True), "my-companion": ("Your companion", False),
    "courses": ("Courses", True), "learn": ("Learning paths", False), "practice": ("Practice", False),
    "flashcards": ("Flashcard review", False), "progress": ("Your progress", False), "profile": ("Your profile", False),
    "pricing": ("Plans and pricing", True), "billing": ("Plan and billing", False),
    "invite-friends": ("Invite friends", False), "privacy": ("Privacy", True), "support": ("Help and support", True),
    "staff": ("Staff", False),
}
DESCRIBED = {
    "guides": "Tutorials and cheat sheets on AI by AI Learning Companions from around the world. Read one, then ask "
              "its author anything.",
    "companions": "AI Learning Companions from around the world, each with their own face, voice and expertise. Chat "
                  "with one, or design your own.",
    "courses": "AI courses taught by AI Learning Companions: lessons in their own voice, hands-on labs, projects they "
               "review, graded quizzes and a certificate when you finish.",
    "pricing": "AI Every Time is free to use, with limits. Plans open every guide, full courses and more AI requests.",
}
# Pages for one of something of the learner's own: its name, and what a valid one looks like.
OWN = {"reset-password": ("Choose a new password", r"[\w-]{1,200}"), "lessons": ("Lesson", r"[\w-]{1,200}"),
       "quizzes": ("Quiz", r"[0-9]{1,15}"), "flashcards": ("Flashcard review", r"[0-9]{1,15}"),
       "staff": ("Staff", r"[a-z]{1,20}")}
# Within a course: its grades, its project, and each module's lessons, lab and quiz.
COURSE_PARTS = re.compile(r"(/grades|/project|/module-\d{1,3}/(lesson-\d{1,3}|lab|quiz))?")


@dataclass
class Page:
    title: str | None = None  # None for the site's own
    description: str | None = None
    image: str | None = None
    size: tuple[int, int] | None = None  # the picture's, where it's known: previews show it at once
    kind: str = "website"
    public: bool = True  # in search engines
    status: int = 200
    move_to: str | None = None  # the address it has now


def describe(path: str, user: dict | None) -> Page:
    """What's at this path, for the page's head and its status."""
    first, _, rest = path.strip("/").partition("/")
    if first not in paths.PAGES:
        return Page("Page not found", public=False, status=404)
    if not rest:
        title, public = NAMED.get(first, ("Page not found", False))
        if first not in NAMED:  # /posts, /tracks and the like: only with something after them
            return Page(title, public=False, status=404)
        return Page(title, DESCRIBED.get(first), public=public)
    ref, _, more = rest.partition("/")
    more = f"/{more}" if more else ""
    if first in ("guides", "posts"):
        return _post(first, ref, more, user)
    if first == "companions":
        return _companion(ref, more, user)
    if first == "courses":
        return _course(ref, more)
    if first == "tracks" and not more:
        t = catalog.data().tracks_by_slug.get(ref)
        return Page(f"{t.title} {t.kind}", t.subtitle) if t else Page("Page not found", public=False, status=404)
    if first == "certificates" and not more:
        return _certificate(ref)
    if first in OWN and not more and re.fullmatch(OWN[first][1], ref):
        return Page(OWN[first][0], public=False)
    return Page("Page not found", public=False, status=404)


def _visible(companion: dict, user: dict | None) -> bool:
    """Whether the visitor may see this companion: anyone, if it's public; its designer, if it's theirs."""
    if companions.public(companion):
        return True
    return (user is not None and companions.visible_to(companion, user["id"])
            and not companions.designer_blocked(companion, user["id"]))


def _post(section: str, ref: str, more: str, user: dict | None) -> Page:
    post = content.find_post(ref)
    by = companions.get(post["companion_id"]) if post else None
    if post is None or not _visible(by, user):
        return Page("Page not found", public=False, status=404)
    if more and not (post["kind"] and re.fullmatch(r"/skills(/[a-z0-9-]{1,60})?", more)):
        return Page("Page not found", public=False, status=404)
    address = paths.post(post) + more
    if f"/{section}/{ref}{more}" != address:  # by its id, or a guide under /posts
        return Page(move_to=address)
    if not companions.public(by):
        return Page(public=False)  # a learner's own companion's: nothing about it in the head
    title = f"{post['title']}: skill library" if more else post["title"]
    return Page(title, post.get("summary") or plain(post["body"]), post.get("graphic"),
                (1024, 1024) if post.get("graphic") else None, kind="article")


def _companion(ref: str, more: str, user: dict | None) -> Page:
    if ref == "new" and not more:
        return Page("Design a companion", public=False)
    try:
        c = companions.get_by_ref(ref)
    except HTTPException:
        return Page("Page not found", public=False, status=404)
    if more not in ("", "/edit", "/customise") or not _visible(c, user):
        return Page("Page not found", public=False, status=404)
    if ref != c["slug"]:
        return Page(move_to=paths.companion(c) + more)
    if more or not companions.public(c):
        return Page(public=False)
    p = c["persona"]
    with db.connect() as conn:
        photo = conn.execute("SELECT photo_url != '' OR EXISTS (SELECT 1 FROM companion_photos WHERE companion_id = c.id) "
                             "FROM companions c WHERE id = ?", (c["id"],)).fetchone()[0]
    # A photo, which link previews show; a drawn face is a picture they can't.
    image = f"/api/companions/{c['id']}/avatar" if photo and p["avatar"].get("use_photo") is not False else None
    who = f"{p['name']}, AI Learning Companion" + (f" from {p['country']}" if p.get("country") else "")
    return Page(who, p.get("tagline") or plain(p.get("bio") or ""), image)


def _course(slug: str, more: str) -> Page:
    c = catalog.course(slug)
    if c is None or not COURSE_PARTS.fullmatch(more):
        return Page("Page not found", public=False, status=404)
    if more:
        return Page(c.title, public=False)  # inside the course, for those taking it
    return Page(c.title, c.subtitle)


def _certificate(code: str) -> Page:
    try:
        cert = courses.certificate(code)
    except HTTPException:
        return Page("There's no certificate with that code", public=False, status=404)
    if code != cert["code"]:
        return Page(move_to=paths.certificate(cert["code"]))
    f = cert["for"]
    title = (f"{cert['name']} earned the {f['title']} {f['kind']}" if cert["kind"] == "track"
             else f"{cert['name']} completed {f['title']}")
    issued = datetime.fromisoformat(cert["issued_at"])
    return Page(title, f"A certificate from AI Every Time, issued on {issued.day} {issued:%B %Y}. "
                       f"Check it at {cert['url'].split('://', 1)[-1]}", "/static/icons/certificate-card.png", (1200, 630))


def plain(markdown: str, most: int = 200) -> str:
    """The start of some markdown as plain text, for a description: no marks, links or code."""
    text = re.sub(r"```.*?```", " ", markdown or "", flags=re.S)
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"[#>*_`|]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > most:
        text = text[:most].rsplit(" ", 1)[0].rstrip(",;:.") + "…"
    return text


def _absolute(address: str) -> str:
    return address if address.startswith("https://") else f"{settings.public_url}{address}"


def render(path: str, page: Page) -> str:
    """static/index.html, its head made this page's."""
    doc = INDEX.read_text(encoding="utf-8")
    title = f"{page.title} · {SITE}" if page.title else SITE
    head = ['<meta property="og:site_name" content="AI Every Time">', f'<meta property="og:type" content="{page.kind}">',
            f'<meta property="og:title" content="{html.escape(page.title or SITE)}">']
    if page.description:
        head.append(f'<meta property="og:description" content="{html.escape(page.description)}">')
    if page.status == 200:
        url = _absolute(path)
        head += [f'<link rel="canonical" href="{html.escape(url)}">', f'<meta property="og:url" content="{html.escape(url)}">']
    head.append(f'<meta property="og:image" content="{html.escape(_absolute(page.image or DEFAULT_IMAGE))}">')
    if page.image and page.size:
        head += [f'<meta property="og:image:width" content="{page.size[0]}">',
                 f'<meta property="og:image:height" content="{page.size[1]}">']
    head.append(f'<meta name="twitter:card" content="{"summary_large_image" if page.image else "summary"}">')
    if not page.public or page.status != 200:
        head.append('<meta name="robots" content="noindex">')
    if settings.ios_app_id:
        # Safari's banner for the iPhone app, once it's on the App Store. Apps' web views, the app's own included,
        # never show it.
        head.append(f'<meta name="apple-itunes-app" content="app-id={settings.ios_app_id}">')
    doc = doc.replace("<title>AI Every Time</title>", f"<title>{html.escape(title)}</title>", 1)
    if page.description:
        doc = re.sub(r'(<meta name="description" content=")[^"]*(">)',
                     lambda m: m[1] + html.escape(page.description) + m[2], doc, count=1)
    return doc.replace('<meta name="theme-color"', "\n  ".join(head) + '\n  <meta name="theme-color"', 1)
