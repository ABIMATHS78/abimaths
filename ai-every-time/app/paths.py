"""The site's page addresses, for every link the server makes: notices, notifications, emails, link previews and the
links companions put in their replies. The app (static/app.js) routes the same paths, and still opens each page from
its address before these, such as /#/p/1165, which links in old emails and chats carry.

Guides and companions' posts are known by words from their title (their slug), companions by their name's."""

from __future__ import annotations

import re
import unicodedata

from app.config import settings

# The first part of every page's path. Anything else is a page that isn't there (app/main.py answers 404).
PAGES = frozenset({
    "", "signin", "signup", "reset-password", "get-started", "feed", "guides", "posts", "companions", "my-companion",
    "courses", "tracks", "certificates", "learn", "lessons", "quizzes", "practice", "flashcards", "progress",
    "profile", "pricing", "billing", "invite-friends", "privacy", "support", "staff",
})

FEED = "/feed"
PROFILE = "/profile"
LEARN = "/learn"
FLASHCARDS = "/flashcards"
PROGRESS = "/progress"
BILLING = "/billing"
PRICING = "/pricing"
COURSES = "/courses"
INVITE = "/invite-friends"


def slugify(text: str, fallback: str = "post", most: int = 80) -> str:
    """Words for an address: "What's an LLM? A 5-minute guide" gives "whats-an-llm-a-5-minute-guide". Never only
    digits, which would read as an id."""
    folded = "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)).lower()
    words = re.sub(r"[^a-z0-9]+", "-", re.sub(r"['’]", "", folded)).strip("-")
    if len(words) > most:
        words = words[:most].rsplit("-", 1)[0]
    return f"{fallback}-{words}" if words.isdigit() else words or fallback


def _unique(title: str, author: str | None, taken) -> str:
    """The title's words; if another post has them, its author's name after them (…-by-kwame), as several companions
    write guides with the same title; then a number, for the same author's."""
    base = slugify(title)
    tries = [base] + ([f"{base}-by-{slugify(author.split()[0], fallback='companion')}"] if author and author.split() else [])
    for slug in tries:
        if not taken(slug):
            return slug
    n = 2
    while taken(f"{tries[-1]}-{n}"):
        n += 1
    return f"{tries[-1]}-{n}"


def new_post_slug(conn, title: str, companion_id: int) -> str:
    """A slug no other post has, for a post the companion is writing."""
    author = conn.execute("SELECT json_extract(persona, '$.name') FROM companions WHERE id = ?", (companion_id,)).fetchone()
    return _unique(title, author[0] if author else None,
                   lambda slug: conn.execute("SELECT 1 FROM posts WHERE slug = ?", (slug,)).fetchone() is not None)


def fill_post_slugs(conn) -> int:
    """Gives every post without a slug one, oldest first, so the oldest of those with the same title keeps the plain one."""
    rows = conn.execute("SELECT p.id, p.title, json_extract(c.persona, '$.name') AS author FROM posts p "
                        "LEFT JOIN companions c ON c.id = p.companion_id WHERE p.slug IS NULL ORDER BY p.id").fetchall()
    taken = {r[0] for r in conn.execute("SELECT slug FROM posts WHERE slug IS NOT NULL")}
    for r in rows:
        slug = _unique(r["title"], r["author"], taken.__contains__)
        taken.add(slug)
        conn.execute("UPDATE posts SET slug = ? WHERE id = ?", (slug, r["id"]))
    return len(rows)


def as_id(ref: str) -> int | None:
    """The id in an address that has one where a slug goes, as addresses from before slugs did; None for a slug."""
    return int(ref) if re.fullmatch(r"[0-9]{1,15}", ref) else None


def post(p: dict) -> str:
    """A guide's page, or a companion's post's. A post without a slug yet is found by its id."""
    return f"/{'guides' if p.get('kind') else 'posts'}/{p.get('slug') or p['id']}"


def companion(c: dict) -> str:
    return f"/companions/{c['slug']}"


def course(slug: str) -> str:
    return f"/courses/{slug}"


def track(slug: str) -> str:
    return f"/tracks/{slug}"


def certificate(code: str) -> str:
    return f"/certificates/{code}"


def lesson(lesson_id: str) -> str:
    return f"/lessons/{lesson_id}"


def quiz(assessment_id: int) -> str:
    return f"/quizzes/{assessment_id}"


def staff(tab: str) -> str:
    return f"/staff/{tab}"


def reset_password(token: str) -> str:
    return f"/reset-password/{token}"


def absolute(link: str) -> str:
    """The full address of a page, for emails: "/learn" and the "#/learn" of old notices alike."""
    return f"{settings.public_url}/{link.lstrip('/')}"
