"""What search engines and AI assistants read: robots.txt, the sitemap, llms.txt, each public page's structured data
(schema.org JSON-LD), and the page itself as plain HTML.

Most crawlers, AI assistants' among them, don't run JavaScript, and Google runs it later than it reads the page, so
every public page comes from the server with what it says already in it: the course, guide or companion, its
headings, text and links. static/app.js draws the page over it as soon as it loads. A visitor who isn't signed in sees
the same: the start of a guide, not all of it, so crawlers are never shown more than people are.

Only pages anyone can open are listed in the sitemap, each at the one address it answers 200 at (its canonical),
so none of them is left out of search results for being a redirect, a 404, a duplicate or marked noindex."""

from __future__ import annotations

import html
import json
import re
from datetime import date
from xml.sax.saxutils import escape as xml_escape

from app import billing, catalog, companions, db, paths
from app.config import settings

SITE = "AI Every Time"
TAGLINE = "The 24/7 AI learning hub"
ABOUT = ("AI Every Time is a 24/7 AI learning hub. Learn AI, data, software and cloud skills from AI Learning "
         "Companions from around the world: chat with them any time, read their guides and cheat sheets, and take "
         "courses with hands-on labs, reviewed projects, graded quizzes and certificates. Start free.")
LOGO = "/static/icons/icon-512.png"

# Public pages that are only their name, for the sitemap, with how often they change.
PLAIN = (("/", "daily"), ("/courses", "weekly"), ("/guides", "daily"), ("/companions", "weekly"),
         ("/pricing", "monthly"), ("/support", "monthly"), ("/privacy", "yearly"))

# Crawlers named so it's plain they're welcome: search engines, and the AI assistants that search and cite the web.
CRAWLERS = ("Googlebot", "Bingbot", "Google-Extended", "GPTBot", "OAI-SearchBot", "ChatGPT-User", "ClaudeBot",
            "Claude-SearchBot", "Claude-User", "PerplexityBot", "Perplexity-User", "Applebot", "Applebot-Extended",
            "DuckDuckBot", "meta-externalagent", "Amazonbot", "CCBot", "YandexBot")
def url(address: str) -> str:
    return address if address.startswith("https://") else f"{settings.public_url}{address}"


# --------------------------------------------------------------------------- robots.txt, sitemap.xml, llms.txt


def robots() -> str:
    """Everyone may read everything. Pages for one learner say noindex themselves, and the app's data (/api/) says so
    in its headers: blocking them here instead would keep Google from seeing that, and from loading what pages
    show, so they'd be listed as "indexed, though blocked by robots.txt"."""
    groups = []
    for bot in ("*", *CRAWLERS):
        groups += [f"User-agent: {bot}", "Allow: /", ""]
    return "\n".join([f"# {SITE}: {TAGLINE}. Search engines and AI assistants are welcome to read and cite it.",
                      "# A summary for AI assistants: " + url("/llms.txt"), "", *groups,
                      f"Sitemap: {url('/sitemap.xml')}", ""])


def _public_companions() -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM companions WHERE retired = 0 AND (is_starter = 1 OR is_public = 1) "
                            "ORDER BY is_starter DESC, id").fetchall()
    return [c for c in map(db.row_to_companion, rows) if companions.public(c)]


def _public_posts(cast: dict[int, dict]) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id, slug, kind, title, summary, topic, created_at, companion_id FROM posts "
                            "ORDER BY created_at DESC, id DESC").fetchall()
    return [dict(r) for r in rows if r["companion_id"] in cast]


def _day(when: str | None) -> str | None:
    return when[:10] if when and re.match(r"\d{4}-\d{2}-\d{2}", when) else None


def entries() -> list[tuple[str, str | None, str]]:
    """Every page for the sitemap: its address, when it last changed (if known) and how often it changes."""
    cat = catalog.data()
    cast = {c["id"]: c for c in _public_companions()}
    posts = _public_posts(cast)
    newest = _day(posts[0]["created_at"]) if posts else None
    out = [(a, newest if a in ("/", "/guides") else None, f) for a, f in PLAIN]
    out += [(paths.course(c.slug), None, "monthly") for c in cat.courses]
    out += [(paths.track(t.slug), None, "monthly") for t in cat.tracks]
    out += [(paths.companion(c), _day(c.get("published_at")), "weekly") for c in cast.values()]
    out += [(paths.post(p), _day(p["created_at"]), "monthly") for p in posts]
    return out


def sitemap() -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for address, changed, often in entries():
        lines.append(f"  <url><loc>{xml_escape(url(address))}</loc>"
                     + (f"<lastmod>{changed}</lastmod>" if changed else "") + f"<changefreq>{often}</changefreq></url>")
    lines.append("</urlset>")
    return "\n".join(lines) + "\n"


def llms() -> str:
    """A plain summary of the site for AI assistants (llmstxt.org): what it is, and links to everything public."""
    cat = catalog.data()
    cast = _public_companions()
    guides = [p for p in _public_posts({c["id"]: c for c in cast}) if p["kind"]]
    out = [f"# {SITE}", "", f"> {ABOUT}", "",
           f"Website: {url('/')}. Free to join; paid plans open every guide, full courses and more AI requests "
           f"({url('/pricing')}). Everyone can take the first module of every course and read the start of every guide.",
           "", "## Main pages", "",
           f"- [Courses]({url('/courses')}): {len(cat.courses)} courses and guided projects on AI, data, software, "
           "networking, security and cloud, with labs, projects, quizzes and certificates",
           f"- [Guides]({url('/guides')}): tutorials and cheat sheets on AI by AI Learning Companions",
           f"- [AI Learning Companions]({url('/companions')}): AI tutors from around the world to chat with and learn from",
           f"- [Plans and pricing]({url('/pricing')})", f"- [Help and support]({url('/support')})",
           f"- [Privacy]({url('/privacy')})", "", "## Professional Certificates and Specializations", ""]
    out += [f"- [{t.title} {t.kind}]({url(paths.track(t.slug))}): {t.subtitle}" for t in cat.tracks]
    for category in catalog.CATEGORIES:
        listed = [c for c in cat.courses if c.category == category]
        if listed:
            out += ["", f"## Courses: {category}", ""]
            out += [f"- [{c.title}]({url(paths.course(c.slug))}): {c.subtitle} ({_level(c)}, about {_hours(c.hours)})"
                    for c in listed]
    out += ["", "## AI Learning Companions", ""]
    for c in cast:
        p = c["persona"]
        what = p.get("tagline") or ", ".join(p.get("expertise") or [])
        out.append(f"- [{p['name']}]({url(paths.companion(c))})" + (f": {what}" if what else ""))
    if guides:
        out += ["", "## Guides", ""]
        out += [f"- [{g['title']}]({url(paths.post(g))})" + (f": {g['summary']}" if g.get("summary") else "")
                for g in guides]
    return "\n".join(out) + "\n"


# --------------------------------------------------------------------------- each page's structured data and HTML


def _level(c) -> str:
    return ("Guided project, " if c.kind == "guided-project" else "") + c.level.capitalize()


def _hours(h: float) -> str:
    return f"{h:g} hour" + ("" if h == 1 else "s")


def _e(text) -> str:
    return html.escape(str(text or ""), quote=True)


def organization() -> dict:
    org = {"@type": "Organization", "@id": url("/#organization"), "name": SITE, "url": url("/"),
           "logo": url(LOGO), "description": ABOUT}
    if settings.support_email:
        org["email"] = settings.support_email
    return org


def website() -> dict:
    return {"@type": "WebSite", "@id": url("/#website"), "name": SITE, "alternateName": "AIET", "url": url("/"),
            "description": ABOUT, "publisher": {"@id": url("/#organization")}, "inLanguage": "en"}


def breadcrumbs(*trail: tuple[str, str]) -> dict:
    items = [("Home", "/"), *trail]
    return {"@type": "BreadcrumbList", "itemListElement": [
        {"@type": "ListItem", "position": i, "name": name, "item": url(address)}
        for i, (name, address) in enumerate(items, 1)]}


def json_ld(things: list[dict]) -> str:
    """The page's structured data, safe inside a <script> element."""
    doc = json.dumps({"@context": "https://schema.org", "@graph": things}, ensure_ascii=False, separators=(",", ":"))
    return doc.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _nav() -> str:
    links = [("/", SITE), ("/courses", "Courses"), ("/guides", "Guides"), ("/companions", "AI Learning Companions"),
             ("/pricing", "Pricing"), ("/signup", "Join free")]
    return ('<header class="seo-top"><nav aria-label="Site">'
            + " · ".join(f'<a href="{a}">{_e(t)}</a>' for a, t in links) + "</nav></header>")


def _foot() -> str:
    return ('<footer class="seo-foot"><p>' + _e(ABOUT) + '</p><p><a href="/privacy">Privacy</a> · '
            '<a href="/support">Help and support</a> · <a href="/llms.txt">About this site for AI assistants</a></p>'
            "</footer>")


def page(main: str) -> str:
    return f'<div class="seo-page">{_nav()}<main>{main}</main>{_foot()}</div>'


def _course_items(listed) -> str:
    return "<ul>" + "".join(
        f'<li><a href="{paths.course(c.slug)}">{_e(c.title)}</a>: {_e(c.subtitle)} '
        f"<small>({_e(_level(c))}, about {_hours(c.hours)})</small></li>" for c in listed) + "</ul>"


def home() -> tuple[str, list[dict]]:
    cat = catalog.data()
    body = [f"<h1>{SITE}: {TAGLINE}</h1>", f"<p>{_e(ABOUT)}</p>",
            '<p><a href="/signup">Create your free account</a> · <a href="/courses">Browse courses</a> · '
            '<a href="/guides">Read guides</a> · <a href="/companions">Meet the AI Learning Companions</a></p>',
            "<h2>Professional Certificates and Specializations</h2><ul>"]
    body += [f'<li><a href="{paths.track(t.slug)}">{_e(t.title)} {_e(t.kind)}</a>: {_e(t.subtitle)}</li>'
             for t in cat.tracks]
    body.append("</ul>")
    for category in catalog.CATEGORIES:
        listed = [c for c in cat.courses if c.category == category]
        if listed:
            body.append(f"<h2>{_e(category)} courses</h2>" + _course_items(listed))
    return page("".join(body)), [organization(), website()]


def courses_page() -> tuple[str, list[dict]]:
    cat = catalog.data()
    body = ["<h1>AI courses</h1>",
            f"<p>{len(cat.courses)} courses and guided projects taught by AI Learning Companions: lessons in their own "
            "voice, hands-on labs, projects they review, graded quizzes and a certificate when you finish. The first "
            "module of every course is free.</p>"]
    for category in catalog.CATEGORIES:
        listed = [c for c in cat.courses if c.category == category]
        if listed:
            body.append(f"<h2>{_e(category)}</h2>" + _course_items(listed))
    body.append("<h2>Professional Certificates and Specializations</h2><ul>" + "".join(
        f'<li><a href="{paths.track(t.slug)}">{_e(t.title)} {_e(t.kind)}</a>: {_e(t.subtitle)}</li>'
        for t in cat.tracks) + "</ul>")
    items = {"@type": "ItemList", "name": "AI Every Time courses", "itemListElement": [
        {"@type": "ListItem", "position": i, "url": url(paths.course(c.slug))} for i, c in enumerate(cat.courses, 1)]}
    return page("".join(body)), [organization(), website(), breadcrumbs(("Courses", "/courses")), items]


def _teacher(slug: str) -> dict | None:
    return companions.starter(slug)


def course_page(c) -> tuple[str, list[dict]]:
    cat = catalog.data()
    teachers = [t for t in map(_teacher, c.instructors) if t]
    body = [f"<h1>{_e(c.title)}</h1>", f"<p><strong>{_e(c.subtitle)}</strong></p>",
            f"<p>{_e(_level(c))} · {_e(c.category)} · about {_hours(c.hours)} · {len(c.modules)} "
            f"module{'s' if len(c.modules) != 1 else ''} · certificate on completion</p>"]
    if teachers:
        body.append("<p>Taught by " + ", ".join(
            f'<a href="{paths.companion(t)}">{_e(t["persona"]["name"])}</a>' for t in teachers)
            + ", AI Learning Companion" + ("s" if len(teachers) > 1 else "") + ".</p>")
    body += ["<h2>About this course</h2>"] + [f"<p>{_e(p)}</p>" for p in c.description.split("\n\n") if p.strip()]
    body.append("<h2>What you'll learn</h2><ul>" + "".join(f"<li>{_e(o)}</li>" for o in c.outcomes) + "</ul>")
    body.append("<h2>Skills you'll gain</h2><p>" + _e(", ".join(c.skills)) + "</p>")
    if c.tools:
        body.append("<h2>Tools</h2><p>" + _e(", ".join(c.tools)) + "</p>")
    body.append(f"<h2>Before you start</h2><p>{_e(c.prerequisites)}</p>")
    if c.aligned_with:
        body.append(f"<p>Aligned with: {_e(c.aligned_with)}</p>")
    body.append("<h2>Modules</h2><ol>")
    for m in c.modules:
        body.append(f"<li><h3>{_e(m.title)}</h3><p>{_e(m.summary)}</p><ul>"
                    + "".join(f"<li>{_e(l.title)} ({l.minutes} min)</li>" for l in m.lessons)
                    + (f"<li>Hands-on lab: {_e(m.lab.title)}</li>" if m.lab else "") + "</ul></li>")
    body.append("</ol>")
    if c.project:
        body.append(f"<h2>Project: {_e(c.project.title)}</h2><p>{_e(c.project.brief)}</p>")
    if tracks := cat.tracks_of.get(c.slug):
        body.append("<h2>Part of</h2><ul>" + "".join(
            f'<li><a href="{paths.track(t.slug)}">{_e(t.title)} {_e(t.kind)}</a></li>' for t in tracks) + "</ul>")
    body += [f"<p><small>{_e(n)}</small></p>" for n in c.notices()]
    body.append(f'<p><a href="/signup?next={paths.course(c.slug)}">Start the course free</a></p>')
    hours = f"PT{max(1, round(c.hours))}H"
    course = {"@type": "Course", "@id": url(paths.course(c.slug)) + "#course", "name": c.title,
              "description": c.subtitle + " " + c.description.split("\n\n")[0], "url": url(paths.course(c.slug)),
              "provider": {"@id": url("/#organization")}, "educationalLevel": c.level.capitalize(),
              "inLanguage": "en", "teaches": c.outcomes, "keywords": ", ".join([*c.skills, *c.tools]),
              "coursePrerequisites": c.prerequisites, "timeRequired": hours,
              "educationalCredentialAwarded": "Certificate of completion",
              "offers": [{"@type": "Offer", "category": "Partially Free", "url": url(paths.course(c.slug))}],
              "hasCourseInstance": [{"@type": "CourseInstance", "courseMode": "Online", "courseWorkload": hours,
                                     "instructor": [{"@type": "Person", "name": t["persona"]["name"],
                                                     "url": url(paths.companion(t)),
                                                     "description": "AI Learning Companion"} for t in teachers]
                                     or [{"@id": url("/#organization")}]}],
              "syllabusSections": [{"@type": "Syllabus", "name": m.title, "description": m.summary}
                                   for m in c.modules]}
    return page("".join(body)), [organization(), website(),
                                 breadcrumbs(("Courses", "/courses"), (c.title, paths.course(c.slug))), course]


def track_page(t) -> tuple[str, list[dict]]:
    cat = catalog.data()
    listed = [cat.by_slug[s] for s in t.courses if s in cat.by_slug]
    body = [f"<h1>{_e(t.title)} {_e(t.kind)}</h1>", f"<p><strong>{_e(t.subtitle)}</strong></p>",
            f"<p>{_e(t.description)}</p>"]
    if t.roles:
        body.append("<h2>Prepares you for</h2><p>" + _e(", ".join(t.roles)) + "</p>")
    body.append(f"<h2>{len(listed)} courses in this {_e(t.kind)}</h2>" + _course_items(listed))
    items = {"@type": "ItemList", "name": f"{t.title} {t.kind}", "itemListElement": [
        {"@type": "ListItem", "position": i, "url": url(paths.course(c.slug))} for i, c in enumerate(listed, 1)]}
    return page("".join(body)), [organization(), website(),
                                 breadcrumbs(("Courses", "/courses"), (f"{t.title} {t.kind}", paths.track(t.slug))), items]


def guides_page() -> tuple[str, list[dict]]:
    cast = {c["id"]: c for c in _public_companions()}
    listed = [p for p in _public_posts(cast) if p["kind"]][:100]
    body = ["<h1>AI guides and cheat sheets</h1>",
            "<p>Tutorials and cheat sheets on AI by AI Learning Companions from around the world. Read one, then ask "
            "its author anything.</p><ul>"]
    body += [f'<li><a href="{paths.post(p)}">{_e(p["title"])}</a>'
             + (f": {_e(p['summary'])}" if p.get("summary") else "")
             + f' <small>by {_e(cast[p["companion_id"]]["persona"]["name"])}</small></li>' for p in listed]
    body.append("</ul>")
    return page("".join(body)), [organization(), website(), breadcrumbs(("Guides", "/guides"))]


def companions_page() -> tuple[str, list[dict]]:
    cast = _public_companions()
    body = ["<h1>AI Learning Companions</h1>",
            "<p>AI Learning Companions from around the world, each with their own face, voice and expertise. Chat "
            "with one, read their guides, take their courses, or design your own.</p><ul>"]
    for c in cast:
        p = c["persona"]
        what = p.get("tagline") or ", ".join(p.get("expertise") or [])
        body.append(f'<li><a href="{paths.companion(c)}">{_e(p["name"])}</a>'
                    + (f" ({_e(p['country'])})" if p.get("country") else "") + (f": {_e(what)}" if what else "") + "</li>")
    body.append("</ul>")
    return page("".join(body)), [organization(), website(), breadcrumbs(("AI Learning Companions", "/companions"))]


def companion_page(c: dict) -> tuple[str, list[dict]]:
    p = c["persona"]
    body = [f"<h1>{_e(p['name'])}, AI Learning Companion</h1>"]
    if p.get("tagline"):
        body.append(f"<p><strong>{_e(p['tagline'])}</strong></p>")
    facts = [x for x in (p.get("country"), ", ".join(p.get("expertise") or [])) if x]
    if facts:
        body.append(f"<p>{_e(' · '.join(facts))}</p>")
    body += [f"<p>{_e(t)}</p>" for t in (p.get("bio") or "").split("\n\n") if t.strip()]
    body.append(f"<p>{_e(p['name'])} is an AI: an AI Learning Companion on {SITE}. "
                f'<a href="/signup">Join free</a> to chat with {_e(p["name"].split()[0])} any time.</p>')
    if c["is_starter"] and (taught := catalog.taught_by(c["slug"])):
        body.append("<h2>Courses</h2>" + _course_items(taught))
    with db.connect() as conn:
        posts = [dict(r) for r in conn.execute(
            "SELECT id, slug, kind, title, summary FROM posts WHERE companion_id = ? ORDER BY created_at DESC, id DESC "
            "LIMIT 50", (c["id"],))]
    if posts:
        body.append("<h2>Guides and posts</h2><ul>" + "".join(
            f'<li><a href="{paths.post(x)}">{_e(x["title"])}</a>' + (f": {_e(x['summary'])}" if x.get("summary") else "")
            + "</li>" for x in posts) + "</ul>")
    profile = {"@type": "ProfilePage", "url": url(paths.companion(c)), "name": f"{p['name']}, AI Learning Companion",
               "mainEntity": {"@type": "Person", "name": p["name"], "description": p.get("tagline") or plain_bio(p),
                         "url": url(paths.companion(c)), "knowsAbout": p.get("expertise") or [],
                         "affiliation": {"@id": url("/#organization")}}}
    return page("".join(body)), [organization(), website(),
                                 breadcrumbs(("AI Learning Companions", "/companions"), (p["name"], paths.companion(c))),
                                 profile]


def plain_bio(p: dict) -> str:
    return (p.get("bio") or "")[:300]


def post_page(post: dict, by: dict) -> tuple[str, list[dict]]:
    """A guide or post as a visitor who isn't signed in sees it: the start of it, with a way to read on."""
    name = by["persona"]["name"]
    shown = billing.preview(post["body"])
    whole = shown == post["body"]
    body = [f"<article><h1>{_e(post['title'])}</h1>",
            f'<p>By <a href="{paths.companion(by)}">{_e(name)}</a>, AI Learning Companion'
            + (f" · {_e(post['topic'])}" if post.get("topic") else "")
            + (f' · <time datetime="{_e(_day(post["created_at"]))}">{_e(_human(post["created_at"]))}</time>'
               if _day(post.get("created_at")) else "") + "</p>"]
    if post.get("summary"):
        body.append(f"<p><strong>{_e(post['summary'])}</strong></p>")
    if post.get("graphic"):
        body.append(f'<img src="{_e(post["graphic"])}" alt="{_e(post["title"])}: infographic" width="1024" '
                    'height="1024" loading="lazy">')
    body.append(markdown(shown))
    if not whole:
        body.append(f'<p><a href="/signup?next={paths.post(post)}">Sign up free to read the whole guide</a></p>')
    body.append("</article>")
    section = ("Guides", "/guides") if post.get("kind") else (name, paths.companion(by))
    article = {"@type": "Article", "headline": post["title"][:110], "url": url(paths.post(post)),
               "mainEntityOfPage": url(paths.post(post)), "inLanguage": "en",
               "author": {"@type": "Person", "name": name, "url": url(paths.companion(by)),
                          "description": "AI Learning Companion"},
               "publisher": {"@id": url("/#organization")}}
    if post.get("summary"):
        article["description"] = post["summary"]
    if post.get("topic"):
        article["about"] = post["topic"]
    if _day(post.get("created_at")):
        article["datePublished"] = post["created_at"]
    if post.get("graphic"):
        article["image"] = url(post["graphic"])
    return page("".join(body)), [organization(), website(), breadcrumbs(section, (post["title"], paths.post(post))),
                                 article]


def simple(title: str, description: str | None, trail: tuple[str, str] | None = None) -> tuple[str, list[dict]]:
    body = f"<h1>{_e(title)}</h1>" + (f"<p>{_e(description)}</p>" if description else "")
    return page(body), [organization(), website()] + ([breadcrumbs(trail)] if trail else [])


def _human(when: str) -> str:
    d = date.fromisoformat(when[:10])
    return f"{d.day} {d:%B %Y}"


# --------------------------------------------------------------------------- markdown, for guides' text


def _inline(text: str) -> str:
    out, last = [], 0
    for m in re.finditer(r"`([^`]+)`|\*\*(.+?)\*\*|\[([^\]]+)\]\(([^)\s]+)\)|(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*", text):
        out.append(_e(text[last:m.start()]))
        if m[1] is not None:
            out.append(f"<code>{_e(m[1])}</code>")
        elif m[2] is not None:
            out.append(f"<strong>{_inline(m[2])}</strong>")
        elif m[3] is not None:
            href = m[4]
            ok = href.startswith(("https://", "http://", "/")) and not href.startswith("//")
            out.append(f'<a href="{_e(href)}"' + ("" if href.startswith("/") else ' rel="nofollow noopener"')
                       + f">{_inline(m[3])}</a>" if ok else _e(m[3]))
        else:
            out.append(f"<em>{_inline(m[5])}</em>")
        last = m.end()
    out.append(_e(text[last:]))
    return "".join(out)


def markdown(text: str) -> str:
    """Enough of Markdown for guides: headings, paragraphs, lists, quotes, code and tables (as text). Everything is
    escaped; links go only to web addresses."""
    out, lines, i = [], (text or "").replace("\r\n", "\n").split("\n"), 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.lstrip().startswith("```"):
            code, i = [], i + 1
            while i < len(lines) and not lines[i].lstrip().startswith("```"):
                code.append(lines[i])
                i += 1
            out.append(f"<pre><code>{_e(chr(10).join(code))}</code></pre>")
            i += 1
            continue
        if h := re.match(r"(#{1,6})\s+(.*)", line):
            level = min(6, len(h[1]) + 1)  # the page's title is its <h1>
            out.append(f"<h{level}>{_inline(h[2].strip().rstrip('#').strip())}</h{level}>")
            i += 1
            continue
        if re.match(r"\s*([-*+]|\d+[.)])\s+", line):
            ordered = bool(re.match(r"\s*\d+[.)]\s+", line))
            items = []
            while i < len(lines) and re.match(r"\s*([-*+]|\d+[.)])\s+", lines[i]):
                items.append(re.sub(r"\s*([-*+]|\d+[.)])\s+", "", lines[i], count=1))
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{_inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        if line.lstrip().startswith(">"):
            quote = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                quote.append(lines[i].lstrip()[1:].strip())
                i += 1
            out.append(f"<blockquote><p>{_inline(' '.join(quote))}</p></blockquote>")
            continue
        if line.lstrip().startswith("|"):
            rows = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells if c):
                    rows.append(cells)
                i += 1
            out.append("<table>" + "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in r) + "</tr>"
                                           for r in rows) + "</table>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"(#{1,6}\s|```|\s*([-*+]|\d+[.)])\s+|\s*[>|])",
                                                                    lines[i]):
            para.append(lines[i].strip())
            i += 1
        if para:
            out.append(f"<p>{_inline(' '.join(para))}</p>")
        else:
            i += 1
    return "".join(out)


# --------------------------------------------------------------------------- one address for each page


class Middleware:
    """Keeps search engines on the site's own address. Pages asked for at Render's onrender.com address move to the
    public one (AIET_PUBLIC_URL), so Google doesn't find each page twice; the app's data (/api/) and the health check
    stay where they are, for the iPhone app and Render. And the app's data is marked noindex: it's for the app, not
    for search results."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope.get("path", "")
        if path.startswith("/api/"):
            async def noindex(message):
                if message["type"] == "http.response.start":
                    message["headers"] = [*message.get("headers", []), (b"x-robots-tag", b"noindex")]
                await send(message)
            return await self.app(scope, receive, noindex)
        target = _move_to(scope)
        if target is None:
            return await self.app(scope, receive, send)
        await send({"type": "http.response.start", "status": 301,
                    "headers": [(b"location", target.encode("latin-1")), (b"content-length", b"0"),
                                (b"cache-control", b"public, max-age=3600")]})
        await send({"type": "http.response.body", "body": b""})


def _move_to(scope) -> str | None:
    if scope.get("method") not in ("GET", "HEAD") or scope.get("path") in ("/healthz",):
        return None
    public = settings.public_url
    if not public.startswith("https://") or ".onrender.com" in public:
        return None
    host = dict(scope.get("headers") or []).get(b"host", b"").decode("latin-1").split(":")[0].lower()
    if not host.endswith(".onrender.com"):
        return None
    query = scope.get("query_string", b"").decode("latin-1")
    return public + quote_path(scope.get("raw_path") or scope["path"].encode()) + (f"?{query}" if query else "")


def quote_path(raw: bytes) -> str:
    text = raw.decode("latin-1")
    return text if text.startswith("/") and not text.startswith("//") else "/"
