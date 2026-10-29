"""What the AI Learning Companions know about AI Every Time's courses, in chat.

Every companion sees the whole catalogue (each course with its level, length, lead instructor, skills and the
address of its page), the courses it teaches itself, and what the learner's plan opens. So it can recommend its
own courses, point learners to the courses other companions teach, and, when no course fits, offer to design a
learning path on the topic. Replies link a course as [Title](/courses/slug), which the chat shows as a link to
its page, and a path to design as [Design a path on …](/learn?design=topic-words&with=companion-slug), which
opens the Learning paths page with the topic filled in.
"""

from __future__ import annotations

import re
from functools import cache

from app import billing, catalog, paths, plans
from app.curriculum import PATHS
from app.starter_companions import STARTERS

RULES = """# Courses on AI Every Time, and recommending them
You know every course on AI Every Time: they're all listed below. The ones you teach yourself are under "Courses you teach", after your persona.

When a learner asks what to study, whether there's a course on something, or would clearly be helped by one:
- Recommend the courses on the list that fit best: your own first when they fit, then any other companion's. For a course you don't teach, say who does, so they know whom they'll learn with there.
- Link every course you name the way the list does, [Course title](/courses/slug), so the learner can open it from this chat. Name only courses on the list, with their links exactly as given; never make up a course, a link or a detail.
- Give two or three good matches rather than a long list, each with why it fits, its level and roughly how long it takes. Name a Professional Certificate or Specialization when they want a whole series of courses.
- Be clear about what their plan opens; it's under "About this learner".
- When nothing on the list covers what they want, say so plainly, then offer two things: to design a learning path on exactly that topic for them (link it as "Courses you teach" shows), and to start teaching it right here in the chat.
- Never say you can't see the courses: they're all below."""

LEVELS = {"beginner": 0, "intermediate": 1, "advanced": 2}


def _names(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _teacher(slug: str) -> str:
    s = STARTERS.get(slug)
    return s.persona.name if s else slug


def _hours(c: catalog.Course) -> str:
    h = c.hours
    return f"{h:g} h"


def index() -> str:
    """Every course by subject, the series they make up, and the free learning paths: the same for every companion
    and learner, so it sits in the cached part of the prompt."""
    return _index(catalog.data())


@cache
def _index(data: catalog.Catalogue) -> str:  # written once for each catalogue the app reads
    lines = [f"# Every course, by subject\n{len(data.courses)} courses. Each has its own page with its modules, labs, "
             "project and study plan; anyone can take the first module of any course free."]
    for category in catalog.CATEGORIES:
        courses = sorted((c for c in data.courses if c.category == category), key=lambda c: (LEVELS[c.level], c.title))
        if not courses:
            continue
        lines.append(f"\n## {category}")
        for c in courses:
            kind = f"{c.level} guided project" if c.kind == "guided-project" else c.level
            lines.append(f"- [{c.title}]({paths.course(c.slug)}): {kind}, {_hours(c)}, taught by "
                         f"{_teacher(c.instructors[0])}. Skills: {', '.join(c.skills[:6])}.")
    if data.tracks:
        lines.append("\n# Professional Certificates and Specializations\n"
                     "Series of courses, each with its own certificate.")
        for t in data.tracks:
            lines.append(f"- [{t.title}]({paths.track(t.slug)}): {t.kind}, {len(t.courses)} courses. {t.subtitle}")
    lines.append(f"\n# Free learning paths\nShort paths of lessons on the [Learning paths]({paths.LEARN}) page, taught by the "
                 "companion the learner chooses: " + "; ".join(f"{p.title} ({p.level})" for p in PATHS) + ".")
    return "\n".join(lines)


def taught(companion: dict) -> str:
    """The courses a companion teaches, and how it links a learning path for the learner to design, for its part of
    the prompt. A learner's own companion teaches none."""
    slug = companion["slug"] if companion.get("is_starter") else ""
    lines = ["# Courses you teach"]
    mine = catalog.taught_by(slug) if slug else []
    for c in mine:
        role = "you lead it" if c.instructors[0] == slug else "you teach part of it"
        others = [_teacher(s) for s in c.instructors if s != slug]
        lines.append(f"- [{c.title}]({paths.course(c.slug)}): {role}{', with ' + _names(others) if others else ''}.")
    if not mine:
        lines.append("None: you don't teach any of the platform's courses, so recommend the ones other companions "
                     "teach.")
    link = f"{paths.LEARN}?design=<topic-words>&with={companion['slug']}"
    lines.append("To offer a learning path designed for the learner, on a topic no course covers, link it like this, "
                 "with the topic's words in lowercase joined by hyphens. It opens the page where they confirm it, and "
                 f"you design it: [Design a path on <topic> with me]({link})")
    return "\n".join(lines)


def access(user: dict) -> str:
    """What the learner's plan opens, as their companion is told it."""
    full, designed = billing.unlocked(user, "courses"), billing.unlocked(user, "custom_paths")
    if full and designed:
        return "What their plan opens: every course in full, and learning paths designed for them."
    parts = [f"Their plan: {plans.plan(billing.tier(user)).name}."]
    parts.append("Every course is included in full." if full else
                 "They can take the first module of any course free; whole courses (every module, the labs, the "
                 f"project and the certificate) come with the {plans.lowest_with('courses').name} plan and above.")
    parts.append("Learning paths designed for them are included." if designed else
                 f"Learning paths designed for them come with the {plans.lowest_with('custom_paths').name} plan and "
                 "above, so say so when you offer one; the free learning paths, and learning with you here, are open "
                 "to them now.")
    return " ".join(parts)


# Words a question about courses has that say nothing about what it's on.
STOP = frozenset("""about any are can course courses does for from get have how into just learn learning like
looking maybe more need offer one please related some take teach that the there this want what which with would you
your""".split())


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9+#]+", text.lower()) if len(w) > 2 and w not in STOP}


def topic(text: str, most: int = 6) -> str:
    """What a learner asked about, as the words of a path link: "regression-machine-data-analytics"."""
    seen: list[str] = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if len(w) > 2 and w not in STOP and w not in seen:
            seen.append(w)
    return "-".join(seen[:most])


def _same(w: str, h: str) -> bool:
    """Whether a learner's word is a course's: the same, the start of it from five letters on ("regress",
    "regression"), or it with up to three letters more ("analytics", "analytic"), but not "underwater" for "under"."""
    return w == h or (len(w) >= 5 and h.startswith(w)) or (len(h) >= 5 and w.startswith(h) and len(w) - len(h) <= 3)


def _hits(wanted: set[str], have: set[str]) -> int:
    """How many of the wanted words a course's text has."""
    return sum(1 for w in wanted if any(_same(w, h) for h in have))


def search(text: str, limit: int = 3) -> list[catalog.Course]:
    """The courses that best match what a learner wrote, by the words they share with each course's title and
    skills (most), subject, and subtitle and outcomes (least); one word only in a course's description isn't
    enough. For demo-mode replies, which have no Claude to read the catalogue."""
    wanted = _words(text)
    if not wanted:
        return []
    scored = []
    for c in catalog.data().courses:
        score = (3 * _hits(wanted, _words(c.title + " " + " ".join(c.skills)))
                 + 2 * _hits(wanted, _words(c.category))
                 + _hits(wanted, _words(c.subtitle + " " + " ".join(c.outcomes))))
        if score >= 2:
            scored.append((-score, LEVELS[c.level], c.title, c))
    return [c for *_, c in sorted(scored)[:limit]]
