"""The prompt for writing an issue or a guide, and the schema of the answer."""

from __future__ import annotations

from .llm import array, obj, string
from .profile import render_style_guide

KINDS = ("issue", "guide")


def author_block(site) -> str:
    a = site["author"]
    lines = [f"Name: {a.get('name', '')}", f"Headline: {a.get('headline', '')}", f"About: {a.get('bio', '')}",
             f"Publication: {site['name']} ({site.absolute('')})", f"What readers are promised: {site['newsletter'].get('promise', '')}"]
    if site.get("never_mention"):
        lines.append("Never mention, anywhere: " + ", ".join(site["never_mention"]))
    return "\n".join(lines)


def catalogue(library) -> str:
    """The existing issues and guides, so new pieces can link to them with issue:<slug> / guide:<slug>."""
    rows = [f"- {item.ref}: {item.title} {('— ' + item.subtitle) if item.subtitle else ''}".rstrip()
            for item in library.all() if item.published]
    return "\n".join(rows) or "(nothing published yet)"


def write_system(profile: dict, site, library) -> str:
    return f"""You write newsletter issues and evergreen guides for one author's publication. You learned the shape of a good \
issue from another creator's examples: write new pieces in that shape, in the author's own voice, about the author's material.

<style_guide>
{render_style_guide(profile)}
</style_guide>

<author>
{author_block(site)}
</author>

<published>
{catalogue(library)}
</published>

How to work:
1. Find the piece's one promise in the brief's material and choose the kind of issue (or the guide blueprint) that fits.
2. Write the title and subtitle from the formulas, filled with the brief's specifics.
3. Write the body in Markdown with the marks from the style guide: the intro section, then the sections, each ending with a \
divider, then the close. Link to published pieces above where they help (issue:<slug>, guide:<slug>).
4. Put a ::screenshot[...] slot wherever the reader must find something on screen, describing exactly what to capture.
5. List what the author must check before publishing.

Rules that always apply:
- Facts, numbers, names, quotes, stories, prices, dates and links come only from the brief. If a sentence needs one the \
brief doesn't give, write around it or put [CHECK: what's needed].
- The examples belong to another creator: never their sentences, names, links, products or sign-offs.
- Never teach cheating on courses, quizzes, certifications or AI detectors, and never suggest fooling anyone."""


def brief_block(brief: dict) -> str:
    parts = [f"Topic: {brief.get('topic', '').strip()}"]
    for key, label in [("audience", "Audience"), ("goal", "What the reader should be able to do"), ("angle", "Angle"),
                       ("length", "Length")]:
        if brief.get(key):
            parts.append(f"{label}: {brief[key]}")
    notes = brief.get("notes")
    if notes:
        notes = "\n".join(f"- {n}" for n in notes) if isinstance(notes, list) else str(notes)
        parts.append(f"Notes and facts (the only facts you may use):\n{notes}")
    if brief.get("links"):
        parts.append("Links you may include:\n" + "\n".join(f"- {link}" for link in brief["links"]))
    return "\n".join(parts)


def write_user(brief: dict, kind: str, examples: list[dict]) -> str:
    shown = "\n\n".join(f'<example key="{e["key"]}" kind="{e.get("archetype", "")}">\n{e["text"][:9000]}\n</example>'
                        for e in examples) or "(no examples on this machine: rely on the style guide)"
    what = "a newsletter issue" if kind == "issue" else "an evergreen guide"
    return f"""<brief>
{brief_block(brief)}
</brief>

<examples note="for shape only; never reuse their words">
{shown}
</examples>

Write {what} for this brief."""


def write_schema(kind: str) -> dict:
    from .knowledge import ISSUE_ARCHETYPES

    archetypes = [a["id"] for a in ISSUE_ARCHETYPES] + (["guide"] if kind == "guide" else [])
    return obj({
        "archetype": string("The kind of issue you chose (or 'guide').", enum=archetypes),
        "slug": string("The page address: lowercase words joined by hyphens, at most 6 words."),
        "title": string("The title."),
        "subtitle": string("One line: who it's for or what they'll be able to do."),
        "description": string("One or two sentences for search results and link previews, under 155 characters."),
        "tags": array(string(), "Two to four lowercase topic tags."),
        "body": string("The whole body in Markdown with the style guide's marks. No # title line."),
        "fact_checks": array(string(), "What the author must check or add before publishing (claims, numbers, links, screenshots)."),
    })


def repair_user(issues: list) -> str:
    lines = "\n".join(f"- {i.message}" for i in issues)
    return f"""The draft has problems that must be fixed before it can be published:
{lines}

Return the same JSON with every problem fixed. Keep everything that was fine."""
