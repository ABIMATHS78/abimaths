"""Prompts for the three Claude calls: write the post, design its graphic, and classify a training graphic."""

from __future__ import annotations

import json

from .markup import plain
from .profile import render_voice_guide
from .schemas import array, obj, string
from .templates import TEMPLATES, get_template
from .themes import THEMES


def author_block(brand: dict) -> str:
    lines = [f"Name: {brand.get('name', '')}", f"Headline: {brand.get('headline', '')}"]
    if brand.get("publication"):
        lines.append(f"Publication: {brand['publication']}")
    lines.append(f"Website (the exact link to use in posts): {brand.get('url') or brand.get('site', '')}")
    cta = brand.get("cta", {})
    if cta.get("resource_line"):
        lines.append(f"Resource line you may use in closings: {cta['resource_line']}")
    if cta.get("share_line"):
        lines.append(f"Share line you may use in closings: {cta['share_line']}")
    if brand.get("never_mention"):
        lines.append("Never mention, in the post or the graphic: " + ", ".join(brand["never_mention"]))
    lines += [f"Voice note: {note}" for note in brand.get("voice_notes", [])]
    return "\n".join(lines)


def catalogue(names: list[str] | None = None) -> str:
    return "\n".join(f"- {t.name}: {t.description}" for t in TEMPLATES.values() if not names or t.name in names)


def write_system(profile: dict, brand: dict) -> str:
    """Stable across requests (cached): the trained voice guide, the author and the template catalogue."""
    return f"""You write LinkedIn posts for one author. You learned a house style from example posts by another creator: \
write new posts in that style's shape, in the author's own voice, about the author's material.

<voice_guide>
{render_voice_guide(profile)}
</voice_guide>

<author>
{author_block(brand)}
</author>

<graphic_templates>
{catalogue()}
</graphic_templates>

How to work:
1. Find the post's one point (the angle) in the brief's material.
2. Pick the hook formula that fits that material and fill it with the brief's real specifics.
3. Pick the body structure that suits the material: steps for a process, a link library for resources, old/new pairs for a fix, and so on.
4. Aim for the measured typical length unless the brief asks for something else. Never go over 2,900 characters.
5. Close with one closing from the guide. When a link is needed, use the author's website (the exact link above) or a link
from the brief.
6. Choose the graphic template that turns the post into a save-worthy image, and whether it should be still (png) or moving \
(gif for a drawing build-up, mp4 for kinetic text or a long build).

Rules that always apply:
- The post is plain text exactly as it will be pasted into LinkedIn: line breaks, no markdown, no hashtags unless asked.
- Facts, numbers, names, quotes, stories, prices, dates and links come only from the brief. If a sentence needs one the brief \
doesn't give, write around it or put [CHECK: what's needed].
- The examples belong to another creator. Borrow their structure and rhythm, never their sentences, facts, names, links or sign-offs.
- Never teach cheating on courses, quizzes or certifications, and never suggest fooling recruiters or anyone else."""


def _example_block(example: dict) -> str:
    return (
        f'<example key="{example["key"]}" template="{example.get("template", "")}" hook="{example.get("hook_type", "")}" '
        f'structure="{example.get("structure", "")}" closing="{example.get("cta_type", "")}">\n'
        f"<post>\n{example.get('text', '')}\n</post>\n"
        f"<graphic>{example.get('graphic_notes', '')}</graphic>\n</example>"
    )


def brief_block(brief: dict) -> str:
    parts = [f"Topic: {brief.get('topic', '').strip()}"]
    for key, label in [("audience", "Audience"), ("goal", "What the reader should think or do"),
                       ("angle", "Angle the author wants"), ("length", "Length")]:
        if brief.get(key):
            parts.append(f"{label}: {brief[key]}")
    notes = brief.get("notes")
    if notes:
        notes = "\n".join(f"- {n}" for n in notes) if isinstance(notes, list) else str(notes)
        parts.append(f"Notes and facts (the only facts you may use):\n{notes}")
    if brief.get("links"):
        parts.append("Links you may include:\n" + "\n".join(f"- {link}" for link in brief["links"]))
    if brief.get("cta"):
        parts.append(f"Closing the author wants: {brief['cta']}")
    return "\n".join(parts)


def write_user(brief: dict, examples: list[dict], template: str | None = None, motion: str | None = None) -> str:
    asks = []
    if template:
        asks.append(f"Use the '{template}' graphic template.")
    if motion and motion != "auto":
        asks.append(f"The graphic will be a {motion}.")
    examples_xml = "\n\n".join(_example_block(e) for e in examples) or "(no examples available: rely on the voice guide)"
    return f"""<brief>
{brief_block(brief)}
</brief>

<examples>
{examples_xml}
</examples>

Write the post for this brief. {' '.join(asks)}""".strip()


def write_schema(template: str | None = None) -> dict:
    names = [template] if template else list(TEMPLATES)
    return obj({
        "angle": string("The post's one point, in one sentence."),
        "hook_type": string("The hook formula id you used from the guide, or 'custom'."),
        "structure": string("The body structure id you used from the guide, or 'custom'."),
        "post": string("The whole post, ready to paste: plain text with \\n line breaks, no markdown."),
        "alternative_hooks": array(string("An alternative opening of one or two lines."), "Two other openings."),
        "template": string("The graphic template that fits best.", enum=names),
        "motion": string("png (still), gif or mp4.", enum=["png", "gif", "mp4"]),
        "graphic_brief": string("What the graphic should say: its headline and the condensed content, in a few lines."),
        "alt_text": string("Alt text describing the graphic for screen readers, one or two sentences."),
        "fact_checks": array(string(), "Claims, numbers or links the author must check before posting. Empty if none."),
    }, ["angle", "hook_type", "structure", "post", "alternative_hooks", "template", "motion", "graphic_brief", "alt_text", "fact_checks"])


def repair_user(issues: list) -> str:
    lines = "\n".join(f"- {i.message}" for i in issues)
    return f"""The post has problems that must be fixed before it can be published:
{lines}

Return the same JSON with the post corrected. Keep everything that was fine."""


def design_system(profile: dict, brand: dict) -> str:
    rules = "\n".join(f"- {r}" for r in profile.get("visual", {}).get("graphic_rules", []))
    return f"""You turn a finished LinkedIn post into the data for one graphic template. The engine draws the template, \
sizes the text and adds the footer ("{brand.get('footer', {}).get('left', '')}" and "{brand.get('footer', {}).get('right', '')}"); \
you only fill the template's fields.

Graphic rules:
{rules}
- Every word on the graphic comes from the post or restates it. Add no new facts, numbers, names or links.
- Respect the counts in the field descriptions. Text is sized automatically, so too much text only makes it small.
- Every field must be present. For a field this graphic doesn't use, give an empty string, an empty list, false, 0, \
or the first choice listed.
- Text markup inside fields: *word* accent colour, **word** bold, ==word== highlight, [[word]] accent pill, __word__ underline, `code` chip. \
Use it sparingly: one accent per headline.
- Never put the author credit, a site footer or the example creator's name inside the content.{_never(brand)}"""


def _never(brand: dict) -> str:
    names = brand.get("never_mention") or []
    return f"\n- Never mention: {', '.join(names)}." if names else ""


def design_user(post: str, graphic_brief: str, template_name: str) -> str:
    template = get_template(template_name)
    return f"""<template name="{template.name}">
{template.description}
Here is valid data for this template on a different topic, to show the shape and the amount of text that fits:
{json.dumps(template.sample, indent=1, ensure_ascii=False)}
</template>

<post>
{post}
</post>

<graphic_brief>
{graphic_brief}
</graphic_brief>

Fill the '{template.name}' template for this post."""


ANNOTATE_SCHEMA = obj({
    "archetype": string("What kind of graphic this is, in a word or two (e.g. cheatsheet, link list, decision tree)."),
    "template": string("The engine template that would reproduce it best.", enum=list(TEMPLATES)),
    "theme": string("The closest colour theme.", enum=list(THEMES)),
    "hook_type": string("The hook formula id from the list, or 'custom'."),
    "structure": string("The body structure id from the list, or 'custom'."),
    "cta_type": string("How the post closes, in two to four words, snake_case."),
    "graphic_notes": string("Two or three sentences describing the layout, type and colours, in your own words."),
    "content_warning": string("Empty, or one sentence if the post teaches something the engine must not copy (cheating, unverifiable numbers)."),
}, ["archetype", "template", "theme", "hook_type", "structure", "cta_type", "graphic_notes", "content_warning"])


def annotate_system(profile_voice: dict | None = None) -> str:
    hooks = ", ".join(h["id"] for h in (profile_voice or {}).get("hook_formulas", []))
    structures = ", ".join(s["id"] for s in (profile_voice or {}).get("structures", []))
    return f"""You classify example LinkedIn posts (text plus graphic) so a content engine can learn from them.
Templates the engine can draw:
{catalogue()}
Colour themes: {', '.join(THEMES)}.
Hook formula ids: {hooks or 'custom'}.
Body structure ids: {structures or 'custom'}.
Describe what you see in your own words; don't copy the post's sentences."""


def annotate_user_text(text: str) -> str:
    return f"<post>\n{plain(text)}\n</post>\n\nClassify this post and its graphic (the image above)."
