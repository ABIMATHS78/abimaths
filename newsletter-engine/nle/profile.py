"""The trained style profile (measurements) and the style guide Claude writes from (measurements + knowledge.py)."""

from __future__ import annotations

import datetime as dt
import json

from . import knowledge
from .paths import PROFILE_PATH, STYLE_GUIDE_PATH


def knowledge_fields() -> dict:
    return {
        "principles": knowledge.PRINCIPLES, "structure": knowledge.STRUCTURE, "archetypes": knowledge.ISSUE_ARCHETYPES,
        "guide_blueprint": knowledge.GUIDE_BLUEPRINT, "title_formulas": knowledge.TITLE_FORMULAS,
        "subtitle_formulas": knowledge.SUBTITLE_FORMULAS, "calls_to_action": knowledge.CALLS_TO_ACTION,
        "marks": knowledge.MARKS, "content_rules": knowledge.CONTENT_RULES, "avoid_phrases": knowledge.AVOID_PHRASES,
    }


def build_profile(stats: dict, archive: dict, annotations: dict) -> dict:
    return {
        "version": 1,
        "trained_at": dt.date.today().isoformat(),
        "stats": stats,
        "archive": archive,
        "archetype_counts": _count(v.get("archetype") for v in annotations.get("issues", {}).values()),
        "guardrails": {"creator_markers": annotations.get("creator_markers", []),
                       "training_domains": annotations.get("training_domains", [])},
        **knowledge_fields(),
    }


def _count(values) -> dict:
    counts: dict = {}
    for value in values:
        if value:
            counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))


def load_profile() -> dict:
    """The saved measurements, with the hand-written knowledge read fresh from knowledge.py."""
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"No trained profile at {PROFILE_PATH}. Run: python -m nle train")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    profile.update(knowledge_fields())
    return profile


def _d(d: dict, unit: str = "") -> str:
    if not d:
        return "n/a"
    fmt = (lambda v: f"{v:.0%}") if unit == "%" else (lambda v: f"{v:,.0f}")
    return f"{fmt(d['median'])} typical (most between {fmt(d['p10'])} and {fmt(d['p90'])})"


def render_style_guide(profile: dict) -> str:
    s, a = profile.get("stats", {}), profile.get("archive", {})
    lines = [
        "# Style guide for issues and guides", "",
        f"Learned from {a.get('issues', 0)} example issues ({s.get('complete_issues', 0)} read in full) "
        f"(trained {profile.get('trained_at', '')}). Use it for the shape of new issues and guides, written in the "
        "author's own voice about the author's own material.", "",
        "## Principles", *[f"- {p}" for p in profile["principles"]], "",
        "## Structure of an issue", *[f"- {p}" for p in profile["structure"]], "",
        "## Measured shape of the examples",
        f"- Length: {_d(s.get('words', {}))} words.",
        f"- Paragraphs: {_d(s.get('paragraphs', {}))} per issue, {_d(s.get('paragraph_words', {}))} words each; "
        f"{_d(s.get('short_paragraph_share', {}), '%')} of paragraphs are 12 words or fewer.",
        f"- Sections (split by dividers): {_d(s.get('sections', {}))}; ## headings {_d(s.get('headings', {}))}; "
        f"### headings {_d(s.get('subheadings', {}))}.",
        f"- Per issue: screenshots {_d(s.get('images', {}))}, prompt boxes {_d(s.get('prompts', {}))}, "
        f"lists {_d(s.get('lists', {}))}, buttons {_d(s.get('buttons', {}))}, quotes {_d(s.get('quotes', {}))}.",
        f"- Opening: {_d(s.get('opening_paragraphs', {}))} paragraphs before the first divider; "
        f"{s.get('opening_list_share', 0):.0%} include a list, {s.get('opening_button_share', 0):.0%} a button.",
        f"- Titles: {_d(a.get('title_words', {}))} words; {a.get('title_period_share', 0):.0%} end with a full stop; "
        f"{a.get('title_number_share', 0):.0%} contain a number.",
        f"- Subtitles: {_d(a.get('subtitle_words', {}))} words; {a.get('subtitle_colon_share', 0):.0%} end with a colon "
        "that leads into the issue.",
        f"- Every issue has a cover image ({a.get('with_cover_share', 0):.0%}). The site draws a title card when there is none.", "",
        "## Kinds of issue (pick one)",
        *[f"- `{x['id']}` ({x['name']}): for when {x['use']}. Sections: " + "; ".join(x["sections"]) + "."
          for x in profile["archetypes"]], "",
        "## Guides (evergreen)", *[f"- {p}" for p in profile["guide_blueprint"]], "",
        "## Titles and subtitles (fill with the brief's specifics)",
        *[f"- Title: {t}" for t in profile["title_formulas"]], *[f"- Subtitle: {t}" for t in profile["subtitle_formulas"]], "",
        "## Calls to action", *[f"- {k}: {v}" for k, v in profile["calls_to_action"].items()], "",
        "## Marks you can use in the body", *[f"- {m}" for m in profile["marks"]], "",
        "## Content rules (always)", *[f"- {r}" for r in profile["content_rules"]], "",
        "## Words that make writing read as AI-generated: never use them",
        "- " + ", ".join(f'"{p}"' for p in profile["avoid_phrases"]) + ".",
        "- No em dashes (—): use a colon, a comma or a new sentence. No 'It's not X, it's Y' sentences.", "",
    ]
    return "\n".join(lines)


def write_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    saved = {k: v for k, v in profile.items() if k not in knowledge_fields()}
    PROFILE_PATH.write_text(json.dumps(saved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    STYLE_GUIDE_PATH.write_text(render_style_guide(profile), encoding="utf-8")
