"""Turn the measurements and the hand-written knowledge into the trained style profile and voice guide."""

from __future__ import annotations

import datetime as dt
import json
import re
from collections import Counter

from . import knowledge
from .analyze import URL_RE, aggregate
from .media import shape_name, summarize_media
from .paths import PROFILE_PATH, VOICE_GUIDE_PATH

COMMON_DOMAINS = {
    "claude.ai", "claude.com", "anthropic.com", "github.com", "google.com", "chatgpt.com", "openai.com",
    "linkedin.com", "lnkd.in", "vercel.com", "youtube.com", "netflix.com", "tiktok.com", "notion.so", "slack.com",
    "miro.com", "wikipedia.org", "microsoft.com", "apple.com", "x.com", "twitter.com", "medium.com",
}


def training_domains(texts: list[str]) -> list[str]:
    """Sites the training posts link to that aren't everyday platforms: likely the creator's own."""
    found = Counter()
    for text in texts:
        for raw in URL_RE.findall(text):
            host = re.sub(r"^https?://", "", raw.lower()).split("/")[0].removeprefix("www.")
            if host and host not in COMMON_DOMAINS:
                found[host] += 1
    return [host for host, _ in found.most_common()]


def build_profile(examples, analyses, medias, annotations) -> dict:
    notes = annotations.get("posts", {})
    noted = [notes.get(e.key, {}) for e in examples]
    count = lambda field: Counter(n.get(field) for n in noted if n.get(field))  # noqa: E731
    hook_counts, structure_counts, cta_counts = count("hook_type"), count("structure"), count("cta_type")
    template_counts = count("template")
    media_by_file = {m["file"]: m for m in medias}

    examples_index = []
    for example, analysis, note in zip(examples, analyses, noted):
        media = media_by_file.get(example.media_file.name) if example.media_file else None
        examples_index.append({
            "key": example.key,
            "template": note.get("template", "unknown"),
            "style": note.get("style", ""),
            "theme": note.get("theme", ""),
            "hook_type": note.get("hook_type", ""),
            "structure": note.get("structure", ""),
            "cta_type": note.get("cta_type", ""),
            "chars": analysis["chars"],
            "media_kind": media.get("kind", "") if media else "",
            "shape": shape_name(media.get("aspect", 0)) if media else "",
        })

    voice, visual = knowledge_fields(hook_counts)
    return {
        "version": 1,
        "trained_at": dt.date.today().isoformat(),
        "source": {
            "posts": len(examples),
            "with_graphics": sum(1 for e in examples if e.media_file),
            "annotated": sum(1 for n in noted if n),
        },
        "voice": {
            **voice,
            "structure_counts": dict(structure_counts.most_common()),
            "cta_counts": dict(cta_counts.most_common()),
        },
        "stats": aggregate(analyses),
        "visual": {
            **summarize_media(medias),
            "templates": dict(template_counts.most_common()),
            **visual,
        },
        "examples_index": examples_index,
        "guardrails": {
            "creator_markers": annotations.get("creator_markers", []),
            "training_domains": training_domains([e.text for e in examples]),
            "content_warnings": {e.key: n["content_warning"] for e, n in zip(examples, noted) if n.get("content_warning")},
        },
    }


def knowledge_fields(seen: dict) -> tuple[dict, dict]:
    """The hand-written parts of the profile (knowledge.py), with hooks ordered by how often the examples used them."""
    hooks = sorted(knowledge.HOOK_FORMULAS, key=lambda h: -seen.get(h["id"], 0))
    voice = {
        "principles": knowledge.VOICE_PRINCIPLES,
        "hook_formulas": [{**h, "seen_in": seen.get(h["id"], 0)} for h in hooks],
        "structures": knowledge.STRUCTURES,
        "cta_formulas": knowledge.CTA_FORMULAS,
        "formatting": knowledge.FORMATTING,
        "avoid_phrases": knowledge.AVOID_PHRASES,
        "content_rules": knowledge.CONTENT_RULES,
    }
    return voice, {"graphic_rules": knowledge.GRAPHIC_RULES, "archetype_guide": knowledge.ARCHETYPES}


def load_profile() -> dict:
    """The trained profile: the measurements saved by `lpe train`, with the hand-written knowledge read fresh from
    knowledge.py, so edits there apply without retraining (and without the training files)."""
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"No trained profile at {PROFILE_PATH}. Run: python -m lpe train --drive <folder link>")
    profile = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    seen = {h["id"]: h.get("seen_in", 0) for h in profile.get("voice", {}).get("hook_formulas", [])}
    voice, visual = knowledge_fields(seen)
    profile.setdefault("voice", {}).update(voice)
    profile.setdefault("visual", {}).update(visual)
    return profile


def _fmt_dist(d: dict, unit: str = "") -> str:
    return f"{d['median']:,.0f}{unit} typical (most between {d['p10']:,.0f} and {d['p90']:,.0f})"


def render_voice_guide(profile: dict) -> str:
    """The trained style as a readable brief. It is Claude's system prompt and the Claude Code skill's reference."""
    v, s, vis = profile["voice"], profile["stats"], profile["visual"]
    pct = lambda x: f"{round(x * 100)}%"  # noqa: E731
    lines = [
        "# LinkedIn voice guide",
        "",
        f"Learned from {profile['source']['posts']} example posts and {profile['source']['with_graphics']} graphics "
        f"(trained {profile['trained_at']}). Use it to write new posts in the same shape, in the author's own voice.",
        "",
        "## Voice",
        *[f"- {p}" for p in v["principles"]],
        "",
        "## Measured shape of the example posts",
        f"- Length: {_fmt_dist(s['chars'], ' characters')}. LinkedIn's limit is 3,000.",
        f"- Lines: {_fmt_dist(s['lines'])} non-empty lines; average line {s['avg_line_chars']['median']:.0f} characters; "
        f"{pct(s['short_line_share']['median'])} of lines are 60 characters or less.",
        f"- Hook (first one or two lines): {_fmt_dist(s['hook_chars'], ' characters')}. LinkedIn cuts off around "
        "210 characters on desktop and 140 on phones, so the promise must land before that.",
        f"- {pct(s['share_hook_ends_with_colon'])} of hooks end with a colon that opens the list; "
        f"{pct(s['share_hook_has_number'])} contain a number; {pct(s['share_hook_has_caps_word'])} put one word in CAPS; "
        f"{pct(s['share_hook_has_aside'])} have a parenthetical aside.",
        f"- {pct(s['share_with_numbered_list'])} use a numbered or 'Step N' list; {pct(s['share_with_glyph_bullets'])} use glyph bullets "
        f"(most used: {', '.join(list(s['bullet_glyphs'])[:6])}); {pct(s['share_with_quoted_prompt'])} include a prompt to copy in quotes; "
        f"{pct(s['share_with_arrows'])} use → menu paths; {pct(s['share_using_ampersand'])} use '&'.",
        f"- They talk to 'you' about {s['you_vs_i']['median']:.1f} times as often as about 'I'. "
        f"{pct(s['share_with_emoji'])} use emoji (usually {s['emoji_per_post']['median']:.0f} per post); "
        f"{pct(s['share_with_hashtags'])} use hashtags.",
        "",
        "## Hooks that worked (pick one, fill it with the brief's real specifics)",
        *[
            f"- `{h['id']}`: {h['formula'].replace(chr(10), ' / ')}"
            + (f" (only if {h['only_if']})" if h.get("only_if") else "")
            + (f" [seen {h['seen_in']}x]" if h.get("seen_in") else "")
            for h in v["hook_formulas"]
        ],
        "",
        "## Body structures",
        *[f"- `{st['id']}`: {st['shape']}" for st in v["structures"]],
        "",
        "## Closings",
        *[
            f"- `{c['id']}`: {c['formula'].replace(chr(10), ' / ')}" + (f" (only if {c['only_if']})" if c.get("only_if") else "")
            for c in v["cta_formulas"]
        ],
        "",
        "## Formatting",
        *[f"- {f}" for f in v["formatting"]],
        "",
        "## Content rules (always)",
        *[f"- {r}" for r in v["content_rules"]],
        "",
        "## Words and habits that make a post read as AI-written: never use them",
        "- " + ", ".join(f'"{p}"' for p in v["avoid_phrases"]) + ".",
        "- No em dashes (—). Use a colon, a comma or a new line.",
        "- No 'It's not X, it's Y' sentences, no 'X quietly does Y', no grand pronouncements.",
        "",
        "## Graphics",
        *[f"- {g}" for g in vis["graphic_rules"]],
        f"- Shapes seen: {', '.join(f'{k} ({n})' for k, n in vis['shapes'].items())}. "
        f"{pct(vis['dark_share'])} of the graphics are dark.",
        "- Recurring backgrounds and inks: " + ", ".join(c["hex"] for c in vis["learned_neutrals"][:6]) + ".",
        "- Recurring accent colours: " + ", ".join(c["hex"] for c in vis["learned_accents"][:8]) + ".",
        "",
        "### Which template fits",
        *[f"- `{name}`: {desc}" for name, desc in vis["archetype_guide"].items()],
        "",
    ]
    return "\n".join(lines)


def write_profile(profile: dict) -> None:
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    VOICE_GUIDE_PATH.write_text(render_voice_guide(profile), encoding="utf-8")
