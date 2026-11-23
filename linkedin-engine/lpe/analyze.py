"""Measure the writing in the training posts: length, rhythm, lists, hooks and calls to action."""

from __future__ import annotations

import re
import statistics
from collections import Counter

FOLD_CHARS = 210  # roughly what LinkedIn shows on desktop before "...see more" (about 140 on phones)

NUMBERED_RE = re.compile(r"^\s*\d{1,3}[.)]\s+\S")
LABELLED_RE = re.compile(r"^\s*(step|lesson|level|min|mins|day|week|tip|rule|part|stage)\s*\d+\b", re.I)
GLYPH_RE = re.compile(r"^\s*(✦|☑️|☑|✅|✔️|✔|→|➜|👉|>|•|▪️|▪|-|\*|⚠️|♻️|🔹|❌|⛔)\s*")
SEPARATOR_RE = re.compile(r"^\s*([-_=*~]{3,}|—{2,})\s*$")
URL_RE = re.compile(r"\b(?:https?://)?(?:[a-z0-9-]+\.)+(?:com|org|io|ai|co|net|guide|free|de|in|app|dev|so|me)(?:/[^\s)\]]*)?", re.I)
CAPS_RE = re.compile(r"\b[A-Z][A-Z']{2,}\b")
ASIDE_RE = re.compile(r"\(([^()\n]{2,60})\)")
# Real emoji only: pictographs, or a symbol shown as emoji (followed by U+FE0F). Plain ☑ → ✦ are typography.
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF]|[←-⯿]️|[✅❌]")
HASHTAG_RE = re.compile(r"(?<![\w&])#[A-Za-z][\w]+")
QUOTE_LINE_RE = re.compile(r'^\s*["“].{20,}')
OLD_NEW_RE = re.compile(r"^\s*(old|new|bad|better|best|before|after)\s*(prompt)?\s*:", re.I)
COMMON_ACRONYMS = {
    "AI", "API", "MCP", "HTML", "CSV", "PDF", "PNG", "SAAS", "IPO", "URL", "NDA", "CRM", "DCF", "SEC", "USA",
    "UK", "CEO", "CTO", "HR", "SEO", "UX", "UI", "GPT", "LLM", "JSON", "XLSX", "DOCX", "OTP", "KPI", "ROI",
    "Q1", "Q2", "Q3", "Q4", "FYI", "PS", "ASD", "STE", "NCR", "NCRS", "MBA", "OK", "IT", "AND", "B2B", "SQL",
}

CTA_PATTERNS = {
    "save": re.compile(r"\bsave (this|it|the|my)\b|\bbookmark\b|three dots", re.I),
    "repost_or_share": re.compile(r"repost|♻️|\bshare (this|it)\b|send (it|this) to|with a friend|your team", re.I),
    "newsletter_or_site": re.compile(r"newsletter|subscribe|free guide|\bguide\b.*(https?://|\.[a-z]{2,})", re.I),
    "ps_line": re.compile(r"^\s*(PS|P\.S\.)\b", re.I | re.M),
    "comment_prompt": re.compile(r"\bcomment\b|\bin the comments\b|\breply\b", re.I),
    "follow": re.compile(r"\bfollow (me|for)\b", re.I),
    "download": re.compile(r"\bdownload\b", re.I),
}


def paragraphs(text: str) -> list[str]:
    return [block.strip() for block in re.split(r"\n\s*\n", text) if block.strip()]


def hook_of(text: str) -> str:
    """The opening the reader sees first: the first paragraph, up to two lines."""
    blocks = paragraphs(text)
    if not blocks:
        return ""
    lines = [line.strip() for line in blocks[0].splitlines() if line.strip()]
    return "\n".join(lines[:2])


def analyze_text(text: str) -> dict:
    lines = text.splitlines()
    nonblank = [line for line in lines if line.strip()]
    lengths = [len(line.strip()) for line in nonblank] or [0]
    words = re.findall(r"\b[\w'’]+\b", text)
    lower = text.lower()
    hook = hook_of(text)
    you = len(re.findall(r"\b(you|your|you're|yourself|you'll|you've)\b", lower))
    me = len(re.findall(r"\b(i|me|my|i'm|i've|i'd|mine)\b", lower))
    glyphs = Counter()
    for line in nonblank:
        match = GLYPH_RE.match(line)
        if match:
            glyphs[match.group(1)] += 1
    caps = [w for w in CAPS_RE.findall(text) if w.upper().replace("'", "") not in COMMON_ACRONYMS]
    return {
        "chars": len(text),
        "words": len(words),
        "lines": len(nonblank),
        "paragraphs": len(paragraphs(text)),
        "avg_line_chars": round(statistics.mean(lengths), 1),
        "max_line_chars": max(lengths),
        "short_line_share": round(sum(1 for n in lengths if n <= 60) / len(lengths), 2),
        "hook": hook,
        "hook_chars": len(hook),
        "hook_ends_with_colon": hook.rstrip().endswith(":"),
        "hook_has_number": bool(re.search(r"\d", hook)),
        "hook_has_caps_word": bool([w for w in CAPS_RE.findall(hook) if w not in COMMON_ACRONYMS]),
        "hook_has_aside": bool(ASIDE_RE.search(hook)),
        "fold_text": text[:FOLD_CHARS],
        "numbered_lines": sum(1 for line in nonblank if NUMBERED_RE.match(line)),
        "labelled_lines": sum(1 for line in nonblank if LABELLED_RE.match(line)),
        "glyph_lines": dict(glyphs),
        "separators": sum(1 for line in nonblank if SEPARATOR_RE.match(line)),
        "arrows": text.count("→"),
        "ampersands": len(re.findall(r"\s&\s", text)),
        "caps_words": caps[:12],
        "caps_word_count": len(caps),
        "asides": len(ASIDE_RE.findall(text)),
        "questions": sum(1 for line in nonblank if line.rstrip().endswith("?")),
        "emoji": len(EMOJI_RE.findall(text)),
        "hashtags": len(HASHTAG_RE.findall(text)),
        "links": len(URL_RE.findall(text)),
        "quoted_prompts": sum(1 for line in nonblank if QUOTE_LINE_RE.match(line)),
        "old_new_pairs": sum(1 for line in nonblank if OLD_NEW_RE.match(line)),
        "you_vs_i": round(you / max(1, me), 2),
        "ctas": _ctas(text),
    }


def _ctas(text: str) -> list[str]:
    tail = text[-700:]
    found = {name for name, pattern in CTA_PATTERNS.items() if pattern.search(tail)}
    if URL_RE.search(tail):
        found.add("link_to_resource")
    return sorted(found)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    low, high = int(index), min(int(index) + 1, len(ordered) - 1)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 1)


def aggregate(analyses: list[dict]) -> dict:
    """Summarise per-post measurements into the numbers the style guide quotes."""
    n = len(analyses) or 1

    def dist(key):
        values = [a[key] for a in analyses]
        return {"p10": percentile(values, 0.1), "median": percentile(values, 0.5), "p90": percentile(values, 0.9)}

    glyphs = Counter()
    ctas = Counter()
    for a in analyses:
        glyphs.update(a["glyph_lines"])
        ctas.update(a["ctas"])
    share = lambda pred: round(sum(1 for a in analyses if pred(a)) / n, 2)  # noqa: E731
    return {
        "posts": len(analyses),
        "chars": dist("chars"),
        "words": dist("words"),
        "lines": dist("lines"),
        "avg_line_chars": dist("avg_line_chars"),
        "hook_chars": dist("hook_chars"),
        "short_line_share": dist("short_line_share"),
        "share_hook_ends_with_colon": share(lambda a: a["hook_ends_with_colon"]),
        "share_hook_has_number": share(lambda a: a["hook_has_number"]),
        "share_hook_has_caps_word": share(lambda a: a["hook_has_caps_word"]),
        "share_hook_has_aside": share(lambda a: a["hook_has_aside"]),
        "share_with_numbered_list": share(lambda a: a["numbered_lines"] + a["labelled_lines"] >= 3),
        "share_with_glyph_bullets": share(lambda a: sum(a["glyph_lines"].values()) >= 3),
        "share_with_quoted_prompt": share(lambda a: a["quoted_prompts"] > 0),
        "share_with_links": share(lambda a: a["links"] > 0),
        "share_with_separator": share(lambda a: a["separators"] > 0),
        "share_with_arrows": share(lambda a: a["arrows"] > 0),
        "share_with_emoji": share(lambda a: a["emoji"] > 0),
        "emoji_per_post": dist("emoji"),
        "share_with_hashtags": share(lambda a: a["hashtags"] > 0),
        "share_using_ampersand": share(lambda a: a["ampersands"] > 0),
        "share_with_caps_emphasis": share(lambda a: a["caps_word_count"] > 0),
        "you_vs_i": dist("you_vs_i"),
        "bullet_glyphs": dict(glyphs.most_common(10)),
        "cta_kinds": {name: round(count / n, 2) for name, count in ctas.most_common()},
    }
