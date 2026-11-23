"""Checks every post before it ships: LinkedIn limits, AI giveaways, the training creator's identity,
copying from the training posts, invented links and numbers, and advice the engine must never give."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .analyze import EMOJI_RE, FOLD_CHARS, HASHTAG_RE, hook_of
from .brand import own_hosts
from .knowledge import AVOID_PHRASES
from .markup import plain

LINKEDIN_LIMIT = 3000
URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s)\]\"']*)?", re.I)
NUMBER_RE = re.compile(r"(?<![\w.])[$€£]?\d[\d,.]*\s?(?:%|x|k|m|bn|b|hours?|hrs?|mins?|minutes?|seconds?|days?|weeks?|months?|years?)?(?![\w])", re.I)
NEGATIVE_PARALLEL_RE = re.compile(r"\b(it'?s|this is|that'?s|this isn'?t|it isn'?t)\s+not\b[^.!?\n]{1,60}[,;:.]\s*(it'?s|this is|that'?s)\b", re.I)
BAIT_RE = re.compile(r"\b(comment|type|reply)\s+[\"'“]?\w+[\"'”]?\s+(below|in the comments)|\blike if you\b|\btag (someone|a friend)\b|\bcomment ['\"“]?yes", re.I)
CHEAT_RE = re.compile(
    r"\b(let|have|use|make|get)\s+(it|ai|chatgpt|codex|claude|gemini|an? agent|the agent)\b[^.\n]{0,80}"
    r"\b(quiz|quizzes|exam|exams|test|tests|certificat\w*|assessment)", re.I)
FOOL_RE = re.compile(r"\bfool(ing)?\b[^.\n]{0,40}\b(hr|recruiters?|hiring|employers?|managers?)\b", re.I)
MARKDOWN_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__|^#{1,6}\s|\[[^\]\n]+\]\((https?://)?[^)\n]+\)", re.M)
PLACEHOLDER_RE = re.compile(r"\[CHECK:[^\]]*\]", re.I)
WORD_RE = re.compile(r"[a-z0-9']+")


@dataclass
class Issue:
    level: str        # "error" blocks publishing, "warning" needs a look, "info" is a note
    code: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("’", "'"))


def longest_shared_run(text: str, other: str, n: int = 6) -> int:
    """Length (in words) of the longest word-for-word run the two texts share."""
    a, b = _words(text), _words(other)
    if len(a) < n or len(b) < n:
        return 0
    grams = {tuple(b[i:i + n]) for i in range(len(b) - n + 1)}
    best = run = 0
    for i in range(len(a) - n + 1):
        if tuple(a[i:i + n]) in grams:
            run = run + 1 if run else n
            best = max(best, run)
        else:
            run = 0
    return best


def _numbers(text: str) -> set[str]:
    found = set()
    for match in NUMBER_RE.finditer(text):
        token = match.group(0).strip().lower().replace(",", "").rstrip(".")
        digits = re.sub(r"[^\d.]", "", token).strip(".")
        if digits and not re.fullmatch(r"[1-9]|1[0-9]|20", digits):  # list numbering and small counts are fine
            found.add(token)
    return found


def lint_post(text: str, profile: dict | None = None, corpus: list[dict] | None = None,
              brief_text: str = "", brand: dict | None = None) -> list[Issue]:
    issues: list[Issue] = []
    profile = profile or {}
    brand = brand or {}
    brief_lower = brief_text.lower()
    length = len(text)

    if length > LINKEDIN_LIMIT:
        issues.append(Issue("error", "too_long", f"{length} characters; LinkedIn allows {LINKEDIN_LIMIT}."))
    elif length < 300:
        issues.append(Issue("warning", "too_short", f"Only {length} characters; the examples are usually 1,300 to 2,800."))
    hook = hook_of(text)
    if len(hook) > FOLD_CHARS:
        issues.append(Issue("warning", "hook_past_fold", f"The opening is {len(hook)} characters; LinkedIn shows about {FOLD_CHARS} before 'see more'."))
    if MARKDOWN_RE.search(text):
        issues.append(Issue("error", "markdown", "Markdown (**bold**, # headings or [links](...)) shows up as raw symbols on LinkedIn."))

    phrases = profile.get("voice", {}).get("avoid_phrases") or AVOID_PHRASES
    lower = text.lower().replace("’", "'")
    found = [p for p in phrases if re.search(r"(?<!\w)" + re.escape(p.lower()) + r"(?!\w)", lower)]
    if found:
        issues.append(Issue("warning", "ai_phrases", "Reads as AI-written: " + ", ".join(f'"{p}"' for p in found)))
    if "—" in text:
        issues.append(Issue("warning", "em_dash", f"{text.count('—')} em dash(es): use a colon, a comma or a new line (lint --fix does it)."))
    if NEGATIVE_PARALLEL_RE.search(text):
        issues.append(Issue("warning", "not_x_but_y", "An 'It's not X, it's Y' sentence: a classic AI giveaway."))
    hashtags = HASHTAG_RE.findall(text)
    if len(hashtags) > 3:
        issues.append(Issue("warning", "hashtags", f"{len(hashtags)} hashtags; use three at most (the examples use none)."))
    emoji = EMOJI_RE.findall(text)
    if len(emoji) > 3:
        issues.append(Issue("warning", "emoji", f"{len(emoji)} emoji; the examples use three at most."))
    if BAIT_RE.search(text):
        issues.append(Issue("warning", "engagement_bait", "Engagement bait ('comment YES', 'tag someone') gets posts demoted."))
    if CHEAT_RE.search(text) or FOOL_RE.search(text):
        issues.append(Issue("error", "integrity", "Reads as advice to have AI pass tests or certifications, or to fool recruiters. Remove it."))
    if PLACEHOLDER_RE.search(text):
        issues.append(Issue("warning", "needs_input", "Fill in the [CHECK: ...] placeholders before posting."))

    guard = profile.get("guardrails", {})
    markers = [m for m in guard.get("creator_markers", []) if m and m.lower() in lower]
    if markers:
        issues.append(Issue("error", "third_party", "Names or brands from the training creator: " + ", ".join(markers)))
    domains = [d for d in guard.get("training_domains", []) if d.lower() in lower and d.lower() not in brief_lower]
    if domains:
        issues.append(Issue("error", "training_link", "Links to the training creator's sites: " + ", ".join(domains)))

    banned = [n for n in brand.get("never_mention", []) if n and n.lower() in lower]
    if banned:
        issues.append(Issue("error", "brand_avoid", "Mentions what brand.json says never to use: " + ", ".join(banned)))

    own = own_hosts(brand) | {"linkedin.com"}
    for url in sorted(set(URL_RE.findall(text))):
        host = re.sub(r"^https?://", "", url.lower()).split("/")[0].removeprefix("www.")
        if host in own or any(host.endswith("." + d) for d in own) or url.lower() in brief_lower or host in brief_lower:
            continue
        level = "error" if "lnkd.in" in host else "warning"
        issues.append(Issue(level, "unknown_link", f"{url} isn't in the brief: check it exists and is yours to share."))

    unknown_numbers = sorted(n for n in _numbers(text) if n not in brief_lower and n.rstrip("%xk") not in brief_lower)
    if unknown_numbers:
        issues.append(Issue("info", "check_numbers", "Numbers that aren't in the brief (make sure they're true): " + ", ".join(unknown_numbers[:12])))

    for record in corpus or []:
        run = longest_shared_run(text, record.get("text", ""))
        if run >= 12:
            issues.append(Issue("error", "copied", f"Shares a {run}-word run with training post '{record['key']}'. Rewrite it in new words."))
        elif run >= 9:
            issues.append(Issue("warning", "close_to_example", f"Shares a {run}-word run with training post '{record['key']}'."))
    return issues


def lint_graphic(spec: dict, profile: dict | None = None, schema_warnings: list[str] | None = None,
                 brand: dict | None = None) -> list[Issue]:
    issues = [Issue("warning", "graphic_data", w) for w in (schema_warnings or [])]
    text = plain(_flatten(spec.get("data", {}))).lower()
    text += " " + " ".join(str(spec.get(k, "")) for k in ("footer_left", "footer_right", "alt_text")).lower()
    for name in (brand or {}).get("never_mention", []):
        if name and name.lower() in text:
            issues.append(Issue("error", "brand_avoid", f"The graphic mentions '{name}', which brand.json says never to use."))
    for marker in (profile or {}).get("guardrails", {}).get("creator_markers", []):
        if marker and marker.lower() in text:
            issues.append(Issue("error", "third_party", f"The graphic mentions '{marker}' from the training set."))
    if spec.get("template") == "big_number" and not str(spec.get("data", {}).get("source", "")).strip():
        issues.append(Issue("error", "no_source", "A chart needs the source of its numbers."))
    return issues


def _flatten(value) -> str:
    if isinstance(value, dict):
        return " ".join(_flatten(v) for v in value.values())
    if isinstance(value, list):
        return " ".join(_flatten(v) for v in value)
    return str(value)


def fix_post(text: str) -> str:
    """Safe automatic fixes: em dashes and stray markdown bold."""
    text = re.sub(r"\s*—\s*", " - ", text)
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)
    return text


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


def format_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No issues found."
    marks = {"error": "✗", "warning": "!", "info": "·"}
    return "\n".join(f"{marks.get(i.level, '-')} [{i.level}] {i.message}" for i in issues)
