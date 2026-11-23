"""Checks every post before it ships: LinkedIn limits, AI giveaways, names the author never uses, the training
creator's identity, copying from the training posts, invented links and numbers, and advice the engine must
never give. The same identity and link checks run on everything else a reader sees: the other openings, the alt
text, the graphic brief and the graphic itself."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .analyze import EMOJI_RE, FOLD_CHARS, HASHTAG_RE, hook_of
from .brand import own_hosts
from .knowledge import AVOID_PHRASES
from .markup import plain

LINKEDIN_LIMIT = 3000
URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s)\]\"'<>]*)?", re.I)
FILE_SUFFIXES = {"md", "txt", "pdf", "html", "htm", "json", "js", "ts", "py", "csv", "tsv", "xlsx", "xls", "docx", "doc",
                 "pptx", "png", "jpg", "jpeg", "gif", "webp", "svg", "mp4", "mov", "zip", "yaml", "yml", "css", "sh", "ipynb"}
NUMBER_RE = re.compile(r"(?<![\w.])[$€£]?\d[\d,.]*\s?(?:%|x|k|m|bn|b|hours?|hrs?|mins?|minutes?|seconds?|days?|weeks?|months?|years?)?(?![\w])", re.I)
NEGATIVE_PARALLEL_RE = re.compile(r"\b(it'?s|this is|that'?s|this isn'?t|it isn'?t)\s+not\b[^.!?\n]{1,60}[,;:.]\s*(it'?s|this is|that'?s)\b", re.I)
BAIT_RE = re.compile(r"\b(comment|type|reply)\s+[\"'“]?\w+[\"'”]?\s+(below|in the comments)|\blike if you\b|\btag (someone|a friend)\b|\bcomment ['\"“]?yes", re.I)
CHEAT_RE = re.compile(
    r"\b(let|have|use|make|get)\s+(it|ai|chatgpt|codex|claude|gemini|an? agent|the agent)\b[^.\n]{0,80}"
    r"\b(quiz|quizzes|exam|exams|test|tests|certificat\w*|assessment)\b", re.I)
FOOL_RE = re.compile(r"\bfool(ing)?\b[^.\n]{0,40}\b(hr|recruiters?|hiring|employers?|managers?)\b", re.I)
MARKDOWN_RE = re.compile(r"\*\*[^*\n]+\*\*|__[^_\n]+__|^#{1,6}\s|\[[^\]\n]+\]\((https?://)?[^)\n]+\)", re.M)
PLACEHOLDER_RE = re.compile(r"\[CHECK:[^\]]*\]", re.I)
WORD_RE = re.compile(r"[a-z0-9']+")
ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿­]")
SPACES_RE = re.compile(r"[\s  -   　]+")
NAME_SEP = r"[\s\-‐-―_.·/]*"   # what may sit between the words of a name: spaces, dashes, dots, nothing


@dataclass
class Issue:
    level: str        # "error" blocks publishing, "warning" needs a look, "info" is a note
    code: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    """Fold look-alike characters, drop invisible ones and collapse every kind of space and line break."""
    text = unicodedata.normalize("NFKC", str(text or ""))
    text = ZERO_WIDTH_RE.sub("", text)
    return SPACES_RE.sub(" ", text).replace("’", "'")


def _token_pattern(token: str) -> str:
    if token.isalpha() and token.isupper() and len(token) <= 4:  # AI also matches A.I.
        return r"\.?".join(re.escape(c) for c in token) + r"\.?"
    return re.escape(token)


def _reads_as_prose(matched: str, tokens: list[str]) -> bool:
    """True for ordinary words that happen to spell a name: 'AI every time' mid-sentence, or 'AI. Every time'
    across a sentence break. A capitalised, hyphenated or run-together form reads as the name itself."""
    if len(tokens) < 2:
        return False
    head = re.match(r"\.?".join(re.escape(c) for c in tokens[0]) + r"(?P<sep>[^A-Za-z0-9]*)(?P<rest>.*)$", matched, re.I)
    if not head:
        return False
    sep, rest = head.group("sep"), head.group("rest")
    if not re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)*", rest) or len(rest.split()) not in (1, len(tokens) - 1):
        return False                             # 'every time' or 'everytime', nothing hyphenated or dotted
    if sep == " ":
        return rest == rest.lower()
    if re.fullmatch(r"\.?[.!?] ", sep):          # a sentence ends after the first word
        return rest[1:] == rest[1:].lower()
    return False


def brand_mentions(text: str, names) -> list[tuple[str, str, str]]:
    """(name, matched text, level) for each never-mention name found in the text.

    Any spelling of the name is an error: 'AI Every Time', 'AI Everytime', 'AI-Every-Time', 'A.I. Every Time',
    'aieverytime.org'. The same words written as ordinary prose ('ask AI every time you draft', 'I use AI. Every
    time it helps') are only a warning, since they may not mean the name at all. Overlapping matches of several
    names ('aieverytime.org' is also 'aieverytime') count once, as the longest."""
    norm = normalize(text)
    hits = []
    for name in names or []:
        tokens = re.findall(r"[A-Za-z0-9]+", str(name))
        if not tokens:
            continue
        loose = re.compile(r"(?<![A-Za-z0-9])" + NAME_SEP.join(_token_pattern(t) for t in tokens) + r"(?![A-Za-z0-9])", re.I)
        for match in loose.finditer(norm):
            level = "warning" if _reads_as_prose(match.group(0), tokens) else "error"
            hits.append((match.start(), match.end(), name, match.group(0), level))
    hits.sort(key=lambda h: (h[0], h[0] - h[1], h[4] != "error"))
    found, seen, end = [], set(), -1
    for start, stop, name, matched, level in hits:
        if start < end:
            continue
        end = stop
        if (matched.lower(), level) not in seen:
            seen.add((matched.lower(), level))
            found.append((name, matched, level))
    return found


def _url_key(url: str) -> str:
    return re.sub(r"^https?://", "", url.strip().lower()).removeprefix("www.").rstrip("/.,;:!?")


def check_links(text: str, brand: dict, brief_text: str, where: str = "") -> list[Issue]:
    """Links must come from the brief or be one of the author's known pages."""
    issues = []
    brief_lower = brief_text.lower()
    own = own_hosts(brand)
    known_pages = {_url_key(u) for u in [brand.get("url", ""), *brand.get("pages", [])] if u}
    prefix = f"In {where}: " if where else ""
    for url in sorted(set(URL_RE.findall(normalize(text)))):
        key = _url_key(url)
        host = key.split("/")[0]
        if "/" not in key and not re.match(r"(?i)(https?://|www\.)", url) and host.rsplit(".", 1)[-1] in FILE_SUFFIXES:
            continue                             # a file name such as SKILL.md or notes.txt, not a link
        if key in brief_lower or url.lower() in brief_lower or host == "linkedin.com" or host.endswith(".linkedin.com"):
            continue
        if host in own or any(host.endswith("." + d) for d in own):
            if key != host and known_pages and key not in known_pages:
                issues.append(Issue("warning", "unknown_page", f"{prefix}{url} isn't a page listed in brand.json or the brief: check it exists."))
            continue
        if host in brief_lower:
            continue
        level = "error" if host == "lnkd.in" or host.endswith(".lnkd.in") else "warning"
        issues.append(Issue(level, "unknown_link", f"{prefix}{url} isn't in the brief: check it exists and is yours to share."))
    return issues


def identity_issues(text: str, profile: dict, brand: dict, brief_text: str = "", where: str = "") -> list[Issue]:
    """Never-mention names, the training creator's names and sites, integrity and links: for any text a reader sees."""
    issues = []
    prefix = f"In {where}: " if where else ""
    norm_lower = normalize(text).lower()
    for name, matched, level in brand_mentions(text, brand.get("never_mention")):
        if level == "error":
            issues.append(Issue("error", "brand_avoid", f"{prefix}mentions '{matched}', a name brand.json says never to use."))
        else:
            issues.append(Issue("warning", "brand_like", f"{prefix}'{matched}' reads like '{name}', a name brand.json says "
                                                         "never to use. Reword it if it could be taken for the name."))
    guard = profile.get("guardrails", {})
    markers = [m for m in guard.get("creator_markers", []) if m and m.lower() in norm_lower]
    if markers:
        issues.append(Issue("error", "third_party", f"{prefix}names or brands from the training creator: " + ", ".join(markers)))
    brief_lower = brief_text.lower()
    domains = [d for d in guard.get("training_domains", []) if d.lower() in norm_lower and d.lower() not in brief_lower]
    if domains:
        issues.append(Issue("error", "training_link", f"{prefix}links to the training creator's sites: " + ", ".join(domains)))
    if CHEAT_RE.search(text) or FOOL_RE.search(text):
        issues.append(Issue("error", "integrity", f"{prefix}reads as advice to have AI pass tests or certifications, or to fool recruiters. Remove it."))
    issues += check_links(text, brand, brief_text, where)
    return issues


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
    if PLACEHOLDER_RE.search(text):
        issues.append(Issue("warning", "needs_input", "Fill in the [CHECK: ...] placeholders before posting."))

    issues += identity_issues(text, profile, brand, brief_text)

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


def lint_extras(fields: dict[str, str], profile: dict | None = None, brand: dict | None = None,
                brief_text: str = "") -> list[Issue]:
    """The identity and link checks for the other things a reader sees: other openings, alt text, graphic brief."""
    issues = []
    for where, text in fields.items():
        if text and str(text).strip():
            issues += identity_issues(str(text), profile or {}, brand or {}, brief_text, where)
    return issues


def plan_extras(plan: dict, alt_text: bool = True) -> dict[str, str]:
    """What else of a writer's plan the author reads or posts: the other openings, the graphic brief and the
    alt text (left out when the graphic's own check covers it)."""
    fields = {
        "the other openings": "\n".join(str(h) for h in plan.get("alternative_hooks") or []),
        "the graphic brief": str(plan.get("graphic_brief") or ""),
    }
    if alt_text:
        fields["the alt text"] = str(plan.get("alt_text") or "")
    return fields


def lint_graphic(spec: dict, profile: dict | None = None, schema_warnings: list[str] | None = None,
                 brand: dict | None = None, brief_text: str = "") -> list[Issue]:
    issues = [Issue("warning", "graphic_data", w) for w in (schema_warnings or [])]
    text = plain(_flatten(spec.get("data", {})))
    text += "\n" + "\n".join(str(spec.get(k, "")) for k in ("footer_left", "footer_right"))
    issues += identity_issues(text, profile or {}, brand or {}, brief_text, "the graphic")
    issues += lint_extras({"the alt text": spec.get("alt_text", "")}, profile, brand, brief_text)
    if spec.get("template") == "big_number" and not str(spec.get("data", {}).get("source", "")).strip():
        issues.append(Issue("error", "no_source", "A chart needs the source of its numbers."))
    return issues


def _flatten(value) -> str:
    if isinstance(value, dict):
        return "\n".join(_flatten(v) for v in value.values())
    if isinstance(value, list):
        return "\n".join(_flatten(v) for v in value)
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
