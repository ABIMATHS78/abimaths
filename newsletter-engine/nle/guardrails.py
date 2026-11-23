"""Checks every issue and guide before it's published: names the author never uses (in any spelling), the
training creator's identity, copying from the example issues, invented links and numbers, AI giveaways, advice
the engine must never give, and drafts that still have holes ([CHECK: ...] notes, screenshot slots)."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass

from .markup import plain

URL_RE = re.compile(r"(?:https?://)?(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s)\]\"'<>]*)?", re.I)
FILE_SUFFIXES = {"md", "txt", "pdf", "html", "htm", "json", "js", "ts", "py", "csv", "tsv", "xlsx", "xls", "docx", "doc",
                 "pptx", "png", "jpg", "jpeg", "gif", "webp", "svg", "mp4", "mov", "zip", "yaml", "yml", "css", "sh", "toml"}
NUMBER_RE = re.compile(r"(?<![\w.])[$€£]?\d[\d,.]*\s?(?:%|x|k|m|bn|hours?|hrs?|mins?|minutes?|days?|weeks?|months?|years?)?(?![\w])", re.I)
NEGATIVE_PARALLEL_RE = re.compile(r"\b(it'?s|this is|that'?s)\s+not\b[^.!?\n]{1,60}[,;:.]\s*(it'?s|this is|that'?s)\b", re.I)
BAIT_RE = re.compile(r"\b(comment|reply)\s+[\"'“]?\w+[\"'”]?\s+(below|in the comments)|\blike if you\b|\btag (someone|a friend)\b", re.I)
CHEAT_RE = re.compile(r"\b(let|have|use|make|get)\s+(it|ai|chatgpt|claude|gemini|an? agent)\b[^.\n]{0,80}"
                      r"\b(quiz|quizzes|exam|exams|tests?|certificat\w*|assessment)\b", re.I)
DETECTOR_RE = re.compile(r"\b(bypass|beat|fool|evade|get past|avoid)\b[^.\n]{0,40}\b(ai[- ]detect\w*|detectors?|gptzero|turnitin)\b", re.I)
FOOL_RE = re.compile(r"\bfool(ing)?\b[^.\n]{0,40}\b(hr|recruiters?|hiring|employers?|managers?|teachers?)\b", re.I)
PLACEHOLDER_RE = re.compile(r"\[CHECK:[^\]]*\]", re.I)
WORD_RE = re.compile(r"[a-z0-9']+")
ZERO_WIDTH_RE = re.compile("[​-‍⁠﻿­]")
SPACES_RE = re.compile(r"[\s  -   　]+")
NAME_SEP = r"[\s\-‐-―_.·/]*"


@dataclass
class Issue:
    level: str        # "error" blocks publishing, "warning" needs a look, "info" is a note
    code: str
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text or ""))
    return SPACES_RE.sub(" ", ZERO_WIDTH_RE.sub("", text)).replace("’", "'")


def _token_pattern(token: str) -> str:
    if token.isalpha() and token.isupper() and len(token) <= 4:      # AI also matches A.I.
        return r"\.?".join(re.escape(c) for c in token) + r"\.?"
    return re.escape(token)


def _reads_as_prose(matched: str, tokens: list[str]) -> bool:
    if len(tokens) < 2:
        return False
    head = re.match(r"\.?".join(re.escape(c) for c in tokens[0]) + r"(?P<sep>[^A-Za-z0-9]*)(?P<rest>.*)$", matched, re.I)
    if not head:
        return False
    sep, rest = head.group("sep"), head.group("rest")
    if not re.fullmatch(r"[A-Za-z0-9]+(?: [A-Za-z0-9]+)*", rest) or len(rest.split()) not in (1, len(tokens) - 1):
        return False
    if sep == " ":
        return rest == rest.lower()
    if re.fullmatch(r"\.?[.!?] ", sep):
        return rest[1:] == rest[1:].lower()
    return False


def brand_mentions(text: str, names) -> list[tuple[str, str, str]]:
    """(name, matched text, level). Any spelling of a name is an error ('AI Everytime', 'aieverytime.org'); the same
    words as ordinary prose ('ask AI every time you draft') are a warning."""
    norm = normalize(text)
    hits = []
    for name in names or []:
        tokens = re.findall(r"[A-Za-z0-9]+", str(name))
        if not tokens:
            continue
        loose = re.compile(r"(?<![A-Za-z0-9])" + NAME_SEP.join(_token_pattern(t) for t in tokens) + r"(?![A-Za-z0-9])", re.I)
        for match in loose.finditer(norm):
            hits.append((match.start(), match.end(), name, match.group(0),
                         "warning" if _reads_as_prose(match.group(0), tokens) else "error"))
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


def check_links(urls: list[str], site, brief_text: str) -> list[Issue]:
    """External links must be in the brief, or be the author's own site."""
    issues = []
    brief = brief_text.lower()
    own_host = site.host
    main = _url_key(site["main_site"].get("url", "")).split("/")[0]
    for url in sorted(set(urls)):
        if not url or url.startswith(("#", "/", "mailto:")) or re.match(r"^(issue|guide|page):", url):
            continue
        key = _url_key(url)
        host = key.split("/")[0]
        if "/" not in key and not url.startswith(("http://", "https://", "www.")) and host.rsplit(".", 1)[-1] in FILE_SUFFIXES:
            continue
        if key in brief or host in (own_host, main) or host.endswith("." + own_host) or host == "linkedin.com" or host.endswith(".linkedin.com"):
            continue
        level = "error" if host == "lnkd.in" or host.endswith(".lnkd.in") else "warning"
        issues.append(Issue(level, "unknown_link", f"{url} isn't in the brief: check it exists and is yours to share."))
    return issues


def _words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower().replace("’", "'"))


def longest_shared_run(text: str, other: str, n: int = 8) -> int:
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
        if digits and not re.fullmatch(r"[1-9]|1[0-9]|20", digits):
            found.add(token)
    return found


def lint_item(item, rendered, site, profile: dict | None = None, corpus: list[dict] | None = None,
              brief_text: str = "") -> list[Issue]:
    """All checks for one issue or guide. `rendered` is the body rendered by nle.md (its text, links and slots)."""
    profile = profile or {}
    issues: list[Issue] = []
    visible = "\n".join([item.title, item.subtitle, item.description, item.cover_alt, rendered.text])
    lower = normalize(visible).lower()

    for name, matched, level in brand_mentions(visible, site.get("never_mention")):
        if level == "error":
            issues.append(Issue("error", "brand_avoid", f"mentions '{matched}', a name site.json says never to use."))
        else:
            issues.append(Issue("warning", "brand_like", f"'{matched}' reads like '{name}', a name site.json says never to use. "
                                                         "Reword it if it could be taken for the name."))
    guard = profile.get("guardrails", {})
    markers = [m for m in guard.get("creator_markers", []) if m and re.search(r"(?<![a-z0-9])" + re.escape(m.lower()) + r"(?![a-z0-9])", lower)]
    if markers:
        issues.append(Issue("error", "third_party", "names or sites from the example issues' creator: " + ", ".join(markers)))
    domains = [d for d in guard.get("training_domains", []) if d.lower() in lower and d.lower() not in brief_text.lower()]
    if domains:
        issues.append(Issue("error", "training_link", "links to the example creator's sites: " + ", ".join(domains)))
    if CHEAT_RE.search(visible) or FOOL_RE.search(visible) or DETECTOR_RE.search(visible):
        issues.append(Issue("error", "integrity", "reads as advice to have AI pass tests, get past AI detectors or fool "
                                                  "someone. Remove it."))
    issues += check_links(rendered.links + URL_RE.findall(rendered.text), site, brief_text)

    if item.published:
        if PLACEHOLDER_RE.search(item.body) or PLACEHOLDER_RE.search(item.title + item.subtitle):
            issues.append(Issue("error", "needs_input", "still has [CHECK: ...] notes: fill them in before publishing."))
        if rendered.screenshots:
            issues.append(Issue("error", "screenshots", f"{len(rendered.screenshots)} screenshot slot(s) still empty: add the "
                                                        "images (or remove the slots) before publishing."))
    else:
        if PLACEHOLDER_RE.search(item.body):
            issues.append(Issue("info", "needs_input", "has [CHECK: ...] notes to fill in."))
        if rendered.screenshots:
            issues.append(Issue("info", "screenshots", f"{len(rendered.screenshots)} screenshot(s) to add: "
                                                       + "; ".join(rendered.screenshots[:6])))
    for problem in rendered.problems:
        issues.append(Issue("error", "broken", problem))

    phrases = profile.get("avoid_phrases") or []
    found = [p for p in phrases if re.search(r"(?<!\w)" + re.escape(p.lower()) + r"(?!\w)", lower)]
    if found:
        issues.append(Issue("warning", "ai_phrases", "reads as AI-written: " + ", ".join(f'"{p}"' for p in found)))
    if "—" in visible:
        issues.append(Issue("warning", "em_dash", f"{visible.count('—')} em dash(es): use a colon, a comma or a new sentence."))
    if NEGATIVE_PARALLEL_RE.search(visible):
        issues.append(Issue("warning", "not_x_but_y", "an 'It's not X, it's Y' sentence: a classic AI giveaway."))
    if BAIT_RE.search(visible):
        issues.append(Issue("warning", "engagement_bait", "engagement bait ('comment YES', 'tag someone')."))
    if len(item.title.split()) > 12:
        issues.append(Issue("warning", "long_title", f"the title has {len(item.title.split())} words; the examples use 1 to 7."))
    if not item.subtitle:
        issues.append(Issue("warning", "no_subtitle", "no subtitle: say who it's for or what they'll be able to do."))
    if item.description and len(item.description) > 160:
        issues.append(Issue("warning", "long_description", f"the description is {len(item.description)} characters; "
                                                           "search engines show about 155."))
    long_paragraphs = [p for p in re.split(r"\n\s*\n", item.body) if not p.lstrip().startswith(("```", "|", "-", "*", ">", "#", "::", "1.", "!["))
                       and len(plain(p).split()) > 70]
    if long_paragraphs:
        issues.append(Issue("warning", "long_paragraphs", f"{len(long_paragraphs)} paragraph(s) over 70 words: the examples keep "
                                                          "paragraphs to one or two sentences."))
    if rendered.words < 350:
        issues.append(Issue("warning", "short", f"only {rendered.words} words."))

    unknown = sorted(n for n in _numbers(visible) if n not in brief_text.lower() and n.rstrip("%xk") not in brief_text.lower())
    if unknown and brief_text:
        issues.append(Issue("info", "check_numbers", "numbers that aren't in the brief (make sure they're true): " + ", ".join(unknown[:12])))

    for record in corpus or []:
        run = longest_shared_run(rendered.text, record.get("text", ""))
        if run >= 12:
            issues.append(Issue("error", "copied", f"shares a {run}-word run with example issue '{record['key']}'. Rewrite it in new words."))
        elif run >= 9:
            issues.append(Issue("warning", "close_to_example", f"shares a {run}-word run with example issue '{record['key']}'."))
    return issues


def has_errors(issues: list[Issue]) -> bool:
    return any(i.level == "error" for i in issues)


def format_issues(issues: list[Issue]) -> str:
    if not issues:
        return "No issues found."
    marks = {"error": "✗", "warning": "!", "info": "·"}
    return "\n".join(f"{marks.get(i.level, '-')} [{i.level}] {i.message}" for i in issues)
