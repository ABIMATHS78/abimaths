"""Words the text reader saw in a graphic that aren't words: likely mistakes in the graphic (or misreadings).

    suspects.py <id>        prints them, with likely corrections from the guide's own words

Used by show.py. A word counts as fine if it's an English word, a word used anywhere in the guides, or the reader's
usual confusions of a fine word (l for I, rn for m, no space between words). Repeated words ("free free") are
suspects too.
"""

from __future__ import annotations

import difflib
import functools
import gzip
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import GUIDES, WORK  # noqa: E402
TOKEN = re.compile(r"[A-Za-z][A-Za-z']*(?:[.\-][A-Za-z]+)*")


@functools.lru_cache(None)
def corpus() -> tuple[Counter, dict]:
    cache = os.path.join(WORK, "vocab.json")
    if os.path.exists(cache):
        data = json.load(open(cache))
        return Counter(data["counts"]), data["texts"]
    counts: Counter = Counter()
    forms: dict = {}
    texts = {}
    for g in json.load(gzip.open(GUIDES, "rt"))["guides"]:
        text = f"{g['title']}\n{g['summary']}\n{g['body']}"
        texts[str(g["id"])] = text
        for w in TOKEN.findall(text):
            counts[w.lower()] += 1
            forms.setdefault(w.lower(), Counter())[w] += 1
    # A word written with capitals inside it nearly every time ("LinkedIn", "IoT", "SaaS", "PyTorch").
    cased = {}
    for low, c in forms.items():
        form, n = c.most_common(1)[0]
        inner = any(ch.isupper() for ch in form[1:]) and not form.isupper()
        if inner and n >= 3 and n >= 0.9 * sum(c.values()):
            cased[low] = form
    texts["__cased__"] = json.dumps(cased)
    json.dump({"counts": counts, "texts": texts}, open(cache, "w"))
    return counts, texts


@functools.lru_cache(None)
def casings() -> dict:
    return json.loads(corpus()[1].get("__cased__", "{}"))


@functools.lru_cache(65536)
def zipf(word: str) -> float:
    """How common an English word is: about 6 for "the", 3 for "trophy", under 2 for rare words and typos."""
    from wordfreq import zipf_frequency
    return zipf_frequency(word, "en")


SHORT = frozenset("a an as at by in of on or to & ai al ml db ui ux io it is".split())  # "al": the reader's "AI"


def common(piece: str) -> bool:
    """A word common enough to count as one of several the reader ran together."""
    if len(piece) < 3:
        return piece in SHORT
    counts, _ = corpus()
    return zipf(piece) >= (4.0 if len(piece) <= 4 else 3.0) or counts.get(piece, 0) >= 30


def known(word: str) -> bool:
    w = word.lower().strip("'")
    if len(w) <= 2:
        return True
    counts, _ = corpus()
    if counts.get(w, 0) >= 2 or zipf(w) >= 2.0:
        return True
    if w.endswith("'s") and known(w[:-2]):
        return True
    # Hyphenated or dotted compounds: fine if every part is.
    parts = re.split(r"[.\-]", w)
    if len(parts) > 1 and all(p and known(p) for p in parts):
        return True
    return False


def misread_ok(word: str) -> bool:
    """A fine word the reader got slightly wrong in its usual ways: l/I/1, rn/m, 0/O, a missing space."""
    w = word
    for a, b in (("l", "I"), ("I", "l"), ("rn", "m"), ("0", "O"), ("1", "l"), ("vv", "w")):
        if a in w:
            v = w.replace(a, b)
            natural = v.isupper() or v.islower() or (v[:1].isupper() and v[1:].islower())
            if natural and known(v):
                return True
    # Run together: a long word that is several common ones, and not just one word misspelled.
    low = w.lower()
    if len(low) >= 10 and not close_single(low):
        return bool(split_words(low))
    return False


def close_single(low: str) -> bool:
    """Whether the word is one known word misspelled ("provendance" for "provenance"), rather than several."""
    counts, _ = corpus()
    pool = [c for c, n in counts.items() if n >= 3 and abs(len(c) - len(low)) <= 2]
    return bool(difflib.get_close_matches(low, pool, n=1, cutoff=0.88))


@functools.lru_cache(4096)
def split_words(w: str) -> tuple[str, ...] | None:
    """The word as two to four common words run together (the reader often drops spaces), or None."""
    if len(w) < 6:
        return None
    n = len(w)
    dp: list = [None] * (n + 1)
    dp[0] = ()
    for i in range(1, n + 1):
        for j in range(max(0, i - 20), i):
            if dp[j] is not None and common(w[j:i]):
                cand = dp[j] + (w[j:i],)
                if dp[i] is None or len(cand) < len(dp[i]):
                    dp[i] = cand
    best = dp[n]
    return best if best and 2 <= len(best) <= 4 and sum(len(p) >= 3 for p in best) >= 2 else None


def suggest(word: str, gid: str) -> list[str]:
    counts, texts = corpus()
    own = sorted({w for w in TOKEN.findall(texts.get(gid, "")) if len(w) >= 3}, key=str.lower)
    low = word.lower()
    picks = difflib.get_close_matches(low, [w.lower() for w in own], n=3, cutoff=0.72)
    if len(picks) < 3:
        pool = [w for w, c in counts.items() if c >= 3 and abs(len(w) - len(low)) <= 2]
        picks += [p for p in difflib.get_close_matches(low, pool, n=3, cutoff=0.78) if p not in picks]
    # In the graphic's own capitalisation.
    def cased(p: str) -> str:
        if word.isupper():
            return p.upper()
        if word[:1].isupper():
            return p[:1].upper() + p[1:]
        return p
    out = []
    for p in picks:
        if p.lower() not in {o.lower() for o in out}:
            out.append(cased(p))
    return out[:3]


def find(gid: str, lines: list[dict]) -> list[dict]:
    out = []
    for i, line in enumerate(lines):
        text = line["text"]
        words = TOKEN.findall(text)
        bad = []
        caps_line = sum(c.isupper() for c in text) >= 0.8 * max(1, sum(c.isalpha() for c in text))
        for w in words:
            if not known(w) and not misread_ok(w):
                bad.append({"word": w, "try": suggest(w, gid)})
                continue
            # A name written in the wrong case: "Linkedin", "SaAs", or "IOT" in a line that isn't all capitals.
            form = casings().get(w.lower())
            if form and w != form and len(w) > 2 and not (w.isupper() and caps_line):
                bad.append({"word": w, "try": [form], "case": True})
        lows = [w.lower() for w in words]
        for k in range(len(lows) - 1):
            if lows[k] == lows[k + 1] and len(lows[k]) > 2:
                bad.append({"word": f"{words[k]} {words[k + 1]}", "try": [words[k]], "repeat": True})
        if bad:
            out.append({"line": i, "box": line["box"], "text": text, "bad": bad})
    return out


def main(gid: str) -> None:
    lines = json.load(open(os.path.join(WORK, "ocr", f"{gid}.json")))
    for s in find(gid, lines):
        words = "; ".join(f"{b['word']!r}" + (f" → {' / '.join(b['try'])}?" if b["try"] else "") for b in s["bad"])
        print(f"{s['line']:3d} {words}")


if __name__ == "__main__":
    main(sys.argv[1])
