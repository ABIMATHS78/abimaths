#!/usr/bin/env python3
"""Checks text against the main writing rules of a Simplified Technical English style.

Usage:  python check_ste.py FILE
   or:  python check_ste.py < FILE

Numbered lines ("1. Remove the cover.") are procedure steps: 20 words per sentence at most, and one instruction
each. Other sentences are descriptions: 25 words at most, and no more than six sentences in a paragraph. Every
sentence is also checked for possible passive voice, -ing forms and words that have a simpler alternative.

These are pattern checks, not a grammar parser: they can flag a sentence that is fine, and they can't judge
meaning. The script exits with 1 when it has findings, so other tools can tell.
"""

import re
import sys
from itertools import pairwise

STEP = re.compile(r"^\s*(?:\d+|[a-z])[.)]\s+")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'’/-]*")
PARTICIPLES = ("built broken chosen done drawn driven fallen found given held hidden kept known laid led left lost made "
               "meant paid put read run said seen sent set shown shut sold spent split spread stuck taken taught thrown "
               "told understood worn written").split()
PASSIVE = re.compile(r"\b(?:am|is|are|was|were|be|been|being)\s+(?:\w+ly\s+)?(\w+ed|" + "|".join(PARTICIPLES) + r")\b",
                     re.IGNORECASE)
NOT_ING = set(("anything bring ceiling during evening everything king morning nothing ring sing something spring "
               "sting string swing thing warning wing").split())
SIMPLER = {
    "additional": "more", "approximately": "about", "assist": "help", "commence": "start",
    "due to the fact that": "because", "ensure": "make sure", "facilitate": "help", "in order to": "to",
    "in the event that": "if", "indicate": "show", "initiate": "start", "it is essential that": "you must",
    "modify": "change", "numerous": "many", "obtain": "get", "perform": "do", "prior to": "before",
    "regarding": "about", "subsequently": "then", "sufficient": "enough", "terminate": "stop", "utilise": "use",
    "utilize": "use",
}
ACTIONS = set(("add adjust apply attach check clean click close connect copy disconnect enter fill fit install lift "
               "lock loosen lower move open paste press pull push put release remove replace reset run save select "
               "set start stop tighten turn type unlock wait write").split())
STEP_WORDS, DESCRIPTION_WORDS, PARAGRAPH_SENTENCES = 20, 25, 6


def sentences(text: str) -> list[str]:
    return [s.strip() for s in SENTENCE_END.split(text) if s.strip()]


def words(sentence: str) -> list[str]:
    return WORD.findall(sentence)


def excerpt(sentence: str, most: int = 70) -> str:
    return sentence if len(sentence) <= most else sentence[:most - 1].rstrip() + "…"


def check_sentence(sentence: str, step: bool) -> list[str]:
    found = []
    n, limit = len(words(sentence)), STEP_WORDS if step else DESCRIPTION_WORDS
    if n > limit:
        found.append(f"{n} words ({'steps' if step else 'descriptions'}: {limit} at most)")
    for m in PASSIVE.finditer(sentence):
        found.append(f'possible passive voice ("{m.group(0)}")')
    for w in words(sentence):
        if w.lower().endswith("ing") and len(w) > 4 and w.lower() not in NOT_ING:
            found.append(f'-ing form ("{w}"): can you use another form?')
    lower = sentence.lower()
    for long, short in SIMPLER.items():
        if re.search(rf"\b{re.escape(long)}\b", lower):
            found.append(f'simpler word: "{long}" → "{short}"')
    if step:
        rest = [w.lower() for w in words(sentence)[1:]]
        joined = any(a == "and" and b in ACTIONS for a, b in pairwise(rest))
        if joined or ", then " in lower or "; " in sentence:
            found.append("more than one instruction? Give each its own step")
    return found


def check(text: str) -> list[tuple[int, str, str]]:
    """Findings as (line number, issue, sentence)."""
    findings: list[tuple[int, str, str]] = []
    paragraph: list[tuple[int, str]] = []

    def flush() -> None:
        if not paragraph:
            return
        first = paragraph[0][0]
        body = sentences(" ".join(line for _, line in paragraph))
        for s in body:
            findings.extend((first, issue, s) for issue in check_sentence(s, step=False))
        if len(body) > PARAGRAPH_SENTENCES:
            findings.append((first, f"{len(body)} sentences in one paragraph ({PARAGRAPH_SENTENCES} at most)", body[0]))
        paragraph.clear()

    for number, raw in enumerate(text.splitlines(), 1):
        line = raw.strip().lstrip(">").strip()
        if not line:
            flush()
        elif STEP.match(line):
            flush()
            step = sentences(STEP.sub("", line, count=1))
            for s in step:
                findings.extend((number, issue, s) for issue in check_sentence(s, step=True))
            if len(step) > 1:
                findings.append((number, "more than one sentence in this step: is it one instruction?", step[0]))
        else:
            paragraph.append((number, line))
    flush()
    return findings


def main() -> int:
    if len(sys.argv) > 2 or (len(sys.argv) == 2 and sys.argv[1] in ("-h", "--help")):
        print(__doc__.strip())
        return 2
    name = sys.argv[1] if len(sys.argv) == 2 else "the text"
    with open(sys.argv[1], encoding="utf-8") if len(sys.argv) == 2 else sys.stdin as f:
        findings = check(f.read())
    if not findings:
        print(f"No findings in {name}.")
        return 0
    print(f"{len(findings)} finding{'s' if len(findings) != 1 else ''} in {name}:")
    for number, issue, sentence in findings:
        print(f'  line {number}: {issue}: "{excerpt(sentence)}"')
    return 1


if __name__ == "__main__":
    sys.exit(main())
