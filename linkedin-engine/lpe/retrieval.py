"""Find the training posts closest to a new brief, to show Claude as examples (TF-IDF, no extra libraries)."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path

from .paths import CORPUS_PATH

STOPWORDS = set(
    """a about above after again against all am an and any are as at be because been before being below between
    both but by can could did do does doing down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some such than that the their theirs
    them themselves then there these they this those through to too under until up very was we were what when where
    which while who whom why will with you your yours yourself yourselves it's don't i'm you're that's let's also
    get got go one two three use using used make made like want need way ll ve re""".split()
)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#'-]*")


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in TOKEN_RE.findall(text.lower()):
        token = raw.strip("'-")
        if len(token) >= 2 and token not in STOPWORDS and not token.isdigit():
            tokens.append(token)
    return tokens


class TfidfIndex:
    def __init__(self, docs: list[tuple[str, str]]):
        self.keys = [key for key, _ in docs]
        counts = [Counter(tokenize(text)) for _, text in docs]
        df = Counter(term for count in counts for term in count)
        n = len(docs)
        self.idf = {term: math.log((1 + n) / (1 + freq)) + 1 for term, freq in df.items()}
        self.vectors = [self._vector(count) for count in counts]

    def _vector(self, count: Counter) -> dict[str, float]:
        vec = {term: (1 + math.log(tf)) * self.idf.get(term, 0.0) for term, tf in count.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {term: v / norm for term, v in vec.items()}

    def scores(self, query: str) -> dict[str, float]:
        qvec = self._vector(Counter(tokenize(query)))
        return {
            key: sum(weight * vec.get(term, 0.0) for term, weight in qvec.items())
            for key, vec in zip(self.keys, self.vectors)
        }


def load_corpus(path: Path | None = None) -> list[dict]:
    path = Path(path) if path else CORPUS_PATH
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _doc_text(record: dict) -> str:
    return " ".join([record.get("text", ""), record.get("graphic_notes", ""), record.get("archetype", "")])


def similar_examples(query: str, corpus: list[dict], k: int = 3, template: str | None = None) -> list[dict]:
    """Best matches for the brief, with a nudge towards the requested template and away from repeats."""
    corpus = [r for r in corpus if r.get("use_as_example", True)]
    if not corpus or k <= 0:
        return []
    index = TfidfIndex([(r["key"], _doc_text(r)) for r in corpus])
    base = index.scores(query)
    by_key = {r["key"]: r for r in corpus}
    for key, record in by_key.items():
        if template and record.get("template") == template:
            base[key] += 0.25
    chosen: list[dict] = []
    used_templates: Counter = Counter()
    remaining = dict(base)
    while remaining and len(chosen) < k:
        best = max(remaining, key=lambda key: remaining[key] - 0.08 * used_templates[by_key[key].get("template")])
        record = by_key[best]
        chosen.append({**record, "score": round(base[best], 3)})
        used_templates[record.get("template")] += 1
        del remaining[best]
    return chosen
