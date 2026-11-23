"""`python -m nle train`: fetch the example material (optional), measure it and write the style files.

Writes model/style_profile.json and model/style_guide.md (committed: numbers and patterns only) and
model/corpus.jsonl (local: the examples' text, used to catch copying)."""

from __future__ import annotations

import json
from pathlib import Path

from .analyze import aggregate, analyze_issue, archive_stats, plain_text
from .paths import ANNOTATIONS_PATH, CORPUS_PATH, DATA_DIR
from .profile import build_profile, load_profile, write_profile


def load_annotations() -> dict:
    return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8")) if ANNOTATIONS_PATH.exists() else {}


def train(fetch_first: bool = False, data_dir: Path | None = None, log=print) -> dict:
    data = Path(data_dir) if data_dir else DATA_DIR
    if fetch_first:
        from .sources import fetch, load_sources

        sources = load_sources()
        if not sources:
            raise FileNotFoundError("Nothing to fetch: create training/sources.local.json (see nle/sources.py).")
        log("Fetching the example material ...")
        fetch(sources, data, log=log)
    posts = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((data / "posts").glob("*.json"))]
    archive_path = data / "archive.json"
    archive = json.loads(archive_path.read_text(encoding="utf-8")) if archive_path.exists() else []
    if not posts and not archive:
        raise FileNotFoundError(f"No example material in {data}. Run `python -m nle train --fetch` first.")
    annotations = load_annotations()
    notes = annotations.get("issues", {})

    analyses = [analyze_issue(p) for p in posts]
    profile = build_profile(aggregate(analyses), archive_stats(archive), annotations)
    write_profile(profile)

    CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CORPUS_PATH.open("w", encoding="utf-8") as out:
        for post in posts:
            note = notes.get(post.get("slug", ""), {})
            out.write(json.dumps({"key": post.get("slug", ""), "title": post.get("title", ""),
                                  "archetype": note.get("archetype", ""), "complete": post.get("audience") == "everyone",
                                  "use_as_example": note.get("use_as_example", True) and post.get("audience") == "everyone",
                                  "text": plain_text(post.get("body_html"))}, ensure_ascii=False) + "\n")
    return {"issues_measured": len(analyses), "complete": sum(a["complete"] for a in analyses), "archive": len(archive)}


def refresh_guide() -> None:
    """Re-render the style guide from the saved numbers and the current knowledge.py (no training material needed)."""
    write_profile(load_profile())
