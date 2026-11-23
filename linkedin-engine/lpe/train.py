"""`lpe train`: fetch the example posts, measure them and write the trained style files."""

from __future__ import annotations

import json
from pathlib import Path

from . import drive
from .analyze import analyze_text
from .dataset import load_examples
from .media import analyze_media
from .paths import ANNOTATIONS_PATH, CORPUS_PATH, DATA_DIR, SOURCES_PATH
from .profile import build_profile, write_profile


def load_annotations() -> dict:
    if ANNOTATIONS_PATH.exists():
        return json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8"))
    return {"creator_markers": [], "posts": {}}


def train(drive_link: str | None = None, data_dir: Path = DATA_DIR, annotate: bool = False, log=print) -> dict:
    data_dir = Path(data_dir)
    if not drive_link and SOURCES_PATH.exists() and not any(data_dir.glob("*")):
        drive_link = json.loads(SOURCES_PATH.read_text(encoding="utf-8")).get("drive")
    if drive_link:
        log(f"Downloading the training folder into {data_dir} ...")
        files = drive.sync_folder(drive_link, data_dir, log=log)
        log(f"  {len(files)} files ready")
        SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
        SOURCES_PATH.write_text(json.dumps({"drive": drive_link}, indent=2) + "\n", encoding="utf-8")

    examples, problems = load_examples(data_dir)
    if not examples:
        raise SystemExit(f"No training posts in {data_dir}. Pass --drive <folder link> or put pairs of files there.")
    for problem in problems:
        log(f"  note: {problem}")

    annotations = load_annotations()
    missing = [e for e in examples if e.key not in annotations.get("posts", {})]
    if missing and annotate:
        from .llm import annotate_example  # needs the Claude API

        for example in missing:
            log(f"  asking Claude to classify {example.key} ...")
            annotations.setdefault("posts", {})[example.key] = annotate_example(example)
        ANNOTATIONS_PATH.write_text(json.dumps(annotations, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    elif missing:
        log(f"  {len(missing)} post(s) have no graphic annotation ({', '.join(e.key for e in missing[:5])}); "
            "add them to training/annotations.json or re-run with --annotate")

    log(f"Measuring {len(examples)} posts ...")
    analyses = [analyze_text(e.text) for e in examples]
    medias = [analyze_media(e.media_file) for e in examples if e.media_file]

    profile = build_profile(examples, analyses, medias, annotations)
    write_profile(profile)

    notes = annotations.get("posts", {})
    media_by_file = {m["file"]: m for m in medias}
    with CORPUS_PATH.open("w", encoding="utf-8") as corpus:
        for example, analysis in zip(examples, analyses):
            note = notes.get(example.key, {})
            corpus.write(json.dumps({
                "key": example.key,
                "text": example.text,
                "media_file": str(example.media_file) if example.media_file else "",
                "media": media_by_file.get(example.media_file.name, {}) if example.media_file else {},
                "analysis": {k: analysis[k] for k in ("chars", "lines", "hook", "ctas")},
                **{k: note.get(k, "") for k in ("archetype", "template", "style", "theme", "hook_type", "structure",
                                                "cta_type", "graphic_notes", "content_warning")},
                "use_as_example": note.get("use_as_example", True),
            }, ensure_ascii=False) + "\n")

    summary = {
        "posts": len(examples),
        "graphics": len(medias),
        "templates": profile["visual"]["templates"],
        "median_chars": profile["stats"]["chars"]["median"],
    }
    log(f"Trained on {summary['posts']} posts and {summary['graphics']} graphics.")
    return summary
