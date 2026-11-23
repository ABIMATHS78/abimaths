"""Pair each training post's text with its graphic.

Files that belong together share a name once the role word and extension are removed:
"Post 3 Text.txt" + "Post 3 Image.jpg" -> "Post 3", and "Post 20 Text.txt" + "Post 20.gif" -> "Post 20".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

TEXT_EXTS = {".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
GIF_EXTS = {".gif"}
VIDEO_EXTS = {".mp4", ".mov", ".webm", ".m4v"}
ROLE_SUFFIX_RE = re.compile(
    r"[\s_\-.]+(text|caption|copy|post text|image|img|graphic|visual|gif|video|animation|motion)$", re.I
)


@dataclass
class Example:
    key: str
    text: str = ""
    text_file: Path | None = None
    media_file: Path | None = None
    media_kind: str = ""          # "image", "gif" or "video"
    extra_media: list[Path] = field(default_factory=list)

    @property
    def order(self) -> tuple:
        return natural_key(self.key)


def natural_key(text: str) -> tuple:
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def stem_key(filename: str) -> str:
    stem = Path(filename).stem
    stem = ROLE_SUFFIX_RE.sub("", stem).strip(" _-.")
    return re.sub(r"\s+", " ", stem) or Path(filename).stem


def media_kind(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in GIF_EXTS:
        return "gif"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in IMAGE_EXTS:
        return "image"
    return ""


def load_examples(data_dir: Path) -> tuple[list[Example], list[str]]:
    """Return paired examples (sorted naturally) and a list of problems (unpaired files)."""
    examples: dict[str, Example] = {}
    problems: list[str] = []
    files = sorted(p for p in Path(data_dir).rglob("*") if p.is_file() and not p.name.startswith("."))
    for path in files:
        ext = path.suffix.lower()
        if ext == ".part":
            continue
        key = stem_key(path.name)
        example = examples.setdefault(key, Example(key=key))
        if ext in TEXT_EXTS:
            if example.text_file:
                problems.append(f"{key}: two text files ({example.text_file.name}, {path.name}); kept the first")
                continue
            example.text_file = path
            example.text = path.read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n").strip()
        elif media_kind(path):
            if example.media_file:
                example.extra_media.append(path)
            else:
                example.media_file = path
                example.media_kind = media_kind(path)
        else:
            problems.append(f"{path.name}: not a text, image, GIF or video file; skipped")
    paired = []
    for key, example in examples.items():
        if not example.text_file:
            problems.append(f"{key}: graphic without a text file; skipped")
            continue
        if not example.media_file:
            problems.append(f"{key}: text without a graphic; kept for the writing style only")
        paired.append(example)
    paired.sort(key=lambda e: e.order)
    return paired, problems
