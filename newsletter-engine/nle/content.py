"""The author's content: newsletter issues and guides as Markdown files with a TOML header.

    +++
    title = "Claude skills, from zero"
    subtitle = "Teach Claude a job once, and it does it your way."
    date = 2026-11-23
    status = "published"          # or "draft": drafts are built only with --drafts
    tags = ["claude", "skills"]
    +++
    The body, in Markdown (see nle/render.py for the extra marks).

Issues live in content/issues/, guides in content/guides/. The file name is the page's address (its slug).
content/path.toml lists the "Start here" learning path; content/library.toml the library of prompts and tools.
"""

from __future__ import annotations

import datetime as dt
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from .paths import CONTENT_DIR, GUIDES_DIR, ISSUES_DIR

SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
FRONT_RE = re.compile(r"\A\+\+\+\s*\n(.*?)\n\+\+\+\s*\n?(.*)\Z", re.S)
KINDS = {"issue": ISSUES_DIR, "guide": GUIDES_DIR}
LEVELS = ("beginner", "intermediate", "advanced")


class ContentError(ValueError):
    pass


@dataclass
class Item:
    kind: str                      # "issue" or "guide"
    slug: str
    title: str
    body: str
    subtitle: str = ""
    description: str = ""
    date: dt.date | None = None
    updated: dt.date | None = None
    status: str = "published"
    tags: list[str] = field(default_factory=list)
    archetype: str = ""
    featured: bool = False
    level: str = ""                # guides
    topic: str = ""                # guides
    cover: str = ""                # an image path, or empty for the drawn title card
    cover_alt: str = ""
    linkedin_url: str = ""         # where readers can discuss it (the LinkedIn post that shares it)
    fact_checks: list[str] = field(default_factory=list)
    brief: str = ""
    path: Path | None = None

    @property
    def ref(self) -> str:
        return f"{self.kind}:{self.slug}"

    @property
    def published(self) -> bool:
        return self.status == "published"

    @property
    def sort_date(self) -> dt.date:
        return self.date or self.updated or dt.date.min


def parse(text: str, kind: str, slug: str, path: Path | None = None) -> Item:
    match = FRONT_RE.match(text.lstrip("﻿"))
    if not match:
        raise ContentError(f"{path or slug}: the file must start with a +++ header (TOML) and end it with +++")
    try:
        meta = tomllib.loads(match.group(1))
    except tomllib.TOMLDecodeError as error:
        raise ContentError(f"{path or slug}: the +++ header isn't valid TOML: {error}") from error
    body = match.group(2).strip("\n") + "\n"
    if not SLUG_RE.match(slug):
        raise ContentError(f"{path or slug}: file names must be lowercase letters, numbers and hyphens")
    known = set(Item.__dataclass_fields__) - {"kind", "slug", "body", "path"}
    unknown = sorted(set(meta) - known)
    if unknown:
        raise ContentError(f"{path or slug}: unknown header field(s): {', '.join(unknown)}")
    if not str(meta.get("title", "")).strip():
        raise ContentError(f"{path or slug}: needs a title")
    for key in ("date", "updated"):
        value = meta.get(key)
        if isinstance(value, dt.datetime):
            meta[key] = value.date()
        elif isinstance(value, str) and value:
            try:
                meta[key] = dt.date.fromisoformat(value)
            except ValueError as error:
                raise ContentError(f"{path or slug}: {key} must be a date like 2026-11-23") from error
    if meta.get("status", "published") not in ("published", "draft"):
        raise ContentError(f"{path or slug}: status must be \"published\" or \"draft\"")
    if kind == "guide" and meta.get("level") and meta["level"] not in LEVELS:
        raise ContentError(f"{path or slug}: level must be one of {', '.join(LEVELS)}")
    if kind == "issue" and meta.get("status", "published") == "published" and not meta.get("date"):
        raise ContentError(f"{path or slug}: a published issue needs a date")
    return Item(kind=kind, slug=slug, body=body, path=path, **meta)


def dump(item: Item) -> str:
    """Write an item back as a file (used when Claude writes a draft)."""
    lines = ["+++"]
    for key in ("title", "subtitle", "description", "date", "updated", "status", "tags", "archetype", "featured",
                "level", "topic", "cover", "cover_alt", "linkedin_url", "fact_checks", "brief"):
        value = getattr(item, key)
        if value in (None, "", [], False) and key not in ("status",):
            continue
        if isinstance(value, dt.date):
            lines.append(f"{key} = {value.isoformat()}")
        elif isinstance(value, bool):
            lines.append(f"{key} = {'true' if value else 'false'}")
        else:
            lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
    lines.append("+++")
    return "\n".join(lines) + "\n" + item.body.strip("\n") + "\n"


def load_kind(kind: str, folder: Path | None = None) -> list[Item]:
    folder = folder or KINDS[kind]
    items = []
    for path in sorted(folder.glob("*.md")):
        items.append(parse(path.read_text(encoding="utf-8"), kind, path.stem, path))
    return items


@dataclass
class Library:
    """Everything the site is built from."""

    issues: list[Item]
    guides: list[Item]
    path: dict
    library: dict
    about: str

    def all(self) -> list[Item]:
        return self.issues + self.guides

    def by_ref(self) -> dict[str, Item]:
        return {item.ref: item for item in self.all()}

    def visible(self, drafts: bool = False) -> "Library":
        keep = (lambda item: True) if drafts else (lambda item: item.published)
        return Library([i for i in self.issues if keep(i)], [g for g in self.guides if keep(g)],
                       self.path, self.library, self.about)


def _toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ContentError(f"{path}: isn't valid TOML: {error}") from error


def load_library(content_dir: Path | None = None) -> Library:
    root = Path(content_dir) if content_dir else CONTENT_DIR
    issues = load_kind("issue", root / "issues")
    guides = load_kind("guide", root / "guides")
    issues.sort(key=lambda i: (i.sort_date, i.slug), reverse=True)
    guides.sort(key=lambda g: (not g.featured, g.title.lower()))
    about_path = root / "about.md"
    about = about_path.read_text(encoding="utf-8") if about_path.exists() else ""
    return Library(issues, guides, _toml(root / "path.toml"), _toml(root / "library.toml"), about)
