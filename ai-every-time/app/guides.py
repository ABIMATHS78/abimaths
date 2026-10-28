"""The education guides: tutorials and cheat sheets written by the starter companions.

Most come from aieverytime.org: scripts/import_guides.py turns an export of the site's articles into
guides.json.gz, beside this file, and the app loads it when it starts. Each guide becomes a post by the companion
who wrote it, dated when the site published it, with more than a post has: a summary, its kind (tutorial or cheat
sheet), level, reading time and its graphic. The graphic is the site's own, with the image generator's spelling
mistakes corrected, in static/guides/<article id>.webp (scripts/graphics/README.md tells how); a guide whose
graphic hasn't been corrected yet shows the site's copy.

Guides written here are in guide_data/, a folder each, named for the guide: guide.json says who wrote it, its
title, summary, kind, level, topic, when it was published and what it offers to download; guide.md is its text;
downloads/ holds the files. A download is a file, a skill's folder (sent as a zip with the folder at the top, the
way Claude takes skills) or a folder of skills (a zip with each skill's folder at the top). They load with the
guides from the site, and their graphic is static/guides/<folder name>.webp. A guide that doesn't fit stops the
app with the reason, like a course file (tests/test_guides.py checks them too).

Loading the guides again after they change updates them in place, so their likes, views and the chats that
started from them stay; guides that are new are added and any no longer there are removed. Guides that haven't
changed since they were last loaded aren't loaded again.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import mimetypes
import re
import stat
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from app import db
from app.config import settings

log = logging.getLogger(__name__)
GUIDES_PATH = Path(__file__).with_name("guides.json.gz")
WRITTEN = Path(__file__).with_name("guide_data")  # the guides written here, a folder each
IMAGES = Path(__file__).resolve().parent.parent / "static" / "guides"  # the corrected graphics, <article id>.webp
SOURCE = "aieverytime:"  # posts.source of a guide: this, then the article's id on the site or the guide's folder
KINDS = ("tutorial", "cheat_sheet")
LEVELS = ("beginner", "intermediate", "advanced")
FOLDER = re.compile(r"^(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*$")  # a written guide's: never a number like the site's ids
WORDS_PER_MINUTE = 200


class Download(BaseModel):
    """Something a written guide offers to download, from its downloads/ folder."""

    model_config = ConfigDict(extra="forbid")
    file: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)*$")  # inside downloads/, never above it
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][\w.-]*$")  # saved as; the file's own name if none
    title: str = Field(min_length=1, max_length=80)
    about: str = Field(min_length=1, max_length=200)


class Written(BaseModel):
    """A written guide's guide.json."""

    model_config = ConfigDict(extra="forbid")
    companion: str = Field(min_length=1)
    title: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=400)
    kind: str
    level: str
    topic: str = Field(min_length=1, max_length=80)
    published: datetime
    downloads: list[Download] = []

    @field_validator("kind")
    @classmethod
    def _kind(cls, v: str) -> str:
        if v not in KINDS:
            raise ValueError(f"kind must be one of {', '.join(KINDS)}")
        return v

    @field_validator("level")
    @classmethod
    def _level(cls, v: str) -> str:
        if v not in LEVELS:
            raise ValueError(f"level must be one of {', '.join(LEVELS)}")
        return v

    @field_validator("published")
    @classmethod
    def _zone(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            raise ValueError("published needs a time zone, such as 2026-10-28T09:00:00Z")
        return v


@dataclass(frozen=True)
class File:
    """A download, ready to send."""

    name: str
    title: str
    about: str
    data: bytes
    media_type: str


_files: dict[str, dict[str, File]] = {}  # each written guide's downloads by name, from when the guides last loaded


def path() -> Path | None:
    if settings.guides == "0":
        return None
    return Path(settings.guides) if settings.guides else GUIDES_PATH


def seed() -> int | None:
    """Load the guides if the file changed since they were last loaded: how many there are, or None if
    nothing was loaded."""
    file = path()
    if file is None or not file.exists():
        return None
    raw = file.read_bytes()
    mine = written()
    with db.connect() as conn:
        companion = {r["slug"]: r["id"] for r in conn.execute("SELECT id, slug FROM companions WHERE is_starter = 1")}
        row = conn.execute("SELECT value FROM meta WHERE key = 'guides'").fetchone()
    # The file, the guides written here, the starters they were loaded for and the corrected graphics there are: a
    # starter or a graphic added later is taken up even if the guides are the same.
    corrected = sorted(p.stem for p in IMAGES.glob("*.webp")) if IMAGES.is_dir() else []
    digest = hashlib.sha256(raw + json.dumps(mine, sort_keys=True).encode()
                            + "\n".join(sorted(companion) + ["--"] + corrected).encode()).hexdigest()
    if row and row["value"] == digest:
        return None
    guides = json.loads(gzip.decompress(raw))["guides"] + mine
    corrected = set(corrected)
    with db.connect() as conn:
        existing = {r["source"]: r["id"] for r in conn.execute(
            "SELECT id, source FROM posts WHERE source LIKE ? ESCAPE '\\'", (SOURCE + "%",))}
        kept, unknown, added = set(), set(), 0
        for g in guides:
            cid = companion.get(g["companion"])
            if cid is None:
                unknown.add(g["companion"])
                continue
            source = SOURCE + str(g["id"])
            graphic = f"/static/guides/{g['id']}.webp" if str(g["id"]) in corrected else g.get("image")
            values = (cid, g["title"], g["body"], g["topic"], g["summary"], g["kind"], g["level"], g["minutes"], graphic,
                      _when(g["published"]))
            if source in existing:
                conn.execute("UPDATE posts SET companion_id = ?, title = ?, body = ?, topic = ?, summary = ?, kind = ?, "
                             "level = ?, minutes = ?, graphic = ?, created_at = ? WHERE id = ?", (*values, existing[source]))
            else:
                conn.execute("INSERT INTO posts (companion_id, title, body, topic, summary, kind, level, minutes, graphic, "
                             "created_at, source) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (*values, source))
                added += 1
            kept.add(source)
        gone = [(pid,) for source, pid in existing.items() if source not in kept]
        conn.executemany("DELETE FROM posts WHERE id = ?", gone)
        if added:
            # The guides are the companions' library, not news: they don't arrive as hundreds of "new posts".
            conn.execute("UPDATE users SET feed_seen_post_id = (SELECT COALESCE(MAX(id), 0) FROM posts)")
        conn.execute("INSERT INTO meta (key, value) VALUES ('guides', ?) ON CONFLICT (key) DO UPDATE SET value = excluded.value",
                     (digest,))
    if unknown:
        log.warning("Guides by companions that aren't starters here were skipped: %s", ", ".join(sorted(unknown)))
    log.info("Loaded %d guides (%d new, %d removed) from %s", len(kept), added, len(gone), file)
    return len(kept)


def _when(published: str) -> str:
    """The publication time as the app writes times (db.now()), so guides and other posts sort together."""
    t = datetime.fromisoformat(published.replace("Z", "+00:00"))
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def written(folder: Path | None = None) -> list[dict]:
    """The guides written here, as guides.json.gz has guides, each checked; their downloads are kept for download().
    One that doesn't fit raises ValueError, naming it and saying why."""
    folder = folder or WRITTEN
    found, files = [], {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []:
        try:
            if not FOLDER.match(d.name):
                raise ValueError("name the folder with lowercase letters, numbers and hyphens, with a letter in it")
            meta = Written.model_validate_json((d / "guide.json").read_text(encoding="utf-8"))
            body = (d / "guide.md").read_text(encoding="utf-8").strip()
            if not body:
                raise ValueError("guide.md is empty")
            offered = [_pack(d / "downloads", x, meta.published) for x in meta.downloads]
            names = [f.name for f in offered]
            if len(set(names)) < len(names):
                raise ValueError("two downloads have the same name")
        except (OSError, ValidationError, ValueError) as e:
            raise ValueError(f"The written guide in guide_data/{d.name} doesn't fit: {e}") from e
        files[d.name] = {f.name: f for f in offered}
        found.append({
            "id": d.name, "companion": meta.companion, "title": meta.title, "body": body, "topic": meta.topic,
            "summary": meta.summary, "kind": meta.kind, "level": meta.level,
            "minutes": max(1, math.ceil(len(body.split()) / WORDS_PER_MINUTE)), "image": None,
            "published": meta.published.isoformat(),
        })
    _files.clear()
    _files.update(files)
    return found


def _pack(root: Path, d: Download, when: datetime) -> File:
    """A download, read from the guide's downloads/ folder: a folder becomes a zip."""
    path = root / d.file
    if path.is_dir():
        name = d.name or f"{path.name}.zip"
        if not name.endswith(".zip"):
            raise ValueError(f"downloads/{d.file} is a folder, so it's sent as a zip: name it something.zip")
        skill = (path / "SKILL.md").is_file()
        return File(name, d.title, d.about, _zip(path, path.name if skill else None, when), "application/zip")
    if not path.is_file():
        raise ValueError(f"there's nothing at downloads/{d.file}")
    name = d.name or path.name
    return File(name, d.title, d.about, path.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream")


def _zip(folder: Path, top: str | None, when: datetime) -> bytes:
    """A folder's files as a zip, under `top` if given: a skill's folder goes in whole, with itself at the top, as
    Claude takes skills; a folder of skills puts each skill's folder at the top. Hidden files and Python's caches
    stay out, scripts stay runnable, and every file is dated when the guide was published, so the zip only changes
    when its files do."""
    out = BytesIO()
    stamp = when.astimezone(timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(p for p in folder.rglob("*") if p.is_file()):
            rel = f.relative_to(folder)
            if any(part.startswith(".") or part == "__pycache__" for part in rel.parts) or f.suffix == ".pyc":
                continue
            info = zipfile.ZipInfo((Path(top) / rel if top else rel).as_posix(), date_time=stamp)
            info.external_attr = (stat.S_IFREG | stat.S_IMODE(f.stat().st_mode)) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, f.read_bytes())
    return out.getvalue()


def _written_id(post: dict) -> str | None:
    source = post.get("source") or ""
    return source[len(SOURCE):] if source.startswith(SOURCE) else None


def downloads(post: dict) -> list[dict]:
    """What a guide offers to download, for its page: none for most."""
    return [{"name": f.name, "title": f.title, "about": f.about, "size": len(f.data)}
            for f in _files.get(_written_id(post) or "", {}).values()]


def download(post: dict, name: str) -> File | None:
    """One of a guide's downloads by the name it's saved as, if it offers one."""
    return _files.get(_written_id(post) or "", {}).get(name)
