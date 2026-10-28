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
# A written guide's folder: never just a number, like the site's ids.
FOLDER = re.compile(r"^(?=[a-z0-9-]*[a-z])[a-z0-9]+(?:-[a-z0-9]+)*$")
WORDS_PER_MINUTE = 200


class Download(BaseModel):
    """Something a written guide offers to download, from its downloads/ folder. A skill's title and what it's
    about come from its SKILL.md unless given here."""

    model_config = ConfigDict(extra="forbid")
    file: str = Field(pattern=r"^[A-Za-z0-9][\w.-]*(?:/[A-Za-z0-9][\w.-]*)*$")  # inside downloads/, never above it
    name: str | None = Field(default=None, pattern=r"^[A-Za-z0-9][\w.-]*$")  # saved as; the file's own name if none
    title: str | None = Field(default=None, min_length=1, max_length=80)
    about: str | None = Field(default=None, min_length=1, max_length=200)
    group: str | None = Field(default=None, pattern=r"^[a-z0-9-]+$")  # "essentials", or a job in the library
    tag: str | None = Field(default=None, min_length=1, max_length=24)  # a word or two shown on its card


class Job(BaseModel):
    """A job in a guide's skill library: its skills are the folders in downloads/library/<slug>/."""

    model_config = ConfigDict(extra="forbid")
    slug: str = Field(pattern=r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
    name: str = Field(min_length=1, max_length=40)


class Library(BaseModel):
    """A guide's skill library: the essentials (downloads in the group "essentials") and the skills by job."""

    model_config = ConfigDict(extra="forbid")
    essentials: str = Field(min_length=1, max_length=80)  # their heading
    essentials_about: str = Field(min_length=1, max_length=200)
    jobs: list[Job] = Field(min_length=1)


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
    library: Library | None = None

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
    group: str | None = None  # "essentials" or a library job's slug
    tag: str | None = None
    command: str | None = None  # a skill's, such as /learn-anything
    bundle: bool = False  # several skills in one zip


_files: dict[str, dict[str, File]] = {}  # each written guide's downloads by name, from when the guides last loaded
_groups: dict[str, list[dict]] = {}  # each written guide's library: the essentials, then its jobs


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
    found, files, groups = [], {}, {}
    for d in sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []:
        try:
            if not FOLDER.match(d.name):
                raise ValueError("name the folder with lowercase letters, numbers and hyphens, with a letter in it")
            meta = Written.model_validate_json((d / "guide.json").read_text(encoding="utf-8"))
            body = (d / "guide.md").read_text(encoding="utf-8").strip()
            if not body:
                raise ValueError("guide.md is empty")
            offered = [_pack(d / "downloads", x, meta.published) for x in meta.downloads]
            library = _library(d / "downloads", meta) if meta.library else ([], [])
            offered += library[0]
            names = [f.name for f in offered]
            if len(set(names)) < len(names):
                twice = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(f"two downloads have the same name: {', '.join(twice)}")
            known = {g["id"] for g in library[1]}
            if stray := sorted({f.group for f in offered if f.group and f.group not in known}):
                raise ValueError(f"downloads are in groups the library doesn't have: {', '.join(stray)}")
        except (OSError, ValidationError, ValueError) as e:
            raise ValueError(f"The written guide in guide_data/{d.name} doesn't fit: {e}") from e
        files[d.name] = {f.name: f for f in offered}
        groups[d.name] = library[1]
        found.append({
            "id": d.name, "companion": meta.companion, "title": meta.title, "body": body, "topic": meta.topic,
            "summary": meta.summary, "kind": meta.kind, "level": meta.level,
            "minutes": max(1, math.ceil(len(body.split()) / WORDS_PER_MINUTE)), "image": None,
            "published": meta.published.isoformat(),
        })
    _files.clear()
    _files.update(files)
    _groups.clear()
    _groups.update(groups)
    return found


def _skill(folder: Path) -> tuple[str, str, str]:
    """A skill's name, what it does (its description) and its title (its first heading), from its SKILL.md."""
    lines = (folder / "SKILL.md").read_text(encoding="utf-8").splitlines()
    end = None
    if lines and lines[0].strip() == "---":
        end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        raise ValueError(f"{folder.name}/SKILL.md doesn't start with frontmatter between lines of three dashes")
    fields = {}
    for line in lines[1:end]:
        key, sep, value = line.partition(":")
        if sep and line[:1].strip():
            value = value.strip()
            quoted = len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'"
            fields[key.strip()] = value[1:-1] if quoted else value
    if fields.get("name") != folder.name:
        raise ValueError(f"{folder.name}/SKILL.md is named {fields.get('name')!r}: a skill's name is its folder's")
    if not fields.get("description"):
        raise ValueError(f"{folder.name}/SKILL.md has no description")
    title = next((line[2:].strip() for line in lines[end + 1:] if line.startswith("# ")), folder.name)
    return folder.name, fields["description"], title


def _pack(root: Path, d: Download, when: datetime) -> File:
    """A download, read from the guide's downloads/ folder: a folder becomes a zip."""
    path = root / d.file
    if path.is_dir():
        name = d.name or f"{path.name}.zip"
        if not name.endswith(".zip"):
            raise ValueError(f"downloads/{d.file} is a folder, so it's sent as a zip: name it something.zip")
        if (path / "SKILL.md").is_file():
            skill, about, title = _skill(path)
            return File(name, d.title or title, d.about or about, _zip([(path, path.name)], when), "application/zip",
                        d.group, d.tag, f"/{skill}")
        if not (d.title and d.about):
            raise ValueError(f"give downloads/{d.file} a title and a line about it")
        return File(name, d.title, d.about, _zip([(path, None)], when), "application/zip", d.group, d.tag, bundle=True)
    if not path.is_file():
        raise ValueError(f"there's nothing at downloads/{d.file}")
    if not (d.title and d.about):
        raise ValueError(f"give downloads/{d.file} a title and a line about it")
    name = d.name or path.name
    return File(name, d.title, d.about, path.read_bytes(), mimetypes.guess_type(name)[0] or "application/octet-stream",
                d.group, d.tag)


def _library(root: Path, meta: Written) -> tuple[list[File], list[dict]]:
    """A guide's skill library: each job's skills from downloads/library/<job>/, a zip of each and one of them all,
    then one zip of every skill, the essentials included; and its groups, the essentials first."""
    lib, when = meta.library, meta.published
    essentials = [root / x.file for x in meta.downloads
                  if x.group == "essentials" and (root / x.file / "SKILL.md").is_file()]
    files, every = [], list(essentials)
    groups = [{"id": "essentials", "name": lib.essentials, "about": lib.essentials_about, "count": len(essentials)}]
    for job in lib.jobs:
        folder = root / "library" / job.slug
        skills = sorted(p for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []
        if not skills:
            raise ValueError(f"downloads/library/{job.slug} has no skills")
        for s in skills:
            if not (s / "SKILL.md").is_file():
                raise ValueError(f"downloads/library/{job.slug}/{s.name} has no SKILL.md")
            name, description, title = _skill(s)
            files.append(File(f"{name}.zip", title, _what(description), _zip([(s, s.name)], when), "application/zip",
                              job.slug, command=f"/{name}"))
        files.append(File(f"{job.slug}-skills.zip", f"All {job.name} skills",
                          f"The {len(skills)} {job.name} skills in one zip, each in its own folder.",
                          _zip([(folder, None)], when), "application/zip", job.slug, bundle=True))
        groups.append({"id": job.slug, "name": job.name, "count": len(skills)})
        every += skills
    files.append(File("every-skill.zip", f"Every skill in the library ({len(every)})",
                      f"All {len(every)} skills in one zip, each in its own folder, ready to unzip into "
                      "~/.claude/skills/.",
                      _zip([(s, s.name) for s in every], when), "application/zip", bundle=True))
    return files, groups


def _what(description: str) -> str:
    """What a skill does, from its description: the part before "Use when…", for its card."""
    return re.split(r"(?<=\.)\s+(?=Use\b)", description, maxsplit=1)[0]


def _zip(parts: list[tuple[Path, str | None]], when: datetime) -> bytes:
    """Folders' files as one zip, each folder's under the name given with it, if any: a skill's folder goes in whole,
    with itself at the top, as Claude takes skills; a folder of skills puts each skill's folder at the top. Hidden
    files and Python's caches stay out, scripts stay runnable, and every file is dated when the guide was
    published, so the zip only changes when its files do."""
    out = BytesIO()
    stamp = when.astimezone(timezone.utc).timetuple()[:6]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for folder, top in parts:
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
    return [{"name": f.name, "title": f.title, "about": f.about, "size": len(f.data), "group": f.group, "tag": f.tag,
             "command": f.command, "bundle": f.bundle}
            for f in _files.get(_written_id(post) or "", {}).values()]


def download_groups(post: dict) -> list[dict]:
    """A guide's skill library, if it has one: the essentials, then each job, with how many skills it holds."""
    return _groups.get(_written_id(post) or "", [])


def download(post: dict, name: str) -> File | None:
    """One of a guide's downloads by the name it's saved as, if it offers one."""
    return _files.get(_written_id(post) or "", {}).get(name)
