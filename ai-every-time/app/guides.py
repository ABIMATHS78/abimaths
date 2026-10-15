"""The education guides from aieverytime.org: tutorials and cheat sheets written by the starter companions.

scripts/import_guides.py turns an export of the site's articles into guides.json.gz, beside this file, and the
app loads it when it starts. Each guide becomes a post by the companion who wrote it, dated when the site
published it, with more than a post has: a summary, its kind (tutorial or cheat sheet), level, reading time and
the description of its infographic, which static/infographic.js draws.

Loading the file again after it changes updates the guides in place, so their likes, views and the chats that
started from them stay; guides new to the file are added and any no longer in it are removed. A file that
hasn't changed since it was last loaded isn't read at all.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from app import db
from app.config import settings

log = logging.getLogger(__name__)
GUIDES_PATH = Path(__file__).with_name("guides.json.gz")
SOURCE = "aieverytime:"  # posts.source of a guide: this, then the article's id on the site
KINDS = ("tutorial", "cheat_sheet")
LEVELS = ("beginner", "intermediate", "advanced")


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
    with db.connect() as conn:
        companion = {r["slug"]: r["id"] for r in conn.execute("SELECT id, slug FROM companions WHERE is_starter = 1")}
        row = conn.execute("SELECT value FROM meta WHERE key = 'guides'").fetchone()
    # The file and the starters it was loaded for: a starter added later gets its guides even if the file is the same.
    digest = hashlib.sha256(raw + "\n".join(sorted(companion)).encode()).hexdigest()
    if row and row["value"] == digest:
        return None
    guides = json.loads(gzip.decompress(raw))["guides"]
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
            values = (cid, g["title"], g["body"], g["topic"], g["summary"], g["kind"], g["level"], g["minutes"],
                      json.dumps(g["graphic"], ensure_ascii=False, separators=(",", ":")), _when(g["published"]))
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
