"""Starter companions' photos, kept small on this server.

Each starter's photo is fetched once from its address in starter_companions.csv, cut to a square and saved as
a small WebP image in companion_photos, the table learners' uploaded photos use. Pages then load every face from
this site, quickly, whatever size the original is. Until a photo is here, or if it can't be fetched, browsers are
sent to the original address, and a drawn face stands in if that fails too (see the avatar endpoint in app.main).
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
import time

import httpx
from PIL import Image, ImageOps

from app import db
from app.config import settings

log = logging.getLogger(__name__)
SIZE = 320  # pixels square: sharp at the largest size pages show a face, on high-density screens too
MAX_BYTES = 25 * 1024 * 1024  # originals larger than this are skipped
RETRY_SECONDS = 600  # after a failed fetch, wait this long before trying that photo again
# Tests point this at an httpx.MockTransport; None means the network.
transport: httpx.AsyncBaseTransport | None = None
_tried: dict[int, float] = {}
_DRIVE = re.compile(r"^https://drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^#]*&)?id=)([\w-]{10,})")


def image_type(data: bytes) -> str | None:
    """Identify an image by its bytes, never by the name or header it came with."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def direct_url(url: str) -> str:
    """The address of the picture itself. A Google Drive sharing link opens a page about the file, so it's
    swapped for the file's thumbnail, which is the picture (if the file is shared with anyone who has the link)."""
    url = url.strip()
    m = _DRIVE.match(url)
    return f"https://drive.google.com/thumbnail?id={m.group(1)}&sz=w1000" if m else url


def thumbnail(data: bytes) -> bytes:
    """A square WebP from the middle of the picture, nearer the top for tall ones, where faces usually are."""
    with Image.open(io.BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGBA" if im.mode in ("RGBA", "LA", "PA") or "transparency" in im.info else "RGB")
        w, h = im.size
        side = min(w, h)
        left, top = (w - side) // 2, (h - side) // 4
        im = im.crop((left, top, left + side, top + side)).resize((SIZE, SIZE), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        im.save(out, "WEBP", quality=82, method=6)
    return out.getvalue()


def _client(timeout: float = 30) -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=timeout, follow_redirects=True, transport=transport,
                             headers={"User-Agent": "AI Every Time (starter companion photos)"})


async def fetch(client: httpx.AsyncClient, url: str) -> bytes:
    """The picture at `url` as a small WebP. Raises on anything that isn't a PNG, JPEG or WebP picture."""
    async with client.stream("GET", direct_url(url)) as r:
        r.raise_for_status()
        if int(r.headers.get("content-length") or 0) > MAX_BYTES:
            raise ValueError("the picture is too large")
        data = bytearray()
        async for chunk in r.aiter_bytes():
            data += chunk
            if len(data) > MAX_BYTES:
                raise ValueError("the picture is too large")
    if image_type(bytes(data)) is None:
        raise ValueError("it isn't a PNG, JPEG or WebP picture (is the file shared with anyone who has the link?)")
    return await asyncio.to_thread(thumbnail, bytes(data))


def _store(companion_id: int, url: str, webp: bytes) -> None:
    with db.connect() as conn:
        # Only if the address is still the one fetched: the CSV may have changed in the meantime.
        conn.execute("INSERT OR REPLACE INTO companion_photos (companion_id, mime, data, updated_at) "
                     "SELECT id, 'image/webp', ?, ? FROM companions WHERE id = ? AND is_starter = 1 AND photo_url = ?",
                     (webp, db.now(), companion_id, url))


def _missing(companion_id: int | None = None) -> list:
    with db.connect() as conn:
        return conn.execute(
            "SELECT c.id, c.photo_url, json_extract(c.persona, '$.name') AS name FROM companions c "
            "WHERE c.is_starter = 1 AND c.photo_url != '' AND (:id IS NULL OR c.id = :id) "
            "AND NOT EXISTS (SELECT 1 FROM companion_photos p WHERE p.companion_id = c.id) ORDER BY c.id",
            {"id": companion_id}).fetchall()


async def _get(client: httpx.AsyncClient, row) -> str | None:
    """Fetch and keep one starter's photo. Returns why it couldn't be, or None once it's stored."""
    _tried[row["id"]] = time.monotonic()
    try:
        _store(row["id"], row["photo_url"], await fetch(client, row["photo_url"]))
    except httpx.HTTPStatusError as e:
        return f"the address answered {e.response.status_code}"
    except Exception as e:  # noqa: BLE001 - any failure leaves the browser fetching the original instead
        return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
    return None


async def ensure_starter_photos() -> int:
    """At start-up: fetch every starter's photo that isn't here yet, a few at a time. Returns how many it stored."""
    if not settings.fetch_photos:
        return 0
    rows = _missing()
    if not rows:
        return 0
    gate = asyncio.Semaphore(4)

    async def one(row) -> str | None:
        async with gate:
            return await _get(client, row)

    async with _client() as client:
        problems = await asyncio.gather(*(one(r) for r in rows))
    failed = [(r["name"], p) for r, p in zip(rows, problems) if p]
    if failed:
        log.warning("Couldn't fetch %d of %d starter companion photos (%s: %s). Browsers load those from their original "
                    "addresses instead; the server tries again later.", len(failed), len(rows), *failed[0])
    else:
        log.info("Stored %d starter companion photos", len(rows))
    return len(rows) - len(failed)


async def fetch_later(companion_id: int) -> None:
    """Fetch one starter's photo that isn't here yet, after a page asked for it, unless that was tried recently."""
    if time.monotonic() - _tried.get(companion_id, -RETRY_SECONDS) < RETRY_SECONDS:
        return
    for row in _missing(companion_id):
        async with _client() as client:
            if problem := await _get(client, row):
                log.debug("couldn't fetch the photo of %s: %s", row["name"], problem)
