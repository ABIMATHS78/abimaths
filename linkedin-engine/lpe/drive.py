"""Download a Google Drive folder that is shared as "Anyone with the link".

Drive's embedded folder view lists every file in a shared folder without an API key, so
training can re-fetch its data on any machine from the folder link alone.
"""

from __future__ import annotations

import html
import os
import re
import ssl
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

FOLDER_ID_RE = re.compile(r"(?:/folders/|[?&]id=)([A-Za-z0-9_-]{10,})")
BARE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")
ENTRY_RE = re.compile(
    r'<div class="flip-entry" id="entry-([A-Za-z0-9_-]+)".*?<a href="([^"]+)".*?'
    r'<div class="flip-entry-title">([^<]*)</div>',
    re.S,
)
USER_AGENT = "linkedin-post-engine/1.0"


@dataclass
class DriveFile:
    id: str
    name: str
    is_folder: bool = False


def parse_folder_id(url_or_id: str) -> str:
    """Accept a folder link (any of Drive's URL shapes) or a bare folder ID."""
    text = url_or_id.strip()
    match = FOLDER_ID_RE.search(text)
    if match:
        return match.group(1)
    if BARE_ID_RE.match(text):
        return text
    raise ValueError(f"Not a Google Drive folder link or ID: {url_or_id!r}")


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:
                return response.read()
        except Exception as error:  # network errors: retry with backoff
            last_error = error
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Could not download {url}: {last_error}")


def parse_listing(page: str) -> list[DriveFile]:
    files = []
    for file_id, href, name in ENTRY_RE.findall(page):
        files.append(DriveFile(id=file_id, name=html.unescape(name).strip(), is_folder="/folders/" in href))
    return files


def list_folder(folder_id: str) -> list[DriveFile]:
    page = _get(f"https://drive.google.com/embeddedfolderview?id={folder_id}").decode("utf-8", "replace")
    files = parse_listing(page)
    if not files and "flip-entries" not in page:
        raise RuntimeError(
            "Drive didn't return a file list. Check the folder is shared as 'Anyone with the link'."
        )
    return files


def download_file(file_id: str, dest: Path) -> int:
    data = _get(f"https://drive.google.com/uc?export=download&id={file_id}")
    if data[:15].lower().startswith(b"<!doctype html") or data[:5].lower() == b"<html":
        raise RuntimeError(f"Drive returned a web page instead of {dest.name}; is the file shared?")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(dest)
    return len(data)


def sync_folder(url_or_id: str, dest_dir: Path, log=print, recursive: bool = True) -> list[Path]:
    """Download every file in the folder (and its subfolders) that isn't already in dest_dir."""
    folder_id = parse_folder_id(url_or_id)
    dest_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for item in list_folder(folder_id):
        if item.is_folder:
            if recursive:
                saved += sync_folder(item.id, dest_dir / _safe_name(item.name), log=log)
            continue
        target = dest_dir / _safe_name(item.name)
        if target.exists() and target.stat().st_size > 0:
            saved.append(target)
            continue
        size = download_file(item.id, target)
        log(f"  downloaded {item.name} ({size / 1024:.0f} KB)")
        saved.append(target)
    return saved


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|\x00-\x1f]", "_", name).strip().strip(".")
    return cleaned or "untitled"
