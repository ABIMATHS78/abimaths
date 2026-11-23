"""The author's brand (brand.json): name, website, footer, theme and the names posts must never use."""

from __future__ import annotations

import json
import re
from pathlib import Path

from .paths import BRAND_PATH, ENGINE_DIR

DEFAULT_BRAND = {
    "name": "Your Name",
    "headline": "",
    "publication": "",
    "site": "",                   # shown on graphics, e.g. example.com
    "url": "",                    # the full link used in posts, e.g. https://example.com/
    "linkedin": "",
    "avatar": "",
    "footer": {"left": "", "right": ""},
    "theme": "learned",           # or a theme name to use on every post
    "cta": {"resource_line": "", "share_line": ""},
    "never_mention": [],          # names and sites no post or graphic may contain
    "voice_notes": [],
}


def load_brand(path: Path | None = None) -> dict:
    path = Path(path) if path else BRAND_PATH
    brand = json.loads(json.dumps(DEFAULT_BRAND))
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, dict) and isinstance(brand.get(key), dict):
                brand[key].update(value)
            else:
                brand[key] = value
    if not brand["site"] and brand["url"]:
        brand["site"] = site_host(brand["url"])
    if not brand["url"] and brand["site"]:
        brand["url"] = f"https://{brand['site']}/"
    if not brand["footer"].get("left") and brand["name"]:
        pub = brand.get("publication")
        brand["footer"]["left"] = f"from {brand['name']}" + (f", on “{pub}”" if pub else "")
    if not brand["footer"].get("right") and brand.get("site"):
        brand["footer"]["right"] = brand["site"]
    return brand


def site_host(url: str) -> str:
    """'https://www.example.com/page' -> 'example.com'."""
    return re.sub(r"^https?://", "", url.strip().lower()).split("/")[0].removeprefix("www.")


def own_hosts(brand: dict) -> set[str]:
    """Sites the author may always link to."""
    hosts = {site_host(brand.get(key, "")) for key in ("site", "url")}
    return {h for h in hosts if h}


def avatar_path(brand: dict) -> Path | None:
    """The author's photo for the feed preview, if brand.json points to one that exists."""
    raw = brand.get("avatar") or ""
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (ENGINE_DIR / path).resolve()
    return path if path.exists() else None
