"""The author's brand (brand.json): name, links, footer and default theme used on every post."""

from __future__ import annotations

import json
from pathlib import Path

from .paths import BRAND_PATH, ENGINE_DIR

DEFAULT_BRAND = {
    "name": "Your Name",
    "headline": "",
    "publication": "",
    "site": "",
    "linkedin": "",
    "avatar": "",
    "footer": {"left": "", "right": ""},
    "theme": "learned",           # or a theme name to use on every post
    "cta": {"resource_line": "", "share_line": ""},
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
    if not brand["footer"].get("left") and brand["name"]:
        pub = brand.get("publication")
        brand["footer"]["left"] = f"from {brand['name']}" + (f", on “{pub}”" if pub else "")
    if not brand["footer"].get("right") and brand.get("site"):
        brand["footer"]["right"] = f"more at {brand['site']}"
    return brand


def avatar_path(brand: dict) -> Path | None:
    """The author's photo for the feed preview, if brand.json points to one that exists."""
    raw = brand.get("avatar") or ""
    if not raw:
        return None
    path = Path(raw)
    if not path.is_absolute():
        path = (ENGINE_DIR / path).resolve()
    return path if path.exists() else None
