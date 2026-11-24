"""The publication's settings (site.json): its name, the author, where it lives and how readers subscribe."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from .paths import SITE_CONFIG

DEFAULTS = {
    "name": "My publication",
    "tagline": "",
    "description": "",
    "language": "en",
    "base_url": "https://example.com",
    "base_path": "/",
    "author": {"name": "Your Name", "headline": "", "bio": "", "avatar": "", "url": "", "person_id": "",
               "linkedin": "", "same_as": []},
    "main_site": {"name": "", "url": "", "links": []},
    "hero_image": {"src": "", "width": 0, "height": 0, "alt": ""},   # the photo beside the home page headline
    # The name in the header and footer as a wordmark: small bold lead, small light join, big wide name.
    # Empty name: the author's photo and the site name instead.
    "wordmark": {"lead": "", "join": "", "name": ""},
    # The ending of every issue and guide, after the Substack screenshot: a short pitch for the paid subscription,
    # the subscribe form, one line on what's behind the paywall, then the paywall box. "{kind}" in the heading
    # becomes "post" or "guide". Off when enabled is false: the old "Ready for more?" box comes back.
    "paywall": {"enabled": False, "pitch": [], "teaser": "", "heading": "The rest of this {kind} is for paid subscribers",
                "button": "Subscribe", "subscribe_url": "", "sign_in_url": ""},
    "newsletter": {"provider": "none", "username": "", "form_action": "", "embed_url": "", "promise": ""},
    "never_mention": [],
    "pages": {"newsletter_label": "Newsletter", "guides_label": "Guides", "path_label": "Start here",
              "library_label": "Library"},
}

PROVIDERS = ("none", "buttondown", "substack", "beehiiv", "kit", "form")


class Site:
    """Settings plus the two link helpers every page uses."""

    def __init__(self, data: dict):
        self.data = data
        self.base_url = data["base_url"].rstrip("/")
        self.base_path = "/" + data["base_path"].strip("/") + "/" if data["base_path"].strip("/") else "/"

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def link(self, path: str = "") -> str:
        """A link inside the publication: 'p/my-issue/' -> '/learn/p/my-issue/'."""
        path = path.lstrip("/")
        return self.base_path + path

    def absolute(self, path: str = "") -> str:
        return self.base_url + self.link(path)

    @property
    def host(self) -> str:
        return re.sub(r"^https?://", "", self.base_url).split("/")[0].removeprefix("www.")


def load_site(path: Path | None = None) -> Site:
    data = copy.deepcopy(DEFAULTS)
    path = Path(path) if path else SITE_CONFIG
    if path.exists():
        for key, value in json.loads(path.read_text(encoding="utf-8")).items():
            if isinstance(value, dict) and isinstance(data.get(key), dict):
                data[key].update(value)
            else:
                data[key] = value
    provider = data["newsletter"].get("provider") or "none"
    if provider not in PROVIDERS:
        raise ValueError(f"site.json: newsletter.provider must be one of {', '.join(PROVIDERS)}, not {provider!r}")
    return Site(data)
