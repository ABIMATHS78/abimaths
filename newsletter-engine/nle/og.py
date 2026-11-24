"""Social images (1200x630 PNG): the title card drawn in the site's style, so a shared link shows a picture.

Uses headless Chromium through Playwright when it is installed (set NLE_CHROMIUM to a Chrome or Chromium binary
to choose one); the build skips this step, and pages use the default image, when it isn't.
"""

from __future__ import annotations

import base64
import html
import os
from pathlib import Path

from .paths import ENGINE_DIR, STATIC_DIR

FONTS = STATIC_DIR / "fonts"
CARD = """<!doctype html><html><head><meta charset="utf-8"><style>
@font-face{{font-family:"Anton";src:url("{anton}") format("woff2");font-weight:400}}
@font-face{{font-family:"Oswald";src:url("{oswald}") format("woff2");font-weight:400 700}}
@font-face{{font-family:"Open Sans";src:url("{open_sans}") format("woff2");font-weight:400 800}}
*{{box-sizing:border-box;margin:0}}
body{{width:1200px;height:630px;background:#d02e2e;color:#fffae7;font-family:"Open Sans",sans-serif;overflow:hidden;position:relative}}
.shade{{position:absolute;inset:0;background:linear-gradient(160deg,rgba(255,255,255,.07) 0%,rgba(255,255,255,0) 45%,rgba(33,17,14,.24) 100%)}}
.glow{{position:absolute;right:-170px;top:-250px;width:760px;height:760px;border-radius:50%;background:radial-gradient(circle,rgba(208,110,46,.8),rgba(208,110,46,0) 66%)}}
.band{{position:absolute;left:0;right:0;bottom:0;height:14px;background:#21110e}}
.wrap{{position:absolute;inset:0;padding:60px 72px 74px;display:flex;flex-direction:column}}
.kicker{{align-self:flex-start;font-family:Oswald,sans-serif;font-weight:700;font-size:22px;letter-spacing:.22em;text-transform:uppercase;color:#fffae7;
  border:2px solid rgba(255,250,231,.55);border-radius:999px;padding:8px 20px}}
.title{{font-family:Anton,Impact,sans-serif;font-weight:400;font-size:{size}px;line-height:.98;letter-spacing:.01em;text-transform:uppercase;margin-top:auto;max-width:1040px;text-wrap:balance}}
.sub{{font-size:27px;line-height:1.35;color:rgba(255,250,231,.9);margin-top:18px;max-width:940px}}
.foot{{display:flex;align-items:center;gap:16px;margin-top:30px;font-family:Oswald,sans-serif;font-weight:700;font-size:22px;letter-spacing:.14em;text-transform:uppercase;color:rgba(255,250,231,.85)}}
.foot img{{width:60px;height:60px;border-radius:50%;object-fit:cover;border:3px solid #fffae7}}
.foot b{{color:#fffae7;font-weight:700}}
</style></head><body><div class="shade"></div><div class="glow"></div><div class="band"></div><div class="wrap">
<div class="kicker">{kicker}</div><div class="title">{title}</div>{sub}
<div class="foot">{avatar}<span><b>{author}</b> · {host}</span></div></div></body></html>"""


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def card_html(entry, site) -> str:
    title = entry.title
    size = 112 if len(title) <= 24 else 98 if len(title) <= 36 else 84 if len(title) <= 52 else 70 if len(title) <= 76 else 60
    subtitle = entry.subtitle if len(entry.subtitle) <= 120 else ""
    avatar_path = ENGINE_DIR / site["author"].get("avatar", "")
    avatar = ""
    if site["author"].get("avatar") and avatar_path.is_file():
        mime = "image/webp" if avatar_path.suffix == ".webp" else "image/png" if avatar_path.suffix == ".png" else "image/jpeg"
        avatar = f'<img src="{_data_uri(avatar_path, mime)}" alt="">'
    return CARD.format(
        anton=_data_uri(FONTS / "anton.woff2", "font/woff2"), oswald=_data_uri(FONTS / "oswald.woff2", "font/woff2"),
        open_sans=_data_uri(FONTS / "open-sans.woff2", "font/woff2"),
        size=size, kicker=html.escape(entry.kind_label), title=html.escape(title),
        sub=f'<div class="sub">{html.escape(subtitle)}</div>' if subtitle else "",
        avatar=avatar, author=html.escape(site["author"]["name"]), host=html.escape(site.host))


def _launch(playwright):
    executable = os.environ.get("NLE_CHROMIUM") or None
    candidates = [executable] if executable else [None, "/opt/pw-browsers/chromium"]
    last = None
    for candidate in candidates:
        if candidate and not Path(candidate).exists():
            continue
        try:
            return playwright.chromium.launch(executable_path=candidate)
        except Exception as error:
            last = error
    raise RuntimeError("Chromium isn't available: run `python -m playwright install chromium` or set NLE_CHROMIUM") from last


def render_og_images(entries, site, out_dir: Path, env=None) -> list[str]:
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    with sync_playwright() as p:
        browser = _launch(p)
        try:
            page = browser.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
            for entry in entries:
                page.set_content(card_html(entry, site), wait_until="load")
                page.evaluate("document.fonts.ready.then(() => true)")
                page.screenshot(path=str(out_dir / f"{entry.slug}.png"), clip={"x": 0, "y": 0, "width": 1200, "height": 630})
                made.append(entry.ref)
        finally:
            browser.close()
    return made
