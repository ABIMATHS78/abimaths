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
@font-face{{font-family:"Fraunces";src:url("{fraunces}") format("woff2");font-weight:100 900}}
@font-face{{font-family:"Inter";src:url("{inter}") format("woff2");font-weight:100 900}}
*{{box-sizing:border-box;margin:0}}
body{{width:1200px;height:630px;background:#000;color:#f4f1ea;font-family:Inter,sans-serif;overflow:hidden;position:relative}}
.dots{{position:absolute;inset:0;background-image:radial-gradient(rgba(255,255,255,.11) 1.3px,transparent 1.6px);background-size:28px 28px;
  -webkit-mask-image:linear-gradient(100deg,#000 0%,rgba(0,0,0,.5) 55%,transparent 90%)}}
.glow{{position:absolute;right:-160px;top:-220px;width:720px;height:720px;border-radius:50%;background:radial-gradient(circle,rgba(69,217,156,.38),rgba(69,217,156,0) 68%)}}
.wrap{{position:absolute;inset:0;padding:64px 72px;display:flex;flex-direction:column}}
.kicker{{font-weight:700;font-size:22px;letter-spacing:.16em;text-transform:uppercase;color:#45d99c}}
.title{{font-family:Fraunces,serif;font-weight:600;font-size:{size}px;line-height:1.04;letter-spacing:-.025em;margin-top:auto;max-width:1000px;text-wrap:balance}}
.sub{{font-size:27px;line-height:1.35;color:#c9cfd8;margin-top:20px;max-width:900px}}
.foot{{display:flex;align-items:center;gap:16px;margin-top:36px;font-size:24px;color:#a6adb7}}
.foot img{{width:60px;height:60px;border-radius:50%;object-fit:cover;border:2px solid #1d2228}}
.foot b{{color:#f4f1ea;font-weight:600}}
</style></head><body><div class="dots"></div><div class="glow"></div><div class="wrap">
<div class="kicker">{kicker}</div><div class="title">{title}</div>{sub}
<div class="foot">{avatar}<span><b>{author}</b> · {host}</span></div></div></body></html>"""


def _data_uri(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def card_html(entry, site) -> str:
    title = entry.title
    size = 92 if len(title) <= 28 else 78 if len(title) <= 48 else 64 if len(title) <= 72 else 54
    subtitle = entry.subtitle if len(entry.subtitle) <= 120 else ""
    avatar_path = ENGINE_DIR / site["author"].get("avatar", "")
    avatar = ""
    if site["author"].get("avatar") and avatar_path.is_file():
        mime = "image/webp" if avatar_path.suffix == ".webp" else "image/png" if avatar_path.suffix == ".png" else "image/jpeg"
        avatar = f'<img src="{_data_uri(avatar_path, mime)}" alt="">'
    return CARD.format(
        fraunces=_data_uri(FONTS / "fraunces.woff2", "font/woff2"), inter=_data_uri(FONTS / "inter.woff2", "font/woff2"),
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
