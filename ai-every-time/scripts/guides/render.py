"""Renders the graphics and printable downloads of the guides written here (app/guide_data) from the HTML in
scripts/guides/<guide>/: graphic.html becomes static/guides/<guide>.webp, a 1024 × 1024 design saved at twice the
pixels so its words stay sharp, and each NAME.pdf.html becomes NAME.pdf in the guide's downloads/ folder.

Run from ai-every-time/:  python scripts/guides/render.py [GUIDE ...]

It needs Playwright and its Chromium: pip install -r scripts/screenshots/requirements.txt, then
playwright install chromium. Each page waits for its fonts, so no text is drawn in a stand-in typeface.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent  # ai-every-time/
GRAPHICS = APP / "static" / "guides"
GUIDES = APP / "app" / "guide_data"
SIZE, SCALE = 1024, 2


def render(page: Page, folder: Path) -> None:
    gid = folder.name
    if not (GUIDES / gid).is_dir():
        raise SystemExit(f"There's no written guide called {gid} in app/guide_data")
    graphic = folder / "graphic.html"
    if graphic.exists():
        page.set_viewport_size({"width": SIZE, "height": SIZE})
        page.goto(graphic.as_uri())
        page.evaluate("document.fonts.ready.then(() => true)")
        out = GRAPHICS / f"{gid}.webp"
        Image.open(io.BytesIO(page.screenshot())).convert("RGB").save(out, "WEBP", quality=90, method=6)
        print(f"{out.relative_to(APP)}: {out.stat().st_size:,} bytes")
    for html in sorted(folder.glob("*.pdf.html")):
        page.goto(html.as_uri())
        page.evaluate("document.fonts.ready.then(() => true)")
        out = GUIDES / gid / "downloads" / html.name.removesuffix(".html")
        page.pdf(path=str(out), prefer_css_page_size=True, print_background=True)
        print(f"{out.relative_to(APP)}: {out.stat().st_size:,} bytes")


def main() -> None:
    names = sys.argv[1:]
    folders = [HERE / n for n in names] if names else sorted(p for p in HERE.iterdir() if p.is_dir() and p.name[0] != "_")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(device_scale_factor=SCALE)
        for folder in folders:
            render(page, folder)
        browser.close()


if __name__ == "__main__":
    main()
