"""A close look at part of a graphic, with a ruler to measure boxes: crop.py <id> x0 y0 x1 y1 [--fixed] [--both]

The region (in the graphic's own pixels) is enlarged with a ruler along the top and left: a tick every 10 pixels and
a number every 50, in the graphic's own coordinates. --fixed shows the corrected graphic, --both the original above
the corrected one. Saved as work/view/<id>/crop_<x0>_<y0>[_fixed|_both].png.
"""

import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK  # noqa: E402
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


def ruled(img: Image.Image, x0: int, y0: int, x1: int, y1: int, scale: float) -> Image.Image:
    crop = img.crop((x0, y0, x1, y1))
    w, h = round((x1 - x0) * scale), round((y1 - y0) * scale)
    crop = crop.resize((w, h), Image.LANCZOS)
    m = 34
    canvas = Image.new("RGB", (w + m, h + m), "white")
    canvas.paste(crop, (m, m))
    d = ImageDraw.Draw(canvas)
    f = ImageFont.truetype(FONT, 11)
    for x in range((x0 // 10) * 10, x1 + 1, 10):
        if x < x0:
            continue
        px = m + round((x - x0) * scale)
        major = x % 50 == 0
        d.line((px, m - (10 if major else 5), px, m - 1), fill=(200, 0, 0) if major else (120, 120, 120))
        if major:
            d.text((px + 2, 2), str(x), font=f, fill=(200, 0, 0))
    for y in range((y0 // 10) * 10, y1 + 1, 10):
        if y < y0:
            continue
        py = m + round((y - y0) * scale)
        major = y % 50 == 0
        d.line((m - (10 if major else 5), py, m - 1, py), fill=(200, 0, 0) if major else (120, 120, 120))
        if major:
            d.text((1, py + 1), str(y), font=f, fill=(200, 0, 0))
    return canvas


def main(argv: list[str]) -> None:
    if len([a for a in argv if not a.startswith("--")]) < 5:
        raise SystemExit("usage: crop.py <id> x0 y0 x1 y1 [--both | --fixed]")
    argv = [a for a in argv if not a.startswith("--")] + [a for a in argv if a.startswith("--")]
    gid = argv[0]
    x0, y0, x1, y1 = (int(v) for v in argv[1:5])
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(1024, x1), min(1024, y1)
    if x1 <= x0 or y1 <= y0:
        raise SystemExit("empty region")
    which = "both" if "--both" in argv else "fixed" if "--fixed" in argv else "orig"
    scale = min(8.0, 1400 / (x1 - x0), 900 / (y1 - y0) if which != "both" else 450 / (y1 - y0))
    scale = max(1.0, scale)
    out = os.path.join(WORK, "view", gid)
    os.makedirs(out, exist_ok=True)
    orig = Image.open(os.path.join(WORK, "orig", f"{gid}.png")).convert("RGB")
    if which == "orig":
        result = ruled(orig, x0, y0, x1, y1, scale)
        name = f"crop_{x0}_{y0}.png"
    else:
        fixed_path = os.path.join(WORK, "fixed", f"{gid}.png")
        if not os.path.exists(fixed_path):
            raise SystemExit("no corrected graphic yet: run fix_one.py first")
        fixed = Image.open(fixed_path).convert("RGB")
        if which == "fixed":
            result = ruled(fixed, x0, y0, x1, y1, scale)
            name = f"crop_{x0}_{y0}_fixed.png"
        else:
            a, b = ruled(orig, x0, y0, x1, y1, scale), ruled(fixed, x0, y0, x1, y1, scale)
            result = Image.new("RGB", (a.width, a.height * 2 + 6), (230, 40, 40))
            result.paste(a, (0, 0))
            result.paste(b, (0, a.height + 6))
            name = f"crop_{x0}_{y0}_both.png"
    path = os.path.join(out, name)
    result.save(path)
    print(path, f"(enlarged {scale:.1f}x)")


if __name__ == "__main__":
    main(sys.argv[1:])
