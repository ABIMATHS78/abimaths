"""Look at a graphic before correcting it: show.py <id>

Writes, in work/view/<id>/:
  full.png          the graphic as it is (1024 x 1024)
  suspects.png      (and suspects_2.png, ... if there are many) the lines with likely mistakes, enlarged, each
                    labelled with its line number
  tile_RC.png       the graphic enlarged 2x, in four quarters (row R, column C), every line the text reader found
                    outlined in magenta with its number
and prints every line the reader found (number, box [x0, y0, x1, y1] in the graphic's own pixels, what the reader
made of it, how sure it was), then the likely mistakes with corrections suggested from the guide's own words.
The reader's text is often wrong (that's why the graphic needs correcting): read the picture itself.
"""

import json
import os
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import suspects as sus  # noqa: E402

from paths import WORK  # noqa: E402
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
SHEET_W, SHEET_H = 1400, 1500


def tiles(img: Image.Image, lines: list, out: str) -> None:
    scale = 2
    big = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
    draw = ImageDraw.Draw(big)
    f = ImageFont.truetype(FONT, 15)
    for i, line in enumerate(lines):
        x0, y0, x1, y1 = [v * scale for v in line["box"]]
        draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), outline=(255, 0, 200), width=1)
        draw.text((x0 - 2, y0 - 19), str(i), font=f, fill=(255, 0, 200), stroke_width=2, stroke_fill="white")
    half = big.width // 2
    for r in range(2):
        for c in range(2):
            big.crop((c * half, r * half, (c + 1) * half, (r + 1) * half)).save(os.path.join(out, f"tile_{r}{c}.png"))


def sheets(img: Image.Image, found: list, out: str) -> list[str]:
    """The suspect lines, each enlarged so its letters are about 34 pixels tall, stacked with their numbers."""
    f = ImageFont.truetype(FONT, 22)
    strips = []
    for s in found:
        x0, y0, x1, y1 = s["box"]
        x0, y0, x1, y1 = max(0, x0 - 6), max(0, y0 - 4), min(1024, x1 + 6), min(1024, y1 + 4)
        h = max(1, y1 - y0)
        scale = max(1.0, min(4.0, 44 / h, (SHEET_W - 70) / max(1, x1 - x0)))
        crop = img.crop((x0, y0, x1, y1)).resize((round((x1 - x0) * scale), round(h * scale)), Image.LANCZOS)
        strips.append((s["line"], crop))
    paths, page, used = [], [], 0
    def flush():
        if not page:
            return
        height = sum(c.height + 10 for _, c in page) + 10
        sheet = Image.new("RGB", (SHEET_W, height), (236, 236, 240))
        d = ImageDraw.Draw(sheet)
        y = 10
        for n, crop in page:
            d.text((8, y + max(0, crop.height // 2 - 12)), str(n), font=f, fill=(200, 0, 150))
            sheet.paste(crop, (64, y))
            y += crop.height + 10
        name = "suspects.png" if not paths else f"suspects_{len(paths) + 1}.png"
        path = os.path.join(out, name)
        sheet.save(path)
        paths.append(path)
    for n, crop in strips:
        if page and used + crop.height + 10 > SHEET_H:
            flush()
            page, used = [], 0
        page.append((n, crop))
        used += crop.height + 10
    flush()
    return paths


def main(gid: str) -> None:
    src = os.path.join(WORK, "orig", f"{gid}.png")
    if not os.path.exists(src):
        raise SystemExit(f"no graphic {gid}")
    lines = json.load(open(os.path.join(WORK, "ocr", f"{gid}.json")))
    out = os.path.join(WORK, "view", gid)
    os.makedirs(out, exist_ok=True)
    for f in os.listdir(out):
        if f.startswith("suspects"):
            os.remove(os.path.join(out, f))
    img = Image.open(src).convert("RGB")
    img.save(os.path.join(out, "full.png"))
    tiles(img, lines, out)
    found = sus.find(gid, lines)
    sheet_paths = sheets(img, found, out)
    print(f"Graphic: {out}/full.png")
    print(f"Enlarged quarters, lines numbered: {out}/tile_00.png (top left), tile_01.png (top right), "
          "tile_10.png (bottom left), tile_11.png (bottom right)")
    if sheet_paths:
        print("Lines with likely mistakes, enlarged: " + ", ".join(sheet_paths))
    guide = os.path.join(WORK, "guide", gid + ".md")
    print(f"Guide text: {guide}")
    if os.path.exists(guide):
        head = open(guide, encoding="utf-8").read().split("\n")
        print(f"  title: {head[0].lstrip('# ').strip()}")
        print(f"  summary: {head[2].strip() if len(head) > 2 else ''}")
    print(f"\n{len(lines)} lines found (number, box, what the reader made of it, how sure):")
    for i, line in enumerate(lines):
        print(f"{i:3d} {line['box']} {line['text']!r} {line['score']}")
    print(f"\nLikely mistakes ({len(found)} lines; the reader's misreadings show up here too, and it misses some):")
    for s in found:
        words = "; ".join(
            f"{b['word']!r}" + (" (case)" if b.get("case") else " (repeated)" if b.get("repeat") else "")
            + (f" → {' / '.join(b['try'])}?" if b["try"] else "") for b in s["bad"])
        print(f"{s['line']:3d} {words}")


if __name__ == "__main__":
    main(sys.argv[1])
