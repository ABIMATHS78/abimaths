"""Draw the app icons in static/icons: an "Ai" whose green dot is the companion who's always on.

Needs Pillow and a bold font (Inter ExtraBold by default):

    pip install pillow
    python scripts/make_icons.py [path/to/font.otf]
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent.parent / "static" / "icons"
FONT = sys.argv[1] if len(sys.argv) > 1 else "/usr/share/fonts/opentype/inter/Inter-ExtraBold.otf"
BRAND, BRAND_2, DOT = (91, 76, 219), (138, 125, 255), (25, 211, 154)
MASTER = 1024


def gradient(size: int) -> Image.Image:
    """Brand purple, lighter towards the bottom right."""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x + y) / (2 * (size - 1))
            px[x, y] = tuple(round(a + (b - a) * t) for a, b in zip(BRAND, BRAND_2))
    return img


def mark(size: int, scale: float, ink=(255, 255, 255, 255), dot=DOT + (255,)) -> Image.Image:
    """An "A" and a dotless "ı" with a round dot above it, centred together; `scale` sets the size."""
    font = ImageFont.truetype(FONT, round(size * 0.56 * scale))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    advance = probe.textlength("A", font=font)
    a_box = probe.textbbox((0, 0), "A", font=font)
    i_box = probe.textbbox((advance, 0), "ı", font=font)
    radius = (a_box[3] - a_box[1]) * 0.15
    cx, cy = (i_box[0] + i_box[2]) / 2, i_box[1] - 1.6 * radius
    box = (min(a_box[0], cx - radius), min(a_box[1], cy - radius), max(i_box[2], cx + radius), max(a_box[3], i_box[3]))
    dx, dy = (size - (box[2] - box[0])) / 2 - box[0], (size - (box[3] - box[1])) / 2 - box[1]
    layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    draw.text((dx, dy), "A", font=font, fill=ink)
    draw.text((dx + advance, dy), "ı", font=font, fill=ink)
    draw.ellipse((cx + dx - radius, cy + dy - radius, cx + dx + radius, cy + dy + radius), fill=dot)
    return layer


def icon(size: int, *, rounded: bool, scale: float) -> Image.Image:
    img = gradient(MASTER).convert("RGBA")
    img.alpha_composite(mark(MASTER, scale))
    if rounded:
        mask = Image.new("L", (MASTER, MASTER), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, MASTER - 1, MASTER - 1), radius=MASTER * 0.22, fill=255)
        img.putalpha(mask)
    return img.resize((size, size), Image.LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    icon(512, rounded=True, scale=1).save(OUT / "icon-512.png")
    icon(192, rounded=True, scale=1).save(OUT / "icon-192.png")
    icon(32, rounded=True, scale=1.1).save(OUT / "favicon-32.png")
    # Maskable and Apple icons are full squares: the system cuts the shape, so keep the mark in the safe zone.
    icon(512, rounded=False, scale=0.78).save(OUT / "icon-maskable-512.png")
    icon(180, rounded=False, scale=0.9).convert("RGB").save(OUT / "apple-touch-icon.png")
    # Android shows the badge in the status bar using only its shape, so it's white on transparent.
    mark(MASTER, 1.15, dot=(255, 255, 255, 255)).resize((96, 96), Image.LANCZOS).save(OUT / "badge-96.png")


if __name__ == "__main__":
    main()
