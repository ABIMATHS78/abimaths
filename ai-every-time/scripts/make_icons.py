"""Make the app's logo image and icons in static/icons, and the iPhone app's icon and launch logo in
ios/AIEveryTime/Assets.xcassets, from the AI Every Time logo, scripts/logo.webp.

Run it again whenever the logo changes:

    pip install pillow
    python scripts/make_icons.py
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

SOURCE = Path(__file__).resolve().with_name("logo.webp")
OUT = Path(__file__).resolve().parent.parent / "static" / "icons"
IOS = Path(__file__).resolve().parent.parent / "ios" / "AIEveryTime" / "Assets.xcassets"
LANCZOS = Image.Resampling.LANCZOS
WHITE = (255, 255, 255, 255)


def logo() -> Image.Image:
    """The logo with everything outside the clock's outer ring made transparent, so it sits on any background."""
    im = Image.open(SOURCE).convert("RGBA")
    big = im.width * 4  # a smooth edge: the circle is drawn large, then shrunk
    mask = Image.new("L", (big, big), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, big - 1, big - 1), fill=255)
    im.putalpha(mask.resize(im.size, LANCZOS))
    return im


def save_png(img: Image.Image, name: str) -> None:
    """A PNG of at most 128 colours: the logo has few, and it keeps the files a tenth of the size."""
    img.convert("RGBA").quantize(colors=128, method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE).save(
        OUT / name, optimize=True)


def centred(clock: Image.Image, size: int, across: float, background=None) -> Image.Image:
    """The clock, `across` of the width, in the middle of a square of `background` (transparent without one)."""
    canvas = Image.new("RGBA", (size, size), background or (0, 0, 0, 0))
    d = round(size * across)
    canvas.alpha_composite(clock.resize((d, d), LANCZOS), ((size - d) // 2, (size - d) // 2))
    return canvas


def badge(size: int) -> Image.Image:
    """Android shows the notification badge using only its shape, so it's the clock's ring and hands, at ten past
    ten as in the logo, white on transparent."""
    big = size * 8
    shape = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(shape)
    c, ring, width = big / 2, big * 0.43, big * 0.085
    draw.ellipse((c - ring, c - ring, c + ring, c + ring), outline=255, width=round(width))
    # The hands as the logo has them: the long one towards two o'clock, the short one towards ten.
    for degrees, length in ((52, 0.66), (-57, 0.44)):
        a = math.radians(degrees)
        end = (c + ring * length * math.sin(a), c - ring * length * math.cos(a))
        draw.line((c, c, *end), fill=255, width=round(width))
        r = width / 2
        draw.ellipse((end[0] - r, end[1] - r, end[0] + r, end[1] + r), fill=255)  # round ends
    draw.ellipse((c - width, c - width, c + width, c + width), fill=255)
    img = Image.new("RGBA", (big, big), WHITE)
    img.putalpha(shape)
    return img.resize((size, size), LANCZOS)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clock = logo()
    # Shown beside the name on the site's pages, up to 64 pixels across on screens up to three times as dense.
    clock.resize((256, 256), LANCZOS).save(OUT / "logo-256.webp", quality=90, method=6)
    for size, name in ((32, "favicon-32.png"), (192, "icon-192.png"), (512, "icon-512.png")):
        save_png(clock.resize((size, size), LANCZOS), name)
    # Phones cut their own shape out of these, so they're full squares with the clock well inside: Android keeps a
    # circle 80% across (maskable), and iOS rounds the corners of a square it fills with black where it's clear.
    save_png(centred(clock, 512, 0.78, WHITE), "icon-maskable-512.png")
    save_png(centred(clock, 180, 0.9, WHITE), "apple-touch-icon.png")
    badge(96).save(OUT / "badge-96.png", optimize=True)
    # The iPhone app's icon: a full square with no transparency, as the App Store requires; iOS rounds the corners.
    centred(clock, 1024, 0.86, WHITE).convert("RGB").save(IOS / "AppIcon.appiconset" / "AppIcon.png", optimize=True)
    # The launch screen's logo, 160 points across, on screens of one, two and three pixels to the point.
    for scale in (1, 2, 3):
        name = "LaunchLogo.png" if scale == 1 else f"LaunchLogo@{scale}x.png"
        clock.resize((160 * scale, 160 * scale), LANCZOS).save(IOS / "LaunchLogo.imageset" / name, optimize=True)


if __name__ == "__main__":
    main()
