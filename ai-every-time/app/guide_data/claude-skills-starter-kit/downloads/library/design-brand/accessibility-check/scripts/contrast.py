#!/usr/bin/env python3
"""Checks the contrast between text and background colours against WCAG 2.2.

Usage:  python contrast.py FOREGROUND BACKGROUND [FOREGROUND BACKGROUND ...]
        python contrast.py "#767676" "#ffffff" 1a1a1a f5f5f5

Colours are hex codes, with or without "#", in three or six digits. For each pair it prints the contrast ratio,
and whether it passes levels AA and AAA for normal text, large text, and the parts of buttons, form fields,
icons and charts that people need to see. Large text is at least 24 CSS pixels (18 point), or 18.66 pixels
(14 point) when bold.

It uses only Python's standard library, so it runs wherever Python does.
"""

import re
import sys

HEX = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
# The lowest passing ratio for each kind of content, at levels AA and AAA. WCAG has no AAA level for
# components and graphics.
NEEDS = {
    "normal text": (4.5, 7.0),
    "large text": (3.0, 4.5),
    "components and graphics": (3.0, None),
}


def rgb(colour: str) -> tuple[int, int, int]:
    match = HEX.match(colour.strip())
    if not match:
        raise ValueError(f'"{colour}" is not a hex colour such as #1a1a1a or #fff')
    digits = match.group(1)
    if len(digits) == 3:
        digits = "".join(d * 2 for d in digits)
    return int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16)


def luminance(colour: str) -> float:
    """WCAG's relative luminance: 0 for black, 1 for white."""
    def linear(channel: int) -> float:
        c = channel / 255
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (linear(c) for c in rgb(colour))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(foreground: str, background: str) -> float:
    lighter, darker = sorted((luminance(foreground), luminance(background)), reverse=True)
    return (lighter + 0.05) / (darker + 0.05)


def main() -> int:
    colours = sys.argv[1:]
    if not colours or len(colours) % 2:
        print(__doc__.split("\n\n")[1])
        return 2
    for foreground, background in zip(colours[::2], colours[1::2]):
        try:
            value = ratio(foreground, background)
        except ValueError as e:
            print(e)
            return 2
        # WCAG doesn't round: 4.49:1 fails 4.5:1, so the ratio is cut to two places, never rounded up.
        print(f"{foreground} on {background}: {int(value * 100) / 100:.2f}:1")
        for kind, (aa, aaa) in NEEDS.items():
            results = [f"AA {'pass' if value >= aa else 'fail'}"]
            if aaa:
                results.append(f"AAA {'pass' if value >= aaa else 'fail'}")
            print(f"  {kind}: {', '.join(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
