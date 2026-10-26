"""Correct the wording of lines of text in an image, leaving everything else in it as it was.

    fix_text.py spec.json in_dir out_dir [--only id,id] [--debug]

spec.json maps an image id to its edits, or to {"family": ..., "edits": [...]}. An edit replaces the text of one line:

    "box": [x0, y0, x1, y1]    the line, in image pixels (from OCR)
    "was": "..."               what it says now, as near as can be read: used to match the font
    "text": "..."              what it should say; "" erases it
    "replace": "old words"     replace only these words of the line (as they are in "was"); the others stay as they are
    "words": [i, j]            the same by position: words i..j of the line, counted from 0
    "align": left|center|right where the new text sits in the old one's place (default left)
    "style": "name"            lines of one style share a font (chosen from all of them) and, if alike, a size
    "family": sans|condensed|serif|mono   the fonts to choose from (default: the image's, else sans and condensed)
    "typeface": "Roboto"       choose among this typeface's weights only (also for the whole image)
    "squeeze": 0.9             draw the new text this much narrower (or wider, above 1); default: as the old text
    "slant": 0.2               lean the new letters right (0.2 is about 11 degrees): italic text in a font without one
    "ref": true                a line that's right as it is, only there to show its style's font (it isn't changed)
    "font", "size", "color", "pad", "threshold", "room", "dy"   overrides

The old letters are found by their contrast with the background, the background is rebuilt under them (a flat colour,
a fitted gradient, or inpainting where it is neither) and the new text is drawn on the old baseline, in the font,
weight, size, colour and softness that best match the old letters.
"""

from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys
from functools import lru_cache

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paths import WORK  # noqa: E402

# The fonts to choose from: those get_fonts.py downloads, and the system's (Debian/Ubuntu packages fonts-inter,
# fonts-liberation, fonts-crosextra-carlito, fonts-crosextra-caladea, fonts-dejavu-core, fonts-freefont-ttf).
FONT_DIRS = [os.path.join(WORK, "fonts"), os.path.join(HERE, "..", "fonts"), os.path.join(HERE, "..", "fonts2"),
             "/usr/share/fonts"]

FAMILY_PREFIXES = {
    "sans": ["LiberationSans-", "Carlito-", "DejaVuSans.", "DejaVuSans-Bold.", "FreeSans.", "FreeSansBold.", "Inter-Light",
             "Inter-Regular", "Inter-Medium", "Inter-SemiBold", "Inter-Bold", "Inter-ExtraBold", "Inter-Black", "Roboto-",
             "OpenSans-", "Lato-", "Montserrat-", "Poppins-", "SourceSans3-", "NunitoSans-", "Nunito-", "IBMPlexSans-",
             "WorkSans-", "DMSans-", "Manrope-", "Barlow-", "Archivo-", "Rubik-", "Raleway-", "SpaceGrotesk-",
             "JosefinSans-", "FiraSans-", "PTSans-", "NotoSans-", "Ubuntu-", "TitilliumWeb-", "Mulish-", "Heebo-",
             "Karla-", "Hind-", "Cabin-", "Exo2-", "MerriweatherSans-", "LibreFranklin-", "PublicSans-", "Figtree-",
             "Outfit-", "PlusJakartaSans-", "Lexend-", "Urbanist-", "Sora-", "Kanit-", "Prompt-"],
    "condensed": ["RobotoCondensed-", "Oswald-", "BebasNeue-", "Anton-", "ArchivoBlack-", "BarlowCondensed-",
                  "BarlowSemiCondensed-", "FiraSansCondensed-", "PTSansNarrow-", "ArchivoNarrow-", "IBMPlexSansCondensed-",
                  "SairaCondensed-", "AsapCondensed-", "EncodeSansCondensed-", "Rajdhani-", "Teko-", "FjallaOne-",
                  "PathwayGothicOne-", "Antonio-", "LeagueGothic-"],
    "serif": ["Merriweather-", "Lora-", "PlayfairDisplay-", "LibreBaskerville-", "NotoSerif-", "PTSerif-", "CrimsonPro-",
              "Fraunces-", "DMSerifDisplay-", "LiberationSerif-", "Caladea-", "DejaVuSerif", "FreeSerif", "SourceSerif4-",
              "EBGaramond-"],
    "mono": ["RobotoMono-", "Inconsolata-", "DejaVuSansMono", "LiberationMono-", "SourceCodePro-", "JetBrainsMono-",
             "FiraCode-", "IBMPlexMono-"],
}
ITALIC = ("Italic", "Oblique")


@lru_cache(None)
def all_fonts() -> tuple[str, ...]:
    found = []
    for d in FONT_DIRS:
        for ext in ("woff", "ttf", "otf"):
            found += glob.glob(os.path.join(d, "**", f"*.{ext}"), recursive=True)
    seen, unique = set(), []
    for f in sorted(set(found)):
        if os.path.basename(f) not in seen:
            seen.add(os.path.basename(f))
            unique.append(f)
    return tuple(unique)


@lru_cache(None)
def family(name: str) -> tuple[str, ...]:
    names = name.split("+")
    out = []
    for f in all_fonts():
        base = os.path.basename(f)
        if any(s in base for s in ITALIC):
            continue
        if any(base.startswith(p) for n in names for p in FAMILY_PREFIXES[n]):
            out.append(f)
    return tuple(out)


def find_font(name: str) -> str:
    for f in all_fonts():
        if os.path.basename(f) == name or os.path.basename(f).rsplit(".", 1)[0] == name:
            return f
    raise SystemExit(f"no font {name}")


@lru_cache(4096)
def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(None)
def has_glyph(path: str, ch: str) -> bool:
    """Whether the font draws this character itself, rather than its stand-in for missing ones."""
    if ch.isspace():
        return True
    f = font(path, 40)
    def ink(c):
        m = f.getmask(c)
        return m.size, bytes(m)
    return ink(ch) != ink("\ue000")


def covers(path: str, *texts: str) -> bool:
    return all(has_glyph(path, ch) for t in texts for ch in set(t))


def hexcolor(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(round(c)) for c in rgb[:3])


# --- Reading the old line -------------------------------------------------------------------------------------------

def edge_background(region: np.ndarray) -> tuple[np.ndarray, float]:
    """The colour behind the letters, and how much it varies (0 for a flat colour).

    Of the region's commonest colours, the one that fills areas rather than thin strokes: letters and the borders
    of boxes are strokes a few pixels wide, which an opening (erode, then dilate) wipes out, while the background
    is left. How much it varies is measured on the edge pixels of about its colour (a gradient varies there)."""
    px = region.reshape(-1, 3).astype(float)
    q = (px // 16).astype(int)
    keys = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    values, counts = np.unique(keys, return_counts=True)
    best, bg = -1, None
    for key in values[np.argsort(-counts)[:3]]:
        mean = px[keys == key].mean(axis=0)
        near = (np.abs(region.astype(float) - mean).sum(axis=2) < 40).astype(np.uint8)
        area = int(cv2.morphologyEx(near, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).sum())
        if area > best:
            best, bg = area, np.median(px[near.reshape(-1).astype(bool)], axis=0)
    edge = np.concatenate([region[0], region[-1], region[:, 0], region[:, -1]]).astype(float)
    close = edge[np.abs(edge - bg).sum(axis=1) < 60]
    spread = float(np.median(np.abs(close - bg).sum(axis=1))) if len(close) else 99.0
    return bg, spread


def ink(region: np.ndarray, bg: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """How much each pixel is letter rather than background (0..1), and the letters' colour."""
    diff = region.astype(float) - bg
    dist = np.abs(diff).sum(axis=2)
    strong = dist > threshold
    if strong.sum() < 4:
        return np.zeros(dist.shape), np.array([0.0, 0.0, 0.0])
    # The letters' own colour: their most contrasting pixels (the edges are blended with the background).
    d = dist[strong]
    color = np.median(region[strong][d >= np.percentile(d, 85)].astype(float), axis=0)
    axis = color - bg
    soft = np.clip((diff @ axis) / max(1e-6, axis @ axis), 0, 1)
    return soft, color


def drop_strays(binary: np.ndarray, top: int, bottom: int, baseline: int | None = None) -> np.ndarray:
    """The letters only: not the borders, rules and boxes the region's edges caught, nor the letters of the lines
    above and below that reach into it (whatever is centred outside the line's own rows, top to bottom)."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    h, w = binary.shape
    keep = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        touches = x == 0 or y == 0 or x + bw == w or y + bh == h
        long = bw > 0.6 * w and bh < 0.25 * h or bh > 0.92 * h and bw < 0.08 * w
        centre = y + bh / 2
        # Cut off by the region's top or bottom without reaching far into the line: a neighbouring line's letters.
        reach = 0.45 * (bottom - top)
        cut = y == 0 and y + bh < top + reach or y + bh == h and y > bottom - reach
        if cut and baseline is not None and y + bh == h and y <= baseline + 1:
            cut = False  # it hangs from this line's baseline: a comma or a descender
        # A thin rule (a connector, an underline) or a sliver of something at the region's left or right edge.
        rule = touches and bh <= 2 and bw >= 3 * bh and area >= 0.6 * bw * bh  # (a hyphen touches nothing)
        sliver = (x == 0 or x + bw == w) and bh < 0.35 * (bottom - top)
        keep[i] = (not (touches and (long or bh >= h - 1 or bw >= w - 1)) and top - 1 <= centre <= bottom + 1
                   and not cut and not rule and not sliver)
    return keep[labels]


def profile(binary: np.ndarray) -> dict | None:
    """Where the line's letters are: their left and right, top, baseline and x-height, and how heavy they are."""
    ys, xs = np.nonzero(binary)
    if len(xs) == 0:
        return None
    rows = binary.sum(axis=1).astype(float)
    m = rows.max()
    body = np.nonzero(rows >= 0.3 * m)[0]
    baseline = int(body.max()) + 1
    top = int(np.nonzero(rows > 0.02 * m)[0].min())
    dense = np.nonzero(rows >= 0.55 * m)[0]
    xh = max(1, baseline - int(dense.min()))
    area = binary.sum()
    edge = binary & ~cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    stroke = 2.0 * area / max(1, edge.sum())
    return {"left": int(xs.min()), "right": int(xs.max()) + 1, "top": top, "baseline": baseline,
            "bottom": int(ys.max()) + 1, "height": baseline - top, "xh": xh, "stroke": stroke}


def softness(soft: np.ndarray) -> float:
    """How blurred the letters' edges are: partly covered pixels per fully covered one."""
    return float(((soft > 0.12) & (soft < 0.88)).sum() / max(1, (soft >= 0.5).sum()))


def word_spans(binary: np.ndarray, prof: dict, expected: int | None = None) -> list[tuple[int, int]]:
    """The columns each word of the line spans: split where the gap between letters is a space's width.

    With the number of words the line has, its widest gaps are the spaces; otherwise a space is taken to be about
    half an x-height wide (in capitals, whose x-height is their height, about a third of it)."""
    cols = binary[prof["top"]:prof["bottom"]].any(axis=0)
    runs, start = [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x))
            start = None
    if not runs:
        return []
    gaps = [runs[k + 1][0] - runs[k][1] for k in range(len(runs) - 1)]
    caps = prof["height"] <= prof["xh"] * 1.15
    gap = max(3, round((0.3 if caps else 0.45) * (prof["height"] if caps else prof["xh"])))
    if expected and 1 < expected <= len(runs):
        ranked = sorted(gaps, reverse=True)
        space, letter = ranked[expected - 2], (ranked[expected - 1] if expected - 1 < len(ranked) else 0)
        if space > letter and space >= 2:
            gap = (space + letter) / 2 + 0.01
    spans = [list(runs[0])]
    for (a, b), g in zip(runs[1:], gaps):
        if g >= gap:
            spans.append([a, b])
        else:
            spans[-1][1] = b
    return [tuple(x) for x in spans]


# --- Drawing the new line -------------------------------------------------------------------------------------------

SHORTLIST = 30  # fonts looked at closely, of those that pass a first rough look
SUPER = 4  # text is drawn this many times larger, then scaled down: letters land between pixels, as in a photo


def draw_mask(text: str, path: str, size: float, squeeze: float = 1.0, blur: float = 0.0,
              slant: float = 0.0) -> tuple[np.ndarray, int, int]:
    """The text as coverage (0..1), with the position of its left edge and baseline in the array."""
    f = font(path, max(4, round(size * SUPER)))
    l, t, r, b = f.getbbox(text, anchor="ls")
    pad = (4 + int(blur * 3)) * SUPER + round(abs(slant) * (b - t))  # with room for leaning letters
    w, h = r - l + 2 * pad, b - t + 2 * pad
    w, h = w + (-w) % SUPER, h + (-h) % SUPER
    img = Image.new("L", (max(SUPER, w), max(SUPER, h)), 0)
    ImageDraw.Draw(img).text((pad - l, pad - t), text, font=f, fill=255, anchor="ls")
    if slant:
        # Leaning letters, for an italic the font lacks: each row moves right by its height above the baseline.
        img = img.transform(img.size, Image.AFFINE, (1, slant, -slant * (pad - t), 0, 1, 0), resample=Image.BICUBIC)
    big = np.asarray(img, dtype=np.float32) / 255
    a = cv2.resize(big, (max(1, round(big.shape[1] / SUPER * squeeze)), big.shape[0] // SUPER),
                   interpolation=cv2.INTER_AREA).astype(float)
    x0, base = round(pad / SUPER * squeeze), (pad - t) // SUPER
    # The strokes solid: thin strokes rarely cover a whole pixel. Then softened, if asked, as a photo softens them.
    if (a > 0.05).any():
        peak = np.percentile(a[a > 0.05], 98)
        a = np.clip(a / max(0.5, peak), 0, 1)
    if blur > 0:
        a = cv2.GaussianBlur(a, (0, 0), blur)
    return a, x0, base


def text_profile(text: str, path: str, size: float) -> dict | None:
    a, _, _ = draw_mask(text, path, size)
    return profile(a >= 0.5)


def quarter(x: float) -> float:
    return max(5.0, round(x * 4) / 4)


def size_for(text: str, path: str, height: int, measure: str) -> float:
    """The size at which the font's letters are as tall as the old ones (top to baseline, or the x-height)."""
    size = quarter(height * 1.4)
    for _ in range(5):
        p = text_profile(text, path, size)
        if not p or p[measure] < 1:
            return size
        new = quarter(size * height / p[measure])
        if new == size:
            break
        size = new
    return size


def overlap(a: np.ndarray, ax: int, ab: int, b: np.ndarray, bx: int, bb: int) -> float:
    """How much two lines of letters coincide, laid on the same left edge and baseline."""
    left, base = max(ax, bx), max(ab, bb)
    h = base + max(a.shape[0] - ab, b.shape[0] - bb)
    w = left + max(a.shape[1] - ax, b.shape[1] - bx)
    A, B = np.zeros((h, w)), np.zeros((h, w))
    A[base - ab:base - ab + a.shape[0], left - ax:left - ax + a.shape[1]] = a
    B[base - bb:base - bb + b.shape[0], left - bx:left - bx + b.shape[1]] = b
    return float(np.minimum(A, B).sum() / max(1e-6, np.maximum(A, B).sum()))


@lru_cache(20000)
def glance(path: str, sample: str) -> tuple[dict, float] | None:
    """The sample drawn once, large and plainly: its profile and how much ink it has."""
    f = font(path, 48)
    l, t, r, b = f.getbbox(sample, anchor="ls")
    img = Image.new("L", (max(1, r - l + 8), max(1, b - t + 8)), 0)
    ImageDraw.Draw(img).text((4 - l, 4 - t), sample, font=f, fill=255, anchor="ls")
    a = np.asarray(img, dtype=float) / 255
    p = profile(a >= 0.5)
    return (p, float(a.sum())) if p else None


def quick(path: str, sample: str, old: dict, letters: np.ndarray) -> float:
    """A first, rough look at how well a font fits: scaled to the old letters' height, how wide is it and how
    much ink has it. Only the fonts that pass it are looked at closely."""
    g = glance(path, sample)
    if g is None:
        return 99.0
    p, mass = g
    measure = "height" if old["height"] > old["xh"] * 1.15 else "xh"
    k = old[measure] / max(1, p[measure])
    width = math.log(max(1, (p["right"] - p["left"]) * k) / max(1, old["right"] - old["left"]))
    ink = math.log(max(1.0, mass * k * k) / max(1.0, letters.sum()))
    xh = math.log(max(1, p["xh"] * k) / max(1, old["xh"]))
    return 3.0 * abs(width) + 3.0 * abs(ink) + 1.5 * abs(xh)


def judge(path: str, sample: str, old: dict, letters: np.ndarray, old_words: list | None = None,
          allow_squeeze: bool = True) -> dict:
    """How well a font matches the old letters: same height, then compare width, stroke, ink and shapes.

    Shapes are compared word by word, each word laid on its old one: along a whole line, small differences in the
    letters' widths add up and would put the end of the line out of step whatever the font."""
    measure = "height" if old["height"] > old["xh"] * 1.15 else "xh"
    size = size_for(sample, path, old[measure], measure)
    a, x0, base = draw_mask(sample, path, size)
    p = profile(a >= 0.5)
    if p is None:
        return {"path": path, "size": size, "by_width": size, "score": 99}
    width = math.log(max(1, p["right"] - p["left"]) / max(1, old["right"] - old["left"]))
    stroke = math.log(max(0.3, p["stroke"]) / max(0.3, old["stroke"]))
    xh = math.log(max(1, p["xh"]) / max(1, old["xh"]))
    # The size the line's width asks for: of a long line, a much finer measure than its letters' height in pixels.
    by_width = size / math.exp(width)
    # A font of the right height and shapes but the wrong width can be drawn a little narrower or wider, as a
    # condensed or extended cut of it would be (up to 18% narrower, 10% wider); that costs something too.
    squeeze = 1.0
    if abs(width) > 0.04 and allow_squeeze:
        squeeze = float(min(1.10, max(0.82, math.exp(-width))))
        a, x0, base = draw_mask(sample, path, size, squeeze=squeeze)
        p = profile(a >= 0.5)
        if p is None:
            return {"path": path, "size": size, "by_width": by_width, "score": 99, "squeeze": squeeze}
        width = math.log(max(1, p["right"] - p["left"]) / max(1, old["right"] - old["left"]))
        stroke = math.log(max(0.3, p["stroke"]) / max(0.3, old["stroke"]))
    # Shapes are compared as blurred as the old letters are: a sharp bold font would otherwise fit soft letters best.
    target, soft = softness(letters), a
    for sigma in (0.5, 0.8, 1.1, 1.5):
        if softness(soft) >= target * 0.85:
            break
        soft = cv2.GaussianBlur(a, (0, 0), sigma)
        soft = soft / max(0.3, soft.max())
    iou = None
    n = len(sample.split())
    if old_words and n == len(old_words) > 1:
        new_words = word_spans(a >= 0.5, p, n)
        if len(new_words) == n:
            parts = [(overlap(soft[:, nl:nr], 0, p["baseline"], letters[:, ol:orr], 0, old["baseline"]), orr - ol)
                     for (nl, nr), (ol, orr) in zip(new_words, old_words)]
            iou = sum(v * w for v, w in parts) / max(1, sum(w for _, w in parts))
    if iou is None:
        iou = overlap(soft, p["left"], p["baseline"], letters, old["left"], old["baseline"])
    # How much ink: blurring spreads the letters' ink but doesn't change how much there is, so this tells their weight
    # even when they're small and soft.
    mass = math.log(max(1.0, a.sum()) / max(1.0, letters.sum()))
    small = old["height"] < 16
    # Small letters' shapes are a few pixels each: there, how much ink and how wide tell the font better.
    score = (3.0 * abs(width) + (0.5 if small else 2.0) * abs(stroke) + 1.5 * abs(xh) + (0.75 if small else 2.0) * (1 - iou)
             + (3.0 if small else 2.0) * abs(mass) + 1.5 * abs(math.log(squeeze)))
    return {"path": path, "size": size, "by_width": by_width, "score": score, "width": width, "stroke": stroke, "xh": xh,
            "iou": iou, "mass": mass, "squeeze": squeeze}


# --- Rebuilding the background ----------------------------------------------------------------------------------------

def fill(out: np.ndarray, mask: np.ndarray, box: tuple[int, int, int, int], bg: np.ndarray, spread: float) -> np.ndarray:
    """Paint over the masked pixels with what's behind them: the background's colour where it is flat, otherwise a
    smooth blend of the nearby background pixels (only those of about its colour, so a box's border or an icon
    next to the letters doesn't bleed into the gap they leave)."""
    if not mask.any():
        return out
    if spread < 4:
        out[mask] = bg
        return out
    X0, Y0, X1, Y1 = box
    m = 12
    x0, y0, x1, y1 = max(0, X0 - m), max(0, Y0 - m), min(out.shape[1], X1 + m), min(out.shape[0], Y1 + m)
    sub = out[y0:y1, x0:x1].astype(np.float32)
    hole = mask[y0:y1, x0:x1]
    near = np.abs(sub - bg).sum(axis=2) < max(36.0, 4 * spread)
    w = (near & ~hole).astype(np.float32)
    result = sub.copy()
    todo = hole.copy()
    for sigma in (1.5, 3, 6, 12, 24):
        if not todo.any():
            break
        den = cv2.GaussianBlur(w, (0, 0), sigma)
        num = cv2.GaussianBlur(sub * w[..., None], (0, 0), sigma)
        ok = todo & (den > 0.05)
        result[ok] = num[ok] / den[ok][..., None]
        todo &= ~ok
    result[todo] = bg
    out[y0:y1, x0:x1][hole] = np.clip(result[hole], 0, 255).astype(np.uint8)
    return out


def match_tone(cover: np.ndarray, letters: np.ndarray) -> np.ndarray:
    """The new letters as dark as the old ones: their strokes' darkest tenth matched to the old letters' (a little,
    no more: the font's own weight is already the old letters')."""
    old = letters[letters > 0.15]
    new = cover[cover > 0.15]
    if len(old) < 20 or len(new) < 20:
        return cover
    gain = float(np.clip(np.quantile(old, 0.9) / max(0.05, np.quantile(new, 0.9)), 0.85, 1.15))
    return np.clip(cover * gain, 0, 1)


def paint(out: np.ndarray, cover: np.ndarray, left: int, top: int, color: np.ndarray) -> None:
    h, w = cover.shape
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(out.shape[1], left + w), min(out.shape[0], top + h)
    if x1 <= x0 or y1 <= y0:
        return
    c = cover[y0 - top:y1 - top, x0 - left:x1 - left][..., None]
    sub = out[y0:y1, x0:x1].astype(float)
    out[y0:y1, x0:x1] = np.clip(sub * (1 - c) + color * c, 0, 255).astype(np.uint8)


# --- Edits --------------------------------------------------------------------------------------------------------------

def read_line(img: np.ndarray, e: dict) -> dict | None:
    """The old line: its background, letters (how much each pixel is letter), colour, profile and words."""
    x0, y0, x1, y1 = e["box"]
    pad = e.get("pad", 3)
    H, W = img.shape[:2]
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    region = img[Y0:Y1, X0:X1]
    bg, spread = edge_background(region)
    soft, color = ink(region, bg, e.get("threshold", 60))
    binary = drop_strays(soft >= 0.5, y0 - Y0, y1 - Y0)
    prof = profile(binary)
    if prof is None:
        return None
    binary = drop_strays(soft >= 0.5, y0 - Y0, y1 - Y0, prof["baseline"])
    # The letters' colour from the letters alone, not from the outlines and icons the region also caught.
    if "color" in e:
        color = np.array(bytes.fromhex(e["color"][1:]), float)
    elif binary.sum() >= 4:
        d = np.abs(region[binary].astype(float) - bg).sum(axis=1)
        color = np.median(region[binary][d >= np.percentile(d, 85)].astype(float), axis=0)
    axis = color - bg
    soft = np.clip(((region.astype(float) - bg) @ axis) / max(1e-6, axis @ axis), 0, 1)
    binary = drop_strays(soft >= 0.5, y0 - Y0, y1 - Y0, prof["baseline"])
    prof = profile(binary)
    if prof is None:
        return None
    # What is erased reaches fainter marks too (a pale full stop), if they stand out from the background's grain.
    faint = drop_strays(soft >= 0.3, y0 - Y0, y1 - Y0, prof["baseline"])
    n, labels, _, _ = cv2.connectedComponentsWithStats(faint.astype(np.uint8), connectivity=8)
    peaks = np.zeros(n)
    np.maximum.at(peaks, labels.ravel(), soft.ravel())
    faint = ((peaks >= 0.42)[labels] & faint) | binary
    ring = np.ones((3, 3), np.uint8)
    letters = soft * cv2.dilate(binary.astype(np.uint8), ring, iterations=2)
    soft = soft * cv2.dilate(faint.astype(np.uint8), ring, iterations=2)
    expected = len(e["was"].split()) if e.get("was") else None
    return {"X0": X0, "Y0": Y0, "X1": X1, "Y1": Y1, "bg": bg, "spread": spread, "color": color,
            "soft": soft, "binary": binary, "prof": prof, "letters": letters,
            "words": word_spans(binary, prof, expected)}


def free_space(img: np.ndarray, rows: slice, start: int, step: int, bg: np.ndarray, limit: int = 80) -> int:
    """How many columns of plain background there are beside a line, from start, going left (step -1) or right (+1):
    room the line can grow into before it meets a border, a box's edge or anything else."""
    n = 0
    x = start
    while 0 <= x < img.shape[1] and n < limit:
        col = img[rows, x].astype(float)
        if (np.abs(col - bg).sum(axis=1) > 45).any():
            break
        n += 1
        x += step
    return n


def which_words(e: dict, spans: list, i: int) -> tuple[int, int] | None:
    """The words of the line an edit changes (first, last), or None for the whole line. "replace" names them by
    their text in "was"; "words" by their position."""
    if "words" in e:
        a, b = (k + len(spans) if k < 0 else k for k in e["words"])
    elif "replace" in e:
        have, want = e["was"].split(), e["replace"].split()
        found = [k for k in range(len(have) - len(want) + 1) if have[k:k + len(want)] == want]
        if len(found) != 1:
            raise SystemExit(f"edit {i}: {e['replace']!r} is in {e['was']!r} {len(found)} times")
        if len(have) != len(spans):
            raise SystemExit(f"edit {i}: {len(have)} words in {e['was']!r} but {len(spans)} in the picture: {spans}")
        a, b = found[0], found[0] + len(want) - 1
    else:
        return None
    if b >= len(spans):
        raise SystemExit(f"edit {i}: only {len(spans)} words in {e['box']}: {spans}")
    return a, b


CACHE_DIR: str | None = None  # where font choices are kept between runs (set by fix_one.py)
CACHE_VERSION = 2
MARGIN = 0.35  # how much worse a style's best font in the graphic's main typeface may score and still be chosen


def family_of(path: str) -> str:
    """A font's typeface, from its file name: "Roboto-500.woff" and "Roboto-700.woff" are both Roboto."""
    name = os.path.basename(path).rsplit(".", 1)[0]
    name = name.split("-")[0]
    for suffix in ("BoldOblique", "BoldItalic", "Bold", "Oblique", "Italic"):
        if name.endswith(suffix) and name != suffix:
            name = name[: -len(suffix)]
    return name


def _ranking(key: str, compute):
    """The fonts a style's lines were judged against, best first: [(score, path, verdicts)], from the cache if
    the same lines were judged before."""
    if not CACHE_DIR:
        return compute()
    path = os.path.join(CACHE_DIR, f"v{CACHE_VERSION}", key[:2], key + ".json")
    if os.path.exists(path):
        try:
            data = json.load(open(path))
            return [(sc, find_font(f), [{**v, "path": find_font(f)} for v in vs]) for sc, f, vs in data]
        except (OSError, ValueError, KeyError, SystemExit):
            pass
    ranked = compute()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clean = [[float(sc), os.path.basename(p),
              [{k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in d.items() if k != "path"}
               for d in vs]] for sc, p, vs in ranked]
    with open(path + ".part", "w") as f:
        json.dump(clean, f)
    os.replace(path + ".part", path)
    return ranked


def typeface_fonts(name: str) -> tuple[str, ...]:
    """Every weight of a typeface ("Roboto": Roboto-300 … Roboto-900), upright."""
    want = name.replace(" ", "").lower()
    out = tuple(f for f in all_fonts() if family_of(f).lower() == want and not any(x in os.path.basename(f) for x in ITALIC))
    if not out:
        raise SystemExit(f"no typeface {name!r}: use a font file's name before the dash, e.g. Roboto, SourceSans3, "
                         "RobotoCondensed, PlayfairDisplay, Inter, LiberationSans")
    return out


def choose_fonts(edits: list[dict], lines: list, default_family: str | None, image_key: str = "",
                 default_typeface: str | None = None) -> dict[int, dict]:
    """The font, size and softness for each edit: one font per style (judged on all of its lines, the correctly
    spelled reference lines counting most), else one per line. A graphic mostly uses one typeface: a style takes
    the weight of it that fits best unless another typeface fits it clearly better."""
    groups: dict[str, list[int]] = {}
    for i, e in enumerate(edits):
        if lines[i] is not None:
            groups.setdefault(e.get("style") or f"#{i}", []).append(i)
    rankings: dict[str, list] = {}
    info: dict[str, tuple] = {}
    for name, idx in groups.items():
        # A font named on any line of the style is the style's font.
        forced = next((edits[i]["font"] for i in idx if edits[i].get("font")), None)
        fam = next((edits[i]["family"] for i in idx if edits[i].get("family")), None) or default_family or "sans+condensed"
        face = next((edits[i]["typeface"] for i in idx if edits[i].get("typeface")), None) or default_typeface
        texts = [edits[i].get("was") or "" for i in idx] + [edits[i].get("text") or "" for i in idx]
        weights = [3.0 if edits[i].get("ref") else 1.0 for i in idx]
        squeezable = not any("squeeze" in edits[i] for i in idx)

        def compute(idx=idx, forced=forced, fam=fam, face=face, texts=texts, weights=weights, squeezable=squeezable):
            paths = [find_font(forced)] if forced else typeface_fonts(face) if face else family(fam)
            paths = [p for p in paths if covers(p, *texts)] or paths
            if len(paths) > SHORTLIST:
                rough = sorted(paths, key=lambda p: sum(quick(p, edits[i].get("was") or edits[i]["text"],
                                                              lines[i]["prof"], lines[i]["letters"]) for i in idx))
                paths = rough[:SHORTLIST]
            ranked = []
            for path in paths:
                verdicts = [judge(path, edits[i].get("was") or edits[i]["text"], lines[i]["prof"], lines[i]["letters"],
                                  lines[i]["words"], squeezable) for i in idx]
                score = sum(w * v["score"] for w, v in zip(weights, verdicts)) / sum(weights)
                ranked.append((score, path, verdicts))
            ranked.sort(key=lambda t: t[0])
            return ranked

        key_data = [CACHE_VERSION, image_key, fam, face, forced, squeezable, SHORTLIST, len(all_fonts()),
                    sorted(set("".join(texts))),
                    [[edits[i]["box"], edits[i].get("was"), bool(edits[i].get("ref")), edits[i].get("pad", 3),
                      edits[i].get("threshold", 60), edits[i].get("color")] for i in idx]]
        key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
        rankings[name] = _ranking(key, compute) if image_key else compute()
        info[name] = (idx, forced or face, weights)
    # The graphic's main typeface: the one that fits its styles best, all told.
    free = [n for n in groups if not info[n][1] and rankings[n]]
    main_family = None
    if len(free) > 1:
        cost: dict[str, float] = {}
        families = {family_of(p) for n in free for _, p, _ in rankings[n]}
        for f in families:
            total = 0.0
            for n in free:
                ranked = rankings[n]
                mine = [sc for sc, p, _ in ranked if family_of(p) == f]
                total += sum(info[n][2]) * (min(mine) if mine else ranked[-1][0] + 0.5)
            cost[f] = total
        main_family = min(cost, key=cost.get)
    chosen: dict[int, dict] = {}
    for name, idx in groups.items():
        ranked = rankings[name]
        _, _, weights = info[name]
        pick = ranked[0]
        if main_family and not info[name][1]:
            mine = [r for r in ranked if family_of(r[1]) == main_family]
            if mine and mine[0][0] <= ranked[0][0] + MARGIN:
                pick = mine[0]
        _, path, verdicts = pick
        runners = [(os.path.basename(p), round(sc, 3)) for sc, p, _ in ranked if p != path][:3]
        # Each line's size: from its width (precise, if its letters were read right) and its letters' height.
        own = {}
        for i, v in zip(idx, verdicts):
            n = len((edits[i].get("was") or edits[i]["text"]).replace(" ", ""))
            trust = 0.75 if n >= 10 else 0.5 if n >= 5 else 0.25
            by_width = v["by_width"] if v.get("squeeze", 1.0) == 1.0 else v["size"]
            own[i] = quarter(math.exp(trust * math.log(by_width) + (1 - trust) * math.log(v["size"])))
        # Lines of one style are one size: the size their correctly spelled lines show, else most of them. A word
        # changed within a line keeps that line's own size, which its other words show.
        sizes = list(own.values())
        refs = [own[i] for i in idx if edits[i].get("ref")]
        common = quarter(float(np.median(refs or sizes)))
        alike = max(sizes) <= min(sizes) * 1.25
        squeezes = [v.get("squeeze", 1.0) for v in verdicts]
        squeeze = float(np.median(squeezes))
        # How soft the letters are: the least blur that makes the font's rendering of the line about as soft (a
        # little blur at most: more makes small letters look heavier, not softer).
        for i, v in zip(idx, verdicts):
            word_edit = "words" in edits[i] or "replace" in edits[i]
            size = own[i] if word_edit or not alike else common
            sample = edits[i].get("was") or edits[i]["text"]
            target = softness(lines[i]["letters"])
            blur = 0.0
            for sigma in (0.0, 0.25, 0.4):
                blur = sigma
                if softness(draw_mask(sample, path, size, blur=sigma)[0]) >= target * 0.8:
                    break
            chosen[i] = {**v, "path": path, "size": size, "blur": blur, "squeeze": squeeze, "runners_up": runners,
                         "main_family": main_family}
    return chosen


def apply(image: Image.Image, edits: list[dict], default_family: str | None = None, log: list | None = None,
          typeface: str | None = None) -> Image.Image:
    img = np.array(image.convert("RGB"))
    out = img.copy()
    lines = [read_line(img, e) for e in edits]
    image_key = hashlib.sha1(img.tobytes()).hexdigest() if CACHE_DIR else ""
    chosen = choose_fonts(edits, lines, default_family, image_key, typeface)
    # First every old line goes (and the words after a changed one move), then every new line is drawn: drawn
    # first, a line's descenders would be wiped out with the letters of the line below it.
    drawings = []
    for i, e in enumerate(edits):
        if e.get("ref"):
            continue
        L = lines[i]
        if L is None:
            if log is not None:
                log.append({"edit": i, "error": "no letters found in the box"})
            continue
        X0, Y0, X1, Y1, prof = L["X0"], L["Y0"], L["X1"], L["Y1"], L["prof"]
        c = chosen[i]
        path, size, blur = c["path"], e.get("size", c["size"]), e.get("blur", c["blur"])
        base = float(e.get("squeeze", c.get("squeeze", 1.0)))
        color = L["color"]
        spans = L["words"]
        words = which_words(e, spans, i)
        # Which of the old letters go: the whole line, or some of its words.
        cols = np.zeros(L["binary"].shape[1], bool)
        if words:
            a, b = words
            old_left, old_right = spans[a][0], spans[b][1]
            cols[max(0, old_left - 2):old_right + 2] = True  # with the letters' soft edges
        else:
            old_left, old_right = prof["left"], prof["right"]
            cols[:] = True
        gone = (L["soft"] > 0.06) & cols[None, :]
        gone = cv2.dilate(gone.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & cols[None, :]
        full = np.zeros(out.shape[:2], bool)
        full[Y0:Y1, X0:X1] = gone
        out = fill(out, full, (X0, Y0, X1, Y1), L["bg"], L["spread"])
        text = e["text"]
        slant = e.get("slant", 0.0)
        cover, p, width, left = None, None, 0, X0 + old_left
        align = e.get("align", "left")
        if text:
            cover = match_tone(draw_mask(text, path, size, squeeze=base, blur=blur, slant=slant)[0], L["letters"])
            p = profile(cover >= 0.5) or profile(cover >= 0.3)
            width = p["right"] - p["left"]
            # Room to grow: into the plain background beside the line, short of any border or box edge (on both
            # sides, for a centred line), and for a changed word, by pushing the rest of its line along.
            rows = slice(Y0 + prof["top"], Y0 + prof["bottom"])
            right_free = free_space(img, rows, X0 + prof["right"] + 1, 1, L["bg"]) - 3
            left_free = free_space(img, rows, X0 + prof["left"] - 1, -1, L["bg"]) - 3
            if align == "center":
                grow = 2 * max(0, min(left_free, right_free))
            elif align == "right":
                grow = max(0, left_free)
            else:
                grow = max(0, right_free)
            grow = min(grow, max(20, 0.15 * (prof["right"] - prof["left"])))
            room = e.get("room", (old_right - old_left) + grow)
            squeeze = base
            if width > room:
                fit = max(0.9, room / width)
                if width * fit > room:
                    size = quarter(size * room / (width * fit) - 0.125)
                squeeze = base * fit
                cover = match_tone(draw_mask(text, path, size, squeeze=squeeze, blur=blur, slant=slant)[0],
                                   L["letters"])
                p = profile(cover >= 0.5) or profile(cover >= 0.3)
                width = p["right"] - p["left"]
            if align == "center":
                left = X0 + (old_left + old_right) / 2 - width / 2
            elif align == "right":
                left = X0 + old_right - width
        # Words after the changed ones move along so the line still reads with one space between words.
        shift = 0
        if words and words[1] + 1 < len(spans):
            nxt = spans[words[1] + 1][0]
            gap = nxt - old_right
            shift = round(left + width + gap - (X0 + nxt)) if text else old_left - nxt
        if shift:
            a = spans[words[1] + 1][0]
            r0, r1 = max(0, prof["top"] - 3), min(Y1 - Y0, prof["bottom"] + 3)
            c0, c1 = max(old_right + 2, a - 2), min(X1 - X0, prof["right"] + 3)  # with their soft left edges
            keep = L["soft"][r0:r1, c0:c1] > 0.06
            keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool)
            src = img[Y0 + r0:Y0 + r1, X0 + c0:X0 + c1]
            clear = np.zeros(out.shape[:2], bool)
            clear[Y0 + r0:Y0 + r1, X0 + c0:X0 + c1] = keep
            out = fill(out, clear, (X0, Y0, min(out.shape[1], X1 + max(0, shift)), Y1), L["bg"], L["spread"])
            d0 = X0 + c0 + shift
            w = min(c1 - c0, out.shape[1] - d0)
            dst = out[Y0 + r0:Y0 + r1, d0:d0 + w]
            dst[keep[:, :w]] = src[:, :w][keep[:, :w]]
        drawn = [X0 + old_left, Y0 + prof["top"], X0 + max(old_right, prof["right"] + max(0, shift)), Y0 + prof["bottom"]]
        if text:
            baseline = Y0 + prof["baseline"] + e.get("dy", 0)
            dl, dt = round(left - p["left"]), round(baseline - p["baseline"])
            drawings.append((cover, dl, dt, color))
            ys, xs = np.nonzero(cover > 0.1)
            if len(xs):
                drawn = [min(drawn[0], dl + int(xs.min())), min(drawn[1], dt + int(ys.min())),
                         max(drawn[2], dl + int(xs.max()) + 1), max(drawn[3], dt + int(ys.max()) + 1)]
        if log is not None:
            log.append({"edit": i, "text": text, "words": words, "of": len(spans), "font": os.path.basename(path),
                        "size": size, "planned": c["size"], "start_size": e.get("size", c["size"]), "blur": blur,
                        "squeeze": base, "main_family": c.get("main_family"), "drawn": [int(v) for v in drawn], "score": round(c["score"], 3), "width": round(c.get("width", 0), 3),
                        "stroke": round(c.get("stroke", 0), 3), "iou": round(c.get("iou", 0), 3),
                        "mass": round(c.get("mass", 0), 3), "runners_up": c["runners_up"], "color": hexcolor(color),
                        "bg": hexcolor(L["bg"]), "spread": round(L["spread"], 1), "shift": shift})
    for cover, left, top, color in drawings:
        paint(out, cover, left, top, color)
    return Image.fromarray(out)


def load_spec(path: str) -> dict:
    spec = json.load(open(path, encoding="utf-8"))
    return {k: (v if isinstance(v, dict) else {"edits": v}) for k, v in spec.items()}


def main(argv: list[str]) -> None:
    args = [a for a in argv if not a.startswith("--")]
    only = next((a.split("=", 1)[1].split(",") for a in argv if a.startswith("--only=")), None)
    spec_path, src, dst = args[:3]
    os.makedirs(dst, exist_ok=True)
    report = {}
    for gid, item in load_spec(spec_path).items():
        if only and gid not in only:
            continue
        log: list = []
        img = apply(Image.open(os.path.join(src, f"{gid}.png")), item["edits"], item.get("family"), log)
        img.save(os.path.join(dst, f"{gid}.png"))
        report[gid] = log
    json.dump(report, sys.stdout, indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
