"""Measure the training graphics: size, shape, motion and the colours they use."""

from __future__ import annotations

import json
import shutil
import subprocess
from collections import Counter
from pathlib import Path


def _hex(rgb) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb[:3])


def _palette(image, colors: int = 8) -> list[dict]:
    from PIL import Image

    small = image.convert("RGB")
    small.thumbnail((220, 220), Image.Resampling.LANCZOS)
    quantized = small.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
    raw = quantized.getpalette()[: colors * 3]
    counts = sorted(quantized.getcolors(), reverse=True)
    total = sum(count for count, _ in counts) or 1
    swatches = []
    for count, index in counts:
        rgb = tuple(raw[index * 3: index * 3 + 3])
        swatches.append({"hex": _hex(rgb), "share": round(count / total, 3)})
    return swatches


def _luminance(image) -> float:
    gray = image.convert("L")
    gray.thumbnail((200, 200))
    pixels = list(gray.getdata())
    return round(sum(pixels) / (255 * len(pixels)), 3) if pixels else 0.5


def analyze_media(path: Path) -> dict:
    path = Path(path)
    info: dict = {"file": path.name, "kind": "", "width": 0, "height": 0}
    ext = path.suffix.lower()
    if ext in {".mp4", ".mov", ".webm", ".m4v"}:
        info["kind"] = "video"
        info.update(_probe_video(path))
        return info
    from PIL import Image

    with Image.open(path) as image:
        info["width"], info["height"] = image.size
        frames = getattr(image, "n_frames", 1)
        info["kind"] = "gif" if frames > 1 else "image"
        info["frames"] = frames
        if frames > 1:
            total = 0
            for index in range(frames):
                image.seek(index)
                total += image.info.get("duration", 100) or 100
            info["duration_s"] = round(total / 1000, 2)
            info["fps"] = round(frames / max(total / 1000, 0.001), 1)
            image.seek(frames - 1)  # the finished frame shows the full design
        rgb = image.convert("RGB")
        info["palette"] = _palette(rgb)
        info["luminance"] = _luminance(rgb)
        info["dark"] = info["luminance"] < 0.42
    info["aspect"] = round(info["width"] / info["height"], 3) if info["height"] else 0
    return info


def _probe_video(path: Path) -> dict:
    if not shutil.which("ffprobe"):
        return {}
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,avg_frame_rate:format=duration", "-of", "json", str(path)],
        capture_output=True, text=True, check=False,
    )
    try:
        data = json.loads(out.stdout)
        stream = data["streams"][0]
        num, den = (stream.get("avg_frame_rate") or "0/1").split("/")
        return {
            "width": stream.get("width", 0),
            "height": stream.get("height", 0),
            "duration_s": round(float(data["format"]["duration"]), 2),
            "fps": round(float(num) / float(den or 1), 1),
        }
    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
        return {}


def shape_name(aspect: float) -> str:
    if not aspect:
        return "unknown"
    if 0.75 <= aspect <= 0.85:
        return "portrait 4:5"
    if aspect < 0.75:
        return "tall (9:16 or taller)"
    if 0.95 <= aspect <= 1.05:
        return "square 1:1"
    return "landscape"


def is_accent(hex_color: str) -> bool:
    """A saturated, mid-light colour (terracotta, red, green...) rather than a background neutral."""
    import colorsys

    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    _, lightness, saturation = colorsys.rgb_to_hls(r, g, b)
    return saturation > 0.3 and 0.18 < lightness < 0.85


def learned_palette(analyses: list[dict], limit: int = 14, accents: bool | None = None) -> list[dict]:
    """Colours that recur across the training graphics, merged when they are near each other.

    accents=True keeps only saturated colours, accents=False only neutrals, None keeps both."""
    buckets: list[dict] = []
    for media in analyses:
        seen_here = set()
        for swatch in media.get("palette", []):
            if accents is not None and is_accent(swatch["hex"]) != accents:
                continue
            rgb = tuple(int(swatch["hex"][i:i + 2], 16) for i in (1, 3, 5))
            for bucket in buckets:
                if sum((a - b) ** 2 for a, b in zip(rgb, bucket["rgb"])) < 30 ** 2:
                    bucket["weight"] += swatch["share"]
                    if id(bucket) not in seen_here:
                        bucket["graphics"] += 1
                        seen_here.add(id(bucket))
                    break
            else:
                bucket = {"rgb": rgb, "weight": swatch["share"], "graphics": 1}
                buckets.append(bucket)
                seen_here.add(id(bucket))
    buckets.sort(key=lambda b: (b["graphics"], b["weight"]), reverse=True)
    return [
        {"hex": _hex(b["rgb"]), "in_graphics": b["graphics"], "weight": round(b["weight"], 2)}
        for b in buckets[:limit]
    ]


def summarize_media(analyses: list[dict]) -> dict:
    shapes = Counter(shape_name(m.get("aspect", 0)) for m in analyses)
    kinds = Counter(m.get("kind", "") for m in analyses)
    motion = [m for m in analyses if m.get("kind") in {"gif", "video"}]
    return {
        "graphics": len(analyses),
        "shapes": dict(shapes.most_common()),
        "kinds": dict(kinds.most_common()),
        "dark_share": round(sum(1 for m in analyses if m.get("dark")) / max(1, len(analyses)), 2),
        "motion_examples": [
            {k: m.get(k) for k in ("file", "frames", "duration_s", "fps", "width", "height")} for m in motion
        ],
        "learned_palette": learned_palette(analyses),
        "learned_neutrals": learned_palette(analyses, limit=8, accents=False),
        "learned_accents": learned_palette(analyses, limit=10, accents=True),
    }
