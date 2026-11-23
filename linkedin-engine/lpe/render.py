"""Render a graphic spec to PNG, GIF and MP4.

A spec is {"template": name, "data": {...}, "theme": optional, "post": optional text}.
Static images are screenshots of the template page. Motion is captured frame by frame: the page's
CSS animations are paused and seeked to each frame's time, so every frame is exact and repeatable,
then ffmpeg turns the frames into an MP4 and a GIF.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .markup import plain, rich, words
from .paths import FONTS_DIR, TEMPLATES_DIR
from .schemas import clean_data
from .templates import get_template
from .themes import theme_css

FONT_FACES = [
    ("Fraunces", "fraunces.woff2", "100 900", "normal"),
    ("Fraunces", "fraunces-italic.woff2", "100 900", "italic"),
    ("Inter", "inter.woff2", "100 900", "normal"),
    ("Inter", "inter-italic.woff2", "100 900", "italic"),
    ("JetBrains Mono", "jetbrains-mono.woff2", "100 800", "normal"),
    ("Anton", "anton.woff2", "400", "normal"),
    ("Kalam", "kalam-400.woff2", "400", "normal"),
    ("Kalam", "kalam-700.woff2", "700", "normal"),
    ("Caveat", "caveat.woff2", "400 700", "normal"),
]
GIF_MAX_BYTES = 5 * 1024 * 1024


def _font_faces() -> str:
    return "\n".join(
        f'@font-face{{font-family:"{family}";src:url("{(FONTS_DIR / file).as_uri()}") format("woff2");'
        f"font-weight:{weight};font-style:{style};font-display:block;}}"
        for family, file, weight, style in FONT_FACES
    )


def _env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "j2"], default_for_string=True),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(rich=rich, plain=plain, words=words, sin=math.sin, cos=math.cos)
    return env


def resolve_theme(spec: dict, template, data: dict, brand: dict) -> str:
    """Explicit theme in the spec, else the brand's forced theme, else the look learned for this template."""
    if spec.get("theme"):
        return spec["theme"]
    forced = (brand or {}).get("theme")
    if forced and forced != "learned":
        return forced
    return template.theme_for(data)


def build_html(spec: dict, brand: dict, motion: bool = False) -> tuple[str, dict]:
    template = get_template(spec["template"])
    data, warnings = clean_data(spec.get("data") or {}, template.schema)
    theme = resolve_theme(spec, template, data, brand)
    width, height = template.size_for(data)
    footer = (brand or {}).get("footer", {})
    context = {
        "data": data,
        "theme": theme,
        "width": width,
        "height": height,
        "motion": motion,
        "brand": brand or {},
        "footer_left": spec.get("footer_left", footer.get("left", "")),
        "footer_right": spec.get("footer_right", footer.get("right", "")),
        "page_title": plain(data.get("title") or spec["template"]),
        "font_faces": _font_faces(),
        "theme_css": theme_css(theme),
        "base_css": (TEMPLATES_DIR / "base.css").read_text(encoding="utf-8"),
        "runtime_js": (TEMPLATES_DIR / "runtime.js").read_text(encoding="utf-8"),
        "motion_config": template.motion_config(data) if motion else {},
    }
    context.update(template.prepare(data, context))
    html = _env().get_template(template.file).render(**context)
    return html, {"theme": theme, "width": width, "height": height, "warnings": warnings}


class Renderer:
    """One headless Chromium for any number of renders."""

    def __init__(self):
        from playwright.sync_api import sync_playwright

        self._pw = sync_playwright().start()
        # LPE_CHROMIUM first, then Playwright's own browser, then the shared Chromium of Claude Code cloud sessions.
        candidates = [os.environ.get("LPE_CHROMIUM") or None]
        if candidates[0] is None and Path("/opt/pw-browsers/chromium").exists():
            candidates.append("/opt/pw-browsers/chromium")
        last_error = None
        for executable in candidates:
            try:
                self.browser = self._pw.chromium.launch(executable_path=executable, args=["--font-render-hinting=none"])
                break
            except Exception as error:
                last_error = error
        else:
            self._pw.stop()
            raise RuntimeError(
                "Couldn't start Chromium. Run `python -m playwright install chromium` "
                "(or set LPE_CHROMIUM to a Chrome/Chromium binary)."
            ) from last_error

    def close(self):
        self.browser.close()
        self._pw.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self, html_path: Path, width: int, height: int, scale: float):
        page = self.browser.new_page(viewport={"width": width, "height": height}, device_scale_factor=scale)
        page.goto(html_path.resolve().as_uri())
        page.wait_for_function("window.__ready === true", timeout=60_000)
        report = page.evaluate("window.__report")
        return page, report

    def png(self, html_path: Path, out_path: Path, width: int, height: int, scale: float = 2) -> dict:
        page, report = self._open(html_path, width, height, scale)
        page.screenshot(path=str(out_path), clip={"x": 0, "y": 0, "width": width, "height": height})
        page.close()
        return report

    def frames(self, html_path: Path, frames_dir: Path, width: int, height: int, fps: int = 30, scale: float = 1) -> dict:
        page, report = self._open(html_path, width, height, scale)
        duration = report["timeline"]["duration"]
        count = max(1, math.ceil(duration * fps))
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index in range(count):
            page.evaluate("t => window.__seek(t)", index / fps)
            page.screenshot(path=str(frames_dir / f"{index:05d}.png"), clip={"x": 0, "y": 0, "width": width, "height": height})
        page.close()
        report["frames"] = count
        return report


def _ffmpeg(args: list[str]) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg isn't installed: it's needed for GIF and MP4 output.")
    result = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", *args], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[:800]}")


def encode_mp4(frames_dir: Path, out_path: Path, fps: int) -> None:
    _ffmpeg([
        "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p",
        "-c:v", "libx264", "-preset", "slow", "-crf", "18", "-tune", "animation",
        "-movflags", "+faststart", "-r", str(fps), str(out_path),
    ])


def encode_gif(frames_dir: Path, out_path: Path, fps: int, width: int, max_bytes: int = GIF_MAX_BYTES) -> dict:
    """Encode a looping GIF, stepping the size and frame rate down until it fits LinkedIn's limit."""
    attempts = [(width, 15, 160), (min(width, 900), 15, 128), (min(width, 760), 12, 96), (min(width, 640), 10, 64)]
    for gif_width, gif_fps, colors in attempts:
        _ffmpeg([
            "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"),
            "-vf", (
                f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];"
                f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle"
            ),
            "-loop", "0", str(out_path),
        ])
        size = out_path.stat().st_size
        if size <= max_bytes:
            break
    return {"width": gif_width, "fps": gif_fps, "colors": colors, "bytes": size}


def render(spec: dict, brand: dict, out_dir: Path, formats=("png",), scale: float = 2, fps: int = 30,
           renderer: Renderer | None = None, log=print) -> dict:
    """Render a spec into out_dir. Returns paths and a report (fit scales, overflow problems, timings)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    formats = [f.lower() for f in formats]
    own = renderer is None
    renderer = renderer or Renderer()
    result: dict = {"files": {}, "problems": []}
    try:
        html, meta = build_html(spec, brand, motion=False)
        result.update(theme=meta["theme"], size=[meta["width"], meta["height"]], warnings=meta["warnings"])
        static_html = out_dir / "graphic.html"
        static_html.write_text(html, encoding="utf-8")
        result["files"]["html"] = str(static_html)

        started = time.time()
        report = renderer.png(static_html, out_dir / "graphic.png", meta["width"], meta["height"], scale=scale)
        result["files"]["png"] = str(out_dir / "graphic.png")
        result["fit"] = report["fit"]
        result["problems"] += report["problems"]
        log(f"  graphic.png ({time.time() - started:.1f}s)")

        if {"gif", "mp4"} & set(formats):
            motion_html, _ = build_html(spec, brand, motion=True)
            with tempfile.TemporaryDirectory(prefix="lpe-frames-") as tmp:
                tmp = Path(tmp)
                page_path = out_dir / "graphic.motion.html"
                page_path.write_text(motion_html, encoding="utf-8")
                started = time.time()
                motion_report = renderer.frames(page_path, tmp / "frames", meta["width"], meta["height"], fps=fps)
                timeline = motion_report["timeline"]
                log(f"  {motion_report['frames']} frames for {timeline['duration']:.1f}s of motion ({time.time() - started:.1f}s)")
                result["motion"] = {"duration_s": timeline["duration"], "frames": motion_report["frames"], "fps": fps}
                if "mp4" in formats:
                    encode_mp4(tmp / "frames", out_dir / "graphic.mp4", fps)
                    result["files"]["mp4"] = str(out_dir / "graphic.mp4")
                    log("  graphic.mp4")
                if "gif" in formats:
                    gif = encode_gif(tmp / "frames", out_dir / "graphic.gif", fps, meta["width"])
                    result["files"]["gif"] = str(out_dir / "graphic.gif")
                    result["gif"] = gif
                    log(f"  graphic.gif ({gif['bytes'] / 1024 / 1024:.1f} MB, {gif['width']}px, {gif['fps']} fps)")
                    if gif["bytes"] > GIF_MAX_BYTES:
                        result["problems"].append("GIF is over 5 MB even at the smallest setting; post the MP4 instead")
    finally:
        if own:
            renderer.close()
    (out_dir / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
