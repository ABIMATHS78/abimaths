"""Renders with the real headless Chromium (skipped when it isn't installed)."""

import shutil

import pytest
from PIL import Image

from lpe.brand import load_brand
from lpe.pipeline import finish
from lpe.render import render
from lpe.templates import TEMPLATES


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_every_template_renders_without_overflow(name, renderer, tmp_path):
    result = render({"template": name, "data": TEMPLATES[name].sample}, load_brand(), tmp_path, scale=1,
                    renderer=renderer, log=lambda m: None)
    assert result["problems"] == []
    with Image.open(tmp_path / "graphic.png") as image:
        assert image.size == tuple(result["size"])


@pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg isn't installed")
def test_motion_renders_gif_and_mp4(renderer, tmp_path):
    spec = {"template": "badge", "data": TEMPLATES["badge"].sample}
    result = render(spec, load_brand(), tmp_path, formats=["png", "gif", "mp4"], scale=1, fps=12,
                    renderer=renderer, log=lambda m: None)
    assert (tmp_path / "graphic.mp4").stat().st_size > 1000
    with Image.open(tmp_path / "graphic.gif") as gif:
        assert gif.n_frames > 10
    assert result["motion"]["duration_s"] > 2


def test_finish_writes_a_complete_post_folder(renderer, tmp_path):
    spec = {
        "template": "split",
        "data": TEMPLATES["split"].sample,
        "post": "Clean up your AI setup today.\n\n" + "\n".join(f"{i}. One small, real step." for i in range(1, 15)),
        "alt_text": "Two columns: delete and keep.",
    }
    report = finish(spec, tmp_path, profile={}, corpus=[], renderer=renderer, scale=1, log=lambda m: None)
    for name in ("graphic.png", "post.txt", "post.md", "preview.html", "report.json", "spec.json"):
        assert (tmp_path / name).exists(), name
    assert report["ok"] and report["characters"] == len(spec["post"])
    assert "see more" in (tmp_path / "preview.html").read_text(encoding="utf-8")



def _final_frame(renderer, html_path, width, height, out_path):
    page, report = renderer._open(html_path, width, height, 1)
    page.evaluate("t => window.__seek(t)", report["timeline"]["duration"] + 1)
    page.screenshot(path=str(out_path), clip={"x": 0, "y": 0, "width": width, "height": height})
    page.close()


def _moved_share(still, last) -> float:
    """Share of pixels that differ even when the last frame may sit a pixel off (sub-pixel text smoothing)."""
    from PIL import ImageChops, ImageFilter

    a, b = still.convert("L"), last.convert("L")
    low, high = b.filter(ImageFilter.MinFilter(3)), b.filter(ImageFilter.MaxFilter(3))
    diff = ImageChops.add(ImageChops.subtract(a, high), ImageChops.subtract(low, a))
    return sum(diff.point(lambda v: 255 if v > 48 else 0).histogram()[255:]) / (diff.width * diff.height)


MOTION_CASES = [(n, TEMPLATES[n].sample) for n in TEMPLATES if n != "kinetic"]  # kinetic is scenes, not a build-up
if TEMPLATES["mindmap"].sample.get("style") != "whiteboard":
    MOTION_CASES.append(("mindmap", {**TEMPLATES["mindmap"].sample, "style": "whiteboard"}))


@pytest.mark.parametrize("name,data", MOTION_CASES, ids=[f"{n}-{d.get('style', '')}".rstrip("-") for n, d in MOTION_CASES])
def test_motion_ends_on_the_still_image(name, data, renderer, tmp_path):
    """An animation that overrides an element's position (as one once moved the whiteboard hub off-centre) leaves
    a last frame that differs from the PNG. A correct one differs by sub-pixel smoothing at most."""
    from lpe.render import build_html

    spec = {"template": name, "data": data}
    brand = load_brand()
    render(spec, brand, tmp_path, scale=1, renderer=renderer, log=lambda m: None)
    html, meta = build_html(spec, brand, motion=True)
    motion_page = tmp_path / "graphic.motion.html"
    motion_page.write_text(html, encoding="utf-8")
    _final_frame(renderer, motion_page, meta["width"], meta["height"], tmp_path / "last.png")
    with Image.open(tmp_path / "graphic.png") as still, Image.open(tmp_path / "last.png") as last:
        share = _moved_share(still, last)
    assert share < 0.0005, f"{share:.3%} of the last frame differs from the still image"
