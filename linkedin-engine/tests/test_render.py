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
