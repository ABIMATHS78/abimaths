"""Make the app's corrected graphics from the corrections: build.py [id ...]

For each corrections/<id>.json beside this file (or the ids given): downloads the original graphic if it isn't in
the work folder yet, applies the corrections and saves the result to static/guides/<id>.webp in the app. The fonts
must be in place first (get_fonts.py). A graphic whose corrections haven't changed since its WebP was made is left.
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download  # noqa: E402
import fix_text as ft  # noqa: E402
from paths import HERE, REPO, WORK  # noqa: E402
from PIL import Image  # noqa: E402

CORRECTIONS = os.path.join(HERE, "corrections")
DEST = os.path.join(REPO, "static", "guides")
QUALITY = 85  # WebP: small text stays as crisp as in the PNG


def build(gid: str) -> str:
    spec_path = os.path.join(CORRECTIONS, f"{gid}.json")
    dest = os.path.join(DEST, f"{gid}.webp")
    if os.path.exists(dest) and os.path.getmtime(dest) >= os.path.getmtime(spec_path):
        return "up to date"
    src = os.path.join(WORK, "orig", f"{gid}.png")
    if not os.path.exists(src):
        download.main([gid])
    spec = json.load(open(spec_path, encoding="utf-8"))
    img = Image.open(src)
    if [e for e in spec["edits"] if not e.get("ref")]:
        img = ft.apply(img, spec["edits"], spec.get("family"), None, spec.get("typeface"))
    img.convert("RGB").save(dest + ".part", "WEBP", quality=QUALITY, method=6)
    os.replace(dest + ".part", dest)
    return "made"


def main(ids: list[str]) -> None:
    ft.CACHE_DIR = os.path.join(WORK, "cache")
    os.makedirs(DEST, exist_ok=True)
    ids = ids or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CORRECTIONS, "*.json")))
    made = 0
    for gid in ids:
        made += build(gid) == "made"
    print(f"{made} made, {len(ids) - made} up to date, in {DEST}")


if __name__ == "__main__":
    main(sys.argv[1:])
