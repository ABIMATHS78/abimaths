"""Copy finished graphics into the app as WebP: publish.py [app_static_guides_dir]

Every graphic marked done (work/done/<id>.json) is saved from work/fixed/<id>.png as <dir>/<id>.webp, if that is
missing or older than the corrected graphic.
"""
import glob, os, sys
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK  # noqa: E402
DEST = sys.argv[1] if len(sys.argv) > 1 else "/home/user/abimaths/ai-every-time/static/guides"
QUALITY = 85
os.makedirs(DEST, exist_ok=True)
made = 0
for done in sorted(glob.glob(os.path.join(WORK, "done", "*.json"))):
    gid = os.path.basename(done)[:-5]
    src = os.path.join(WORK, "fixed", f"{gid}.png")
    dst = os.path.join(DEST, f"{gid}.webp")
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src):
        continue
    Image.open(src).convert("RGB").save(dst + ".part", "WEBP", quality=QUALITY, method=6)
    os.replace(dst + ".part", dst)
    made += 1
total = glob.glob(os.path.join(DEST, "*.webp"))
print(f"{made} saved; {len(total)} graphics in {DEST}, {sum(os.path.getsize(p) for p in total) // 1024} KB")
