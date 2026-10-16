"""Download the guides' original graphics from aieverytime.org: download.py [id ...]

Each guide's graphic (its "image" address in app/guides.json.gz) is saved as orig/<id>.png in the work folder (see
paths.py), unless it's there already. With no ids, every guide's.
"""
import gzip
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import GUIDES, WORK  # noqa: E402

OUT = os.path.join(WORK, "orig")


def get(item: tuple[str, str]) -> tuple[str, str]:
    gid, url = item
    dest = os.path.join(OUT, f"{gid}.png")
    if os.path.exists(dest) and os.path.getsize(dest) > 10000:
        return gid, "had"
    err = ""
    for attempt in range(4):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            if data[:8] != b"\x89PNG\r\n\x1a\n":
                return gid, "not a PNG"
            with open(dest + ".part", "wb") as f:
                f.write(data)
            os.replace(dest + ".part", dest)
            return gid, "got"
        except OSError as ex:
            err = str(ex)
            time.sleep(2 ** attempt)
    return gid, "failed: " + err


def main(ids: list[str]) -> None:
    guides = json.load(gzip.open(GUIDES, "rt"))["guides"]
    urls = {str(g["id"]): g["image"] for g in guides if g.get("image")}
    todo = [(i, urls[i]) for i in (ids or sorted(urls)) if i in urls]
    os.makedirs(OUT, exist_ok=True)
    with ThreadPoolExecutor(8) as pool:
        results = list(pool.map(get, todo))
    done = sum(r in ("got", "had") for _, r in results)
    print(f"{done} of {len(todo)} graphics in {OUT}")
    for gid, r in results:
        if r not in ("got", "had"):
            print(gid, r)


if __name__ == "__main__":
    main(sys.argv[1:])
