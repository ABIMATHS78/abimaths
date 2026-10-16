"""Record that a graphic is finished: mark_done.py <id> ["a note about anything left as it is"]

Checks the corrected graphic was made from the current corrections, then writes work/done/<id>.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK  # noqa: E402


def pin(gid: str, spec: dict, made: dict, spec_path: str, fixed: str, log_path: str) -> None:
    """Write each correction's font, size, softness and width into the spec, as they were when the graphic was
    checked, so it is made again exactly as it is however the tool chooses fonts later. Checked by making it again."""
    import numpy as np
    from PIL import Image
    import fix_one
    import fix_text as ft
    pinned = json.loads(json.dumps(spec))
    for entry in made.get("edits", []):
        if "error" in entry:
            continue
        e = pinned["edits"][entry["edit"]]
        e.setdefault("font", entry["font"])
        e.setdefault("size", entry["start_size"])
        e.setdefault("blur", entry["blur"])
        e.setdefault("squeeze", entry.get("squeeze", 1.0))
    if pinned == spec:
        return
    ft.CACHE_DIR = os.path.join(WORK, "cache")
    again = ft.apply(Image.open(os.path.join(WORK, "orig", f"{gid}.png")), pinned["edits"], pinned.get("family"), None,
                     pinned.get("typeface"))
    if not np.array_equal(np.asarray(again.convert("RGB")), np.asarray(Image.open(fixed).convert("RGB"))):
        raise SystemExit("made again with its fonts written in, the graphic came out different: run fix_one.py again")
    with open(spec_path + ".part", "w", encoding="utf-8") as f:
        json.dump(pinned, f, ensure_ascii=False, indent=1)
    os.replace(spec_path + ".part", spec_path)
    made["spec"] = fix_one.spec_hash(pinned)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(made, f, ensure_ascii=False, indent=1)


def main(argv: list[str]) -> None:
    gid = argv[0]
    note = " ".join(argv[1:]).strip()
    spec_path = os.path.join(WORK, "spec", f"{gid}.json")
    fixed = os.path.join(WORK, "fixed", f"{gid}.png")
    if not os.path.exists(spec_path):
        raise SystemExit("no spec: write work/spec/<id>.json first (with \"edits\": [] if nothing needs correcting)")
    spec = json.load(open(spec_path, encoding="utf-8"))
    log_path = os.path.join(WORK, "view", gid, "log.json")
    made = json.load(open(log_path, encoding="utf-8")) if os.path.exists(log_path) else {}
    import fix_one
    if not os.path.exists(fixed) or made.get("spec") != fix_one.spec_hash(spec):
        raise SystemExit("the corrected graphic wasn't made from these corrections: run fix_one.py again and check it")
    pin(gid, spec, made, spec_path, fixed, log_path)
    edits = [e for e in spec["edits"] if not e.get("ref")]
    record = {"id": gid, "corrections": len(edits), "skipped": spec.get("skipped", []), "note": note,
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(record, open(os.path.join(WORK, "done", f"{gid}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{gid}: done, {len(edits)} corrections" + (f", {len(record['skipped'])} left as they are" if record["skipped"] else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
