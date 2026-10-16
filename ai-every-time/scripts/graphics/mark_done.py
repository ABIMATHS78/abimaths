"""Record that a graphic is finished: mark_done.py <id> ["a note about anything left as it is"]

Checks the corrected graphic was made from the current corrections, then writes work/done/<id>.json.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK  # noqa: E402


def main(argv: list[str]) -> None:
    gid = argv[0]
    note = " ".join(argv[1:]).strip()
    spec_path = os.path.join(WORK, "spec", f"{gid}.json")
    fixed = os.path.join(WORK, "fixed", f"{gid}.png")
    if not os.path.exists(spec_path):
        raise SystemExit("no spec: write work/spec/<id>.json first (with \"edits\": [] if nothing needs correcting)")
    if not os.path.exists(fixed) or os.path.getmtime(fixed) < os.path.getmtime(spec_path):
        raise SystemExit("the corrected graphic is older than the spec: run fix_one.py again and check it")
    spec = json.load(open(spec_path, encoding="utf-8"))
    edits = [e for e in spec["edits"] if not e.get("ref")]
    record = {"id": gid, "corrections": len(edits), "skipped": spec.get("skipped", []), "note": note,
              "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(record, open(os.path.join(WORK, "done", f"{gid}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"{gid}: done, {len(edits)} corrections" + (f", {len(record['skipped'])} left as they are" if record["skipped"] else ""))


if __name__ == "__main__":
    main(sys.argv[1:])
