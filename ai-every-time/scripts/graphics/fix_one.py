"""Correct one graphic from its list of corrections, and show the result: fix_one.py <id>

Reads work/spec/<id>.json, checks it, writes the corrected graphic to work/fixed/<id>.png and, in work/view/<id>/,
pair_N.png: each changed area enlarged, the original above the corrected version (a red line between them).
Prints what it did with each correction, and anything to look at closely.

work/spec/<id>.json:
  {"family": "sans",            optional: fonts to try for the whole graphic (see AGENT.md)
   "edits": [ ... ],            the corrections, one per line of text
   "skipped": ["..."]}          optional: mistakes left as they are, and why
"""

import json
import os
import shutil
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fix_text as ft  # noqa: E402
from PIL import Image  # noqa: E402

from paths import WORK  # noqa: E402
ft.CACHE_DIR = os.path.join(WORK, "cache")
KEYS = {"box", "was", "text", "replace", "words", "align", "style", "family", "typeface", "ref", "font", "size", "pad",
        "threshold", "room", "dy", "color", "blur", "squeeze", "slant", "fill", "note"}
FAMILIES = set(ft.FAMILY_PREFIXES)


def check(spec: dict) -> list[str]:
    problems = []
    if not isinstance(spec, dict) or not isinstance(spec.get("edits"), list):
        return ['the file must be {"edits": [...]} (with "family" and "skipped" if wanted)']
    unknown = set(spec) - {"edits", "family", "typeface", "skipped", "notes"}
    if unknown:
        problems.append(f"unknown keys at the top: {sorted(unknown)}")
    fam = spec.get("family")
    if fam and not set(fam.split("+")) <= FAMILIES:
        problems.append(f"family {fam!r}: use {sorted(FAMILIES)} or several joined with +")
    for i, e in enumerate(spec["edits"]):
        where = f"edit {i}"
        if not isinstance(e, dict):
            problems.append(f"{where}: not an object")
            continue
        bad = set(e) - KEYS
        if bad:
            problems.append(f"{where}: unknown keys {sorted(bad)}")
        box = e.get("box")
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int) for v in box)
                and 0 <= box[0] < box[2] <= 1024 and 0 <= box[1] < box[3] <= 1024):
            problems.append(f"{where}: box must be [x0, y0, x1, y1], whole pixels inside 0..1024, x0 < x1, y0 < y1")
        if not isinstance(e.get("was"), str) or not e["was"].strip():
            problems.append(f"{where}: 'was' (the line as the picture shows it) is required")
        if not e.get("ref") and not isinstance(e.get("text"), str):
            problems.append(f"{where}: 'text' (what it should say; \"\" to remove it) is required")
        if "replace" in e and isinstance(e.get("was"), str):
            have, want = e["was"].split(), str(e["replace"]).split()
            hits = [k for k in range(len(have) - len(want) + 1) if have[k:k + len(want)] == want]
            if len(hits) != 1:
                problems.append(f"{where}: 'replace' {e['replace']!r} must be in 'was' exactly once, as whole words")
        if "words" in e and not (isinstance(e["words"], list) and len(e["words"]) == 2
                                 and all(isinstance(v, int) for v in e["words"])):
            problems.append(f"{where}: 'words' must be [first, last] (counted from 0; -1 is the last word)")
        if e.get("align", "left") not in ("left", "center", "right"):
            problems.append(f"{where}: align must be left, center or right")
        if e.get("family") and not set(e["family"].split("+")) <= FAMILIES:
            problems.append(f"{where}: family {e['family']!r}: use {sorted(FAMILIES)}")
        if e.get("ref") and "text" in e:
            problems.append(f"{where}: a 'ref' line isn't changed, so it has no 'text'")
    return problems


def spec_hash(spec: dict) -> str:
    """What the corrected graphic depends on in the spec: its edits and family (not notes or what was skipped)."""
    import hashlib
    parts = [spec.get("family"), spec["edits"]] if not spec.get("typeface") else \
        [spec.get("family"), spec["typeface"], spec["edits"]]
    return hashlib.sha1(json.dumps(parts, sort_keys=True).encode()).hexdigest()


def groups(boxes: list[list[int]]) -> list[list[int]]:
    out = []
    for b in sorted(boxes, key=lambda b: (b[1], b[0])):
        for g in out:
            near_y = b[1] - g[3] < 40 and g[1] - b[3] < 40
            near_x = b[0] - g[2] < 200 and g[0] - b[2] < 200
            if near_y and near_x:
                g[:] = [min(g[0], b[0]), min(g[1], b[1]), max(g[2], b[2]), max(g[3], b[3])]
                break
        else:
            out.append(list(b))
    return out


SHEET_W, SHEET_H = 1400, 1900


def pairs(gid: str, boxes: list[list[int]]) -> list[str]:
    """Each changed area, original above corrected, as pair_N.png; and all of them on one or two sheets,
    check.png (and check_2.png), to look at together."""
    view = os.path.join(WORK, "view", gid)
    for f in os.listdir(view) if os.path.isdir(view) else []:
        if f.startswith(("pair_", "check")):
            os.remove(os.path.join(view, f))
    os.makedirs(view, exist_ok=True)
    a = Image.open(os.path.join(WORK, "orig", f"{gid}.png")).convert("RGB")
    b = Image.open(os.path.join(WORK, "fixed", f"{gid}.png")).convert("RGB")
    paths, made = [], []
    for n, (x0, y0, x1, y1) in enumerate(groups(boxes)):
        x0, y0, x1, y1 = max(0, x0 - 40), max(0, y0 - 16), min(1024, x1 + 40), min(1024, y1 + 16)
        scale = max(1.0, min(3.0, 1500 / (x1 - x0), 520 / (y1 - y0)))
        w, h = round((x1 - x0) * scale), round((y1 - y0) * scale)
        canvas = Image.new("RGB", (w, 2 * h + 8), (230, 40, 40))
        canvas.paste(a.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (0, 0))
        canvas.paste(b.crop((x0, y0, x1, y1)).resize((w, h), Image.LANCZOS), (0, h + 8))
        path = os.path.join(view, f"pair_{n}.png")
        canvas.save(path)
        made.append(canvas)
        paths.append(f"pair_{n}.png: region {x0},{y0} to {x1},{y1}, enlarged {scale:.1f}x")
    # The sheets: the pairs stacked, each shrunk if needed to the sheet's width, with a gap between them.
    sheets, page, used = [], [], 0
    for c in made:
        if c.width > SHEET_W:
            c = c.resize((SHEET_W, round(c.height * SHEET_W / c.width)), Image.LANCZOS)
        if page and used + c.height + 24 > SHEET_H:
            sheets.append(page)
            page, used = [], 0
        page.append(c)
        used += c.height + 24
    if page:
        sheets.append(page)
    sheet_paths = []
    for k, page in enumerate(sheets):
        width = max(c.width for c in page)
        sheet = Image.new("RGB", (width, sum(c.height + 24 for c in page)), (40, 40, 48))
        y = 0
        for c in page:
            sheet.paste(c, (0, y))
            y += c.height + 24
        name = "check.png" if k == 0 else f"check_{k + 1}.png"
        sheet.save(os.path.join(view, name))
        sheet_paths.append(os.path.join(view, name))
    return sheet_paths, paths


def main(gid: str) -> None:
    spec_path = os.path.join(WORK, "spec", f"{gid}.json")
    try:
        spec = json.load(open(spec_path, encoding="utf-8"))
    except FileNotFoundError:
        raise SystemExit(f"no {spec_path}: write the corrections first")
    except json.JSONDecodeError as ex:
        raise SystemExit(f"{spec_path} isn't valid JSON: {ex}")
    problems = check(spec)
    if problems:
        print("Fix these in the spec first:")
        for p in problems:
            print(" -", p)
        raise SystemExit(1)
    src = os.path.join(WORK, "orig", f"{gid}.png")
    dst = os.path.join(WORK, "fixed", f"{gid}.png")
    edits = spec["edits"]
    if not [e for e in edits if not e.get("ref")]:
        shutil.copyfile(src, dst)
        print("No corrections: the graphic is kept exactly as it is.")
        return
    log: list = []
    try:
        img = ft.apply(Image.open(src), edits, spec.get("family"), log, spec.get("typeface"))
    except SystemExit as ex:
        raise SystemExit(f"Couldn't apply the corrections: {ex}\n(For 'replace'/'words', 'was' must have as many "
                         "words as the picture shows; the numbers in brackets are where the tool found words.)")
    except Exception:
        traceback.print_exc()
        raise SystemExit("The tool failed on this spec (see above).")
    img.save(dst)
    # What was done, with each correction's font, size and softness: kept so a finished graphic can be made again
    # exactly, whatever later changes to how fonts are chosen.
    view = os.path.join(WORK, "view", gid)
    os.makedirs(view, exist_ok=True)
    with open(os.path.join(view, "log.json"), "w", encoding="utf-8") as f:
        json.dump({"spec": spec_hash(spec), "edits": log}, f, ensure_ascii=False, indent=1)
    print(f"Corrected graphic: {dst}")
    mains = {entry.get("main_family") for entry in log if entry.get("main_family")}
    if mains:
        print(f"Main typeface (the one most styles fit): {', '.join(sorted(mains))}. If the new words' letters look "
              "wrong, try another for the whole graphic, e.g. \"typeface\": \"Roboto\" at the top of the spec.")
    for entry in log:
        i = entry["edit"]
        e = edits[i]
        if "error" in entry:
            print(f"  edit {i}: NOT DONE: {entry['error']} (check the box covers the line's letters)")
            continue
        what = repr(entry["text"]) if entry["text"] else "(removed)"
        part = f" words {entry['words'][0]}-{entry['words'][1]} of {entry['of']}" if entry["words"] else ""
        notes = []
        if entry.get("planned") and entry["size"] < entry["planned"] * 0.88:
            notes.append(f"shrunk from {entry['planned']} to {entry['size']} to fit: check it doesn't look too small")
        if entry["score"] > 2.0:
            notes.append("no close font: check it looks like its neighbours")
        if entry["shift"]:
            notes.append(f"the rest of the line moved {entry['shift']:+d} px")
        squeezed = f" x{entry['squeeze']:.2f} wide" if abs(entry.get("squeeze", 1.0) - 1) > 0.005 else ""
        print(f"  edit {i}: {what}{part}  [{entry['font']} {entry['size']}{squeezed}]"
              + ("  NOTE: " + "; ".join(notes) if notes else ""))
    # Each changed area: the old line's box, and everything the new text and the words moved after it cover.
    areas = []
    for entry in log:
        box = list(edits[entry["edit"]]["box"])
        d = entry.get("drawn")
        if d:
            box = [min(box[0], d[0]), min(box[1], d[1]), max(box[2], d[2]), max(box[3], d[3])]
        areas.append(box)
    sheet_paths, pair_paths = pairs(gid, areas)
    print("To check: every changed area, original above corrected (red line between), on "
          + " and ".join(sheet_paths))
    print(f"The same areas one by one, larger, in {os.path.join(WORK, 'view', gid)}/:")
    for p in pair_paths:
        print(" ", p)


if __name__ == "__main__":
    main(sys.argv[1])
