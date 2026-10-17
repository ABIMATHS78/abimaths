"""Check course files: that each fits the format (app/catalog.py), that its instructors are starter companions,
and a few things worth a second look (printed as warnings). See README.md in this folder.

Run from ai-every-time/:  python scripts/courses/check.py [FILE.json ...]     (every course when none is given)
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app import catalog  # noqa: E402
from app.starter_companions import CSV_PATH, REAL_PEOPLE  # noqa: E402

STARTERS = {row["slug"] for row in csv.DictReader(CSV_PATH.open(encoding="utf-8"))}
MODEL_ID = re.compile(r"claude-(?:opus|sonnet|haiku|fable|instant)-\d", re.I)
JETBRAINS_TOOL = re.compile(r"PyCharm|IntelliJ|WebStorm|GoLand|Rider|CLion|DataGrip|RustRover|PhpStorm|JetBrains")


def _shortcuts() -> set[tuple[str, str]]:
    """The JetBrains shortcuts courses may use: the table in README.md, as (macOS, Windows and Linux) pairs."""
    guide = Path(__file__).with_name("README.md").read_text(encoding="utf-8")
    table = guide[guide.index("| Action | macOS |"):]
    pairs = set()
    for row in table.splitlines()[2:]:
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) != 3:
            break
        _, mac, win = cells
        pairs.add((mac, win))
        macs, wins = mac.split(" / "), win.split(" / ")
        if len(macs) == len(wins):
            pairs.update(zip((m.strip() for m in macs), (w.strip() for w in wins)))
    return pairs


SHORTCUTS = _shortcuts()


def check(path: Path) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    text = path.read_text(encoding="utf-8")
    try:
        json.loads(text)
    except json.JSONDecodeError as e:
        return [f"not valid JSON: {e}"], []
    try:
        c = catalog.Course.model_validate_json(text)
    except ValueError as e:
        return [str(e)], []
    if c.slug != path.stem:
        errors.append(f"slug {c.slug!r} doesn't match the file name")
    for s in c.instructors:
        if s not in STARTERS:
            errors.append(f"instructor {s!r} isn't a starter companion (see app/starter_companions.csv)")
        elif s in REAL_PEOPLE:
            errors.append(f"instructor {s!r} is modelled on a real person; choose another")
    if MODEL_ID.search(text):
        errors.append("names a Claude model ID; let learners pick a current one from Anthropic's docs instead")
    shown = [(f"lesson {m}.{n}", l.walkthrough.steps) for m, mod in enumerate(c.modules, 1) for n, l in enumerate(mod.lessons, 1)
             if l.walkthrough and JETBRAINS_TOOL.search(l.walkthrough.tool)]
    shown += [(f"module {m} lab", lab.steps) for m, lab in c.labs if JETBRAINS_TOOL.search(lab.tool)]
    for where, steps in shown:
        for s in steps:
            if s.keys and (s.keys.mac, s.keys.win) not in SHORTCUTS:
                errors.append(f"{where}: shortcut {s.keys.mac} / {s.keys.win} isn't in the table in README.md; "
                              "use one from it, or the menu path")
    lessons = [l for m in c.modules for l in m.lessons]
    if c.jetbrains:
        shown = sum(l.walkthrough is not None for l in lessons)
        if shown < len(lessons) / 2:
            warnings.append(f"a JetBrains IDE course with walkthroughs in only {shown} of {len(lessons)} lessons")
    for m, lab in c.labs:
        for i, ch in enumerate(lab.checks, 1):
            if not ch.look_for:
                warnings.append(f"module {m} lab, check {i}: no look_for, so demo mode can't decide it")
    if c.kind == "course" and c.project is None:
        warnings.append("a course without a capstone project")
    if c.kind == "course" and not c.labs and c.category in ("Software Development", "Networking", "Cybersecurity"):
        warnings.append("a technical course without a lab")
    for m, module in enumerate(c.modules, 1):
        for n, l in enumerate(module.lessons, 1):
            for k in l.key_points:
                if len(k.split()) < 6:
                    warnings.append(f"lesson {m}.{n}: a very short key point: {k!r}")
    return errors, warnings


def main() -> None:
    files = [Path(a) for a in sys.argv[1:]] or sorted(p for p in catalog.DATA.glob("*.json") if p.name != "tracks.json")
    bad = 0
    for f in files:
        errors, warnings = check(f)
        status = "ERROR" if errors else "ok"
        if errors:
            bad += 1
        print(f"{status:5} {f.name}")
        for e in errors:
            print(f"      error: {e}")
        for w in warnings:
            print(f"      warning: {w}")
    if not sys.argv[1:] and not bad:
        try:
            courses, tracks = catalog.load()
        except ValueError as e:
            print(f"ERROR {e}")
            raise SystemExit(1)
        hours = sum(c.hours for c in courses)
        print(f"\n{len(courses)} courses ({sum(c.kind == 'guided-project' for c in courses)} guided projects), "
              f"{sum(len(m.lessons) for c in courses for m in c.modules)} lessons, {sum(len(c.labs) for c in courses)} labs, "
              f"{sum(c.project is not None for c in courses)} projects, about {hours:.0f} hours; {len(tracks)} tracks")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
