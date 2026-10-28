#!/usr/bin/env python3
"""Zips a skill folder with the folder itself at the top of the zip, as the Claude apps expect.

Usage:  python package_skill.py path/to/my-skill [my-skill.zip]

It runs check_skill.py first and stops if that finds a problem. The zip goes in the current folder unless you
name another place. Hidden files, __pycache__ folders and .pyc files are left out.
"""

import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_skill import check  # noqa: E402  (beside this script)


def wanted(path: Path, folder: Path) -> bool:
    parts = path.relative_to(folder).parts
    return path.is_file() and not any(p.startswith(".") or p == "__pycache__" for p in parts) and path.suffix != ".pyc"


def package(folder: Path, out: Path) -> int:
    """Writes the zip; how many files it holds."""
    files = sorted(p for p in folder.rglob("*") if wanted(p, folder))
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, Path(folder.name) / f.relative_to(folder))
    return len(files)


def main() -> int:
    if len(sys.argv) not in (2, 3) or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    folder = Path(sys.argv[1]).resolve()
    problems = check(folder)
    if problems:
        print(f"Not packaged. Fix {'these' if len(problems) > 1 else 'this'} first:")
        for p in problems:
            print(f"  - {p}")
        return 1
    out = Path(sys.argv[2]) if len(sys.argv) == 3 else Path.cwd() / f"{folder.name}.zip"
    count = package(folder, out)
    print(f"Made {out} ({count} file{'s' if count != 1 else ''}, {out.stat().st_size:,} bytes), "
          f"with {folder.name}/ at the top.")
    print("Upload it in Claude: Customize → Skills → + → Create skill → Upload a skill.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
