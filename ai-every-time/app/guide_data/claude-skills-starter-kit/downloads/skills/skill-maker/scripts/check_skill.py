#!/usr/bin/env python3
"""Checks a skill folder before you upload it.

Usage:  python check_skill.py path/to/my-skill

It checks that SKILL.md exists and starts with frontmatter between lines of three dashes, and that:
- the name uses only lowercase letters, numbers and hyphens, has 64 characters at most, doesn't contain
  "claude" or "anthropic", and matches the folder's name;
- the description isn't empty, has 200 characters at most (the limit in the Claude apps; the API allows 1,024)
  and has no XML tags;
- SKILL.md has fewer than 500 lines;
- every file SKILL.md links to, or names in backticks under scripts/, templates/ and the like, exists.

Exits with 1 if anything fails, so other tools (such as package_skill.py) can tell.
"""

import re
import sys
from pathlib import Path

NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
TAG = re.compile(r"<[^>]+>")
LINK = re.compile(r"\]\(([^)\s#]+)\)")
NAMED = re.compile(r"`((?:scripts|templates|references?|resources|assets|examples)/[^`\s]+)`")
RESERVED = ("anthropic", "claude")
APP_DESCRIPTION, API_DESCRIPTION, MAX_LINES = 200, 1024, 500


def frontmatter(text: str) -> tuple[dict[str, str], list[str]]:
    """The frontmatter's top-level fields, and problems with its shape. Handles plain, quoted and folded values."""
    lines, problems = text.splitlines(), []
    if not lines or lines[0].strip() != "---":
        return {}, ["SKILL.md must start with a line of three dashes (---), then the frontmatter"]
    end = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if end is None:
        return {}, ["the frontmatter has no closing line of three dashes (---)"]
    fields: dict[str, str] = {}
    i = 1
    while i < end:
        line = lines[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#") or line[0].isspace():
            continue
        key, sep, value = line.partition(":")
        if not sep:
            problems.append(f'this frontmatter line has no "key: value": {line.strip()}')
            continue
        key, value = key.strip(), value.strip()
        if value in (">", ">-", "|", "|-"):  # a value folded over the indented lines that follow
            parts = []
            while i < end and (not lines[i].strip() or lines[i][0].isspace()):
                parts.append(lines[i].strip())
                i += 1
            value = (" " if value.startswith(">") else "\n").join(p for p in parts if p)
        elif len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif ": " in value or value.endswith(":"):
            problems.append(f'the {key} contains a colon followed by a space, which YAML misreads: '
                            f'wrap the value in double quotes')
        fields[key] = value
    return fields, problems


def check(folder: Path) -> list[str]:
    """Problems with the skill in `folder`; none means it's ready to package."""
    if not folder.is_dir():
        return [f"{folder} isn't a folder"]
    skill = folder / "SKILL.md"
    if not skill.is_file():
        lower = folder / "skill.md"
        return ["rename skill.md to SKILL.md"] if lower.is_file() else [f"there's no SKILL.md in {folder}"]
    text = skill.read_text(encoding="utf-8")
    fields, problems = frontmatter(text)
    name, description = fields.get("name", ""), fields.get("description", "")
    if not name:
        problems.append("the frontmatter has no name")
    else:
        if not NAME.match(name):
            problems.append(f'the name "{name}" must use only lowercase letters, numbers and single hyphens')
        if len(name) > 64:
            problems.append(f"the name has {len(name)} characters (64 at most)")
        if any(word in name for word in RESERVED):
            problems.append('the name must not contain "claude" or "anthropic"')
        if name != folder.resolve().name:
            problems.append(f'the folder is called "{folder.resolve().name}", but the name is "{name}": make them match')
    if not description.strip():
        problems.append("the frontmatter has no description: say what the skill does and when to use it")
    elif len(description) > API_DESCRIPTION:
        problems.append(f"the description has {len(description)} characters ({API_DESCRIPTION} at most)")
    elif len(description) > APP_DESCRIPTION:
        problems.append(f"the description has {len(description)} characters: the Claude apps take "
                        f"{APP_DESCRIPTION} at most")
    for key in ("name", "description"):
        if TAG.search(fields.get(key, "")):
            problems.append(f"the {key} must not contain XML tags such as <example>")
    lines = len(text.splitlines())
    if lines >= MAX_LINES:
        problems.append(f"SKILL.md has {lines} lines: keep it under {MAX_LINES} and move detail into linked files")
    for target in sorted(set(LINK.findall(text)) | set(NAMED.findall(text))):
        if "://" in target or target.startswith("mailto:"):
            continue
        if not (folder / target).exists():
            problems.append(f"SKILL.md mentions {target}, but there's no such file in the folder")
    return problems


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__.strip())
        return 2
    folder = Path(sys.argv[1])
    problems = check(folder)
    if problems:
        print(f"{len(problems)} problem{'s' if len(problems) != 1 else ''} with {folder}:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"{folder} looks ready to package.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
