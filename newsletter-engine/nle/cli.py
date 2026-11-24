"""Command line: python -m nle <command> --help"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import http.server
import json
import shutil
import sys
from pathlib import Path

EXAMPLES = """
examples:
  python -m nle new issue "Claude skills, from zero"      start a draft from the structure
  python -m nle write issue --brief briefs/claude-skills.json   Claude writes a draft (needs an API key)
  python -m nle check                                       run every check on every issue and guide
  python -m nle serve --drafts                              build and preview at http://localhost:8000/learn/
  python -m nle build                                       build the site into dist/
  python -m nle publish --into ../mojeedabisiga.com/site    copy the built site into the main website
  python -m nle train --fetch                               re-learn the structure from the example issues
"""


def _brief(path: str | None, base: Path | None = None) -> dict | None:
    if not path:
        return None
    from .paths import ENGINE_DIR

    candidates = [Path(path)] if Path(path).is_absolute() else [Path(path), ENGINE_DIR / path] + ([base / path] if base else [])
    for candidate in candidates:
        if candidate.is_file():
            brief = json.loads(candidate.read_text(encoding="utf-8"))
            brief.setdefault("_path", path)
            return brief
    raise FileNotFoundError(f"Brief not found: {path}")


def cmd_build(args) -> int:
    from .build import Builder

    report = Builder(drafts=args.drafts, out_root=Path(args.out) if args.out else None,
                     og=False if args.no_og else (True if args.og else None)).build()
    print(f"Built {len(report.pages)} files into {report.out_dir}" + (f" ({report.og_images} social images)" if report.og_images else ""))
    for note in report.notes:
        print(f"  · {note}")
    for problem in report.problems:
        print(f"  ✗ {problem}")
    if report.problems:
        print("Fix the problems above before publishing (drafts are left out unless you pass --drafts).")
    return 0 if report.ok else 2


def cmd_serve(args) -> int:
    from .build import Builder
    from .paths import DIST_DIR

    report = Builder(drafts=args.drafts, og=False).build()
    for problem in report.problems:
        print(f"  ✗ {problem}")
    site_path = report.out_dir.relative_to(DIST_DIR).as_posix()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(DIST_DIR))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"Preview: http://localhost:{args.port}/{site_path}/  (Ctrl+C to stop)")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
    return 0


def check_all(only: list[str] | None = None) -> tuple[list[tuple[str, list]], bool]:
    from .build import Builder
    from .config import load_site
    from .guardrails import has_errors, lint_item
    from .pipeline import brief_text, load_corpus
    from .profile import load_profile

    site, profile, corpus = load_site(), load_profile(), load_corpus()
    builder = Builder(drafts=True, og=False)
    builder._prepare_entries()
    from .build import BuildReport

    report = BuildReport(out_dir=builder.out)
    builder._render_bodies(report)
    results, failed = [], False
    for entry in builder.entries.values():
        item = entry.item
        if only and not any(Path(o).stem == item.slug for o in only):
            continue
        brief = None
        if item.brief:
            try:
                brief = _brief(item.brief)
            except FileNotFoundError:
                pass
        issues = lint_item(item, entry.rendered, site, profile, corpus, brief_text(brief))
        if item.published:
            published_refs = {i.ref for i in builder.full.all() if i.published}
            for ref in entry.rendered.refs:
                if ref.startswith(("issue:", "guide:")) and ref not in published_refs:
                    from .guardrails import Issue

                    issues.append(Issue("error", "links_draft", f"links to {ref}, which isn't published"))
        failed = failed or (item.published and has_errors(issues))
        results.append((f"{item.kind} {item.slug} ({item.status})", issues))
    return results, failed


def cmd_check(args) -> int:
    from .guardrails import format_issues

    results, failed = check_all(args.files)
    for name, issues in results:
        print(f"== {name}")
        print(format_issues(issues))
    if not results:
        print("No issues or guides yet. Start one: python -m nle new issue \"Title\"")
    return 2 if failed else 0


NEW_ISSUE = """Open with the problem, in one short line.

Then the payoff, in one more.

What's inside:

1. [first thing]
2. [second thing]
3. [third thing]

::subscribe

---

## [First section heading]

[Short paragraphs. One or two sentences each.]

::screenshot[[what the reader must see on screen]]

```prompt [What this prompt does]
[the prompt to copy]
```

---

## [Next section heading]

[...]

---

## What to do today

[One clear action.]

::button[Share it with one person who needs it](page:subscribe)

[One-line sign-off.]
"""

NEW_GUIDE = """[Who this guide is for, in one line.]

By the end you'll have [the outcome].

**You need:**

- [what they need before starting]

---

## 1. [First step]

[What to do, where to click, how to check it worked.]

::screenshot[[what the reader must see on screen]]

---

## 2. [Next step]

> [!WARNING]
> [The one mistake that would cost the reader time.]

---

## Before you finish

- [ ] [check one]
- [ ] [check two]

## Questions

### [A common question]

[The answer.]
"""


def cmd_new(args) -> int:
    from .content import Item, dump
    from .paths import GUIDES_DIR, ISSUES_DIR
    from .pipeline import slugify

    folder = ISSUES_DIR if args.kind == "issue" else GUIDES_DIR
    slug = args.slug or slugify(args.title)
    path = folder / f"{slug}.md"
    if path.exists():
        print(f"{path} already exists.")
        return 1
    today = dt.date.today()
    item = Item(kind=args.kind, slug=slug, title=args.title, subtitle="", body=NEW_ISSUE if args.kind == "issue" else NEW_GUIDE,
                status="draft", date=today if args.kind == "issue" else None, updated=today if args.kind == "guide" else None,
                level="beginner" if args.kind == "guide" else "")
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(item), encoding="utf-8")
    print(f"Created {path}. Fill it in, then: python -m nle check && python -m nle serve --drafts")
    return 0


def cmd_write(args) -> int:
    from .llm import Claude
    from .pipeline import write

    brief = _brief(args.brief) or {}
    if args.topic:
        brief["topic"] = args.topic
    if args.notes:
        notes = Path(args.notes[1:]).read_text(encoding="utf-8") if args.notes.startswith("@") else args.notes
        brief["notes"] = notes
    result = write(brief, args.kind, claude=Claude(model=args.model, effort=args.effort))
    print(f"\nDraft saved: {result['path']}")
    print(result["report"])
    print(f"Cost: about ${result['cost']['usd_estimate']:.2f} ({result['cost']['calls']} Claude calls)")
    print("Next: add the screenshots, fill any [CHECK: ...] notes, set status = \"published\", then python -m nle check.")
    return 0 if result["ok"] else 2


def cmd_train(args) -> int:
    from .train import train

    print(json.dumps(train(fetch_first=args.fetch), indent=2))
    print("Wrote model/style_profile.json, model/style_guide.md and model/corpus.jsonl.")
    return 0


def cmd_guide(args) -> int:
    from .profile import load_profile, render_style_guide
    from .train import refresh_guide

    if args.write:
        refresh_guide()
        print("Refreshed model/style_guide.md and model/style_profile.json from knowledge.py.")
        return 0
    print(render_style_guide(load_profile()))
    return 0


def cmd_publish(args) -> int:
    from .build import Builder
    from .config import load_site

    target_root = Path(args.into).resolve()
    if not (target_root / "index.html").is_file():
        print(f"{target_root} doesn't look like a website folder (no index.html). Point --into at the site's root folder.")
        return 1
    results, failed = check_all()
    if failed:
        print("The checks found errors in published pieces. Run `python -m nle check` and fix them first.")
        return 2
    report = Builder(drafts=False).build()
    if not report.ok:
        for problem in report.problems:
            print(f"  ✗ {problem}")
        return 2
    site = load_site()
    rel = site.base_path.strip("/")
    if not rel:
        print("base_path in site.json is '/', which would overwrite the whole website. Set a folder such as /learn/.")
        return 1
    destination = target_root / rel
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(report.out_dir, destination)
    print(f"Copied {len(report.pages)} files to {destination}.")
    print(f"Next: add {site.absolute('sitemap.xml')} to the main site's robots.txt (Sitemap: line), check the main site, "
          "then commit and merge so it goes live.")
    return 0


def cmd_catalog(args) -> int:
    from .config import load_site
    from .content import load_library
    from .md import render

    site, library = load_site(), load_library().visible()
    pages = []
    for item in library.all():
        rendered = render(item.body)
        path = ("p/" if item.kind == "issue" else "guides/") + item.slug + "/"
        pages.append({"kind": item.kind, "title": item.title, "subtitle": item.subtitle, "url": site.absolute(path),
                      "minutes": rendered.minutes, "tags": list(item.tags),
                      "sections": [{"title": title, "url": site.absolute(path) + "#" + anchor} for level, anchor, title in rendered.toc if level == 2]})
    for key, label in (("start/", "Start here: the guides in reading order"), ("library/", "Library: prompts and templates to copy"),
                       ("guides/", "All guides"), ("archive/", "Newsletter archive"), ("", "Home")):
        pages.append({"kind": "page", "title": label, "subtitle": "", "url": site.absolute(key), "minutes": 0, "tags": [], "sections": []})
    out = Path(args.out)
    out.write_text(json.dumps({"site": site.absolute(""), "pages": pages}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(pages)} pages to {out}")
    return 0


def cmd_substack(args) -> int:
    from .substack import export

    path = export(Path(args.out) if args.out else None)
    print(f"Substack kit: {path}")
    print("Open it in a browser on a computer, then for each post: Copy title, Copy subtitle, Copy post, paste into Substack.")
    print("Images load from the live site, so publish the site (python -m nle publish) before pasting.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m nle", description="Newsletter engine: your guides and issues, as a website.",
                                     epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build", help="build the website into dist/")
    p.add_argument("--drafts", action="store_true", help="include drafts (for previewing)")
    p.add_argument("--out", help="output folder (default: dist/)")
    p.add_argument("--og", action="store_true", help="fail if the social images can't be made")
    p.add_argument("--no-og", action="store_true", help="skip the social images")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("serve", help="build and preview the site locally")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--drafts", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("check", help="run the checks on every issue and guide (or the files given)")
    p.add_argument("files", nargs="*")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("new", help="start a draft issue or guide from the learned structure")
    p.add_argument("kind", choices=["issue", "guide"])
    p.add_argument("title")
    p.add_argument("--slug")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("write", help="have Claude write a draft from a brief (needs ANTHROPIC_API_KEY)")
    p.add_argument("kind", choices=["issue", "guide"])
    p.add_argument("topic", nargs="?")
    p.add_argument("--brief", help="a JSON brief (topic, audience, goal, notes, links)")
    p.add_argument("--notes", help="facts the piece may use; @file.txt to read a file")
    p.add_argument("--model")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    p.set_defaults(func=cmd_write)

    p = sub.add_parser("train", help="measure the example issues and write the style files")
    p.add_argument("--fetch", action="store_true", help="download the material first (training/sources.local.json)")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("guide", help="print the style guide")
    p.add_argument("--write", action="store_true", help="save it after editing nle/knowledge.py")
    p.set_defaults(func=cmd_guide)

    p = sub.add_parser("catalog", help="list every published page (for the LinkedIn engine to link to)")
    p.add_argument("--out", default="../linkedin-engine/learn_catalog.json")
    p.set_defaults(func=cmd_catalog)

    p = sub.add_parser("substack", help="export every issue and guide ready to paste into Substack, with a copy-button kit page")
    p.add_argument("--out", help="output folder (default: dist/substack/)")
    p.set_defaults(func=cmd_substack)

    p = sub.add_parser("publish", help="build, check and copy the site into the main website's folder")
    p.add_argument("--into", required=True, help="the main website's folder, e.g. ../mojeedabisiga.com/site")
    p.set_defaults(func=cmd_publish)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:
        name = type(error).__name__
        if name == "AuthenticationError" or "Could not resolve authentication" in str(error):
            print("Claude API: no valid credentials. Set ANTHROPIC_API_KEY.", file=sys.stderr)
        elif name in {"RateLimitError", "OverloadedError", "InternalServerError", "APIConnectionError", "APITimeoutError"}:
            print(f"Claude API is busy or unreachable ({name}). Try again in a minute.", file=sys.stderr)
        elif name in {"FileNotFoundError", "ValueError", "ContentError", "LLMError", "RuntimeError", "JSONDecodeError"}:
            print(f"{name}: {error}", file=sys.stderr)
        else:
            raise
        return 1
