"""Command line: python -m lpe <command> --help"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .paths import DATA_DIR

EXAMPLES = """
examples:
  python -m lpe train --drive "https://drive.google.com/drive/folders/<id>"
  python -m lpe generate "Why your AI drafts sound generic" --notes @notes.txt --format gif
  python -m lpe generate --brief briefs/skills-starter-kit.json
  python -m lpe render output/2026-11-23-my-post/spec.json --format png,mp4
  python -m lpe lint output/2026-11-23-my-post/post.txt --fix
  python -m lpe templates cycle --sample
"""


def _read_arg(value: str | None) -> str | None:
    """Values starting with @ are read from that file."""
    if value and value.startswith("@"):
        return Path(value[1:]).read_text(encoding="utf-8")
    return value


def _load_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cmd_train(args) -> int:
    from .train import train

    summary = train(drive_link=args.drive, data_dir=Path(args.data), annotate=args.annotate)
    print(json.dumps(summary, indent=2))
    print("Wrote model/style_profile.json, model/voice_guide.md and model/corpus.jsonl.")
    return 0


def _brief_from_args(args) -> dict:
    brief = _load_json(args.brief) if args.brief else {}
    if args.topic:
        brief["topic"] = args.topic
    for key in ("audience", "goal", "angle", "cta"):
        if getattr(args, key, None):
            brief[key] = getattr(args, key)
    if args.notes:
        brief["notes"] = _read_arg(args.notes)
    if args.link:
        brief["links"] = list(brief.get("links", [])) + args.link
    return brief


def cmd_generate(args) -> int:
    from .llm import Claude
    from .pipeline import generate

    brief = _brief_from_args(args)
    template = args.template or brief.get("template")
    theme = args.theme or brief.get("theme")
    formats = args.format or brief.get("format")
    claude = Claude(model=args.model, effort=args.effort)
    result = generate(brief, template=template, theme=theme, formats=formats, out_dir=Path(args.out) if args.out else None,
                      claude=claude, examples_k=args.examples, scale=args.scale, render_graphic=not args.no_render)
    _print_result(result)
    return 0 if result.get("ok", True) else 2


def cmd_render(args) -> int:
    from .pipeline import finish, resolve_formats

    spec_path = Path(args.spec)
    spec = _load_json(args.spec)
    brief = _load_json(args.brief) if args.brief else spec.get("brief")
    out_dir = Path(args.out) if args.out else spec_path.parent
    formats = resolve_formats(args.format or spec.get("formats"), None, spec["template"])
    result = finish(spec, out_dir, brief=brief, formats=formats, scale=args.scale,
                    plan={k: spec[k] for k in ("alternative_hooks", "fact_checks") if k in spec})
    _print_result(result)
    return 0 if result.get("ok", True) else 2


def _print_result(result: dict) -> None:
    from .guardrails import Issue, format_issues

    issues = [Issue(**i) for i in result.get("post_issues", []) + result.get("graphic_issues", [])]
    print()
    print(f"Done: {result.get('out_dir')}")
    for kind, name in (result.get("files") or {}).items():
        print(f"  {kind:5} {name}")
    if result.get("characters"):
        print(f"  post  {result['characters']:,} characters")
    if result.get("cost"):
        print(f"  cost  about ${result['cost']['usd_estimate']:.2f} ({result['cost']['calls']} Claude calls)")
    if issues or result.get("render_problems"):
        print("\nChecks:")
        print(format_issues(issues))
        for problem in result.get("render_problems", []):
            print(f"! [graphic] {problem}")


def cmd_lint(args) -> int:
    from .brand import load_brand
    from .guardrails import fix_post, format_issues, has_errors, lint_post
    from .pipeline import brief_text
    from .profile import load_profile
    from .retrieval import load_corpus

    path = Path(args.file)
    text = path.read_text(encoding="utf-8").strip()
    if path.suffix == ".json":
        text = json.loads(text).get("post", "")
    if args.fix:
        fixed = fix_post(text)
        if fixed != text and path.suffix != ".json":
            path.write_text(fixed + "\n", encoding="utf-8")
            print(f"Fixed em dashes and stray markdown in {path}.")
        text = fixed
    brief = _load_json(args.brief) if args.brief else None
    issues = lint_post(text, load_profile(), load_corpus(), brief_text(brief), load_brand())
    print(f"{len(text):,} characters")
    print(format_issues(issues))
    return 2 if has_errors(issues) else 0


def cmd_examples(args) -> int:
    from .retrieval import load_corpus, similar_examples

    corpus = load_corpus()
    if not corpus:
        print("No training texts on this machine. Run: python -m lpe train --drive <folder link>")
        return 1
    for example in similar_examples(args.query, corpus, k=args.k, template=args.template):
        print(f"=== {example['key']}  template={example.get('template')}  hook={example.get('hook_type')}  "
              f"structure={example.get('structure')}  match={example['score']}")
        print(f"graphic: {example.get('graphic_notes', '')}")
        print(example["text"] if args.full else example["text"][:600] + (" ..." if len(example["text"]) > 600 else ""))
        print()
    return 0


def cmd_templates(args) -> int:
    from .schemas import api_schema
    from .templates import TEMPLATES, get_template

    if not args.name:
        for template in TEMPLATES.values():
            print(f"{template.name:19} [{template.default_theme}, {template.motion_default}] {template.description}")
        return 0
    template = get_template(args.name)
    print(f"{template.name}: {template.description}")
    print(f"default theme: {template.default_theme}; usual format: {template.motion_default}; shapes: {', '.join(template.shapes)}")
    if args.schema or not args.sample:
        print("\nschema:")
        print(json.dumps(api_schema(template.schema), indent=1, ensure_ascii=False))
    if args.sample:
        print("\nsample spec:")
        print(json.dumps({"template": template.name, "data": template.sample}, indent=1, ensure_ascii=False))
    return 0


def cmd_themes(args) -> int:
    from .themes import THEMES, theme_vars

    for name in THEMES:
        v = theme_vars(name)
        print(f"{name:11} background {v['bg']}  ink {v['ink']}  accent {v['accent']}")
    return 0


def cmd_guide(args) -> int:
    from .paths import VOICE_GUIDE_PATH

    print(VOICE_GUIDE_PATH.read_text(encoding="utf-8"))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m lpe", description="LinkedIn Post Engine: posts and graphics in a learned style.",
                                     epilog=EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="learn the style from a folder of example posts")
    p.add_argument("--drive", help="link to a Google Drive folder shared as 'Anyone with the link'")
    p.add_argument("--data", default=str(DATA_DIR), help="local folder of training files (default: training/data)")
    p.add_argument("--annotate", action="store_true", help="ask Claude to classify posts that have no annotation yet")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("generate", help="write a post and render its graphic with Claude")
    p.add_argument("topic", nargs="?", help="what the post is about")
    p.add_argument("--brief", help="a JSON brief (topic, audience, goal, notes, links, template, format)")
    p.add_argument("--notes", help="facts and material the post may use; @file.txt to read a file")
    p.add_argument("--audience")
    p.add_argument("--goal", help="what the reader should think or do")
    p.add_argument("--angle", help="the point you want to make")
    p.add_argument("--cta", help="the closing you want")
    p.add_argument("--link", action="append", help="a link the post may include (repeatable)")
    p.add_argument("--template", help="force a template (see `templates`)")
    p.add_argument("--theme", help="force a colour theme (see `themes`)")
    p.add_argument("--format", help="auto, png, gif, mp4, all, or a list like png,mp4 (default: auto)")
    p.add_argument("--out", help="output folder (default: output/<date>-<topic>)")
    p.add_argument("--model", help="Claude model (default: claude-opus-5-5 or $LPE_MODEL)")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], help="thinking effort (default: high)")
    p.add_argument("--examples", type=int, default=3, help="training posts to show Claude (default 3)")
    p.add_argument("--scale", type=float, default=2, help="PNG pixel density (2 gives 2160x2700)")
    p.add_argument("--no-render", action="store_true", help="write post and spec.json only")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("render", help="render a spec.json (post + graphic data) without calling Claude")
    p.add_argument("spec")
    p.add_argument("--format", help="png, gif, mp4, all or a list (default: the spec's or the template's)")
    p.add_argument("--out", help="output folder (default: the spec's folder)")
    p.add_argument("--brief", help="the brief, so the checks know which facts and links were given")
    p.add_argument("--scale", type=float, default=2)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("lint", help="check a post for limits, AI giveaways, copying and invented links")
    p.add_argument("file", help="post.txt, or a spec.json with a 'post' field")
    p.add_argument("--brief")
    p.add_argument("--fix", action="store_true", help="fix em dashes and markdown in place")
    p.set_defaults(func=cmd_lint)

    p = sub.add_parser("examples", help="show the training posts closest to a topic")
    p.add_argument("query")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--template")
    p.add_argument("--full", action="store_true", help="print whole posts")
    p.set_defaults(func=cmd_examples)

    p = sub.add_parser("templates", help="list templates, or show one's schema and sample")
    p.add_argument("name", nargs="?")
    p.add_argument("--schema", action="store_true")
    p.add_argument("--sample", action="store_true")
    p.set_defaults(func=cmd_templates)

    sub.add_parser("themes", help="list colour themes").set_defaults(func=cmd_themes)
    sub.add_parser("guide", help="print the trained voice guide").set_defaults(func=cmd_guide)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130
    except Exception as error:  # turn common failures into plain advice
        name = type(error).__name__
        if name == "AuthenticationError" or "Could not resolve authentication" in str(error):
            print("Claude API: no valid credentials. Set ANTHROPIC_API_KEY (or run `ant auth login`).", file=sys.stderr)
        elif name in {"RateLimitError", "OverloadedError", "InternalServerError", "APIConnectionError", "APITimeoutError"}:
            print(f"Claude API is busy or unreachable ({name}). Try again in a minute.", file=sys.stderr)
        elif name in {"FileNotFoundError", "ValueError", "LLMError", "RuntimeError", "JSONDecodeError"}:
            print(f"{name}: {error}", file=sys.stderr)
        else:
            raise
        return 1
