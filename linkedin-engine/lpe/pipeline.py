"""From a brief to a finished post folder: post text, graphic (PNG, GIF and/or MP4), preview and report.

generate() does the whole job with the Claude API. finish() does everything after the writing, so the
same output comes from `lpe render spec.json` when the post and graphic data were written by hand or by
Claude Code (see the linkedin-post-engine skill).
"""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from .brand import load_brand
from .guardrails import fix_post, format_issues, has_errors, lint_extras, lint_graphic, lint_post, plan_extras
from .paths import ENGINE_DIR, OUTPUT_DIR
from .profile import load_profile
from .retrieval import load_corpus, similar_examples
from .schemas import clean_data
from .templates import get_template


def slugify(text: str, limit: int = 48) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "post"


def brief_text(brief: dict | None) -> str:
    if not brief:
        return ""
    parts = [str(brief.get(k, "")) for k in ("topic", "audience", "goal", "angle", "cta")]
    notes = brief.get("notes", "")
    parts.append("\n".join(notes) if isinstance(notes, list) else str(notes))
    parts += list(brief.get("links", []))
    return "\n".join(parts)


def load_brief(value, base: Path | None = None) -> dict | None:
    """A brief given inline (a dict) or as a path to a JSON file, looked up next to the spec, then in the engine folder."""
    if not value:
        return None
    if isinstance(value, dict):
        return value
    path = Path(value)
    candidates = [path] if path.is_absolute() else [Path(base or ".") / path, ENGINE_DIR / path, path]
    for candidate in candidates:
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"Brief not found: {value}")


def post_issues_for(post: str, plan: dict, profile: dict, corpus: list[dict], brief: str, brand: dict,
                    alt_text: bool = True) -> list:
    """Every check for the writer's output: the post itself, then the other fields a reader or the author sees."""
    issues = lint_post(post, profile, corpus, brief, brand) if post else []
    return issues + lint_extras(plan_extras(plan, alt_text), profile, brand, brief)


def default_out_dir(topic: str) -> Path:
    return OUTPUT_DIR / f"{dt.date.today().isoformat()}-{slugify(topic)}"


def resolve_formats(requested, suggested: str | None, template_name: str) -> list[str]:
    """png always; plus gif/mp4 as requested, or the writer's suggestion, or the template's default."""
    if isinstance(requested, str):
        requested = [f.strip() for f in requested.split(",") if f.strip()]
    if requested and "auto" not in requested:
        chosen = ["gif", "mp4", "png"] if "all" in requested else list(requested)
    else:
        motion = suggested or get_template(template_name).motion_default
        chosen = [motion]
    return ["png"] + [f for f in ("gif", "mp4") if f in chosen]


def finish(spec: dict, out_dir: Path, *, brand: dict | None = None, profile: dict | None = None,
           corpus: list[dict] | None = None, brief: dict | None = None, formats=("png",), plan: dict | None = None,
           scale: float = 2, renderer=None, cost: dict | None = None, log=print) -> dict:
    from .preview import write_preview
    from .render import render

    brand = brand or load_brand()
    profile = profile if profile is not None else _load_profile_or_empty()
    corpus = corpus if corpus is not None else load_corpus()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    post = spec.get("post", "")
    plan = plan or {}

    template = get_template(spec["template"])
    _, schema_warnings = clean_data(spec.get("data") or {}, template.schema)
    facts = brief_text(brief)
    post_issues = post_issues_for(post, plan, profile, corpus, facts, brand, alt_text=False)
    graphic_issues = lint_graphic(spec, profile, schema_warnings, brand, facts)

    log(f"Rendering the {spec['template']} graphic ({', '.join(formats)}) ...")
    rendered = render(spec, brand, out_dir, formats=formats, scale=scale, renderer=renderer, log=log)
    files = rendered["files"]
    if post:
        (out_dir / "post.txt").write_text(post + "\n", encoding="utf-8")
        write_preview(out_dir, post, brand, files, spec.get("alt_text", ""))

    report = {
        "template": spec["template"],
        "theme": rendered.get("theme"),
        "formats": list(formats),
        "characters": len(post),
        "post_issues": [i.as_dict() for i in post_issues],
        "graphic_issues": [i.as_dict() for i in graphic_issues],
        "render_problems": rendered.get("problems", []),
        "fit": rendered.get("fit", []),
        "motion": rendered.get("motion"),
        "gif": rendered.get("gif"),
        "files": {k: Path(v).name for k, v in files.items()},
        "plan": {k: plan.get(k) for k in ("angle", "hook_type", "structure", "motion") if plan.get(k)},
        "examples_used": plan.get("_examples", []),
        "cost": cost,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "post.md").write_text(_post_markdown(spec, plan, report, post_issues + graphic_issues), encoding="utf-8")
    report["out_dir"] = str(out_dir)
    report["ok"] = not has_errors(post_issues + graphic_issues) and not rendered.get("problems")
    return report


def _post_markdown(spec: dict, plan: dict, report: dict, issues: list) -> str:
    post = spec.get("post", "")
    title = post.splitlines()[0] if post else spec["template"]
    if len(title) > 80:
        title = title[:80].rsplit(" ", 1)[0] + "…"
    shareable = [name for kind, name in report["files"].items() if kind != "html"]
    lines = [f"# {title}", ""]
    lines.append(f"Template: {report['template']} · Theme: {report['theme']} · Files: {', '.join(shareable)}")
    if post:
        lines.append(f"Characters: {len(post):,} of 3,000")
    lines += ["", "## Post", "", "Copy everything between the lines.", "", "```text", post, "```", ""]
    if plan.get("alternative_hooks"):
        lines += ["## Other openings", ""] + [f"{n}. {h}" for n, h in enumerate(plan["alternative_hooks"], 1)] + [""]
    checks = list(plan.get("fact_checks") or [])
    lines += ["## Before you post", ""]
    lines += [f"- [ ] {c}" for c in checks] or ["- [ ] Read it once out loud."]
    if issues:
        lines += ["", "Checks found:", "", format_issues(issues)]
    if spec.get("alt_text"):
        lines += ["", "## Alt text", "", spec["alt_text"]]
    if report.get("motion"):
        m = report["motion"]
        lines += ["", f"Motion: {m['duration_s']} s at {m['fps']} fps."]
    return "\n".join(lines) + "\n"


def _load_profile_or_empty() -> dict:
    try:
        return load_profile()
    except FileNotFoundError:
        return {}


def generate(brief: dict, *, template: str | None = None, theme: str | None = None, formats=None,
             out_dir: Path | None = None, claude=None, examples_k: int = 3, scale: float = 2,
             render_graphic: bool = True, repairs: int = 2, log=print) -> dict:
    """Write a post with Claude in the trained style, design its graphic, check both, and render it.

    Every field the author sees is checked. Errors go back to Claude to fix, up to `repairs` times for the post
    and once for the graphic; whatever is still wrong is reported, and the result's "ok" is False."""
    from .llm import Claude, design_graphic, repair_graphic, repair_post, write_post

    if not str(brief.get("topic", "")).strip():
        raise ValueError("The brief needs a topic.")
    brand = load_brand()
    profile = load_profile()
    corpus = load_corpus()
    if not corpus:
        log("  note: no training texts on this machine, so Claude gets the style guide but no example posts. "
            "Run `python -m lpe train --drive <folder link>` for better results.")
    claude = claude or Claude()
    query = brief_text(brief)
    examples = similar_examples(query, corpus, k=examples_k, template=template)
    if examples:
        log(f"Examples from training: {', '.join(e['key'] for e in examples)}")

    motion = formats if isinstance(formats, str) and formats in ("png", "gif", "mp4") else None
    log(f"Writing the post with {claude.model} ...")
    plan, messages = write_post(claude, brief, brand, profile, examples, template, motion)
    plan["post"] = fix_post(plan["post"])
    issues = post_issues_for(plan["post"], plan, profile, corpus, query, brand)
    for _ in range(repairs):
        errors = [i for i in issues if i.level == "error"]
        if not errors:
            break
        log("  fixing: " + "; ".join(i.message for i in errors))
        plan, messages = repair_post(claude, messages, errors, brand, profile, template)
        plan["post"] = fix_post(plan["post"])
        issues = post_issues_for(plan["post"], plan, profile, corpus, query, brand)
    if has_errors(issues):
        log("  still wrong after the repairs (see the checks below): " + "; ".join(i.message for i in issues if i.level == "error"))

    template_name = template or plan["template"]
    log(f"Designing the {template_name} graphic ...")
    data, design_messages = design_graphic(claude, plan["post"], plan["graphic_brief"], template_name, brand, profile)
    errors = [i for i in lint_graphic({"template": template_name, "data": data}, profile, [], brand, query) if i.level == "error"]
    if errors:
        log("  fixing the graphic: " + "; ".join(i.message for i in errors))
        data, design_messages = repair_graphic(claude, design_messages, errors, template_name, brand, profile)

    spec = {"template": template_name, "data": data, "post": plan["post"], "alt_text": plan.get("alt_text", "")}
    if theme:
        spec["theme"] = theme
    spec["alternative_hooks"] = list(plan.get("alternative_hooks") or [])
    spec["fact_checks"] = list(plan.get("fact_checks") or [])
    spec["brief"] = brief
    plan["_examples"] = [e["key"] for e in examples]

    out_dir = Path(out_dir) if out_dir else default_out_dir(brief["topic"])
    chosen = resolve_formats(formats, plan.get("motion"), template_name)
    if not render_graphic:                      # (a render writes spec.json itself)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "spec.json").write_text(json.dumps(spec, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (out_dir / "post.txt").write_text(plan["post"] + "\n", encoding="utf-8")
        _, schema_warnings = clean_data(data or {}, get_template(template_name).schema)
        post_issues = post_issues_for(plan["post"], plan, profile, corpus, query, brand, alt_text=False)
        graphic_issues = lint_graphic(spec, profile, schema_warnings, brand, query)
        return {"out_dir": str(out_dir), "spec": spec, "plan": plan, "cost": claude.cost(),
                "characters": len(plan["post"]), "files": {"spec": "spec.json", "post": "post.txt"},
                "post_issues": [i.as_dict() for i in post_issues], "graphic_issues": [i.as_dict() for i in graphic_issues],
                "ok": not has_errors(post_issues + graphic_issues)}
    report = finish(spec, out_dir, brand=brand, profile=profile, corpus=corpus, brief=brief, formats=chosen,
                    plan=plan, scale=scale, cost=claude.cost(), log=log)
    report["spec"] = spec
    return report
