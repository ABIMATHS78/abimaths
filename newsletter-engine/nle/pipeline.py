"""From a brief to a draft file: Claude writes the issue or guide, the checks run, errors go back to Claude to fix,
and the draft is saved in content/issues/ or content/guides/ with status = "draft" for the author to finish."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

from .config import load_site
from .content import Item, dump, load_library
from .guardrails import format_issues, has_errors, lint_item
from .md import Renderer
from .paths import CORPUS_PATH, GUIDES_DIR, ISSUES_DIR
from .profile import load_profile


def brief_text(brief: dict | None) -> str:
    if not brief:
        return ""
    parts = [str(brief.get(k, "")) for k in ("topic", "audience", "goal", "angle")]
    notes = brief.get("notes", "")
    parts.append("\n".join(notes) if isinstance(notes, list) else str(notes))
    parts += list(brief.get("links", []))
    return "\n".join(parts)


def load_corpus() -> list[dict]:
    if not CORPUS_PATH.exists():
        return []
    return [json.loads(line) for line in CORPUS_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def pick_examples(corpus: list[dict], brief: dict, k: int = 2) -> list[dict]:
    """The complete example issues closest to the brief (shared words), for shape."""
    words = set(re.findall(r"[a-z]{4,}", brief_text(brief).lower()))
    usable = [r for r in corpus if r.get("use_as_example") and r.get("complete")]
    scored = sorted(usable, key=lambda r: -len(words & set(re.findall(r"[a-z]{4,}", r["text"].lower()))))
    return scored[:k]


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60].rstrip("-") or "untitled"


def write(brief: dict, kind: str = "issue", *, claude=None, repairs: int = 2, examples_k: int = 2, log=print) -> dict:
    """Write a draft with Claude, check it, repair it, and save it. Returns the path, the checks and the cost."""
    from .llm import Claude
    from .prompts import repair_user, write_schema, write_system, write_user

    if kind not in ("issue", "guide"):
        raise ValueError("kind must be 'issue' or 'guide'")
    if not str(brief.get("topic", "")).strip():
        raise ValueError("The brief needs a topic.")
    site, profile, library, corpus = load_site(), load_profile(), load_library(), load_corpus()
    claude = claude or Claude()
    examples = pick_examples(corpus, brief, examples_k)
    if examples:
        log(f"Examples for shape: {', '.join(e['key'] for e in examples)}")
    system = write_system(profile, site, library)
    messages = [{"role": "user", "content": write_user(brief, kind, examples)}]
    log(f"Writing the {kind} with {claude.model} ...")
    data, content = claude.json_call(system, messages, write_schema(kind), label=kind)
    facts = brief_text(brief)
    item, issues = _check(data, kind, brief, site, profile, library, corpus, facts)
    for _ in range(repairs):
        errors = [i for i in issues if i.level == "error" and i.code not in ("screenshots", "needs_input")]
        if not errors:
            break
        log("  fixing: " + "; ".join(i.message for i in errors))
        messages = messages + [{"role": "assistant", "content": content}, {"role": "user", "content": repair_user(errors)}]
        data, content = claude.json_call(system, messages, write_schema(kind), label=f"{kind} repair")
        item, issues = _check(data, kind, brief, site, profile, library, corpus, facts)
    folder = ISSUES_DIR if kind == "issue" else GUIDES_DIR
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{item.slug}.md"
    n = 2
    while path.exists():
        path = folder / f"{item.slug}-{n}.md"
        n += 1
    path.write_text(dump(item), encoding="utf-8")
    blocking = [i for i in issues if i.level == "error" and i.code not in ("screenshots", "needs_input")]
    return {"path": str(path), "issues": [i.as_dict() for i in issues], "ok": not blocking, "cost": claude.cost(),
            "report": format_issues(issues)}


def _check(data: dict, kind: str, brief: dict, site, profile, library, corpus, facts) -> tuple[Item, list]:
    item = Item(kind=kind, slug=slugify(data.get("slug") or data.get("title", "")), title=data.get("title", "").strip(),
                subtitle=data.get("subtitle", "").strip(), description=data.get("description", "").strip(),
                body=re.sub(r"^#\s.*\n+", "", data.get("body", "").strip()) + "\n", status="draft",
                date=dt.date.today() if kind == "issue" else None, updated=dt.date.today() if kind == "guide" else None,
                tags=[t.strip().lower() for t in data.get("tags", []) if t.strip()][:4],
                archetype=data.get("archetype", ""), fact_checks=list(data.get("fact_checks", [])),
                brief=brief.get("_path", ""))
    known = {i.ref for i in library.all() if i.published}
    rendered = Renderer(resolve=lambda ref: "#" if ref in known or ref.startswith("page:") else None,
                        card=lambda ref: "" if ref in known else None, media=lambda src: None).render(item.body)
    return item, lint_item(item, rendered, site, profile, corpus, facts)
