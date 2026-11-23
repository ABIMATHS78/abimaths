"""Inline text markup used inside content blocks. Writers (people or Claude) use four marks:

  **bold**    *italic*    `code`    [link text](target)

A target is a web address, or a reference to another page of the publication: `issue:<slug>`, `guide:<slug>`,
`page:subscribe`, `page:archive`, `page:guides`, `page:start`, `page:library` or `page:about`. Everything else is
escaped, so text can never inject HTML.
"""

from __future__ import annotations

import html
import re

LINK_RE = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")
CODE_RE = re.compile(r"`([^`\n]+)`")
BOLD_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
ITALIC_RE = re.compile(r"(?<![\w*])\*(?=\S)(.+?)(?<=\S)\*(?![\w*])")
REF_RE = re.compile(r"^(issue|guide|page):([a-z0-9][a-z0-9-]*)$")
SAFE_SCHEMES = ("http://", "https://", "mailto:", "#", "/")


def resolve_target(target: str, resolver=None) -> str | None:
    """A reference becomes a site link through `resolver`; a web address passes; anything else is refused."""
    if REF_RE.match(target):
        return resolver(target) if resolver else None
    if target.startswith(SAFE_SCHEMES):
        return target
    return None


def inline(text, resolver=None) -> str:
    """Render inline markup to safe HTML."""
    text = str(text or "")
    out, last = [], 0
    for match in re.finditer(r"`[^`\n]+`|\[[^\]\n]+\]\([^)\s]+\)", text):
        out.append(_marks(html.escape(text[last:match.start()], quote=False)))
        token = match.group(0)
        if token.startswith("`"):
            out.append(f"<code>{html.escape(token[1:-1], quote=False)}</code>")
        else:
            label, target = LINK_RE.match(token).groups()
            href = resolve_target(target, resolver)
            label_html = _marks(html.escape(label, quote=False))
            if href is None:
                out.append(label_html)
            else:
                external = href.startswith(("http://", "https://"))
                rel = ' rel="noopener"' if external else ""
                out.append(f'<a href="{html.escape(href)}"{rel}>{label_html}</a>')
        last = match.end()
    out.append(_marks(html.escape(text[last:], quote=False)))
    return "".join(out)


def _marks(escaped: str) -> str:
    escaped = BOLD_RE.sub(r"<strong>\1</strong>", escaped)
    return ITALIC_RE.sub(r"<em>\1</em>", escaped)


def plain(text) -> str:
    """The words without the marks: for search, reading time, checks and descriptions."""
    text = LINK_RE.sub(r"\1", str(text or ""))
    text = CODE_RE.sub(r"\1", text)
    text = BOLD_RE.sub(r"\1", text)
    return ITALIC_RE.sub(r"\1", text)


def links(text) -> list[str]:
    """Every link target in the text, references included."""
    return [target for _, target in LINK_RE.findall(str(text or ""))]
