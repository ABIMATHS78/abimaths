"""A tiny markup for text on graphics. Claude writes it; templates render it safely (HTML is escaped).

  *word*      accent colour        **word**   bold          ==word==   marker highlight
  __word__    accent underline     [[word]]   accent pill   `code`     monospace chip
  A new line in the text becomes a line break.

Line breaks are kept out of the places where a break reads badly: a /command never splits at its hyphens, an arrow
stays on the line of the word before it, and a lone + stays with its arrow ("Skills → +").
"""

from __future__ import annotations

import html
import re

from markupsafe import Markup

_RULES = [
    (re.compile(r"\*\*(.+?)\*\*"), r"<strong>\1</strong>"),
    (re.compile(r"==(.+?)=="), r"<mark>\1</mark>"),
    (re.compile(r"__(.+?)__"), r'<span class="ul">\1</span>'),
    (re.compile(r"\[\[(.+?)\]\]"), r'<span class="pill">\1</span>'),
    (re.compile(r"(?<![\w*])\*(?![\s*])(.+?)(?<![\s*])\*(?![\w*])"), r'<em class="accent">\1</em>'),
]
_COMMAND = re.compile(r"(?<![\w/.:~-])(/[a-z0-9][a-z0-9-]*[a-z0-9])(?![\w/-])")
_STRIP = re.compile(r"\*\*|==|__|\[\[|\]\]|`|(?<![\w*])\*(?![\s*])|(?<![\s*])\*(?![\w*])")


def rich(text) -> Markup:
    if text is None:
        return Markup("")
    parts = str(text).split("`")
    out = []
    for index, part in enumerate(parts):
        escaped = html.escape(part, quote=False)
        if index % 2 == 1 and index < len(parts) - 1:
            out.append(f'<code class="chip">{escaped}</code>')
            continue
        if index % 2 == 1:  # an unmatched backtick: keep it as text
            escaped = "`" + escaped
        escaped = _keep_together(escaped)
        for pattern, replacement in _RULES:
            escaped = pattern.sub(replacement, escaped)
        out.append(escaped)
    return Markup("".join(out).replace("\n", "<br>"))


def _keep_together(escaped: str) -> str:
    escaped = re.sub(r" +→", "\u00a0→", escaped)
    escaped = re.sub(r"→ +\+(?=\s|$)", "→\u00a0+", escaped)
    return _COMMAND.sub(r'<span class="nw">\1</span>', escaped)


def plain(text) -> str:
    return _STRIP.sub("", str(text or ""))


def words(text, anim: str = "pop", dur: float | None = None, cls: str = "w") -> Markup:
    """Wrap every word in a span so motion templates can reveal words one by one.
    Accent markup (*word*) is kept per word."""
    out = []
    for line_index, line in enumerate(str(text or "").split("\n")):
        if line_index:
            out.append("<br>")
        accent = False
        for token in line.split():
            starts, ends = token.startswith("*") and not token.startswith("**"), token.endswith("*") and not token.endswith("**")
            if starts:
                accent = True
            word = plain(token)
            klass = f"{cls} accent" if accent else cls
            timing = f' data-dur="{dur}"' if dur else ""
            out.append(f'<span class="{klass}" data-anim="{anim}"{timing}>{html.escape(word, quote=False)}</span> ')
            if ends:
                accent = False
    return Markup("".join(out).rstrip())
