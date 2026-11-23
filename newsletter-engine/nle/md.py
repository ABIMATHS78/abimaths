"""Markdown for issues and guides, with a few marks of our own.

Standard Markdown: ## and ### headings, paragraphs, lists, > quotes, --- dividers, tables, `code`, **bold**,
*italic*, [links](...), ![images](path "caption") and - [ ] checklists.

Extra marks:
  ```prompt Optional label        a prompt in a box with a Copy button
  ...
  ```
  > [!TIP] / [!NOTE] / [!WARNING] / [!KEY]      a coloured callout box (the rest of the quote is its text)
  ::button[Read the guide](guide:claude-skills)   a button (a line of its own)
  ::subscribe  or  ::subscribe[Custom line]       the subscribe box
  ::card[issue:some-slug]                         a card linking to another issue or guide
  ::screenshot[What the screenshot must show]     a slot for a screenshot the author still has to take
  ::download[Label](path/or/url)                  a download button

Link targets may be web addresses or references: issue:<slug>, guide:<slug>, page:subscribe, page:archive,
page:guides, page:start, page:library, page:about.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass, field

from markdown_it import MarkdownIt
from mdit_py_plugins.tasklists import tasklists_plugin

from .markup import REF_RE, SAFE_SCHEMES

DIRECTIVE_RE = re.compile(r"^::([a-z]+)(?:\[([^\]\n]*)\])?(?:\(([^)\s]*)\))?\s*$")
DIRECTIVES = ("button", "subscribe", "card", "screenshot", "download")
ALERT_RE = re.compile(r"^\[!(TIP|NOTE|WARNING|KEY|IMPORTANT)\]\s*", re.I)
ALERT_TONES = {"tip": "tip", "note": "note", "warning": "warning", "key": "key", "important": "key"}
ALERT_TITLES = {"tip": "Tip", "note": "Note", "warning": "Watch out", "key": "Key point"}


@dataclass
class Rendered:
    html: str
    text: str                                    # the words, for search, reading time and checks
    toc: list[tuple[int, str, str]] = field(default_factory=list)   # (level, id, title)
    links: list[str] = field(default_factory=list)                   # every link target as written
    refs: list[str] = field(default_factory=list)                    # internal references used
    screenshots: list[str] = field(default_factory=list)             # screenshot slots still open
    problems: list[str] = field(default_factory=list)

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def minutes(self) -> int:
        return max(1, round(self.words / 230))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].rstrip("-") or "section"


def _directive_rule(state, start, end, silent):
    if state.sCount[start] - state.blkIndent >= 4:
        return False
    line = state.src[state.bMarks[start] + state.tShift[start]:state.eMarks[start]]
    match = DIRECTIVE_RE.match(line)
    if not match or match.group(1) not in DIRECTIVES:
        return False
    if silent:
        return True
    token = state.push("directive", "", 0)
    token.meta = {"name": match.group(1), "label": match.group(2) or "", "target": match.group(3) or ""}
    token.map = [start, start + 1]
    token.block = True
    state.line = start + 1
    return True


def _make_parser() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": False, "linkify": False, "typographer": False}).enable(["table", "strikethrough"])
    md.use(tasklists_plugin, enabled=True)
    md.block.ruler.before("paragraph", "directive", _directive_rule,
                          {"alt": ["paragraph", "reference", "blockquote", "list"]})
    return md


_MD = _make_parser()


class Renderer:
    """Renders one body. `resolve(ref)` turns 'guide:slug' into a link (or None); `cards(ref)` and
    `subscribe(label)` return the HTML for those boxes. All three come from the site builder."""

    def __init__(self, resolve=None, card=None, subscribe=None, media=None, draft: bool = False):
        self.resolve = resolve or (lambda ref: None)
        self.card = card or (lambda ref: None)
        self.subscribe = subscribe or (lambda label: "")
        self.media = media or (lambda src: src)      # a relative image path -> its published address, or None
        self.draft = draft

    def render(self, body: str) -> Rendered:
        env: dict = {}
        tokens = _MD.parse(body, env)
        out = Rendered(html="", text="")
        self._prepare(tokens, out)
        out.html = self._html(tokens, out)
        return out

    # --- token passes -------------------------------------------------------------------------------------
    def _prepare(self, tokens, out: Rendered) -> None:
        used_ids: set[str] = set()
        words: list[str] = []
        for i, token in enumerate(tokens):
            if token.type == "heading_open" and token.tag in ("h2", "h3"):
                title = tokens[i + 1].content
                base = _slugify(title)
                anchor, n = base, 2
                while anchor in used_ids:
                    anchor, n = f"{base}-{n}", n + 1
                used_ids.add(anchor)
                token.attrSet("id", anchor)
                out.toc.append((int(token.tag[1]), anchor, title))
            if token.type == "blockquote_open":
                self._alert(tokens, i)
            if token.type == "paragraph_open" and i + 2 < len(tokens) and tokens[i + 2].type == "paragraph_close":
                children = tokens[i + 1].children or []
                if len(children) == 1 and children[0].type == "image":
                    token.meta["figure"] = True
                    tokens[i + 2].meta["figure"] = True
            if token.type == "inline":
                parts = []
                for child in token.children or []:
                    if child.type in ("text", "code_inline"):
                        parts.append(child.content)
                    elif child.type in ("softbreak", "hardbreak"):
                        parts.append(" ")
                    elif child.type == "link_open":
                        self._link(child, out)
                    elif child.type == "image":
                        self._image(child, out)
                words.append("".join(parts))
            if token.type == "fence":
                words.append(token.content)
            if token.type == "directive":
                meta = token.meta
                if meta["label"]:
                    words.append(meta["label"])
                if meta["name"] in ("button", "download", "card"):
                    target = meta["target"] if meta["name"] != "card" else meta["label"]
                    out.links.append(target)
                    if REF_RE.match(target):
                        out.refs.append(target)
                if meta["name"] == "screenshot":
                    out.screenshots.append(meta["label"] or "a screenshot")
        # one line per block, so a check never reads across two paragraphs as one sentence
        out.text = "\n".join(re.sub(r"\s+", " ", w).strip() for w in words if w.strip())

    def _alert(self, tokens, i: int) -> None:
        # blockquote_open, paragraph_open, inline: "[!TIP] text"
        if i + 2 >= len(tokens) or tokens[i + 1].type != "paragraph_open" or tokens[i + 2].type != "inline":
            return
        inline = tokens[i + 2]
        match = ALERT_RE.match(inline.content)
        if not match:
            return
        tone = ALERT_TONES[match.group(1).lower()]
        tokens[i].meta["alert"] = tone
        inline.content = inline.content[match.end():]
        if inline.children and inline.children[0].type == "text":
            first = inline.children[0]
            first.content = ALERT_RE.sub("", first.content, count=1)
            # "[!TIP]" alone on its line leaves a soft break first: drop it
            if not first.content and len(inline.children) > 1 and inline.children[1].type in ("softbreak", "hardbreak"):
                inline.children = inline.children[2:]

    def _image(self, token, out: Rendered) -> None:
        src = token.attrGet("src") or ""
        if not src:
            out.problems.append(f"an image has no file: {token.content!r}")
            return
        if not token.content.strip():
            out.problems.append(f"the image {src} has no alt text (describe what it shows)")
        if src.startswith(("http://", "https://", "/")):
            return
        published = self.media(src)
        if published is None:
            out.problems.append(f"the image {src} isn't in content/media/")
        else:
            token.attrSet("src", published)

    def _link(self, token, out: Rendered) -> None:
        href = token.attrGet("href") or ""
        out.links.append(href)
        if REF_RE.match(href):
            out.refs.append(href)
            resolved = self.resolve(href)
            if resolved is None:
                out.problems.append(f"a link points to {href}, which doesn't exist")
                token.attrSet("href", "#")
                token.attrSet("class", "broken-link")
            else:
                token.attrSet("href", resolved)
        elif not href.startswith(SAFE_SCHEMES):
            out.problems.append(f"a link has an address that isn't a web address or a reference: {href}")
            token.attrSet("href", "#")

    # --- html ---------------------------------------------------------------------------------------------
    def _html(self, tokens, out: Rendered) -> str:
        renderer = _MD.renderer
        parts = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token.type == "directive":
                parts.append(self._directive(token.meta, out))
            elif token.type == "fence":
                parts.append(self._fence(token))
            elif token.type == "blockquote_open" and token.meta.get("alert"):
                tone = token.meta["alert"]
                parts.append(f'<aside class="callout callout-{tone}"><p class="callout-title">{ALERT_TITLES[tone]}</p>')
            elif token.type == "blockquote_close" and self._closes_alert(tokens, i):
                parts.append("</aside>\n")
            elif token.type == "paragraph_open" and token.meta.get("figure"):
                image = tokens[i + 1].children[0]
                parts.append(self._figure(image))
                i += 3
                continue
            elif token.type == "inline":
                parts.append(self._inline(token))
            else:
                parts.append(renderer.renderToken(tokens, i, _MD.options, {}))
            i += 1
        return "".join(parts)

    def _closes_alert(self, tokens, close_index: int) -> bool:
        depth = 0
        for j in range(close_index, -1, -1):
            if tokens[j].type == "blockquote_close":
                depth += 1
            elif tokens[j].type == "blockquote_open":
                depth -= 1
                if depth == 0:
                    return bool(tokens[j].meta.get("alert"))
        return False

    def _inline(self, token) -> str:
        children = token.children or []
        for child in children:
            if child.type == "link_open":
                href = child.attrGet("href") or ""
                if href.startswith(("http://", "https://")):
                    child.attrSet("rel", "noopener")
            if child.type == "image":
                child.attrSet("loading", "lazy")
        return _MD.renderer.renderInline(children, _MD.options, {})

    def _figure(self, image) -> str:
        src = html.escape(image.attrGet("src") or "")
        alt = html.escape(image.content or "")
        caption = image.attrGet("title") or ""
        figcaption = f"<figcaption>{html.escape(caption)}</figcaption>" if caption else ""
        return f'<figure class="figure"><img src="{src}" alt="{alt}" loading="lazy">{figcaption}</figure>\n'

    def _fence(self, token) -> str:
        info = (token.info or "").strip()
        kind, _, label = info.partition(" ")
        code = html.escape(token.content.rstrip("\n"))
        if kind == "prompt":
            title = html.escape(label.strip() or "Prompt to copy")
            return (f'<div class="prompt"><div class="prompt-head"><span>{title}</span>'
                    f'<button class="copy-btn" type="button" data-copy>Copy</button></div>'
                    f'<pre><code>{code}</code></pre></div>\n')
        lang = f' class="language-{html.escape(kind)}"' if kind else ""
        return (f'<div class="codeblock"><button class="copy-btn" type="button" data-copy>Copy</button>'
                f'<pre><code{lang}>{code}</code></pre></div>\n')

    def _directive(self, meta: dict, out: Rendered) -> str:
        name, label, target = meta["name"], meta["label"], meta["target"]
        if name == "subscribe":
            return self.subscribe(label) + "\n"
        if name == "screenshot":
            what = html.escape(label or "a screenshot")
            return (f'<figure class="figure figure-missing" role="note"><div class="missing-shot">'
                    f'<strong>Screenshot to add</strong><span>{what}</span></div></figure>\n')
        if name == "card":
            card = self.card(label)
            if card is None:
                out.problems.append(f"a card points to {label}, which doesn't exist")
                return ""
            return card + "\n"
        href = self._target(target, out)
        text = html.escape(label or ("Download" if name == "download" else "Read more"))
        if name == "download":
            return (f'<p class="cta"><a class="btn btn-ghost" href="{html.escape(href)}" download>'
                    f'<svg class="icon" aria-hidden="true"><use href="#i-download"/></svg> {text}</a></p>\n')
        external = ' rel="noopener"' if href.startswith(("http://", "https://")) else ""
        return f'<p class="cta"><a class="btn btn-primary" href="{html.escape(href)}"{external}>{text}</a></p>\n'

    def _target(self, target: str, out: Rendered) -> str:
        if REF_RE.match(target):
            resolved = self.resolve(target)
            if resolved is None:
                out.problems.append(f"a button points to {target}, which doesn't exist")
                return "#"
            return resolved
        if target.startswith(SAFE_SCHEMES) or (target and not re.match(r"^[a-z]+:", target)):
            return target
        out.problems.append(f"a button has an address that isn't a web address or a reference: {target or '(empty)'}")
        return "#"


def render(body: str, **kwargs) -> Rendered:
    return Renderer(**kwargs).render(body)
