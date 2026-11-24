"""An email-ready copy of an issue: one self-contained HTML file with inline styles and absolute links, for pasting
into a newsletter tool (Buttondown, Kit, Beehiiv, Mailchimp...). Email programs ignore stylesheets and scripts,
so the web page's boxes are redrawn with inline styles, and Copy buttons are left out."""

from __future__ import annotations

import html
import re

# The colours of mojeedabisiga.com/learn: cream paper, red, charcoal. Email programs rarely load web fonts, so the
# tall uppercase headings fall back to Impact or Arial Black, and the italic accents to Georgia.
INK, INK2, MUTED, LINE, ACCENT, SOFT, CODE = "#2f2d2d", "#4f4c49", "#6b6762", "#e6e1d1", "#d02e2e", "#fbe8dc", "#2f2d2d"
PAPER, BEIGE, CREAM = "#fffae7", "#f6eed2", "#fffae7"
DISPLAY = "Anton, Impact, 'Arial Narrow Bold', 'Arial Black', sans-serif"
SERIF = "'Playfair Display', Georgia, 'Times New Roman', serif"
SANS = "'Open Sans', -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
MONO = "SFMono-Regular, Menlo, Consolas, monospace"
TONES = {"tip": (SOFT, "#b82828"), "note": (BEIGE, INK2), "warning": ("#fadfd8", "#b82828"), "key": ("#fbead9", "#9c4c17")}

STYLES = {
    "p": f"margin:0 0 16px;font:17px/1.65 {SANS};color:{INK}",
    "h2": f"margin:34px 0 12px;font:800 28px/1.05 {DISPLAY};letter-spacing:.01em;text-transform:uppercase;color:{INK}",
    "h3": f"margin:26px 0 10px;font:800 21px/1.1 {DISPLAY};letter-spacing:.01em;text-transform:uppercase;color:{INK}",
    "ul": f"margin:0 0 16px;padding-left:24px;font:17px/1.6 {SANS};color:{INK}",
    "ol": f"margin:0 0 16px;padding-left:24px;font:17px/1.6 {SANS};color:{INK}",
    "li": "margin:6px 0",
    "a": f"color:{ACCENT};text-decoration:underline",
    "blockquote": f"margin:20px 0;padding:4px 0 4px 16px;border-left:4px solid {ACCENT};color:{INK};font:italic 20px/1.45 {SERIF}",
    "table": f"width:100%;border-collapse:collapse;margin:20px 0;font:15px/1.5 {SANS}",
    "th": f"text-align:left;padding:8px 10px;border-bottom:1px solid {LINE};background:{BEIGE}",
    "td": f"text-align:left;padding:8px 10px;border-bottom:1px solid {LINE};vertical-align:top",
    "code": f"font:15px/1.5 {MONO};background:{BEIGE};padding:1px 5px;border-radius:4px",
}


def _style_tags(body: str) -> str:
    for tag, style in STYLES.items():
        body = re.sub(rf"<{tag}(\s[^>]*)?>", lambda m: f'<{tag}{_strip_class(m.group(1) or "")} style="{style}">', body)
    return body


def _strip_class(attrs: str) -> str:
    return re.sub(r'\s(class|id|loading|data-[a-z-]+)="[^"]*"', "", attrs)


def _code_box(label: str, code: str) -> str:
    return (f'<div style="margin:20px 0;border-radius:10px;overflow:hidden;background:{CODE}">'
            f'<div style="padding:8px 14px;background:{ACCENT};font:700 12px/1.2 {SANS};letter-spacing:.14em;text-transform:uppercase;color:{CREAM}">{label}</div>'
            f'<pre style="margin:0;padding:14px 16px;white-space:pre-wrap;font:15px/1.6 {MONO};color:{CREAM}">{code}</pre></div>')


def _prompts(body: str) -> str:
    body = re.sub(r'<div class="prompt"><div class="prompt-head"><span>(.*?)</span>.*?</div><pre><code>(.*?)</code></pre></div>',
                  lambda m: _code_box(m.group(1), m.group(2)), body, flags=re.S)
    return re.sub(r'<div class="codeblock">.*?<pre><code[^>]*>(.*?)</code></pre></div>',
                  lambda m: _code_box("Code", m.group(1)), body, flags=re.S)


def _callouts(body: str) -> str:
    def box(match):
        tone, inner = match.group(1), match.group(2)
        bg, fg = TONES.get(tone, TONES["note"])
        inner = inner.replace('<p class="callout-title">', f'<p style="margin:0 0 6px;font:700 12px/1.2 {SANS};letter-spacing:.12em;text-transform:uppercase;color:{fg}">', 1)
        return f'<div style="margin:20px 0;padding:14px 16px;border-radius:10px;border-left:4px solid {fg};background:{bg}">{inner}</div>'
    return re.sub(r'<aside class="callout callout-([a-z]+)">(.*?)</aside>', box, body, flags=re.S)


def _buttons(body: str) -> str:
    def button(match):
        href, text = match.group(1), re.sub(r"<[^>]+>", "", match.group(2)).strip()
        return (f'<p style="margin:24px 0;text-align:center"><a href="{href}" style="display:inline-block;padding:13px 26px;border-radius:999px;'
                f'background:{ACCENT};color:{CREAM};font:800 17px/1.2 {DISPLAY};letter-spacing:.06em;text-transform:uppercase;text-decoration:none">{text}</a></p>')
    return re.sub(r'<p class="cta"><a class="btn[^"]*" href="([^"]*)"[^>]*>(.*?)</a></p>', button, body, flags=re.S)


def _cards(body: str) -> str:
    def card(match):
        href, inner = match.group(1), match.group(2)
        title = re.search(r'<span class="ec-title">(.*?)</span>', inner, re.S)
        sub = re.search(r'<span class="ec-sub">(.*?)</span>', inner, re.S)
        return (f'<p style="margin:20px 0;padding:14px 16px;border:1px solid {LINE};border-radius:10px">'
                f'<a href="{href}" style="color:{INK};text-decoration:none;font:800 19px/1.15 {DISPLAY};letter-spacing:.01em;text-transform:uppercase">{title.group(1) if title else "Read more"}</a>'
                + (f'<br><span style="font:15px/1.5 {SANS};color:{MUTED}">{sub.group(1)}</span>' if sub else "") + "</p>")
    return re.sub(r'<a class="embed-card" href="([^"]*)">(.*?)</a>(?=\s*(?:<|$))', card, body, flags=re.S)


def _figures(body: str) -> str:
    body = re.sub(r'<figure class="figure figure-missing".*?</figure>', "", body, flags=re.S)
    body = re.sub(r'<figure class="figure"><img src="([^"]*)" alt="([^"]*)"[^>]*>(?:<figcaption>(.*?)</figcaption>)?</figure>',
                  lambda m: (f'<p style="margin:22px 0;text-align:center"><img src="{m.group(1)}" alt="{m.group(2)}" width="560" '
                             f'style="max-width:100%;height:auto;border-radius:8px;border:1px solid {LINE}">'
                             + (f'<br><span style="font:14px/1.4 {SANS};color:{MUTED}">{m.group(3)}</span>' if m.group(3) else "") + "</p>"),
                  body, flags=re.S)
    return body


def email_html(entry, site, body_html: str | None = None) -> str:
    """`body_html` is the issue rendered without subscribe boxes (readers of the email already subscribed)."""
    base = site.base_url
    body = entry.html if body_html is None else body_html
    body = re.sub(r'<button[^>]*>.*?</button>', "", body, flags=re.S)
    body = re.sub(r'<input class="task-list-item-checkbox"[^>]*checked[^>]*>', "☑", body)
    body = re.sub(r'<input class="task-list-item-checkbox"[^>]*>', "☐", body)
    body = _prompts(body)
    body = _callouts(body)
    body = _buttons(body)
    body = _cards(body)
    body = _figures(body)
    body = re.sub(r"<hr\s*/?>", f'<hr style="border:0;border-top:1px solid {LINE};margin:32px 0">', body)
    body = _style_tags(body)
    body = re.sub(r'(href|src)="(/[^"]*)"', lambda m: f'{m.group(1)}="{base}{m.group(2)}"', body)
    title, subtitle = html.escape(entry.title), html.escape(entry.subtitle)
    return f"""<!doctype html>
<html lang="{site['language']}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title></head>
<body style="margin:0;padding:0;background:{PAPER}">
<!-- Email-ready copy of {html.escape(entry.abs_url)}. Paste it into your newsletter tool; it adds the unsubscribe link. -->
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" style="background:{PAPER}"><tr><td align="center" style="padding:24px 12px">
<table role="presentation" width="600" cellpadding="0" cellspacing="0" style="max-width:600px;width:100%;background:#ffffff;border:1px solid {LINE};border-top:5px solid {ACCENT};border-radius:14px">
<tr><td style="padding:28px 32px 8px">
<p style="margin:0 0 18px;font:700 13px/1.2 {SANS};letter-spacing:.2em;text-transform:uppercase;color:{ACCENT}">{html.escape(site['name'])}</p>
<h1 style="margin:0 0 10px;font:800 36px/1.02 {DISPLAY};letter-spacing:.01em;text-transform:uppercase;color:{INK}">{title}</h1>
{f'<p style="margin:0 0 14px;font:italic 20px/1.4 {SERIF};color:{ACCENT}">{subtitle}</p>' if subtitle else ''}
<p style="margin:0 0 20px;font:14px/1.4 {SANS};color:{MUTED}">{html.escape(site['author']['name'])} · {entry.date_label} · <a href="{entry.abs_url}" style="color:{MUTED}">Read it on the web</a></p>
</td></tr>
<tr><td style="padding:0 32px 24px">{body}</td></tr>
<tr><td style="padding:20px 32px 28px;border-top:1px solid {LINE};font:14px/1.5 {SANS};color:{MUTED}">
You're getting this because you subscribed to {html.escape(site['name'])}. <a href="{site.absolute('')}" style="color:{MUTED}">{html.escape(site.host + site.base_path)}</a>
</td></tr></table></td></tr></table></body></html>
"""
