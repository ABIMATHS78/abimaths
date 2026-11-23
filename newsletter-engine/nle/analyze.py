"""Measures the structure of example issues (Substack HTML): blocks, sections, paragraph lengths, prompts,
screenshots, buttons. Only numbers come out; the text itself stays in the local corpus for the copy check."""

from __future__ import annotations

import datetime as dt
import html as htmllib
import re
import statistics as st
from html.parser import HTMLParser

VOID = {"img", "br", "hr", "input", "source", "meta", "link", "wbr", "col", "area", "embed", "param", "track"}
TRACK = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "ul", "ol", "blockquote", "pre", "figure", "table", "iframe"}
SPECIAL = ("captioned-image-container", "button-wrapper", "subscription-widget-wrap", "subscription-widget",
           "embedded-post-wrap", "captioned-button-wrap", "digest-post-embed", "youtube-wrap", "tweet", "pullquote",
           "image-gallery-embed", "file-embed-wrapper", "poll-embed")
KIND = {"captioned-image-container": "image", "image-gallery-embed": "image", "button-wrapper": "button",
        "captioned-button-wrap": "button", "subscription-widget-wrap": "subscribe", "subscription-widget": "subscribe",
        "embedded-post-wrap": "card", "digest-post-embed": "card", "pre": "prompt", "blockquote": "quote",
        "ul": "list", "ol": "list", "hr": "divider", "p": "paragraph", "h2": "heading", "h3": "subheading",
        "h4": "subheading", "h1": "heading", "table": "table", "figure": "image"}


class _Blocks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.blocks: list[dict] = []
        self.stack: list[str] = []
        self.cur: dict | None = None

    def handle_starttag(self, tag, attrs):
        cls = dict(attrs).get("class") or ""
        if self.cur is None:
            if tag == "hr":
                self.blocks.append({"tag": "hr", "text": ""})
                return
            special = next((k for k in SPECIAL if k in cls), None)
            if tag in TRACK or (tag == "div" and special):
                self.cur = {"tag": special or tag, "text": "", "ordered": tag == "ol"}
                if tag not in VOID:
                    self.stack = [tag]
            return
        if tag not in VOID:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if self.cur is None or tag in VOID:
            return
        while self.stack:
            if self.stack.pop() == tag:
                break
        if not self.stack:
            self.blocks.append(self.cur)
            self.cur = None

    def handle_data(self, data):
        if self.cur is not None:
            self.cur["text"] += data


def blocks(body_html: str) -> list[dict]:
    parser = _Blocks()
    parser.feed(body_html or "")
    for block in parser.blocks:
        block["kind"] = KIND.get(block["tag"], "other")
        block["words"] = len(block["text"].split())
    return parser.blocks


def plain_text(body_html: str) -> str:
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", body_html or "", flags=re.S)
    text = re.sub(r"</(p|h[1-6]|li|blockquote|pre|figcaption)>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"[ \t]+", " ", htmllib.unescape(text)).strip()


def analyze_issue(post: dict) -> dict:
    """Numbers for one issue from the Substack API (title, subtitle, body_html, audience, wordcount)."""
    bl = blocks(post.get("body_html") or "")
    kinds = [b["kind"] for b in bl]
    paragraphs = [b["words"] for b in bl if b["kind"] == "paragraph" and b["words"]]
    first_divider = kinds.index("divider") if "divider" in kinds else len(kinds)
    opening = kinds[:first_divider]
    title, subtitle = post.get("title", ""), post.get("subtitle") or ""
    return {
        "complete": post.get("audience") == "everyone",
        "words": post.get("wordcount") or len(plain_text(post.get("body_html")).split()),
        "title_words": len(title.split()), "title_period": title.rstrip().endswith("."),
        "subtitle_words": len(subtitle.split()), "subtitle_colon": subtitle.rstrip().endswith(":"),
        "paragraphs": len(paragraphs),
        "paragraph_words": st.median(paragraphs) if paragraphs else 0,
        "short_paragraph_share": sum(1 for n in paragraphs if n <= 12) / len(paragraphs) if paragraphs else 0,
        "sections": kinds.count("divider") + 1,
        "headings": kinds.count("heading"), "subheadings": kinds.count("subheading"),
        "lists": kinds.count("list"), "numbered_lists": sum(1 for b in bl if b["kind"] == "list" and b.get("ordered")),
        "prompts": kinds.count("prompt"), "quotes": kinds.count("quote"), "images": kinds.count("image"),
        "buttons": kinds.count("button"), "cards": kinds.count("card"), "tables": kinds.count("table"),
        "opening_paragraphs": opening.count("paragraph"), "opening_has_list": "list" in opening,
        "opening_has_button": "button" in opening,
    }


def _dist(values) -> dict:
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"median": 0, "p10": 0, "p90": 0}
    return {"median": round(st.median(values), 2), "p10": values[int(len(values) * .1)],
            "p90": values[min(len(values) - 1, int(len(values) * .9))]}


def aggregate(analyses: list[dict]) -> dict:
    full = [a for a in analyses if a["complete"]]
    share = lambda key, rows: round(sum(1 for a in rows if a[key]) / len(rows), 2) if rows else 0  # noqa: E731
    out = {"issues": len(analyses), "complete_issues": len(full),
           "title_words": _dist(a["title_words"] for a in analyses), "title_period_share": share("title_period", analyses),
           "subtitle_words": _dist(a["subtitle_words"] for a in analyses), "subtitle_colon_share": share("subtitle_colon", analyses)}
    for key in ("words", "paragraphs", "paragraph_words", "short_paragraph_share", "sections", "headings", "subheadings",
                "lists", "numbered_lists", "prompts", "quotes", "images", "buttons", "opening_paragraphs"):
        out[key] = _dist(a[key] for a in full)
    out["opening_list_share"] = share("opening_has_list", full)
    out["opening_button_share"] = share("opening_has_button", full)
    return out


def archive_stats(archive: list[dict]) -> dict:
    if not archive:
        return {}
    dates = sorted(dt.date.fromisoformat(p["post_date"][:10]) for p in archive if p.get("post_date"))
    gaps = [(b - a).days for a, b in zip(dates, dates[1:])]
    titles = [p.get("title", "") for p in archive]
    subtitles = [p.get("subtitle") or "" for p in archive]
    return {
        "issues": len(archive),
        "from": dates[0].isoformat() if dates else "", "to": dates[-1].isoformat() if dates else "",
        "issues_per_week": round(7 / st.mean(gaps), 1) if gaps and st.mean(gaps) else 0,
        "free_share": round(sum(1 for p in archive if p.get("audience") == "everyone") / len(archive), 2),
        "title_words": _dist(len(t.split()) for t in titles),
        "title_period_share": round(sum(t.rstrip().endswith(".") for t in titles) / len(titles), 2),
        "title_number_share": round(sum(bool(re.search(r"\d", t)) for t in titles) / len(titles), 2),
        "subtitle_words": _dist(len(s.split()) for s in subtitles),
        "subtitle_colon_share": round(sum(s.rstrip().endswith(":") for s in subtitles) / len(subtitles), 2),
        "with_cover_share": round(sum(bool(p.get("cover_image")) for p in archive) / len(archive), 2),
    }
