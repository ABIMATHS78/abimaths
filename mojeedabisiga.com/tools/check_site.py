#!/usr/bin/env python3
"""Checks the mojeedabisiga.com website before it is published.

What it checks, on every HTML page in the site folder:
  - each page's structured data (JSON-LD) is valid JSON
  - the Person entity in the structured data is identical on every page that has one,
    so Google sees one consistent set of facts
  - every internal link, image, stylesheet, script and icon points to a file that exists,
    and every #anchor it points to exists on the target page
  - every page has a <title>; every indexable page also has a meta description and a canonical URL

Uses only the Python standard library.

Usage: python3 check_site.py <site-folder>
Exits with status 1 and lists the problems if anything is wrong.
"""
import json
import os
import re
import sys
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit

EXTERNAL = re.compile(r"^(?:[a-z][a-z0-9+.-]*:|//)", re.IGNORECASE)


class Page(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.refs = []
        self.ids = set()
        self.jsonld = []
        self.title = ""
        self.meta = {}
        self.canonical = None
        self._in_jsonld = False
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if a.get("id"):
            self.ids.add(a["id"])
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])
        if a.get("srcset"):
            self.refs += [part.split()[0] for part in a["srcset"].split(",") if part.strip()]
        if tag == "script" and a.get("type") == "application/ld+json":
            self._in_jsonld = True
            self.jsonld.append("")
        if tag == "title":
            self._in_title = True
        if tag == "meta" and a.get("name"):
            self.meta[a["name"]] = a.get("content", "")
        if tag == "link" and a.get("rel") == "canonical":
            self.canonical = a.get("href")

    def handle_endtag(self, tag):
        if tag == "script":
            self._in_jsonld = False
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_jsonld:
            self.jsonld[-1] += data
        if self._in_title:
            self.title += data


def resolve(site, page_rel, path):
    """Map a link path to a file in the site folder (a folder link means its index.html)."""
    if path.startswith("/"):
        target = os.path.normpath(os.path.join(site, path.lstrip("/")))
    else:
        target = os.path.normpath(os.path.join(site, os.path.dirname(page_rel), path))
    if path.endswith("/") or os.path.isdir(target):
        target = os.path.join(target, "index.html")
    return target


def main(site):
    site = os.path.abspath(site)
    problems = []
    pages = {}
    for folder, _, files in os.walk(site):
        for name in files:
            if name.endswith(".html"):
                rel = os.path.relpath(os.path.join(folder, name), site)
                parser = Page()
                with open(os.path.join(site, rel), encoding="utf-8") as fh:
                    parser.feed(fh.read())
                pages[rel] = parser

    persons = {}
    for rel, page in sorted(pages.items()):
        if not page.title.strip():
            problems.append(f"{rel}: missing <title>")
        indexable = "noindex" not in page.meta.get("robots", "")
        if indexable and not page.meta.get("description"):
            problems.append(f"{rel}: missing meta description")
        if indexable and not page.canonical:
            problems.append(f"{rel}: missing canonical URL")
        for block in page.jsonld:
            try:
                data = json.loads(block)
            except json.JSONDecodeError as err:
                problems.append(f"{rel}: structured data is not valid JSON ({err})")
                continue
            nodes = data.get("@graph", [data]) if isinstance(data, dict) else data
            for node in nodes:
                if isinstance(node, dict) and node.get("@type") == "Person":
                    persons[rel] = json.dumps(node, sort_keys=True)

        for ref in page.refs:
            if EXTERNAL.match(ref):
                continue
            parts = urlsplit(ref)
            path, fragment = unquote(parts.path), parts.fragment
            if not path:
                if fragment and fragment not in page.ids:
                    problems.append(f"{rel}: anchor #{fragment} not found on this page")
                continue
            target = resolve(site, rel, path)
            if not os.path.isfile(target):
                problems.append(f"{rel}: '{ref}' points to a missing file")
                continue
            target_rel = os.path.relpath(target, site)
            if fragment and target_rel in pages and fragment not in pages[target_rel].ids:
                problems.append(f"{rel}: anchor #{fragment} not found on {target_rel}")

    if len(set(persons.values())) > 1:
        problems.append("the Person structured data differs between pages: " + ", ".join(sorted(persons)))

    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: {len(pages)} pages checked; structured data valid; Person entity identical on {len(persons)} pages; all internal links resolve.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python3 check_site.py <site-folder>")
    sys.exit(main(sys.argv[1]))
