"""Builds AI Every Time's copy of DataGlobal Hub's Events pages from what capture.py saved: static/events/index.html
and gdai-2026.html (shown in a frame by the app), their stylesheet, and plain HTML of each for search engines."""
import json, re, sys
from pathlib import Path
from bs4 import BeautifulSoup

DL = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SEO = Path(sys.argv[3])
SRC = "https://news.dataglobalhub.org"
# When the conference runs each day, in its own time zone: the countdown counts down to its opening, and the events
# list says so beside its dates.
OPENS = "2026-11-20T09:00:00-06:00"  # 9:00 AM CST (UTC-6) on Day 1
HOURS = "Day 1–3 · 9:00 AM – 6:00 PM CST"
CLOCK = ('<svg class="lucide lucide-clock h-3.5 w-3.5" fill="none" height="24" stroke="currentColor" '
         'stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewbox="0 0 24 24" width="24" '
         'xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>')
cap = json.loads((DL / "capture.json").read_text())


def fix(soup):
    """Addresses: pictures and files to the copies here, the two pages to this site's, the rest to where they were."""
    for el in soup.find_all(True):
        for attr in ("src", "href"):
            v = el.get(attr)
            if not v:
                continue
            m = re.fullmatch(r"/events/([^/?#]+\.(webp|png|jpg|pdf|xlsx))", v)
            if m:
                el[attr] = ("files/" if m[2] in ("pdf", "xlsx") else "img/") + m[1]
            elif v in ("/events", "/events/gdai-2026"):
                el["target"] = "_top"
            elif v.startswith("/"):
                el[attr] = SRC + v
                el["target"] = "_blank"
                el["rel"] = "noopener noreferrer"
        if el.name == "img":
            # The site shows a picture once it has loaded; here it's simply shown.
            el["class"] = [c for c in el.get("class", []) if c != "invisible"]
            el["data-status"] = "loaded"
    return soup


def button(soup, text):
    return next(b for b in soup.find_all("button") if text in b.get_text())


def gdai():
    soup = BeautifulSoup(cap["main_collapsed"], "lxml").find("main")
    for a in soup.find_all("aside", recursive=False):  # the support box: the app shows its own, fixed to the window
        a.decompose()
    for d in soup.find_all("div", recursive=False):
        if d.get("class") == ["h-20", "lg:hidden"]:
            d.decompose()
    # Countdown: the four numbers.
    days = next(d for d in soup.find_all("div") if d.get_text(strip=True) == "Days").parent.parent
    days["data-countdown"] = OPENS
    for cell, unit in zip(days.find_all("div", recursive=False), ("d", "h", "m", "s")):
        cell.find("div")["data-unit"] = unit
    # Speakers: all of them, those after the first twelve hidden until "View More".
    old = button(soup, "View More Speaking Sessions").find_parent("section")
    full = BeautifulSoup(cap["speakers_expanded"], "lxml").find("section")
    before = len(old.select("div.grid")[0].find_all(recursive=False))
    grid = full.select("div.grid")[0]
    cards = grid.find_all(recursive=False)
    for card in cards[before:]:
        card["data-more"] = ""
        card["hidden"] = ""
    old.select("div.grid")[0].replace_with(grid)
    button(old, "View More Speaking Sessions")["data-show-more"] = ""
    # Agenda: the three days, one shown at a time.
    tabs = button(soup, "Day 1 ·").parent
    for i, b in enumerate(tabs.find_all("button")):
        b["data-day"] = str(i)
    ol = tabs.find_parent("section").find("ol")
    panels = []
    for i, d in enumerate(cap["days"]):
        panel = BeautifulSoup(d["ol"], "lxml").find("ol")
        panel["data-day-panel"] = str(i)
        if i:
            panel["hidden"] = ""
        panels.append(panel)
    ol.replace_with(*panels)
    # Questions: each answer under its question, closed to begin with.
    box = button(soup, "When is the conference?").parent.parent
    opened = BeautifulSoup(cap["faq_open"], "lxml").body.find()
    for item in opened.find_all("div", recursive=False):
        b, p = item.find("button"), item.find("p")
        b["aria-expanded"] = "false"
        b["data-faq"] = ""
        svg = b.find("svg")
        svg["class"] = [c for c in svg.get("class", []) if c != "rotate-180"]
        p["hidden"] = ""
    box.replace_with(opened)
    return fix(soup), len(cards), before


def listing():
    main = BeautifulSoup(cap["list_main"], "lxml").find("main")
    # The conference's hours, in the same style as its dates and right after them.
    dates = next(li for li in main.find_all("li") if "November 20-22" in li.get_text())
    hours = BeautifulSoup(f'<li class="{" ".join(dates["class"])}" title="Each day, Friday 20 to Sunday 22 '
                          f'November 2026: 9:00 AM to 6:00 PM Central Standard Time (UTC-6)">{CLOCK}{HOURS}</li>',
                          "lxml").find("li")
    dates.insert_after(hours)
    return fix(main)


def page(title, main):
    return (f'<!doctype html>\n<html lang="en" data-brand="dataglobalhub">\n<head>\n<meta charset="utf-8">\n'
            f'<meta name="viewport" content="width=device-width, initial-scale=1">\n<meta name="robots" content="noindex">\n'
            f'<title>{title}</title>\n<link rel="stylesheet" href="events.css">\n</head>\n<body>\n{main}\n'
            f'<script src="events.js"></script>\n</body>\n</html>\n')


def plain(main):
    """The page's words and links without its styling, for search engines: every day of the agenda, every speaker and
    every answer, as they're all part of the page."""
    soup = BeautifulSoup(str(main), "lxml").find("main")
    for el in soup.find_all(["svg", "button", "script", "style"]):
        if el.name == "button":
            el.name = "p"
        else:
            el.decompose()
    for el in soup.find_all(True):
        keep = {k: v for k, v in el.attrs.items() if k in ("href", "src", "alt", "width", "height", "id")}
        if el.name == "img" and "src" in keep:
            keep["src"] = "/static/events/" + keep["src"]
            keep["loading"] = "lazy"
        if el.name == "a" and keep.get("href", "").startswith("files/"):
            keep["href"] = "/static/events/" + keep["href"]
        el.attrs = keep
    for el in soup.find_all(["div", "span"]):
        if not el.get_text(strip=True) and not el.find("img"):
            el.decompose()
    for el in soup.find_all("span"):
        el.unwrap()
    out = soup.decode_contents()
    return re.sub(r"\s+", " ", out).replace("> <", "><")


g, total, shown = gdai()
l = listing()
(OUT / "gdai-2026.html").write_text(page("Global Data & AI Virtual Tech Conference 2026", str(g)), encoding="utf-8")
(OUT / "index.html").write_text(page("Events", str(l)), encoding="utf-8")
SEO.mkdir(parents=True, exist_ok=True)
(SEO / "gdai-2026.html").write_text(plain(g), encoding="utf-8")
(SEO / "index.html").write_text(plain(l), encoding="utf-8")
css = (DL / "fonts.css").read_text() + "\n" + (DL / "index-BP1k6Wbv.css").read_text() + """
/* Additions for the copy on AI Every Time: the page sits in a frame as tall as itself. */
html, body { height: auto; min-height: 0; overflow: hidden; }
[hidden] { display: none !important; }
"""
(OUT / "events.css").write_text(css, encoding="utf-8")
print("speakers", total, "shown first", shown, "| sizes", {f.name: f.stat().st_size for f in list(OUT.glob("*.html")) + list(OUT.glob("*.css")) + list(SEO.glob("*.html"))})
