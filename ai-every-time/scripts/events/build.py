"""Builds AI Every Time's copy of DataGlobal Hub's Events pages from what capture.py saved: static/events/index.html
and gdai-2026.html (shown in a frame by the app), their stylesheet, and plain HTML of each for search engines."""
import copy, json, re, sys
from pathlib import Path
from bs4 import BeautifulSoup

DL = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SEO = Path(sys.argv[3])
SRC = "https://news.dataglobalhub.org"
# When the conference runs each day, in its own time zone: the countdown counts down to its opening, and the events
# list says so beside its dates.
OPENS = "2026-11-20T09:00:00-06:00"  # 9:00 AM CST (UTC-6) on Day 1
HOURS = "Day 1–3 · 9:00 AM – 6:00 PM CST"
# Changes AI Every Time has made since copying the pages: what it says on DataGlobal Hub, and what it says here.
CORRECTIONS = {
    # Dr Gaurav Kumar Gupta's session, as the agenda (PDF and Excel) has it.
    "From Bots to Control Rooms: Operationalizing Enterprise AI with Governance and Visibility":
        "Designing AI-Native Enterprises: The Next Evolution of Enterprise Architecture",
}
# Suprajyotsna Dasari gives Day 1's Opening Keynote Address, and her session moves from Day 3 at 2:45 PM to Day 1 at
# 9:45 AM, as the agenda (PDF and Excel) has it.
KEYNOTE = "Suprajyotsna Dasari"
MOVED = ("Future of Work & AI in Society", "Automate Your Tasks by Building Agentic AI", "Suprajyotsna Dasari")
# Sessions withdrawn on 20 November (the agenda PDF and Excel no longer have them), and the
# speaker cards of their speakers, who have no other session.
WITHDRAWN = {
    "When AI Enters Matters: A Behavioral Case for Human-First Reasoning",
    "From Assistant to Operator: Security and Governance Risks in Multi-Agent AI Workflows",
    "From Prompts to Processes: Building Autonomous AI Workflows",
    "Data Security in the Age of Autonomous AI: Strategies for Cybersecurity",
    "Synthetic Lives for Smarter Agents: Generating Personal Context Data for AI Benchmark Design",
    "Designing AI Agents Users Will Trust: Evals, Guardrails, and Feedback Loops",
    "Multi-Agent Preventive Maintenance Diagnostics to Enhance Labor Productivity",
    "The Rise of Vibe-Coding: Empowering the Next Wave of Makers",
    "Build Your First Agent in Copilot Studio",
    "S.T.E.P. Up Your Leadership Strategy with Data-Driven Insights",
}
WITHDRAWN_SPEAKERS = {
    "Devika Toprani", "Cansu Berkem", "Khadija Assbai", "Maitrik Patel", "Utkarsh Tripathi", "Dr. Joe Perez", "Joe Perez",
    "Omar Soussa", "Mihaela-Roxana Ghidersa", "Elayne Jones", "Gbenga Akingbulere",
}
# Sessions moved since copying the agenda (22 November 2026, the owner): from (day panel, slot) to
# (day panel, slot), by title. Already moved, left as is.
MOVES = [
    (1, "2:45 PM - 3:15 PM", 2, "2:45 PM - 3:15 PM", "AI Won't Replace Creators, It Will Replace Creative Limitations"),
]
# A session added on 20 November (the agenda PDF and Excel have it): Day 2 (panel 1), its time,
# track, title and speaker, and the speaker's card (initials, no photo).
ADDED = (1, "12:45 PM - 1:15 PM", "AI & Data Innovation",
         "Diagnosis and Detection of Chromosomal Disorders: AI vs Traditional Methods", "Karinmoluwa Akinrelere")
ADDED_HEADLINE = "Speaker, AI & Data Innovation"
# The host speaker (20 November 2026): every DataGlobal Hub session (panel, start, title) names it, and
# it has a speaker card first in the grid.
HOST = "DataGlobal Hub (By Mojeed Abisiga)"
HOST_HEADLINE = "CEO & Co-founder, DataGlobal Hub · CEO & Founder, AI Every Time"
HOST_INITIALS = "MA"
HOST_SESSIONS = [
    (0, "9:00 AM", "Welcome & Opening Announcements"),
    (0, "11:45 AM", "Virtual Networking / Coffee Break"),
    (0, "2:15 PM", "Fireside Chat with Industry Leaders"),
    (0, "4:45 PM", "Virtual Networking / Coffee Break"),
    (0, "5:15 PM", "Closing Keynote"),
    (0, "5:45 PM", "Closing Remarks - End of Day 1"),
    (1, "9:00 AM", "Welcome & Announcements"),
    (1, "9:15 AM", "Opening Keynote Address"),
    (1, "11:45 AM", "Virtual Networking / Coffee Break"),
    (1, "2:15 PM", "Fireside Chat with Industry Leaders"),
    (1, "4:45 PM", "Virtual Networking / Coffee Break"),
    (1, "5:15 PM", "Closing Keynote"),
    (1, "5:45 PM", "Closing Remarks - End of Day 2"),
    (2, "9:00 AM", "Welcome & Announcements"),
    (2, "9:15 AM", "Opening Keynote Address"),
    (2, "11:45 AM", "Virtual Networking / Coffee Break"),
    (2, "2:15 PM", "Fireside Chat with Industry Leaders"),
    (2, "4:15 PM", "Designing AI-Native Enterprises: The Next Evolution"),
    (2, "4:15 PM", "The Human Side of AI: Leading with Curiosity and Credibility"),
    (2, "4:45 PM", "Virtual Networking / Coffee Break"),
    (2, "5:15 PM", "Closing Keynote"),
    (2, "5:45 PM", "Closing Remarks - End of Conference"),
]
# Speakers' titles as they asked for them (20–21 November), on their cards.
SPEAKER_TITLES = {
    "Swapna Chimanchodkar": "Senior Leader in Process & Digital Transformation, IEEE Senior Member",
}
CLOCK = ('<svg class="lucide lucide-clock h-3.5 w-3.5" fill="none" height="24" stroke="currentColor" '
         'stroke-linecap="round" stroke-linejoin="round" stroke-width="2" viewbox="0 0 24 24" width="24" '
         'xmlns="http://www.w3.org/2000/svg"><circle cx="12" cy="12" r="10"></circle><path d="M12 6v6l4 2"></path></svg>')
cap = json.loads((DL / "capture.json").read_text())


def correct(text: str) -> str:
    for old, new in CORRECTIONS.items():
        assert old in text or new in text, f"not on the page any more: {old}"
        text = text.replace(old, new)
    return text


def slot(panel, time):
    return next(li for li in panel.find_all("li", recursive=False) if li.find("p").get_text(strip=True) == time)


def reschedule(panels):
    """The agenda's changes since copying it (KEYNOTE, MOVED), on the three days' panels; already there, left as is."""
    keynote = slot(panels[0], "9:15 AM - 9:45 AM").find("div")
    title = keynote.find("p")
    if KEYNOTE not in keynote.get_text():
        speaker = BeautifulSoup(f'<p class="mt-1 text-sm text-muted-foreground">{KEYNOTE}</p>', "lxml").find("p")
        title.insert_after(speaker)
    cards = lambda li: li.find("div").find_all("div", recursive=False)
    texts = lambda card: tuple(p.get_text(strip=True) for p in card.find_all("p"))
    for card in cards(slot(panels[2], "2:45 PM - 3:15 PM")):
        if texts(card) == MOVED:
            card.decompose()
    day1 = cards(slot(panels[0], "9:45 AM - 10:15 AM"))
    if MOVED not in map(texts, day1):
        new = copy.copy(day1[-1])
        for p, text in zip(new.find_all("p"), MOVED):
            p.string = text
        day1[-1].insert_after(new)


def move_sessions(panels):
    """MOVES: the session's card goes from its old slot to the end of its new one (an emptied slot goes too)."""
    cards = lambda li: li.find("div").find_all("div", recursive=False)
    for day, time, to_day, to_time, title in MOVES:
        old = slot(panels[day], time)
        card = next((c for c in cards(old) if title in c.get_text()), None)
        new = cards(slot(panels[to_day], to_time))
        if card is None or any(title in c.get_text() for c in new):
            continue
        new[-1].insert_after(card.extract())
        if not cards(old):
            old.decompose()


def withdraw_sessions(panels):
    """Take the WITHDRAWN sessions off the schedule (a slot left empty goes too)."""
    for panel in panels:
        for li in panel.find_all("li", recursive=False):
            box = li.find("div")
            cards = box.find_all("div", recursive=False) if box else []
            for card in cards:
                if any(p.get_text(strip=True) in WITHDRAWN for p in card.find_all("p")):
                    card.decompose()
            if cards and not box.find_all("div", recursive=False):
                li.decompose()


def withdraw_speakers(grid, shown):
    """Remove WITHDRAWN_SPEAKERS' cards; the first `shown` cards stay in view, the rest behind View More."""
    for card in grid.find_all(recursive=False):
        h = card.find("h3")
        if h and h.get_text(strip=True) in WITHDRAWN_SPEAKERS:
            card.decompose()
    for i, card in enumerate(grid.find_all(recursive=False)):
        if i < shown:
            card.attrs.pop("data-more", None)
            card.attrs.pop("hidden", None)
        else:
            card["data-more"] = ""
            card["hidden"] = ""


def add_session(panels):
    """ADDED's session card, last in its slot; already there, left as is."""
    day, time, track, title, name = ADDED
    box = slot(panels[day], time).find("div")
    cards = box.find_all("div", recursive=False)
    if any(title in c.get_text() for c in cards):
        return
    new = copy.copy(cards[-1])
    for p, text in zip(new.find_all("p"), (track, title, name)):
        p.string = text
    cards[-1].insert_after(new)


def add_speaker(grid):
    """ADDED's speaker card, after the others, with her initials as the cards without a photo have; already there, left as is."""
    *_, title, name = ADDED
    cards = grid.find_all(recursive=False)
    if any(c.find("h3").get_text(strip=True) == name for c in cards):
        return
    new = copy.copy(next(c for c in cards if not c.find("img") and "Keynote" not in c.get_text()))
    new.find("span").string = "".join(w[0] for w in name.split()[:2]).upper()
    new.find("h3").string = name
    new.find("h3").find_next_sibling("p").string = ADDED_HEADLINE
    new.find("p", class_="kicker").find_next_sibling("p").string = title
    cards[-1].insert_after(new)


def host_sessions(panels):
    """HOST on each of HOST_SESSIONS: a speaker line under a plenary, or in place of a talk's "TBC"; already there, left as is."""
    for day, time, title in HOST_SESSIONS:
        li = next(li for li in panels[day].find_all("li", recursive=False) if li.find("p").get_text(strip=True).startswith(time + " -"))
        box = li.find("div")
        cards = box.find_all("div", recursive=False)
        if not cards:  # a plenary: title, then its track
            if HOST in box.get_text():
                continue
            line = BeautifulSoup(f'<p class="mt-1 text-sm text-muted-foreground">{HOST}</p>', "lxml").find("p")
            box.find("p").insert_after(line)
            continue
        card = next(c for c in cards if c.find_all("p")[1].get_text(strip=True) == title)
        who = card.find_all("p")[2]
        if who.get_text(strip=True) in ("TBC", ""):
            who.string = HOST


def host_card(grid):
    """HOST's speaker card, first in the grid, with initials; already there, left as is."""
    cards = grid.find_all(recursive=False)
    if any(c.find("h3").get_text(strip=True) == HOST for c in cards):
        return
    new = copy.copy(next(c for c in cards if not c.find("img") and "Keynote" not in c.get_text()))
    new.find("span").string = HOST_INITIALS
    new.find("h3").string = HOST
    new.find("h3").find_next_sibling("p").string = HOST_HEADLINE
    new.find("p", class_="kicker").string = "Host"
    new.find("p", class_="kicker").find_next_sibling("p").string = "Welcome and opening sessions, fireside chats, networking and closing remarks"
    cards[0].insert_before(new)


def speaker_titles(grid):
    """SPEAKER_TITLES on those speakers' cards (the line under the name)."""
    for card in grid.find_all(recursive=False):
        h = card.find("h3")
        if h and h.get_text(strip=True) in SPEAKER_TITLES:
            line = h.find_next_sibling("p")
            if line is not None:
                line.string = SPEAKER_TITLES[h.get_text(strip=True)]


def keynote_card(grid):
    """KEYNOTE's speaker card names the keynote above her session; already there, left as is."""
    card = next(c for c in grid.find_all(recursive=False) if c.find("h3").get_text(strip=True) == KEYNOTE)
    if "Opening Keynote Address" in card.get_text():
        return
    session = card.find("p", class_="kicker")
    session["class"] = session["class"] + ["mt-3"]
    for html in ('<p class="kicker">Keynote</p>',
                 '<p class="mt-1 text-sm font-medium leading-snug">Opening Keynote Address</p>'):
        session.insert_before(BeautifulSoup(html, "lxml").find("p"))


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
    keynote_card(grid)
    speaker_titles(grid)
    add_speaker(grid)
    host_card(grid)
    withdraw_speakers(grid, before)
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
    reschedule(panels)
    move_sessions(panels)
    withdraw_sessions(panels)
    add_session(panels)
    host_sessions(panels)
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
(OUT / "gdai-2026.html").write_text(page("Global Data & AI Virtual Tech Conference 2026", correct(str(g))), encoding="utf-8")
(OUT / "index.html").write_text(page("Events", str(l)), encoding="utf-8")
SEO.mkdir(parents=True, exist_ok=True)
(SEO / "gdai-2026.html").write_text(correct(plain(g)), encoding="utf-8")
(SEO / "index.html").write_text(plain(l), encoding="utf-8")
css = (DL / "fonts.css").read_text() + "\n" + (DL / "index-BP1k6Wbv.css").read_text() + """
/* Additions for the copy on AI Every Time: the page sits in a frame as tall as itself. */
html, body { height: auto; min-height: 0; overflow: hidden; }
[hidden] { display: none !important; }
"""
(OUT / "events.css").write_text(css, encoding="utf-8")
print("speakers", total, "shown first", shown, "| sizes", {f.name: f.stat().st_size for f in list(OUT.glob("*.html")) + list(OUT.glob("*.css")) + list(SEO.glob("*.html"))})
