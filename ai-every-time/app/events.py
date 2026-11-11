"""The Events pages: /events, and each event's own page. They're AI Every Time's copy of DataGlobal Hub's Events
pages, which present the conference AI Every Time puts on. The app shows the copy in a frame (static/events/); the
same words, without their styling, are in app/event_data/ for search engines (app/seo.py)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).resolve().parent / "event_data"

GDAI = {
    "slug": "gdai-2026",
    "name": "Global Data & AI Virtual Tech Conference 2026",
    "summary": "The award-winning AI-native virtual conference returns, November 20-22, 2026: 140+ speakers, 1,000+ "
               "sessions and attendees from 60+ countries. Presented by AI Every Time, hosted by DataGlobal Hub.",
    "dates": "November 20-22, 2026",
    "starts": "2026-11-20T09:00:00-05:00",
    "ends": "2026-11-22T18:00:00-05:00",
    "image": "/static/events/img/GDAI_Event_Cover_november.webp",
    "image_size": (1350, 646),
    "tickets": "https://www.eventeny.com/events/globaldataaivirtualtechconference2026-26260/",
    "price_from": "10.99",
    "price_to": "499",
}


@lru_cache(maxsize=None)
def words(name: str) -> str:
    """A page's words as plain HTML: "index" for the list, or an event's slug."""
    return (DATA / f"{name}.html").read_text(encoding="utf-8")
