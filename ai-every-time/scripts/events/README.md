# The Events pages

`/events` and `/events/gdai-2026` are AI Every Time's copy of DataGlobal Hub's Events pages
(`news.dataglobalhub.org/events`), which present the conference AI Every Time puts on.

To bring the copy up to date when the pages there change:

1. `python capture.py` opens both pages in a browser (Playwright) and saves what's on them to `../dl/capture.json`:
   the page as it loads, each day of the agenda, every speaker and every answer. Save the site's stylesheet beside it
   too, and a `fonts.css` with the fonts' `@font-face` rules.
2. Download any new pictures and files the pages name into `static/events/img/` and `static/events/files/`.
3. `python build.py <folder with capture.json> static/events app/event_data` writes the two pages the app shows in a
   frame, their stylesheet, and the pages' words for search engines.

`static/events/events.js` is written by hand: the countdown, the agenda's days, the rest of the speakers, the answers,
and links that move the page the frame is on.
