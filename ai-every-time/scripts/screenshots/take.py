"""Screenshots of AI Every Time as a learner sees it, for social media: from the welcome page and signing up to
choosing, customising and learning with a companion. See README.md.

It starts the app itself, with a new database in OUT_DIR, the starter companions' real photos and the guides, and
goes through it in a desktop browser (1440 x 900, at twice the pixels) and on a phone (390 x 844, at three times)
as a new learner, Amara. With ANTHROPIC_API_KEY set, it also takes the screens whose words Claude writes: the
companion starting the chat and replying, a lesson, a quiz, the feed and progress. In demo mode (no key) those are
left out, unless --all asks for them to try the steps (their words are then demo text, not for posting), and the
demo banner is hidden, as it is for learners when a key is set.

Run from ai-every-time/:  python scripts/screenshots/take.py OUT_DIR [--only desktop|phone] [--all]
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sqlite3
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
APP = HERE.parent.parent  # ai-every-time/
ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
ap.add_argument("out", type=Path, help="folder for the screenshots (made if missing; its app.db is started afresh)")
ap.add_argument("--only", choices=["desktop", "phone"], help="just one of the two")
ap.add_argument("--all", action="store_true", help="in demo mode too, take the screens whose words Claude writes")
args = ap.parse_args()
OUT: Path = args.out.resolve()
OUT.mkdir(parents=True, exist_ok=True)
DB = OUT / "app.db"
for f in OUT.glob("app.db*"):
    f.unlink()

# The app reads its settings when it's imported.
os.environ.update(AIET_DATABASE=str(DB), AIET_FETCH_PHOTOS="1")
sys.path.insert(0, str(APP))
os.chdir(APP)

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

from app.config import settings  # noqa: E402

AI = not settings.offline  # Claude writes the companions' words
TAKE_AI = AI or args.all


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
BASE = f"http://127.0.0.1:{PORT}"
server = uvicorn.Server(uvicorn.Config("app.main:app", host="127.0.0.1", port=PORT, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.05)


def scalar(sql: str, *params):
    with sqlite3.connect(DB) as conn:
        return conn.execute(sql, params).fetchone()[0]


# The starters' photos are fetched in the background at start-up: wait until no more are coming.
last, still = -1, 0
for _ in range(240):
    have = scalar("SELECT COUNT(*) FROM companion_photos")
    wanted = scalar("SELECT COUNT(*) FROM companions WHERE is_starter = 1 AND photo_url != ''")
    still = still + 1 if have == last else 0
    if have >= wanted or still >= 20:
        break
    last = have
    time.sleep(0.5)
print(f"{have} of {wanted} starter photos, {scalar('SELECT COUNT(*) FROM posts WHERE kind IS NOT NULL')} guides, "
      + ("Claude writing the companions' words" if AI else "demo mode"))

ids = {slug: cid for slug, cid in sqlite3.connect(DB).execute("SELECT slug, id FROM companions WHERE is_starter = 1")}
# A guide whose graphic has been corrected: most still show the site's original, spelling mistakes and all.
GUIDE = scalar("SELECT id FROM posts WHERE graphic LIKE '/static/guides/%' AND title LIKE 'Mastering DuckDB%' "
               "ORDER BY id LIMIT 1")

DEVICES = {
    "desktop": dict(viewport={"width": 1440, "height": 900}, device_scale_factor=2),
    "phone": dict(viewport={"width": 390, "height": 844}, device_scale_factor=3, is_mobile=True, has_touch=True,
                  user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 "
                             "(KHTML, like Gecko) Version/18.0 Mobile/15E148 Safari/604.1"),
}
HIDE_DEMO = ("document.addEventListener('DOMContentLoaded', () => { const s = document.createElement('style'); "
             "s.textContent = '#banner{display:none!important}'; document.head.appendChild(s); });")
WAIT_AI = 240_000  # ms: Claude writing a lesson or a quiz can take a while
errors: list[str] = []

# Pictures from other sites (the guides' graphics) are fetched here, through this machine's proxy settings if it
# has any, and handed to the browser; the app itself the browser reaches directly.
web = httpx.Client(timeout=30, follow_redirects=True)


def outside(route):
    if route.request.url.startswith(BASE):
        return route.continue_()
    try:
        r = web.get(route.request.url)
        route.fulfill(status=r.status_code, body=r.content,
                      headers={"content-type": r.headers.get("content-type", "application/octet-stream")})
    except httpx.HTTPError as e:
        errors.append(f"couldn't fetch {route.request.url}: {e}")
        route.abort()


def settle(page, extra=0.4):
    """Wait until the pictures in view have loaded and any message has gone."""
    page.wait_for_load_state("networkidle")
    page.wait_for_function("""() => [...document.images].filter(i => {
        const r = i.getBoundingClientRect(); return r.bottom > 0 && r.top < innerHeight && r.width > 0;
      }).every(i => i.complete && i.naturalWidth > 0)""", timeout=30_000)
    page.wait_for_function("!document.querySelector('#toast.show')", timeout=15_000)
    time.sleep(extra)


def to_content(page, kind, selector=".main h1"):
    """The page from its top. On a phone the menu comes first on every page: scroll to the page itself, as a
    learner would."""
    if kind == "phone":
        page.evaluate("""(sel) => { const el = document.querySelector(sel);
            if (el) { el.scrollIntoView({block: 'start'}); scrollBy(0, -16); } }""", selector)
    else:
        page.evaluate("scrollTo(0, 0)")


def shot(page, kind, name):
    if kind == "desktop":  # nothing hovered
        page.mouse.move(DEVICES[kind]["viewport"]["width"] - 4, DEVICES[kind]["viewport"]["height"] - 4)
        time.sleep(0.2)
    path = OUT / kind / f"{name}.png"
    path.parent.mkdir(exist_ok=True)
    page.screenshot(path=str(path))
    print("  ", path.relative_to(OUT))


def journey(browser, kind):
    with sqlite3.connect(DB) as conn:  # a new learner each time
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("DELETE FROM users WHERE email = 'amara@example.com'")
    ctx = browser.new_context(**DEVICES[kind])
    if not AI:
        ctx.add_init_script(HIDE_DEMO)
    ctx.route("**/*", outside)
    page = ctx.new_page()
    page.on("pageerror", lambda e: errors.append(f"{kind}: error on {page.url}: {e}"))

    # Signing up.
    page.goto(BASE + "/#/welcome")
    page.wait_for_function("document.querySelectorAll('#faces img').length === 8")
    settle(page)
    shot(page, kind, "01-welcome")
    page.fill("[name=display_name]", "Amara")
    page.fill("[name=email]", "amara@example.com")
    page.fill("[name=password]", "correct horse battery")
    page.select_option("[name=level]", "beginner")
    page.fill("[name=goals]", "Understand how AI works and use it to grow my catering business")
    page.fill("[name=interests]", "football, cooking, Afrobeats")
    if kind == "phone":
        page.evaluate("document.querySelector('#authForm').closest('.card').scrollIntoView({block: 'center'})")
    settle(page)
    shot(page, kind, "02-signup")
    page.click("#authBtn")

    # Choosing a companion, who starts the conversation.
    page.wait_for_url("**/#/onboarding")
    page.wait_for_selector("#starterGrid .ccard img.avatar")
    settle(page, 1.0)
    shot(page, kind, "03-choose")
    page.fill("#starterSearch", "Amahlé")
    page.click("#starterGrid [data-act=choose]")
    page.wait_for_url("**/#/c/*")
    page.wait_for_selector("#log .msg.assistant[data-id]", timeout=WAIT_AI)
    if TAKE_AI:
        to_content(page, kind, ".chat")
        settle(page)
        shot(page, kind, "04-chat-opener")
        page.fill("#input", "Yes! I play five-a-side every Sunday. Could AI help my team play better?")
        page.click("#send")
        page.wait_for_function("document.querySelectorAll('#log .msg.assistant[data-id]').length >= 2", timeout=WAIT_AI)
        page.evaluate("document.querySelector('#log').scrollTop = 1e9")
        to_content(page, kind, ".chat")
        settle(page, 1.0)
        shot(page, kind, "05-chat-reply")

    # Customising a companion.
    page.goto(BASE + f"/#/studio/remix/{ids['agnieszka-kowalska']}")
    page.wait_for_selector("#pv")
    to_content(page, kind)
    settle(page)
    shot(page, kind, "06-customise")
    page.evaluate("document.querySelector('#picChoice').closest('fieldset').scrollIntoView({block: 'start'})")
    settle(page)
    shot(page, kind, "07-customise-face")

    # A guide, the learning paths, and a lesson with a quiz.
    page.goto(BASE + f"/#/p/{GUIDE}")
    page.wait_for_selector(".guide-figure img")
    to_content(page, kind)
    settle(page, 1.0)
    shot(page, kind, "08-guide")
    page.goto(BASE + "/#/learn")
    page.wait_for_selector(".main h1")
    to_content(page, kind)
    settle(page)
    shot(page, kind, "09-learn")
    if TAKE_AI:
        page.goto(BASE + "/#/lesson/what-is-ai")
        page.wait_for_selector(".lesson-layout .prose", timeout=WAIT_AI)
        to_content(page, kind)
        settle(page)
        shot(page, kind, "10-lesson")
        page.click(".lesson-layout [data-study=quiz]")  # before "Mark as complete", which moves on to the next lesson
        page.wait_for_url("**/#/quiz/*", timeout=WAIT_AI)
        page.wait_for_selector("#quiz .question")
        quiz = int(page.url.rsplit("/", 1)[1])
        # Amara knows her stuff: every answer right but the last, as a learner part-way through might.
        answers = [q["answer_index"] for q in json.loads(scalar("SELECT questions FROM assessments WHERE id = ?", quiz))]
        answers[-1] = (answers[-1] + 1) % 4
        page.locator("#quiz .question").first.locator("label.option").nth(answers[0]).click()
        to_content(page, kind)
        settle(page)
        shot(page, kind, "11-quiz")
        for i, a in enumerate(answers):
            page.locator("#quiz .question").nth(i).locator("label.option").nth(a).click()
        page.click("#submitQuiz")
        page.wait_for_selector(".result")
        to_content(page, kind)
        settle(page)
        shot(page, kind, "12-quiz-results")
        page.request.post(BASE + "/api/lessons/what-is-ai/complete")
        page.goto(BASE + "/#/feed")
        page.wait_for_selector(".main h1")
        to_content(page, kind)
        settle(page, 1.0)
        shot(page, kind, "13-feed")
        page.goto(BASE + "/#/progress")
        page.wait_for_selector(".tiles")
        to_content(page, kind)
        settle(page)
        shot(page, kind, "14-progress")

    # The companions from around the world.
    page.goto(BASE + "/#/companions")
    page.wait_for_selector(".ccard img.avatar")
    page.fill("#starterSearch", "")
    page.wait_for_function("document.querySelectorAll('#starterGrid .ccard').length > 20")
    page.locator("#starterSearch").blur()
    to_content(page, kind)
    settle(page, 1.0)
    shot(page, kind, "15-companions")
    ctx.close()


with sync_playwright() as p:
    chromium = os.environ.get("CHROMIUM") or next((c for c in ["/opt/pw-browsers/chromium"] if Path(c).exists()), None)
    browser = p.chromium.launch(executable_path=chromium)
    for kind in [args.only] if args.only else ["desktop", "phone"]:
        print(kind)
        journey(browser, kind)
    browser.close()
server.should_exit = True
for e in errors:
    print("PROBLEM", e)
if not AI:
    print("Demo mode: " + ("the screens Claude writes show demo text, not for posting." if args.all else
                           "set ANTHROPIC_API_KEY to take the screens Claude writes as well."))
