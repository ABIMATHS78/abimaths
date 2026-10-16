"""Slides for social media (1080 x 1350) from the screenshots take.py makes: a caption above the real screens, shown
in a browser window with the phone view beside it. The screenshots themselves are left exactly as taken. A slide
whose screenshots aren't there (those Claude writes, after a run in demo mode) is left out. See README.md.

Run from ai-every-time/:  python scripts/screenshots/slides.py OUT_DIR    (the folder take.py filled)
"""

from __future__ import annotations

import html
import io
import os
import sys
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
LOGO = HERE.parent.parent / "static" / "icons" / "logo-256.webp"
SHOTS = Path(sys.argv[1]).resolve()
OUT = SHOTS / "slides"

# Each slide: its name, the headline (*the word in colour*), the line under it, the desktop and phone screenshots,
# which side the phone goes, and, for the lesson, a stand-in when the lesson itself wasn't taken.
SLIDES = [
    ("meet", "Learn AI with a *companion*", "Always-available AI Learning Companions from around the world, each with "
     "their own face, voice and expertise.", "01-welcome", "01-welcome", "right"),
    ("sign-up", "Create your account in a *minute*", "Tell us your AI level, what you want to learn and what you're "
     "into.", "02-signup", "02-signup", "right"),
    ("choose", "Choose your *companion*", "103 companions from every continent. Pick one as-is, customise one, or "
     "design your own.", "03-choose", "03-choose", "right"),
    ("chat", "Your companion starts the *conversation*", "From what you're into, in their own voice. And they "
     "remember what you tell them.", "05-chat-reply", "04-chat-opener", "right"),
    ("customise", "Make them *yours*", "Change their face, personality, teaching style, writing style and voice "
     "whenever you like.", "06-customise", "07-customise-face", "left"),
    ("guides", "Read their *guides*", "1,162 tutorials and cheat sheets by the companions. Read one, then ask its "
     "author anything.", "08-guide", "08-guide", "right"),
    ("lesson", "Lessons in your companion's *voice*", "Structured courses taught by your companion, with quizzes and "
     "a mock exam at the end of each path.", "10-lesson", "09-learn", "right"),
    ("learn", "Follow a learning *path*", "Structured courses taught by your companion, in their own voice, with "
     "quizzes and mock exams.", "09-learn", "09-learn", "right"),
    ("quiz", "Test *yourself*", "Quizzes and flashcards from any lesson, guide or chat, and a mock exam for each "
     "path.", "11-quiz", "12-quiz-results", "right"),
    ("feed", "A feed that *teaches*", "Fresh lessons and guides from the companions you follow. Every post is a "
     "conversation starter.", "13-feed", "13-feed", "right"),
    ("progress", "See your *progress*", "Your streak, lessons, quiz scores and flashcards, all in one place.",
     "14-progress", "14-progress", "right"),
]

CSS = """
* { box-sizing: border-box; margin: 0; }
html, body { width: 1080px; height: 1350px; overflow: hidden; }
body { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; color: #1c1a29; position: relative;
  background: radial-gradient(900px 700px at 85% 15%, #e3deff 0%, rgba(227,222,255,0) 70%),
              radial-gradient(800px 600px at 0% 100%, #ece9ff 0%, rgba(236,233,255,0) 70%), #f6f5fb; }
.top { position: absolute; left: 72px; right: 72px; top: 64px; display: flex; align-items: center; justify-content: space-between; }
.logo { display: flex; align-items: center; gap: 14px; font-weight: 800; font-size: 30px; letter-spacing: -.02em; }
.logo img { width: 58px; height: 58px; border-radius: 50%; }
.logo .every { color: #9e0c0c; }
.count { font-size: 24px; font-weight: 600; color: #6a6780; }
.text { position: absolute; left: 72px; right: 100px; top: 140px; height: 340px; display: flex; flex-direction: column; justify-content: center; gap: 24px; }
h1 { font-size: 66px; line-height: 1.08; font-weight: 800; letter-spacing: -.025em; }
h1 em { font-style: normal; color: #5b4cdb; }
p.sub { font-size: 29px; line-height: 1.4; color: #5d5a73; }
.browser { position: absolute; left: 72px; top: 500px; width: 936px; border-radius: 18px; overflow: hidden; background: #fff;
  box-shadow: 0 30px 70px rgba(40, 30, 110, .22), 0 4px 14px rgba(40, 30, 110, .10); }
.bar { height: 38px; background: #ecebf2; display: flex; align-items: center; gap: 9px; padding: 0 16px; }
.bar i { width: 13px; height: 13px; border-radius: 50%; display: block; }
.browser img { display: block; width: 100%; }
.phone { position: absolute; top: 640px; width: 318px; padding: 11px; border-radius: 50px; background: #111018;
  box-shadow: 0 34px 80px rgba(20, 15, 60, .35), 0 6px 18px rgba(20, 15, 60, .18); }
.phone.right { right: 44px; } .phone.left { left: 44px; }
.phone img { display: block; width: 100%; border-radius: 40px; }
"""


def slide_html(n: int, total: int, head: str, sub: str, desktop: Path, phone: Path, side: str) -> str:
    head = "".join(f"<em>{part}</em>" if i % 2 else part for i, part in enumerate(html.escape(head).split("*")))
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
      <div class="top"><div class="logo"><img src="{LOGO.as_uri()}"><span>AI <span class="every">Every</span> Time</span></div>
        <div class="count">{n} / {total}</div></div>
      <div class="text"><h1>{head}</h1><p class="sub">{html.escape(sub)}</p></div>
      <div class="browser"><div class="bar"><i style="background:#ff5f57"></i><i style="background:#febc2e"></i><i style="background:#28c840"></i></div>
        <img src="{desktop.as_uri()}"></div>
      <div class="phone {side}"><img src="{phone.as_uri()}"></div>
    </body></html>"""


def main() -> None:
    chosen = []
    for name, head, sub, desk, phone, side in SLIDES:
        files = (SHOTS / "desktop" / f"{desk}.png", SHOTS / "phone" / f"{phone}.png")
        if not all(f.exists() for f in files):
            continue
        if name == "learn" and any(c[0] == "lesson" for c in chosen):
            continue  # the lesson slide shows the paths already
        chosen.append((name, head, sub, *files, side))
    if not chosen:
        raise SystemExit(f"no screenshots in {SHOTS}/desktop and {SHOTS}/phone: run take.py first")
    OUT.mkdir(exist_ok=True)
    (OUT / "html").mkdir(exist_ok=True)
    with sync_playwright() as p:
        chromium = os.environ.get("CHROMIUM") or next((c for c in ["/opt/pw-browsers/chromium"] if Path(c).exists()), None)
        browser = p.chromium.launch(executable_path=chromium)
        tab = browser.new_context(viewport={"width": 1080, "height": 1350}, device_scale_factor=2).new_page()
        for n, (name, head, sub, desktop, phone, side) in enumerate(chosen, 1):
            # A file of its own, so the page may show the screenshots, which are files too.
            src = OUT / "html" / f"{n:02d}-{name}.html"
            src.write_text(slide_html(n, len(chosen), head, sub, desktop, phone, side), encoding="utf-8")
            tab.goto(src.as_uri(), wait_until="load")
            tab.wait_for_function("[...document.images].every(i => i.complete && i.naturalWidth > 0)")
            # Drawn at twice the size, then made 1080 x 1350, so the lettering is sharp.
            big = Image.open(io.BytesIO(tab.screenshot())).convert("RGB")
            out = OUT / f"slide-{n:02d}-{name}.png"
            big.resize((1080, 1350), Image.LANCZOS).save(out, optimize=True)
            print("  ", out.relative_to(SHOTS))
        browser.close()


if __name__ == "__main__":
    main()
