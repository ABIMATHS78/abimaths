"""The App Store screenshots in this folder, made again: the site as the iPhone app shows it, framed with a caption at
the sizes App Store Connect takes, for 6.9-inch iPhones (iphone-6.9/) and 13-inch iPads (ipad-13/).

    python ios/screenshots/make.py        from ai-every-time/, with Playwright's Chromium and the Inter font

It runs the site in demo mode on a throwaway database, makes up a learner, Amara (a chat with her companion, lessons
done, a quiz taken, flashcards and a course under way), and photographs each page as the app shows it: the app's
user agent, a stand-in for its bridge, nothing for sale, and demo mode's banner hidden, as it isn't part of the app.
Change CAPTIONS and SCREENS to change what they show. INTER_FONTS points at the Inter font's .otf files if they
aren't in /usr/share/fonts/opentype/inter.
"""
import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont
from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # ai-every-time
FONTS = Path(os.environ.get("INTER_FONTS", "/usr/share/fonts/opentype/inter"))
PHONE_UA = ("Mozilla/5.0 (iPhone; CPU iPhone OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
            "Mobile/15E148 AIEveryTimeApp/1.0 (iPhone)")
PAD_UA = ("Mozilla/5.0 (iPad; CPU OS 26_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
          "Mobile/15E148 AIEveryTimeApp/1.0 (iPad)")
BRIDGE = """window.webkit = { messageHandlers: { aiet: { postMessage(msg) {
  if (msg.action === 'pushStatus') return Promise.resolve({ permission: 'granted', environment: 'production' });
  return Promise.resolve(true);
} } } };"""
HIDE = "#banner { display: none !important; } .toast { display: none !important; }"
COMPANION_NAME = "Agnieszka"  # a starter companion, not one modelled on a real person

CHAT = [
    ("assistant", "Hi Amara! I'm Agnieszka, from Kraków. You run a bakery, so let's start there: which job eats "
                  "most of your week?"),
    ("user", "Writing our menu descriptions and Instagram posts. Can AI help with that?"),
    ("assistant", "It can, and it's a great first project. A language model has read a huge amount of writing, so it "
                  "predicts words that fit what you ask for. The clearer you ask, the better it fits.\n\n"
                  "**Try this:**\n\n> You're the friendly baker at Amara's Bakery. Write three Instagram captions "
                  "for our new cinnamon buns, each under 25 words, with one emoji.\n\n"
                  "Giving it a role, an audience and a limit is most of the trick. Want to see why each part "
                  "helps?"),
    ("user", "Yes please! And how do I make it sound like me?"),
    ("assistant", "Paste in two or three posts you've written yourself and add: *Match the tone of these.* "
                  "Models copy examples very well. We'll practise it together in today's lesson."),
]
CARDS = [
    ("What's the difference between a program that follows rules and one that learns from data?",
     "People write a rule-based program's rules. A learning system finds the patterns itself, from examples."),
    ("Why do spam filters learn from examples instead of following fixed rules?",
     "Spammers keep changing their tricks. A filter that keeps learning from new examples keeps up; fixed rules go stale."),
    ("When are hand-written rules still the better choice?",
     "When the logic is simple, must never vary, or has to be explained exactly, such as how a tax is worked out."),
    ("What is training data?", "The examples a model learns from. Their gaps and mistakes become the model's."),
    ("Name one risk of learning from data.",
     "The model copies the biases in its examples, so check who and what the data leaves out."),
]

SCREENS = {  # in the order the App Store shows them
    "chat": lambda s: f"#/c/{s['companion']}",
    "studio": lambda s: "#/studio/new",
    "course": lambda s: f"#/course/{s['course']}",
    "lesson": lambda s: f"#/course/{s['course']}/learn/l1-1",
    "flashcards": lambda s: f"#/review?deck={s['deck']}",
    "guides": lambda s: "#/guides",
    "progress": lambda s: "#/progress",
}
# The flashcard page is mostly empty space on an iPad's big screen, so iPads skip it.
SETS = {"iphone": list(SCREENS), "ipad": [n for n in SCREENS if n != "flashcards"]}

CAPTIONS = {
    "chat": ("Ask your AI companion anything", "It explains AI your way, and remembers you"),
    "studio": ("Design a companion of your own", "Face, country, personality and teaching style"),
    "course": ("Courses that end in a certificate", "Lessons, labs, projects and quizzes"),
    "lesson": ("Learn a little every day", "Clear lessons, with a tutor beside each one"),
    "flashcards": ("Remember what you learn", "Flashcards that come back just in time"),
    "guides": ("Over a thousand guides", "Tutorials and cheat sheets by the companions"),
    "progress": ("Keep your streak going", "Your progress, and what to do next"),
}
DEVICES = {
    # The canvas App Store Connect takes, the browser that photographs the page, and the outline drawn around it:
    # the widest the screen may be, the status bar's height and the corners' radius in canvas pixels, the bezel.
    "iphone": dict(size=(1320, 2868), folder="iphone-6.9", ua=PHONE_UA, viewport={"width": 440, "height": 956}, scale=3,
                   screen_w=1000, status=150, radius=150, bezel=26, island=True, title=104, sub=54, top=170),
    "ipad": dict(size=(2064, 2752), folder="ipad-13", ua=PAD_UA, viewport={"width": 1032, "height": 1376}, scale=2,
                 screen_w=1640, status=64, radius=60, bezel=34, island=False, title=112, sub=58, top=150),
}
TOP, BOTTOM = (0x4A, 0x3C, 0xC9), (0x8A, 0x7D, 0xFF)  # the brand's purple, darker at the top
INK = (0x1C, 0x1A, 0x29)


# ------------------------------------------------------------------ the site and its learner

def serve(db: Path) -> tuple[subprocess.Popen, str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = os.environ | {"AIET_DATABASE": str(db), "AIET_OFFLINE": "1", "AIET_PUSH_CONTACT": "mailto:screens@example.com"}
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(port)],
                              cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            urllib.request.urlopen(f"{base}/healthz", timeout=1)
            break
        except OSError:
            time.sleep(0.5)
    else:
        server.kill()
        raise SystemExit("The site didn't start.")
    # Starter companions' photos are fetched in the background: wait a little for them, or drawn faces show.
    for _ in range(60):
        with sqlite3.connect(db) as conn:
            if conn.execute("SELECT COUNT(*) FROM companion_photos").fetchone()[0] >= 100:
                break
        time.sleep(0.5)
    return server, base


def make_learner(request, base: str, db: Path) -> dict:
    r = request.post(f"{base}/api/auth/signup", data={
        "email": "amara@example.com", "password": "correct horse", "display_name": "Amara", "level": "beginner",
        "goals": "Use AI to grow my bakery", "interests": "baking, football, my small business", "ai_consent": True})
    assert r.status == 201, r.text()
    me = r.json()
    companion = next(c["id"] for c in request.get(f"{base}/api/companions").json()
                     if c["is_starter"] and c["persona"]["name"].startswith(COMPANION_NAME))
    request.post(f"{base}/api/companions/{companion}/primary")
    # Lessons done, and a quiz with all but one answer right.
    paths = request.get(f"{base}/api/paths").json()
    lessons = [lesson["id"] for p in paths[:2] for lesson in p["lessons"]][:6]
    for lesson in lessons:
        request.get(f"{base}/api/lessons/{lesson}")
        request.post(f"{base}/api/lessons/{lesson}/complete")
    quiz = request.post(f"{base}/api/assessments", data={"kind": "quiz", "source_type": "lesson",
                                                        "source_ref": lessons[0]}).json()
    with sqlite3.connect(db) as conn:
        questions = json.loads(conn.execute("SELECT questions FROM assessments WHERE id = ?", (quiz["id"],)).fetchone()[0])
    answers = [q["answer_index"] for q in questions]
    answers[-1] = (answers[-1] + 1) % len(questions[-1]["options"])
    request.post(f"{base}/api/assessments/{quiz['id']}/submit", data={"answers": answers})
    # Flashcards: demo mode's are placeholders, so these replace them.
    deck = request.post(f"{base}/api/decks", data={"source_type": "lesson", "source_ref": lessons[1], "count": 8}).json()
    with sqlite3.connect(db) as conn:
        conn.execute("DELETE FROM cards WHERE deck_id = ?", (deck["id"],))
        due = datetime.now(timezone.utc).isoformat(timespec="seconds")
        conn.executemany("INSERT INTO cards (deck_id, front, back, due_at) VALUES (?, ?, ?, ?)",
                         [(deck["id"], front, back, due) for front, back in CARDS])
    # A course under way.
    courses = request.get(f"{base}/api/courses").json()["courses"]
    course = next((c for c in courses if c["title"].startswith("Prompt Engineering")), courses[0])
    request.post(f"{base}/api/courses/{course['slug']}/enroll")
    syllabus = request.get(f"{base}/api/courses/{course['slug']}").json()["syllabus"]
    for item in [i for m in syllabus for i in m["items"] if i["type"] == "lesson"][:4]:
        request.post(f"{base}/api/lessons/{item['id']}/complete")
    # The chat with her companion, and twelve days in a row of learning.
    with sqlite3.connect(db) as conn:
        start = datetime.now(timezone.utc) - timedelta(minutes=20)
        conn.executemany("INSERT INTO messages (user_id, companion_id, role, content, created_at) VALUES (?, ?, ?, ?, ?)",
                         [(me["id"], companion, role, text, (start + timedelta(minutes=2 * i)).isoformat(timespec="seconds"))
                          for i, (role, text) in enumerate(CHAT)])
        conn.executemany("INSERT OR IGNORE INTO activity (user_id, day) VALUES (?, ?)",
                         [(me["id"], (date.today() - timedelta(days=d)).isoformat()) for d in range(12)])
    return {"companion": companion, "deck": deck["id"], "course": course["slug"]}


def photograph(browser, base, kind, learner, cookies, folder: Path) -> list[Path]:
    spec = DEVICES[kind]
    ctx = browser.new_context(viewport=spec["viewport"], device_scale_factor=spec["scale"], is_mobile=True,
                              has_touch=True, user_agent=spec["ua"])
    ctx.add_init_script(BRIDGE)
    ctx.add_cookies(cookies)
    page = ctx.new_page()
    shots = []
    for n, name in enumerate(SETS[kind], 1):
        page.goto(f"{base}/?shot={name}{SCREENS[name](learner)}")
        page.add_style_tag(content=HIDE)
        page.wait_for_timeout(2500)
        page.add_style_tag(content=HIDE)
        if name == "chat":  # the end of the conversation; on phones, the chat card fills the screen
            page.evaluate("""() => {
              const chat = document.querySelector('section.chat'), bar = document.querySelector('.topbar');
              if (bar && getComputedStyle(bar).display !== 'none')
                window.scrollTo(0, chat.getBoundingClientRect().top + window.scrollY - bar.offsetHeight - 8);
              document.querySelector('#log').scrollTop = 1e6;
            }""")
        if name == "flashcards":  # turned over: question and answer
            page.click("#flip")
        if name == "course":  # a throwaway database's enrolment count isn't one to show
            page.evaluate("""() => document.querySelectorAll('span.muted.small').forEach(
              (s) => { if (/enrol/.test(s.textContent)) s.style.display = 'none'; })""")
        page.wait_for_timeout(400)
        shot = folder / f"{kind}-{n}-{name}.png"
        page.screenshot(path=str(shot))
        shots.append(shot)
    ctx.close()
    return shots


# ------------------------------------------------------------------ framing

def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def gradient(size):
    w, h = size
    column = Image.new("RGB", (1, h))
    for y in range(h):
        t = y / (h - 1)
        column.putpixel((0, y), tuple(round(a + (b - a) * t) for a, b in zip(TOP, BOTTOM)))
    return column.resize(size)


def status_bar(width, height, ipad):
    """iOS's status bar over the page's white top bar: the time, signal, Wi-Fi and battery."""
    bar = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(bar)
    s = height / (64 if ipad else 150)  # drawn for these heights, scaled to the one asked for
    y = height * (0.5 if ipad else 0.56)
    d.text((round((44 if ipad else 110) * s), y), "9:41  Thu 29 Oct" if ipad else "9:41",
           font=font("Inter-SemiBold.otf", round((34 if ipad else 52) * s)), fill=INK, anchor="lm")
    right = width - round((44 if ipad else 96) * s)
    bw, bh = round((48 if ipad else 64) * s), round((22 if ipad else 30) * s)  # the battery
    d.rounded_rectangle((right - bw, y - bh / 2, right, y + bh / 2), radius=round(7 * s), outline=INK, width=max(2, round(3 * s)))
    d.rounded_rectangle((right - bw + round(5 * s), y - bh / 2 + round(5 * s), right - round(14 * s), y + bh / 2 - round(5 * s)),
                        radius=round(3 * s), fill=INK)
    d.rectangle((right + round(3 * s), y - round(5 * s), right + round(7 * s), y + round(5 * s)), fill=INK)
    cx, base = right - bw - round(44 * s), y + round(12 * s)  # Wi-Fi
    for r in (10, 20, 30):
        rr = round(r * s)
        d.arc((cx - rr, base - rr, cx + rr, base + rr), 225, 315, fill=INK, width=max(2, round(5 * s)))
    sx = cx - round(94 * s)  # signal
    for i in range(4):
        hgt = round((8 + 7 * i) * s)
        d.rounded_rectangle((sx + i * round(12 * s), base - hgt, sx + i * round(12 * s) + round(8 * s), base),
                            radius=round(2 * s), fill=INK)
    return bar


def device(shot, spec, height):
    """The page under a status bar, with rounded corners, in a dark outline, `height` tall in all."""
    st_h = round(spec["status"] * shot.width / spec["screen_w"])
    screen = Image.new("RGB", (shot.width, shot.height + st_h), "white")
    screen.paste(status_bar(shot.width, st_h, not spec["island"]), (0, 0))
    screen.paste(shot, (0, st_h))
    if spec["island"]:
        iw, ih = round(shot.width * 0.285), round(st_h * 0.5)
        x0, y0 = (shot.width - iw) // 2, round(st_h * 0.18)
        ImageDraw.Draw(screen).rounded_rectangle((x0, y0, x0 + iw, y0 + ih), radius=ih // 2, fill="black")
    r, b = spec["radius"], spec["bezel"]
    w = min(spec["screen_w"], round(screen.width * (height - 2 * b) / screen.height))
    h = round(screen.height * w / screen.width)
    screen = screen.resize((w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), radius=r, fill=255)
    frame = Image.new("RGBA", (w + 2 * b, h + 2 * b), (0, 0, 0, 0))
    ImageDraw.Draw(frame).rounded_rectangle((0, 0, w + 2 * b - 1, h + 2 * b - 1), radius=r + b, fill=(18, 16, 25, 255))
    frame.paste(screen, (b, b), mask)
    return frame


def caption(canvas, spec, title, sub) -> int:
    """The caption at the top, wrapped to the canvas. Returns where it ends."""
    d = ImageDraw.Draw(canvas)
    tf, sf = font("InterDisplay-Bold.otf", spec["title"]), font("Inter-Medium.otf", spec["sub"])
    lines, line = [], ""
    for word in title.split():
        test = f"{line} {word}".strip()
        if line and d.textlength(test, font=tf) > canvas.width * 0.86:
            lines.append(line)
            line = word
        else:
            line = test
    lines.append(line)
    y = spec["top"]
    for text in lines:
        d.text((canvas.width / 2, y), text, font=tf, fill="white", anchor="mt")
        y += round(spec["title"] * 1.16)
    y += round(spec["sub"] * 0.45)
    d.text((canvas.width / 2, y), sub, font=sf, fill=(236, 233, 255), anchor="mt")
    return y + round(spec["sub"] * 1.6)


def frame(shot: Path) -> Path:
    kind, n, name = shot.stem.split("-", 2)
    spec = DEVICES[kind]
    canvas = gradient(spec["size"]).convert("RGBA")
    y = caption(canvas, spec, *CAPTIONS[name]) + round(spec["sub"] * 0.6)
    phone = device(Image.open(shot).convert("RGB"), spec, canvas.height - y - spec["top"] // 2)
    x = (canvas.width - phone.width) // 2
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((x + 10, y + 30, x + phone.width - 10, y + phone.height + 30),
                                             radius=spec["radius"], fill=(20, 10, 60, 110))
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(40)))
    canvas.paste(phone, (x, y), phone)
    out = HERE / spec["folder"] / f"{n}-{name}.png"
    out.parent.mkdir(exist_ok=True)
    canvas.convert("RGB").save(out, optimize=True)
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        server, base = serve(tmp / "screens.db")
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch()
                ctx = browser.new_context()
                learner = make_learner(ctx.request, base, tmp / "screens.db")
                cookies = ctx.cookies()
                shots = [s for kind in DEVICES for s in photograph(browser, base, kind, learner, cookies, tmp)]
                browser.close()
        finally:
            server.terminate()
        for folder in DEVICES.values():
            for old in (HERE / folder["folder"]).glob("*.png"):
                old.unlink()
        for shot in shots:
            out = frame(shot)
            print(out.relative_to(HERE), Image.open(out).size)


if __name__ == "__main__":
    main()
