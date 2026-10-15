"""AI Every Time: HTTP API and static web app."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, EmailStr, Field

from app import account, auth, companions, content, db, digest, insights, learning, llm, push, reminders, staff, usage
from app.avatar import render_svg
from app.config import settings
from app.persona import EXPERTISE_AREAS, Avatar, Persona

log = logging.getLogger(__name__)
STATIC = Path(__file__).resolve().parent.parent / "static"
_background: set[asyncio.Task] = set()


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _background.add(task)
    task.add_done_callback(_background.discard)


@asynccontextmanager
async def lifespan(_: FastAPI):
    db.init_db()
    companions.seed_starters()
    _spawn(content.ensure_initial_posts())
    if settings.autogen_interval_minutes > 0:
        _spawn(content.autogen_loop())
    if digest.email_available():
        _spawn(digest.digest_loop())
    if push.available():
        _spawn(push.push_loop())
    if settings.offline:
        log.warning("No Anthropic credentials found: running in offline demo mode.")
    else:
        _spawn(llm.check_account())
    yield
    for task in list(_background):
        task.cancel()


app = FastAPI(title="AI Every Time", lifespan=lifespan)
app.include_router(learning.router)
app.include_router(reminders.router)
app.include_router(staff.router)
app.include_router(insights.router)
app.include_router(digest.router)
app.include_router(push.router)
app.include_router(account.router)
User = Depends(auth.current_user)


# ------------------------------------------------------------------------- auth


def _known_zone(v: str) -> str:
    if v:
        try:
            if len(v) > 64:
                raise ValueError
            ZoneInfo(v)
        except (ZoneInfoNotFoundError, ValueError, OSError):  # OSError: names that can't be file names
            raise ValueError("unknown time zone")
    return v


def _zone_or_unknown(v: str) -> str:
    try:
        return _known_zone(v)
    except ValueError:
        return ""


# An IANA time zone name such as "Africa/Lagos", as browsers report it. Empty means unknown (UTC).
TimeZone = Annotated[str, Field(max_length=64), AfterValidator(_known_zone)]


class Signup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(min_length=1, max_length=60)
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    goals: str = Field("", max_length=500)
    # Reported by the browser. One the server doesn't know is ignored rather than blocking sign-up.
    timezone: Annotated[str, AfterValidator(_zone_or_unknown)] = ""


class Login(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=60)
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    goals: str | None = Field(None, max_length=500)
    email_reminders: bool | None = None
    timezone: TimeZone | None = None
    # The hour of their own day for the daily reminder, by email and notification.
    reminder_hour: int | None = Field(None, ge=0, le=23)


@app.post("/api/auth/signup", status_code=201)
def signup(body: Signup, response: Response, background: BackgroundTasks):
    try:
        with db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, display_name, password_hash, level, goals, timezone, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (body.email.lower(), body.display_name.strip(), auth.hash_password(body.password), body.level,
                 body.goals.strip(), body.timezone, db.now()),
            )
            # Start every learner following the starter companions so their feed isn't empty.
            conn.execute("INSERT INTO follows (user_id, companion_id, created_at) "
                         "SELECT ?, id, ? FROM companions WHERE is_starter = 1", (cur.lastrowid, db.now()))
            # Posts published before sign-up are part of the welcome feed, not "new".
            reminders.mark_feed_seen(conn, cur.lastrowid)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with that email already exists")
    auth.set_cookie(response, auth.create_session(cur.lastrowid))
    user = _load_user(cur.lastrowid)
    digest.request_confirmation(user, background, per_hour=1)
    return me(user)


@app.post("/api/auth/login")
def login(body: Login, response: Response):
    email = body.email.lower()
    account.check_guesses(email)
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    if row is None or not auth.verify_password(body.password, row["password_hash"]):
        account.wrong_password(email)
        raise HTTPException(401, "Wrong email or password")
    account.right_password(email)
    auth.set_cookie(response, auth.create_session(row["id"]))
    return me(dict(row))


@app.post("/api/auth/logout", status_code=204)
def logout(response: Response, aiet_session: str | None = Cookie(default=None)):
    if aiet_session:
        auth.delete_session(aiet_session)
    response.delete_cookie(auth.COOKIE)


def _load_user(user_id: int) -> dict:
    with db.connect() as conn:
        return dict(conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone())


@app.get("/api/me")
def me(user: dict = User):
    return auth.public_user(user) | {
        "offline_mode": settings.offline, "email_available": digest.email_available(), "push_available": push.available(),
        "timezone": user.get("timezone") or "", "reminder_hour": reminders.reminder_hour(user),
        "email_confirmed": bool(user.get("email_confirmed_at")), "ai_allowance": usage.status(user),
        "ai_problem": _ai_problem(user),
    }


def _ai_problem(user: dict) -> str | None:
    """For a banner while Claude can't be used: what's wrong and the fix for admins, plainly for everyone else."""
    if not llm.account_problem:
        return None
    if auth.role_of(user) != "admin":
        return llm.AI_UNAVAILABLE
    return (f"AI isn't working: {llm.account_problem}. Check ANTHROPIC_API_KEY (and AIET_MODEL) on the server, "
            "then restart it.")


@app.patch("/api/me")
def update_me(body: ProfileUpdate, background: BackgroundTasks, user: dict = User):
    changes = body.model_dump(exclude_none=True)
    if changes.get("email_reminders") and not digest.email_available():
        raise HTTPException(409, "Email reminders aren't set up on this server yet")
    if "email_reminders" in changes:
        changes["email_reminders"] = int(changes["email_reminders"])
        with db.connect() as conn:
            digest.ensure_token(conn, user["id"])
        if changes["email_reminders"] and not user["email_reminders"]:
            # Nothing is emailed until the learner confirms the address is theirs. One link at a time:
            # sign-up may have sent one already.
            digest.request_confirmation(user, background, per_hour=1)
    if changes:
        cols = ", ".join(f"{k} = ?" for k in changes)
        with db.connect() as conn:
            conn.execute(f"UPDATE users SET {cols} WHERE id = ?", (*changes.values(), user["id"]))
    return me(_load_user(user["id"]))


# ------------------------------------------------------------------ companions


class CreateCompanion(BaseModel):
    persona: Persona
    make_primary: bool = True


class RemixCompanion(BaseModel):
    overrides: dict | None = None
    make_primary: bool = True


@app.get("/api/options")
def options():
    """Everything the companion designer needs to render its controls."""
    return {
        "expertise": EXPERTISE_AREAS,
        "teaching_styles": ["socratic", "step_by_step", "storytelling", "examples_first", "challenge_driven"],
        "lengths": ["concise", "balanced", "detailed"],
        "levels": ["beginner", "intermediate", "advanced"],
        "emoji": ["none", "light", "expressive"],
        "hair_styles": ["short", "long", "curly", "bun", "braids", "afro", "waves", "bald"],
        "accessories": ["none", "glasses", "round_glasses", "headphones", "earrings", "headwrap", "cap"],
        "expressions": ["smile", "grin", "calm", "curious"],
    }


@app.get("/api/companions")
def list_companions(user: dict = User):
    return companions.list_for(user["id"])


@app.get("/api/companions/{companion_id}")
def get_companion(companion_id: int, user: dict = User):
    c = companions.get_visible(companion_id, user["id"])
    with db.connect() as conn:
        following = conn.execute("SELECT 1 FROM follows WHERE user_id = ? AND companion_id = ?",
                                 (user["id"], companion_id)).fetchone()
        followers = conn.execute("SELECT COUNT(*) FROM follows WHERE companion_id = ?", (companion_id,)).fetchone()[0]
        designer = conn.execute("SELECT display_name FROM users WHERE id = ?", (c["owner_id"],)).fetchone() if c["owner_id"] else None
    mine = c["owner_id"] == user["id"]
    return companions.for_viewer(c, user["id"]) | {
        "following": bool(following), "followers": followers, "is_mine": mine,
        "designer": designer["display_name"] if designer else None,
        # Only the designer sees why a companion was removed and where their appeal stands.
        "moderation": staff.moderation_for_owner(c) if mine else None,
    }


class ReportIn(BaseModel):
    reason: Literal["impersonation", "harmful", "misleading", "spam", "other"]


@app.post("/api/companions/{companion_id}/appeal", status_code=201)
def appeal_removal(companion_id: int, body: staff.AppealIn, user: dict = User):
    return staff.appeal(user, companion_id, body)


@app.post("/api/companions/{companion_id}/publish")
def publish(companion_id: int, user: dict = User):
    return companions.set_published(user["id"], companion_id, True)


@app.delete("/api/companions/{companion_id}/publish")
def unpublish(companion_id: int, user: dict = User):
    return companions.set_published(user["id"], companion_id, False)


@app.post("/api/companions/{companion_id}/report", status_code=204)
def report_companion(companion_id: int, body: ReportIn, user: dict = User):
    companions.report(user["id"], companion_id, body.reason)


async def _first_post(companion: dict, row: int) -> None:
    with usage.recording(row):
        await content.generate_for(companion)


def _write_first_post(user: dict, companion: dict) -> None:
    """A new companion writes its first post, which counts as one of the designer's AI requests.
    With none left it starts without one, and they can ask for it later."""
    row = usage.try_reserve(user, "post")
    if row is not None:
        _spawn(_first_post(companion, row))


@app.post("/api/companions", status_code=201)
async def create_companion(body: CreateCompanion, user: dict = User):
    c = companions.create(user["id"], body.persona)
    if body.make_primary:
        companions.set_primary(user["id"], c["id"])
    _write_first_post(user, c)
    return c


@app.post("/api/companions/{companion_id}/remix", status_code=201)
async def remix_companion(companion_id: int, body: RemixCompanion, user: dict = User):
    try:
        c = companions.remix(user["id"], companion_id, body.overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if body.make_primary:
        companions.set_primary(user["id"], c["id"])
    _write_first_post(user, c)
    return c


@app.put("/api/companions/{companion_id}")
def update_companion(companion_id: int, persona: Persona, user: dict = User):
    return companions.update(user["id"], companion_id, persona)


@app.delete("/api/companions/{companion_id}", status_code=204)
def delete_companion(companion_id: int, user: dict = User):
    companions.delete(user["id"], companion_id)


@app.get("/api/companions/{companion_id}/versions")
def companion_versions(companion_id: int, user: dict = User):
    return companions.versions(user["id"], companion_id)


@app.post("/api/companions/{companion_id}/versions/{version_id}/restore")
def restore_version(companion_id: int, version_id: int, user: dict = User):
    return companions.restore(user["id"], companion_id, version_id)


@app.post("/api/companions/{companion_id}/follow", status_code=204)
def follow(companion_id: int, user: dict = User):
    companions.set_follow(user["id"], companion_id, True)


@app.delete("/api/companions/{companion_id}/follow", status_code=204)
def unfollow(companion_id: int, user: dict = User):
    companions.set_follow(user["id"], companion_id, False)


@app.post("/api/companions/{companion_id}/primary", status_code=204)
def make_primary(companion_id: int, user: dict = User):
    companions.set_primary(user["id"], companion_id)


# --------------------------------------------------------------------- avatars

SVG = "image/svg+xml"
MAX_PHOTO_BYTES = 2 * 1024 * 1024
# Uploaded photos are served with these headers so a disguised file can't run as a page.
IMAGE_HEADERS = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
                 "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}


def _image_type(data: bytes) -> str | None:
    """Identify an upload by its bytes, never by the name or header the client sent."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


@app.get("/api/companions/{companion_id}/avatar")
@app.get("/api/companions/{companion_id}/avatar.svg", include_in_schema=False)
def companion_avatar(companion_id: int, user: dict | None = Depends(auth.optional_user)):
    # Starter faces are public (they appear on the welcome page); others follow companion visibility.
    c = companions.get(companion_id)
    # Moderators also need to see hidden companions' faces to review them.
    moderator = user is not None and auth.role_of(user) in auth.CAN["moderate"]
    if not companions.visible_to(c, user["id"] if user else None) and not moderator:
        raise HTTPException(404, "Companion not found")
    with db.connect() as conn:
        photo = conn.execute("SELECT mime, data FROM companion_photos WHERE companion_id = ?", (companion_id,)).fetchone()
    if photo:
        return Response(photo["data"], media_type=photo["mime"], headers=IMAGE_HEADERS)
    return Response(render_svg(Avatar.model_validate(c["persona"]["avatar"])), media_type=SVG, headers=IMAGE_HEADERS)


@app.put("/api/companions/{companion_id}/photo", status_code=204)
async def upload_photo(companion_id: int, request: Request, user: dict = User):
    """Raw image bytes in the body (PNG, JPEG or WebP, up to 2 MB)."""
    companions.get_owned(companion_id, user["id"])
    if int(request.headers.get("content-length") or 0) > MAX_PHOTO_BYTES:
        raise HTTPException(413, "Photos can be up to 2 MB")
    data = bytearray()
    async for chunk in request.stream():
        data += chunk
        if len(data) > MAX_PHOTO_BYTES:
            raise HTTPException(413, "Photos can be up to 2 MB")
    mime = _image_type(bytes(data))
    if mime is None:
        raise HTTPException(415, "Upload a PNG, JPEG or WebP image")
    ts = db.now()
    with db.connect() as conn:
        conn.execute("INSERT OR REPLACE INTO companion_photos (companion_id, mime, data, updated_at) VALUES (?, ?, ?, ?)",
                     (companion_id, mime, bytes(data), ts))
        conn.execute("UPDATE companions SET updated_at = ? WHERE id = ?", (ts, companion_id))


@app.delete("/api/companions/{companion_id}/photo", status_code=204)
def delete_photo(companion_id: int, user: dict = User):
    companions.get_owned(companion_id, user["id"])
    with db.connect() as conn:
        conn.execute("DELETE FROM companion_photos WHERE companion_id = ?", (companion_id,))
        conn.execute("UPDATE companions SET updated_at = ? WHERE id = ?", (db.now(), companion_id))


@app.post("/api/avatar/preview")
def avatar_preview(avatar: Avatar):
    return Response(render_svg(avatar), media_type=SVG)


# --------------------------------------------------------------------- content


class Engage(BaseModel):
    kind: Literal["view", "like", "share", "helpful", "ask"]


@app.get("/api/feed")
def get_feed(user: dict = User):
    return content.feed(user["id"])


@app.get("/api/companions/{companion_id}/posts")
def companion_posts(companion_id: int, user: dict = User):
    companions.get_visible(companion_id, user["id"])
    return content.feed(user["id"], companion_id=companion_id)


@app.post("/api/companions/{companion_id}/posts/generate", status_code=201)
async def generate_post(companion_id: int, user: dict = User):
    c = companions.get_owned(companion_id, user["id"])
    with usage.spend(user, "post"):
        post = await content.generate_for(c)
    if post is None:
        raise HTTPException(502, "Couldn't write a post right now. Try again.")
    return post


def _visible_post(post_id: int, user_id: int) -> dict:
    post = content.get_post(post_id)
    if post is None:
        raise HTTPException(404, "Post not found")
    companions.get_visible(post["companion_id"], user_id)
    return post


@app.post("/api/posts/{post_id}/engage", status_code=204)
def engage(post_id: int, body: Engage, user: dict = User):
    _visible_post(post_id, user["id"])
    content.record(user["id"], post_id, body.kind)


@app.delete("/api/posts/{post_id}/engage/{kind}", status_code=204)
def disengage(post_id: int, kind: Literal["like", "helpful"], user: dict = User):
    _visible_post(post_id, user["id"])
    content.unrecord(user["id"], post_id, kind)


@app.get("/api/insights/topics")
def topic_insights(user: dict = User):
    return content.topic_scores()


# ------------------------------------------------------------------------ chat


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    post_id: int | None = None
    lesson_id: str | None = Field(None, max_length=80)


def _memories(user_id: int, companion_id: int) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id, fact, created_at FROM memories WHERE user_id = ? AND companion_id = ? ORDER BY id",
                            (user_id, companion_id)).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/companions/{companion_id}/messages")
def get_messages(companion_id: int, user: dict = User):
    companions.get_visible(companion_id, user["id"])
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, role, content, post_id, created_at FROM messages WHERE user_id = ? AND companion_id = ? "
            "ORDER BY id DESC LIMIT 200", (user["id"], companion_id)).fetchall()
    return [dict(r) for r in reversed(rows)]


@app.post("/api/companions/{companion_id}/chat")
async def chat(companion_id: int, body: ChatIn, user: dict = User):
    c = companions.get_visible(companion_id, user["id"])
    persona = companions.persona_for(c, user["id"])
    post = _visible_post(body.post_id, user["id"]) if body.post_id else None
    lesson = learning.lesson_context(user["id"], companion_id, body.lesson_id) if body.lesson_id and not post else None
    # Before anything is saved, so a message turned away by the allowance leaves no trace.
    row = usage.reserve(user, "chat")
    if post:
        content.record(user["id"], post["id"], "ask")
    post = post or lesson
    learning.record_activity(user)

    with db.connect() as conn:
        conn.execute("INSERT INTO messages (user_id, companion_id, role, content, post_id, created_at) VALUES (?, ?, 'user', ?, ?, ?)",
                     (user["id"], companion_id, body.message, body.post_id, db.now()))
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? AND companion_id = ? ORDER BY id DESC LIMIT ?",
            (user["id"], companion_id, settings.history_turns * 2)).fetchall()
    history = _normalise_history([dict(r) for r in reversed(rows)])
    memories = [m["fact"] for m in _memories(user["id"], companion_id)]

    async def stream():
        # Tokens for the reply, and for the memories taken from it, go on this message's record.
        with usage.recording(row):
            parts: list[str] = []
            async for chunk in llm.stream_reply(persona, user, memories, history, post):
                parts.append(chunk)
                yield chunk
            reply = "".join(parts).strip()
            with db.connect() as conn:
                conn.execute("INSERT INTO messages (user_id, companion_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
                             (user["id"], companion_id, reply, db.now()))
            _spawn(_remember(user["id"], companion_id, persona, memories, body.message, reply))

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


def _normalise_history(rows: list[dict]) -> list[dict]:
    """The API needs a user-first, alternating history; merge any consecutive same-role rows."""
    while rows and rows[0]["role"] != "user":
        rows.pop(0)
    out: list[dict] = []
    for r in rows:
        if out and out[-1]["role"] == r["role"]:
            out[-1]["content"] += "\n\n" + r["content"]
        else:
            out.append({"role": r["role"], "content": r["content"]})
    return out


async def _remember(user_id: int, companion_id: int, persona: Persona, existing: list[str], msg: str, reply: str):
    facts = await llm.extract_memories(persona, existing, msg, reply)
    if facts:
        with db.connect() as conn:
            conn.executemany("INSERT INTO memories (user_id, companion_id, fact, created_at) VALUES (?, ?, ?, ?)",
                             [(user_id, companion_id, f, db.now()) for f in facts])


@app.get("/api/companions/{companion_id}/memories")
def get_memories(companion_id: int, user: dict = User):
    companions.get_visible(companion_id, user["id"])
    return _memories(user["id"], companion_id)


@app.delete("/api/memories/{memory_id}", status_code=204)
def forget(memory_id: int, user: dict = User):
    with db.connect() as conn:
        cur = conn.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, user["id"]))
    if cur.rowcount == 0:
        raise HTTPException(404, "Memory not found")


# ---------------------------------------------------------------------- static

app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz", include_in_schema=False)
def healthz():
    """For hosting platforms and the container's health check: the app is up and its database answers.
    Opened read-write but never created, so a missing or replaced database file shows up here."""
    try:
        conn = sqlite3.connect(f"file:{quote(settings.database_path)}?mode=rw", uri=True, timeout=5)
        try:
            conn.execute("SELECT 1 FROM users LIMIT 1").fetchall()
        finally:
            conn.close()
    except sqlite3.Error:
        log.exception("health check failed")
        return JSONResponse({"ok": False}, status_code=503)
    return {"ok": True}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Browsers ask for this on pages that don't name an icon, such as an email link opened in a new tab.
    return FileResponse(STATIC / "icons" / "favicon-32.png", media_type="image/png")


@app.get("/sw.js", include_in_schema=False)
def service_worker():
    # Served from the root so it can show notifications for the whole app; no-cache so updates reach devices.
    return FileResponse(STATIC / "sw.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"})
