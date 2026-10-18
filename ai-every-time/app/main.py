"""AI Every Time: HTTP API and static web app."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import BackgroundTasks, Cookie, Depends, FastAPI, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import AfterValidator, BaseModel, EmailStr, Field

from app import (account, auth, catalog, companions, content, courses, db, digest, guides, insights, learning, llm,
                 photos, push, reactions, reminders, staff, usage)
from app.avatar import render_initials, render_svg
from app.config import settings
from app.persona import EXPERTISE_AREAS, Avatar, Persona
from app.starter_companions import COUNTRY_CODES, flag

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
    catalog.data()  # read the courses now, so a course file that doesn't fit stops the app here, saying which
    courses.forget_changed_lessons()
    companions.seed_starters()
    guides.seed()  # before first posts: a starter with guides has posted already
    _spawn(content.ensure_initial_posts())
    _spawn(photos.ensure_starter_photos())
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
app.include_router(courses.router)
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
    interests: str = Field("", max_length=300)
    # Reported by the browser. One the server doesn't know is ignored rather than blocking sign-up.
    timezone: Annotated[str, AfterValidator(_zone_or_unknown)] = ""


class Login(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=60)
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    goals: str | None = Field(None, max_length=500)
    interests: str | None = Field(None, max_length=300)
    email_reminders: bool | None = None
    timezone: TimeZone | None = None
    # The hour of their own day for the daily reminder, by email and notification.
    reminder_hour: int | None = Field(None, ge=0, le=23)


@app.post("/api/auth/signup", status_code=201)
def signup(body: Signup, response: Response, background: BackgroundTasks):
    try:
        with db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, display_name, password_hash, level, goals, interests, timezone, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (body.email.lower(), body.display_name.strip(), auth.hash_password(body.password), body.level,
                 body.goals.strip(), body.interests.strip(), body.timezone, db.now()),
            )
            # Start every learner following the featured starter companions, one from each region, so their
            # feed isn't empty.
            conn.execute("INSERT INTO follows (user_id, companion_id, created_at) "
                         "SELECT ?, id, ? FROM companions WHERE is_starter = 1 AND featured = 1 AND retired = 0",
                         (cur.lastrowid, db.now()))
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
    return (f"AI isn't working: {llm.account_problem}. Check ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID and "
            "AIET_MODEL, if set) on the server, then restart it.")


@app.patch("/api/me")
def update_me(body: ProfileUpdate, background: BackgroundTasks, user: dict = User):
    changes = body.model_dump(exclude_none=True)
    for k in ("goals", "interests"):
        if k in changes:
            changes[k] = changes[k].strip()
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
    keep_photo: bool = True  # a starter companion's photo comes along, unless they'd rather draw a face


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
        # The starter companions' countries, as they spell them, with their flags.
        "countries": {name: flag(name) for name in COUNTRY_CODES},
    }


@app.get("/api/companions")
def list_companions(user: dict = User):
    return companions.list_for(user["id"])


@app.get("/api/starters/faces")
def starter_faces():
    """A few starter companions' faces for the welcome page, which is open to everyone: the featured ones, then
    the most followed."""
    with db.connect() as conn:
        rows = conn.execute("SELECT id, json_extract(persona, '$.name') AS name, updated_at FROM companions "
                            "WHERE is_starter = 1 AND retired = 0 ORDER BY featured DESC, base_followers DESC, id LIMIT 8").fetchall()
    return [dict(r) for r in rows]


@app.get("/api/companions/{companion_id}")
def get_companion(companion_id: int, user: dict = User):
    c = companions.get_visible(companion_id, user["id"])
    with db.connect() as conn:
        following = conn.execute("SELECT 1 FROM follows WHERE user_id = ? AND companion_id = ?",
                                 (user["id"], companion_id)).fetchone()
        counts = conn.execute("SELECT is_starter, base_followers, (SELECT COUNT(*) FROM follows WHERE companion_id = c.id) "
                              "AS follows FROM companions c WHERE id = ?", (companion_id,)).fetchone()
        designer = conn.execute("SELECT display_name FROM users WHERE id = ?", (c["owner_id"],)).fetchone() if c["owner_id"] else None
        has_photo = conn.execute("SELECT 1 FROM companion_photos WHERE companion_id = ?", (companion_id,)).fetchone()
    mine = c["owner_id"] == user["id"]
    return companions.for_viewer(c, user["id"]) | {
        "following": bool(following), "followers": companions.followers(counts), "is_mine": mine,
        "designer": designer["display_name"] if designer else None, "has_photo": has_photo is not None,
        # Only the designer sees why a companion was removed and where their appeal stands.
        "moderation": staff.moderation_for_owner(c) if mine else None,
        "courses": [{"slug": x.slug, "title": x.title, "kind": x.kind, "level": x.level}
                    for x in catalog.taught_by(c["slug"])] if c["is_starter"] else [],
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
        c = companions.remix(user["id"], companion_id, body.overrides, body.keep_photo)
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


def _starter_first_post(companion_id: int) -> None:
    """A starter companion that hasn't posted yet writes its first post when a learner starts following it."""
    c = companions.get(companion_id)
    if c["is_starter"]:
        _spawn(content.first_post(c))


@app.post("/api/companions/{companion_id}/follow", status_code=204)
async def follow(companion_id: int, user: dict = User):
    companions.set_follow(user["id"], companion_id, True)
    _starter_first_post(companion_id)


@app.delete("/api/companions/{companion_id}/follow", status_code=204)
def unfollow(companion_id: int, user: dict = User):
    companions.set_follow(user["id"], companion_id, False)


@app.post("/api/companions/{companion_id}/primary", status_code=204)
async def make_primary(companion_id: int, user: dict = User):
    companions.set_primary(user["id"], companion_id)
    _starter_first_post(companion_id)


# --------------------------------------------------------------------- avatars

SVG = "image/svg+xml"
MAX_PHOTO_BYTES = 2 * 1024 * 1024
# Uploaded photos are served with these headers so a disguised file can't run as a page.
IMAGE_HEADERS = {"Cache-Control": "no-cache", "X-Content-Type-Options": "nosniff",
                 "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'"}


@app.get("/api/companions/{companion_id}/avatar")
@app.get("/api/companions/{companion_id}/avatar.svg", include_in_schema=False)
def companion_avatar(companion_id: int, background: BackgroundTasks, v: str = "", face: bool = False,
                     user: dict | None = Depends(auth.optional_user)):
    """The companion's photo, or its drawn face if it has none or its designer chose the drawn face. A starter's
    photo that isn't stored here yet is loaded by the browser from its own address meanwhile; if that fails, the
    page asks again with face=1 for the drawn face. A starter's face, asked for with v (its updated_at, which
    changes with its face), can be kept by the browser for a day. Learners' photos are asked for each time, so
    one that's unpublished stops showing at once."""
    # Starter faces are public (they appear on the welcome page); others follow companion visibility.
    c = companions.get(companion_id)
    # Moderators also need to see hidden companions' faces to review them.
    moderator = user is not None and auth.role_of(user) in auth.CAN["moderate"]
    if not companions.visible_to(c, user["id"] if user else None) and not moderator:
        raise HTTPException(404, "Companion not found")
    keep = c["is_starter"] and v == c["updated_at"]
    headers = IMAGE_HEADERS | ({"Cache-Control": "private, max-age=86400"} if keep else {})
    drawn = face or c["persona"]["avatar"].get("use_photo") is False
    with db.connect() as conn:
        photo = None if drawn else conn.execute("SELECT mime, data FROM companion_photos WHERE companion_id = ?",
                                                (companion_id,)).fetchone()
        source = conn.execute("SELECT photo_url FROM companions WHERE id = ?", (companion_id,)).fetchone()["photo_url"]
    if photo:
        return Response(photo["data"], media_type=photo["mime"], headers=headers)
    if source and not drawn:
        if settings.fetch_photos:
            background.add_task(photos.fetch_later, companion_id)
        return RedirectResponse(photos.direct_url(source), status_code=307, headers={"Cache-Control": "no-cache"})
    avatar = Avatar.model_validate(c["persona"]["avatar"])
    # No guessing at what a real person looks like: their initials stand in for their photo.
    svg = render_initials(c["persona"]["name"], avatar.background) if c["real_person"] else render_svg(avatar)
    return Response(svg, media_type=SVG, headers=headers)


@app.get("/api/companions/{companion_id}/photo")
def companion_photo(companion_id: int, user: dict = User):
    """The photo itself, even while the drawn face is shown instead, for choosing between them in the designer:
    a learner's own companion's, or a starter's they're customising."""
    c = companions.get_visible(companion_id, user["id"])
    with db.connect() as conn:
        photo = conn.execute("SELECT mime, data FROM companion_photos WHERE companion_id = ?", (companion_id,)).fetchone()
    if photo is None or not (c["is_starter"] or c["owner_id"] == user["id"]):
        raise HTTPException(404, "No photo")
    return Response(photo["data"], media_type=photo["mime"], headers=IMAGE_HEADERS)


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
    mime = photos.image_type(bytes(data))
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


# A page of posts starts after the last one on the previous page: "<its created_at>|<its id>".
Before = Annotated[str | None, Query(max_length=100)]


def _page(read, *args, **kwargs) -> list[dict]:
    try:
        return read(*args, **kwargs)
    except ValueError as e:
        raise HTTPException(422, str(e))


@app.get("/api/feed")
def get_feed(user: dict = User, before: Before = None):
    return _page(content.feed, user["id"], before=before)


@app.get("/api/companions/{companion_id}/posts")
def companion_posts(companion_id: int, user: dict = User, before: Before = None):
    companions.get_visible(companion_id, user["id"])
    return _page(content.feed, user["id"], companion_id=companion_id, before=before)


@app.get("/api/posts/{post_id}")
def get_post(post_id: int, user: dict = User):
    _visible_post(post_id, user["id"])
    return content.post_view(user["id"], post_id)


@app.get("/api/guides")
def list_guides(
    user: dict = User,
    region: Annotated[str | None, Query(max_length=40)] = None,
    kind: Literal["tutorial", "cheat_sheet"] | None = None,
    level: Literal["beginner", "intermediate", "advanced"] | None = None,
    topic: Annotated[str | None, Query(max_length=80)] = None,
    q: Annotated[str | None, Query(max_length=120)] = None,
    companion: int | None = None,
    before: Before = None,
):
    """The education guides, newest first, a page at a time, optionally narrowed down."""
    return _page(content.guides, user["id"], region=region, kind=kind, level=level, topic=topic, q=q,
                 companion_id=companion, before=before)


@app.get("/api/guides/topics")
def guide_topics(user: dict = User):
    return content.guide_topics(user["id"])


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
def get_messages(companion_id: int, since: int | None = None, user: dict = User):
    """The conversation's last 200 messages, oldest first, with their reactions. With since (the id the chat
    endpoint's X-Message-Id header gives), that message and the ones after it."""
    companions.get_visible(companion_id, user["id"])
    cols = "SELECT id, role, content, post_id, reaction, created_at FROM messages WHERE user_id = ? AND companion_id = ?"
    with db.connect() as conn:
        if since is None:
            rows = conn.execute(f"{cols} ORDER BY id DESC LIMIT 200", (user["id"], companion_id)).fetchall()[::-1]
        else:
            rows = conn.execute(f"{cols} AND id >= ? ORDER BY id LIMIT 200", (user["id"], companion_id, since)).fetchall()
    return [dict(r) for r in rows]


class ReactionIn(BaseModel):
    emoji: str = Field(max_length=16)


def _own_message(message_id: int, user: dict) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM messages WHERE id = ? AND user_id = ?", (message_id, user["id"])).fetchone()
    if row is None:
        raise HTTPException(404, "Message not found")
    companions.get_visible(row["companion_id"], user["id"])
    return dict(row)


@app.put("/api/messages/{message_id}/reaction", status_code=204)
def react(message_id: int, body: ReactionIn, user: dict = User):
    """The learner's reaction to one of the companion's messages, in place of any they gave it before."""
    emoji = reactions.canonical(body.emoji)
    if emoji is None:
        raise HTTPException(422, f"React with one of {' '.join(reactions.REACTIONS)}")
    if _own_message(message_id, user)["role"] != "assistant":
        raise HTTPException(400, "You can react to your companion's messages; it reacts to yours.")
    with db.connect() as conn:
        conn.execute("UPDATE messages SET reaction = ? WHERE id = ?", (emoji, message_id))


@app.delete("/api/messages/{message_id}/reaction", status_code=204)
def unreact(message_id: int, user: dict = User):
    if _own_message(message_id, user)["role"] != "assistant":
        raise HTTPException(400, "You can react to your companion's messages; it reacts to yours.")
    with db.connect() as conn:
        conn.execute("UPDATE messages SET reaction = NULL WHERE id = ?", (message_id,))


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
        sent = conn.execute("INSERT INTO messages (user_id, companion_id, role, content, post_id, created_at) VALUES (?, ?, 'user', ?, ?, ?)",
                            (user["id"], companion_id, body.message, body.post_id, db.now())).lastrowid
        rows = [dict(r) for r in conn.execute(
            "SELECT id, role, content, reaction FROM messages WHERE user_id = ? AND companion_id = ? ORDER BY id DESC LIMIT ?",
            (user["id"], companion_id, settings.history_turns * 2)).fetchall()[::-1]]
        first = conn.execute("SELECT MIN(id) FROM messages WHERE user_id = ? AND companion_id = ?",
                             (user["id"], companion_id)).fetchone()[0]
    # A chat the companion started: the learner opening it comes first, so its opening message stays in view.
    if rows[0]["id"] == first and rows[0]["role"] == "assistant":
        rows.insert(0, {"role": "user", "content": llm.opened(user), "reaction": None})
    history = _normalise_history(rows)
    # The learner's reaction to the companion's last reply, which the companion can take into account.
    reacted = next((r["reaction"] for r in reversed(rows) if r["role"] == "assistant"), None)
    memories = [m["fact"] for m in _memories(user["id"], companion_id)]

    def react(emoji: str) -> None:  # the companion's reaction to the learner's message
        with db.connect() as conn:
            conn.execute("UPDATE messages SET reaction = ? WHERE id = ?", (emoji, sent))

    async def stream():
        # Tokens for the reply, and for the memories taken from it, go on this message's record.
        with usage.recording(row):
            parts: list[str] = []
            async for chunk in llm.stream_reply(persona, user, memories, history, post, reacted, react):
                parts.append(chunk)
                yield chunk
            reply = reactions.strip("".join(parts)).strip()
            if not reply:  # nothing to keep: an empty message would break the conversation's history
                return
            with db.connect() as conn:
                conn.execute("INSERT INTO messages (user_id, companion_id, role, content, created_at) VALUES (?, ?, 'assistant', ?, ?)",
                             (user["id"], companion_id, reply, db.now()))
            _spawn(_remember(user["id"], companion_id, persona, memories, body.message, reply))

    # The learner's message id, so the page can show the companion's reaction to it and let them react to the reply.
    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8", headers={"X-Message-Id": str(sent)})


class OpeningIn(BaseModel):
    post_id: int | None = None  # the post the learner came from, if any


# When each (learner, companion) chat's first message started being written, while it is. One older than
# OPENING_SECONDS was given up on (a stream that never started has nothing to clear it).
_opening: dict[tuple[int, int], float] = {}
OPENING_SECONDS = 120


@app.post("/api/companions/{companion_id}/opening")
async def opening(companion_id: int, body: OpeningIn, user: dict = User):
    """The companion's first message in a chat the learner hasn't started: a conversation from their interests,
    written by Claude in the companion's voice (from a template in demo mode, or if Claude can't be reached),
    streamed as text and kept as the chat's first message. It costs the learner nothing. Nothing (204) for a chat
    that has begun, or whose first message is already being written."""
    c = companions.get_visible(companion_id, user["id"])
    key = (user["id"], companion_id)
    with db.connect() as conn:
        begun = conn.execute("SELECT 1 FROM messages WHERE user_id = ? AND companion_id = ? LIMIT 1", key).fetchone()
    if begun or time.monotonic() - _opening.get(key, -OPENING_SECONDS) < OPENING_SECONDS:
        return Response(status_code=204)
    _opening[key] = time.monotonic()
    try:
        persona = companions.persona_for(c, user["id"])
        post = _visible_post(body.post_id, user["id"]) if body.post_id else None
        memories = [m["fact"] for m in _memories(user["id"], companion_id)]
        row = None if settings.offline else usage.try_reserve(user, "opening")
    except BaseException:
        _opening.pop(key, None)
        raise

    async def stream():
        try:
            with usage.recording(row):
                parts: list[str] = []
                async for chunk in llm.stream_opening(persona, user, memories, post, c["real_person"], row is not None):
                    parts.append(chunk)
                    yield chunk
            text = "".join(parts).strip()
            if text:
                with db.connect() as conn:
                    # Only as the chat's first message: had the learner written first, theirs would start it.
                    conn.execute("INSERT INTO messages (user_id, companion_id, role, content, created_at) "
                                 "SELECT ?, ?, 'assistant', ?, ? WHERE NOT EXISTS "
                                 "(SELECT 1 FROM messages WHERE user_id = ? AND companion_id = ?)",
                                 (*key, text, db.now(), *key))
        finally:
            _opening.pop(key, None)

    return StreamingResponse(stream(), media_type="text/plain; charset=utf-8")


def _normalise_history(rows: list[dict]) -> list[dict]:
    """The API needs a user-first, alternating history; merge any consecutive same-role rows."""
    rows = [r for r in rows if r["content"].strip()]
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
