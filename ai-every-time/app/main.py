"""AI Every Time: HTTP API and static web app."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Cookie, Depends, FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, EmailStr, Field

from app import auth, companions, content, db, learning, llm
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
    if settings.offline:
        log.warning("No Anthropic credentials found: running in offline demo mode.")
    yield
    for task in list(_background):
        task.cancel()


app = FastAPI(title="AI Every Time", lifespan=lifespan)
app.include_router(learning.router)
User = Depends(auth.current_user)


# ------------------------------------------------------------------------- auth


class Signup(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=200)
    display_name: str = Field(min_length=1, max_length=60)
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    goals: str = Field("", max_length=500)


class Login(BaseModel):
    email: EmailStr
    password: str


class ProfileUpdate(BaseModel):
    display_name: str | None = Field(None, min_length=1, max_length=60)
    level: Literal["beginner", "intermediate", "advanced"] | None = None
    goals: str | None = Field(None, max_length=500)


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(auth.COOKIE, token, httponly=True, samesite="lax", secure=settings.secure_cookies,
                        max_age=settings.session_days * 86400)


@app.post("/api/auth/signup", status_code=201)
def signup(body: Signup, response: Response):
    try:
        with db.connect() as conn:
            cur = conn.execute(
                "INSERT INTO users (email, display_name, password_hash, level, goals, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (body.email.lower(), body.display_name.strip(), auth.hash_password(body.password), body.level,
                 body.goals.strip(), db.now()),
            )
            # Start every learner following the starter companions so their feed isn't empty.
            conn.execute("INSERT INTO follows (user_id, companion_id, created_at) "
                         "SELECT ?, id, ? FROM companions WHERE is_starter = 1", (cur.lastrowid, db.now()))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with that email already exists")
    _set_cookie(response, auth.create_session(cur.lastrowid))
    return me(_load_user(cur.lastrowid))


@app.post("/api/auth/login")
def login(body: Login, response: Response):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE email = ?", (body.email.lower(),)).fetchone()
    if row is None or not auth.verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "Wrong email or password")
    _set_cookie(response, auth.create_session(row["id"]))
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
    return auth.public_user(user) | {"offline_mode": settings.offline}


@app.patch("/api/me")
def update_me(body: ProfileUpdate, user: dict = User):
    changes = body.model_dump(exclude_none=True)
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
    return c | {"following": bool(following), "followers": followers, "is_mine": c["owner_id"] == user["id"]}


@app.post("/api/companions", status_code=201)
async def create_companion(body: CreateCompanion, user: dict = User):
    c = companions.create(user["id"], body.persona)
    if body.make_primary:
        companions.set_primary(user["id"], c["id"])
    _spawn(content.generate_for(c))
    return c


@app.post("/api/companions/{companion_id}/remix", status_code=201)
async def remix_companion(companion_id: int, body: RemixCompanion, user: dict = User):
    try:
        c = companions.remix(user["id"], companion_id, body.overrides)
    except ValueError as e:
        raise HTTPException(422, str(e))
    if body.make_primary:
        companions.set_primary(user["id"], c["id"])
    _spawn(content.generate_for(c))
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


@app.get("/api/companions/{companion_id}/avatar.svg")
def companion_avatar(companion_id: int):
    # Faces are public so they can appear in shared posts.
    c = companions.get(companion_id)
    return Response(render_svg(Avatar.model_validate(c["persona"]["avatar"])), media_type=SVG,
                    headers={"Cache-Control": "no-cache"})


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
def insights(user: dict = User):
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
    persona = Persona.model_validate(c["persona"])
    post = _visible_post(body.post_id, user["id"]) if body.post_id else None
    if post:
        content.record(user["id"], post["id"], "ask")
    elif body.lesson_id:
        post = learning.lesson_context(companion_id, body.lesson_id)
    learning.record_activity(user["id"])

    with db.connect() as conn:
        conn.execute("INSERT INTO messages (user_id, companion_id, role, content, post_id, created_at) VALUES (?, ?, 'user', ?, ?, ?)",
                     (user["id"], companion_id, body.message, body.post_id, db.now()))
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE user_id = ? AND companion_id = ? ORDER BY id DESC LIMIT ?",
            (user["id"], companion_id, settings.history_turns * 2)).fetchall()
    history = _normalise_history([dict(r) for r in reversed(rows)])
    memories = [m["fact"] for m in _memories(user["id"], companion_id)]

    async def stream():
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
