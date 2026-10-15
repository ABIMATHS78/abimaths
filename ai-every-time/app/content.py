"""Automatic content and the content intelligence loop.

Learner engagement (views, likes, "helpful", shares and follow-up questions)
is scored per topic; companions write their next posts toward what learners
genuinely value.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from app import companions, db, llm
from app.config import settings
from app.persona import Persona

log = logging.getLogger(__name__)

# A follow-up question is the strongest signal: the content started a relationship.
ENGAGEMENT_WEIGHTS = {"view": 1, "like": 3, "helpful": 4, "share": 5, "ask": 6}


def topic_scores(days: int = 30) -> dict[str, float]:
    """Average weighted engagement per post, by topic, over the window."""
    cases = " ".join(f"WHEN '{k}' THEN {v}" for k, v in ENGAGEMENT_WEIGHTS.items())
    with db.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT p.topic, COUNT(DISTINCT p.id) AS posts,
                   COALESCE(SUM(CASE e.kind {cases} ELSE 0 END), 0) AS score
            FROM posts p LEFT JOIN engagements e ON e.post_id = p.id
            WHERE p.created_at >= ?
            GROUP BY p.topic
            """,
            ((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(timespec="seconds"),),
        ).fetchall()
    return {r["topic"]: round(r["score"] / r["posts"], 2) for r in rows if r["posts"]}


def top_topics_for(persona: Persona, limit: int = 3) -> list[str]:
    scores = topic_scores()
    ranked = sorted(persona.expertise, key=lambda t: scores.get(t, 0.0), reverse=True)
    platform_wide = [t for t, _ in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if t not in ranked]
    return (ranked + platform_wide)[:limit]


async def generate_for(companion: dict) -> dict | None:
    persona = Persona.model_validate(companion["persona"])
    with db.connect() as conn:
        recent = [r["title"] for r in conn.execute(
            "SELECT title FROM posts WHERE companion_id = ? ORDER BY id DESC LIMIT 15", (companion["id"],)
        )]
    post = await llm.generate_post(persona, top_topics_for(persona), recent)
    if post is None:
        return None
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO posts (companion_id, title, body, topic, created_at) VALUES (?, ?, ?, ?, ?)",
            (companion["id"], post.title[:200], post.body, post.topic[:60], db.now()),
        )
        row = conn.execute("SELECT * FROM posts WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


async def generate_for_starters() -> int:
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM companions WHERE is_starter = 1").fetchall()
    made = 0
    for row in rows:
        if await generate_for(db.row_to_companion(row)):
            made += 1
        elif llm.account_problem:
            break  # every other post would fail the same way
    return made


async def ensure_initial_posts() -> None:
    """Make sure every starter companion has something in its feed on first run."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.* FROM companions c WHERE c.is_starter = 1 "
            "AND NOT EXISTS (SELECT 1 FROM posts p WHERE p.companion_id = c.id)"
        ).fetchall()
    for row in rows:
        if await generate_for(db.row_to_companion(row)) is None and llm.account_problem:
            break  # every other post would fail the same way


async def autogen_loop() -> None:
    interval = settings.autogen_interval_minutes * 60
    while True:
        await asyncio.sleep(interval)
        try:
            made = await generate_for_starters()
            log.info("auto-generated %d posts", made)
        except Exception:
            log.exception("auto-generation run failed")


def record(user_id: int, post_id: int, kind: str) -> None:
    with db.connect() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO engagements (user_id, post_id, kind, created_at) VALUES (?, ?, ?, ?)",
            (user_id, post_id, kind, db.now()),
        )


def unrecord(user_id: int, post_id: int, kind: str) -> None:
    with db.connect() as conn:
        conn.execute("DELETE FROM engagements WHERE user_id = ? AND post_id = ? AND kind = ?", (user_id, post_id, kind))


def feed(user_id: int, companion_id: int | None = None, limit: int = 30) -> list[dict]:
    """Posts from followed companions (or one visible companion), newest first."""
    # Following a companion that was later unpublished or hidden must not keep its posts in the feed.
    where = "p.companion_id = :cid" if companion_id else "p.companion_id IN (SELECT companion_id FROM follows WHERE user_id = :uid)"
    with db.connect() as conn:
        rows = conn.execute(
            f"""
            SELECT p.*,
              (SELECT COUNT(*) FROM engagements e WHERE e.post_id = p.id AND e.kind = 'like') AS likes,
              (SELECT COUNT(*) FROM engagements e WHERE e.post_id = p.id AND e.kind = 'helpful') AS helpful,
              EXISTS (SELECT 1 FROM engagements e WHERE e.post_id = p.id AND e.user_id = :uid AND e.kind = 'like') AS liked,
              EXISTS (SELECT 1 FROM engagements e WHERE e.post_id = p.id AND e.user_id = :uid AND e.kind = 'helpful') AS marked_helpful
            FROM posts p JOIN companions c ON c.id = p.companion_id
            WHERE {where} AND {companions.VISIBLE_SQL} ORDER BY p.id DESC LIMIT :limit
            """,
            {"uid": user_id, "cid": companion_id, "limit": limit},
        ).fetchall()
    return [dict(r) | {"liked": bool(r["liked"]), "marked_helpful": bool(r["marked_helpful"])} for r in rows]


def get_post(post_id: int) -> dict | None:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    return dict(row) if row else None
