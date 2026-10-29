"""Automatic content and the content intelligence loop.

Learner engagement (views, likes, "helpful", shares and follow-up questions)
is scored per topic; companions write their next posts toward what learners
genuinely value.
"""

from __future__ import annotations

import asyncio
import logging
import time
import unicodedata
from datetime import datetime, timedelta, timezone

from app import companions, db, llm, paths
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
            "INSERT INTO posts (companion_id, title, body, topic, slug, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (companion["id"], post.title[:200], post.body, post.topic[:60], paths.new_post_slug(conn, post.title[:200], companion["id"]),
             db.now()),
        )
        row = conn.execute("SELECT * FROM posts WHERE id = ?", (cur.lastrowid,)).fetchone()
    return dict(row)


# Starter companions posting on each automatic run, at most: with a hundred of them, a run stays cheap.
POSTS_PER_RUN = 8
# Starters' first posts are on the platform, not on a learner's allowance, so they're capped: this many an hour
# across the site, and a starter whose first post didn't work out waits an hour before trying again.
FIRST_POSTS_PER_HOUR = 20
_first_posts: set[int] = set()  # starters writing their first post right now
_first_post_tries: dict[int, float] = {}  # when each starter last tried


async def generate_for_starters(limit: int = POSTS_PER_RUN) -> int:
    """New posts from the starter companions learners follow, those that posted longest ago first."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.* FROM companions c WHERE c.is_starter = 1 AND c.retired = 0 "
            "AND EXISTS (SELECT 1 FROM follows f WHERE f.companion_id = c.id) "
            "ORDER BY (SELECT MAX(p.id) FROM posts p WHERE p.companion_id = c.id), c.id LIMIT ?", (limit,)).fetchall()
    made = 0
    for row in rows:
        if await generate_for(db.row_to_companion(row)):
            made += 1
        elif llm.account_problem:
            break  # every other post would fail the same way
    return made


async def ensure_initial_posts() -> None:
    """At start-up, a first post from each featured starter companion (the ones new learners start out following)
    and any other starter someone follows, if it has none yet. The rest write theirs when first followed."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.* FROM companions c WHERE c.is_starter = 1 AND c.retired = 0 "
            "AND (c.featured = 1 OR EXISTS (SELECT 1 FROM follows f WHERE f.companion_id = c.id)) "
            "AND NOT EXISTS (SELECT 1 FROM posts p WHERE p.companion_id = c.id) ORDER BY c.featured DESC, c.id"
        ).fetchall()
    for row in rows:
        if await first_post(db.row_to_companion(row)) is None and llm.account_problem:
            break  # every other post would fail the same way


async def first_post(companion: dict) -> dict | None:
    """A starter companion's first post, unless it has one or is writing it already, within the hourly caps
    above. The platform pays for it, not the learner whose follow prompted it."""
    cid, now = companion["id"], time.monotonic()
    if cid in _first_posts or companion["retired"] or llm.account_problem:
        return None
    if now - _first_post_tries.get(cid, -3600) < 3600 or sum(now - t < 3600 for t in _first_post_tries.values()) >= FIRST_POSTS_PER_HOUR:
        return None
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM posts WHERE companion_id = ?", (cid,)).fetchone():
            return None
    _first_post_tries[cid] = now
    _first_posts.add(companion["id"])
    try:
        return await generate_for(companion)
    finally:
        _first_posts.discard(companion["id"])


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


# What lists of posts show of each. A guide's body runs to thousands of words, so lists carry its summary and
# graphic instead, and its own page fetches the rest; a post a companion wrote is short and comes whole.
_LISTED = ("p.id, p.slug, p.companion_id, p.title, p.topic, p.created_at, p.summary, p.kind, p.level, p.minutes, "
           "p.graphic, CASE WHEN p.summary IS NULL THEN p.body ELSE '' END AS body")
_COUNTS = """
  (SELECT COUNT(*) FROM engagements e WHERE e.post_id = p.id AND e.kind = 'like') AS likes,
  (SELECT COUNT(*) FROM engagements e WHERE e.post_id = p.id AND e.kind = 'helpful') AS helpful,
  EXISTS (SELECT 1 FROM engagements e WHERE e.post_id = p.id AND e.user_id = :uid AND e.kind = 'like') AS liked,
  EXISTS (SELECT 1 FROM engagements e WHERE e.post_id = p.id AND e.user_id = :uid AND e.kind = 'helpful') AS marked_helpful"""
# Newest first, a page at a time: each page after the first starts after the last post of the one before.
_PAGE = "AND (:bt IS NULL OR (p.created_at, p.id) < (:bt, :bid)) ORDER BY p.created_at DESC, p.id DESC LIMIT :limit"


def cursor(before: str | None) -> dict:
    """The query parameters for a page that starts after `before`, a post's "<created_at>|<id>". ValueError if
    it isn't one."""
    if not before:
        return {"bt": None, "bid": None}
    when, _, pid = before.rpartition("|")
    if not when or not pid.isdigit():
        raise ValueError("before: a post's created_at and id, as created_at|id")
    return {"bt": when, "bid": int(pid)}


def _post(row) -> dict:
    post = dict(row)
    # The address of the guide's graphic. (Before graphics were pictures, this held a description of one.)
    graphic = post.get("graphic") or ""
    post["graphic"] = graphic if graphic.startswith(("/static/", "https://")) else None
    if "liked" in post:
        post["liked"], post["marked_helpful"] = bool(post["liked"]), bool(post["marked_helpful"])
    return post


def feed(user_id: int, companion_id: int | None = None, limit: int = 20, before: str | None = None) -> list[dict]:
    """Posts from followed companions (or one visible companion), newest first."""
    # Following a companion that was later unpublished or hidden must not keep its posts in the feed.
    where = "p.companion_id = :cid" if companion_id else "p.companion_id IN (SELECT companion_id FROM follows WHERE user_id = :uid)"
    with db.connect() as conn:
        rows = conn.execute(
            f"SELECT {_LISTED}, {_COUNTS} FROM posts p JOIN companions c ON c.id = p.companion_id "
            f"WHERE {where} AND {companions.VISIBLE_SQL} {_PAGE}",
            {"uid": user_id, "cid": companion_id, "limit": limit, **cursor(before)},
        ).fetchall()
    return [_post(r) for r in rows]


def guides(user_id: int, *, region: str | None = None, kind: str | None = None, level: str | None = None,
           topic: str | None = None, q: str | None = None, companion_id: int | None = None,
           before: str | None = None, limit: int = 24) -> list[dict]:
    """The guides learners can browse, newest first: by their companion's region, kind, level, topic,
    companion, or words in the title, summary, topic or companion's name, accents aside."""
    where = ["p.kind IS NOT NULL", companions.VISIBLE_SQL,
             "(c.retired = 0 OR EXISTS (SELECT 1 FROM follows f WHERE f.companion_id = c.id AND f.user_id = :uid))"]
    params: dict = {"uid": user_id, "limit": limit, **cursor(before)}
    for column, value in (("c.region", region), ("p.kind", kind), ("p.level", level), ("p.topic", topic),
                          ("p.companion_id", companion_id)):
        if value:
            name = column.split(".")[1]
            where.append(f"{column} = :{name}")
            params[name] = value
    for i, word in enumerate((q or "").split()[:6]):
        params[f"w{i}"] = "%" + _fold(word).replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        where.append(f"fold(p.title || ' ' || p.summary || ' ' || p.topic || ' ' || json_extract(c.persona, '$.name')) "
                     f"LIKE :w{i} ESCAPE '\\'")
    with db.connect() as conn:
        conn.create_function("fold", 1, _fold, deterministic=True)
        rows = conn.execute(
            f"SELECT {_LISTED}, {_COUNTS} FROM posts p JOIN companions c ON c.id = p.companion_id "
            f"WHERE {' AND '.join(where)} {_PAGE}", params).fetchall()
    return [_post(r) for r in rows]


def guide_topics(user_id: int) -> list[dict]:
    """The topics the guides cover, with how many there are of each, most first."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT p.topic, COUNT(*) AS guides FROM posts p JOIN companions c ON c.id = p.companion_id "
            f"WHERE p.kind IS NOT NULL AND c.retired = 0 AND {companions.VISIBLE_SQL} "
            "GROUP BY p.topic ORDER BY guides DESC, p.topic", {"uid": user_id}).fetchall()
    return [dict(r) for r in rows]


def _fold(text: str | None) -> str:
    """Text without accents, in lower case: "Aoifè Murphŷ" is found by "aoife murphy"."""
    return "".join(ch for ch in unicodedata.normalize("NFKD", text or "") if not unicodedata.combining(ch)).lower()


def post_view(user_id: int, post_id: int) -> dict | None:
    """A post whole, with its likes and the learner's own engagement."""
    with db.connect() as conn:
        row = conn.execute(f"SELECT p.*, {_COUNTS} FROM posts p WHERE p.id = :pid", {"uid": user_id, "pid": post_id}).fetchone()
    return _post(row) if row else None


def find_post(ref: str) -> dict | None:
    """A post by its slug, as its page's address has it, or by its id, as addresses before slugs did."""
    with db.connect() as conn:
        found = paths.as_id(ref)
        row = (conn.execute("SELECT * FROM posts WHERE id = ?", (found,)) if found is not None
               else conn.execute("SELECT * FROM posts WHERE slug = ?", (ref,))).fetchone()
    return _post(row) if row else None


def get_post(post_id: int) -> dict | None:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM posts WHERE id = ?", (post_id,)).fetchone()
    return _post(row) if row else None
