"""The admin Insights page: how many learners there are and how they learn, and how much AI the
site uses against its allowances, so whoever runs it can see costs coming and tune the limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends

from app import auth, content, db, usage
from app.config import settings

router = APIRouter(prefix="/api/staff")
DAYS = 14
TOKENS = "input_tokens + output_tokens + cache_read_tokens + cache_write_tokens"


@router.get("/insights")
def insights(user: dict = Depends(auth.require("admin"))):
    now = datetime.now(timezone.utc)
    week_ago = (now - timedelta(days=7)).isoformat(timespec="seconds")
    day_ago = (now - timedelta(hours=24)).isoformat(timespec="seconds")
    days = [(now - timedelta(days=i)).date().isoformat() for i in range(DAYS - 1, -1, -1)]  # oldest first, UTC
    since = days[0] + "T00:00:00+00:00"
    with db.connect() as conn:
        def one(sql: str, *args):
            return conn.execute(sql, args).fetchone()[0]

        learners = {
            "total": one("SELECT COUNT(*) FROM users"),
            "new_this_week": one("SELECT COUNT(*) FROM users WHERE created_at > ?", week_ago),
            # Study days follow each learner's own calendar, so this week is the last seven of them.
            "active_this_week": one("SELECT COUNT(DISTINCT user_id) FROM activity WHERE day >= ?", days[-7]),
            "confirmed": one("SELECT COUNT(*) FROM users WHERE email_confirmed_at IS NOT NULL"),
            "with_notifications": one("SELECT COUNT(DISTINCT user_id) FROM push_subscriptions"),
        }
        quizzes = conn.execute("SELECT COUNT(*), AVG(100.0 * score / total) FROM assessments "
                               "WHERE submitted_at > ? AND total > 0", (week_ago,)).fetchone()
        learning = {
            "lessons_completed": one("SELECT COUNT(*) FROM lesson_progress WHERE status = 'completed' AND updated_at > ?",
                                     week_ago),
            "quizzes": quizzes[0], "quiz_average": None if quizzes[1] is None else round(quizzes[1]),
            "chat_messages": one("SELECT COUNT(*) FROM messages WHERE role = 'user' AND created_at > ?", week_ago),
        }
        community = {
            "designed": one("SELECT COUNT(*) FROM companions WHERE owner_id IS NOT NULL AND removed = 0"),
            "published": one("SELECT COUNT(*) FROM companions WHERE owner_id IS NOT NULL AND is_public = 1 "
                             "AND hidden = 0 AND removed = 0"),
            # Everyone starts out following the starter companions, so only learners' own designs say much.
            "most_followed": [dict(r) for r in conn.execute(
                "SELECT c.id, json_extract(c.persona, '$.name') AS name, json_extract(c.persona, '$.flag') AS flag, "
                "u.display_name AS designer, COUNT(f.user_id) AS followers FROM companions c "
                "JOIN follows f ON f.companion_id = c.id LEFT JOIN users u ON u.id = c.owner_id "
                "WHERE c.owner_id IS NOT NULL AND c.is_public = 1 AND c.hidden = 0 AND c.removed = 0 "
                "GROUP BY c.id ORDER BY followers DESC, c.id LIMIT 5")],
        }
        # Requests from learners that went through, by day; scheduled posts and failed calls aside.
        per_day = {r["day"]: (r["requests"], r["tokens"]) for r in conn.execute(
            f"SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS requests, SUM({TOKENS}) AS tokens FROM ai_usage "
            "WHERE created_at >= ? AND kind != 'automatic' AND failed = 0 GROUP BY day", (since,))}
        kinds = [dict(r) for r in conn.execute(
            "SELECT kind, COUNT(*) AS requests, SUM(failed) AS failed, SUM(units) AS units, "
            "SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens, "
            "SUM(cache_read_tokens) AS cache_read_tokens, SUM(cache_write_tokens) AS cache_write_tokens, "
            "SUM(fallback) AS fallbacks FROM ai_usage WHERE created_at >= ? GROUP BY kind "
            "ORDER BY requests DESC, kind", (since,))]
        refused = {r["reason"]: {"requests": r["requests"], "learners": r["learners"]} for r in conn.execute(
            "SELECT reason, SUM(count) AS requests, COUNT(DISTINCT user_id) AS learners FROM ai_refusals "
            "WHERE hour >= ? GROUP BY reason", (since[:13],))}
        last_24h = one("SELECT COALESCE(SUM(units), 0) FROM ai_usage WHERE created_at > ?", day_ago)
    topics = [kv for kv in sorted(content.topic_scores().items(), key=lambda kv: kv[1], reverse=True) if kv[1] > 0][:6]
    none = {"requests": 0, "learners": 0}
    return {
        "learners": learners,
        "learning": learning,
        "community": community | {"topics": [{"topic": t, "score": s} for t, s in topics]},
        "ai": {
            "days": [{"day": d, "requests": per_day.get(d, (0, 0))[0], "tokens": per_day.get(d, (0, 0))[1]} for d in days],
            "kinds": kinds,
            "turned_away": {"allowance": refused.get("allowance", none), "site": refused.get("site", none)},
            "units_last_24h": last_24h,
            "limits": {"allowance": settings.ai_allowance, "unconfirmed": usage.unconfirmed_allowance(),
                       "site": settings.ai_site_limit},
        },
    }
