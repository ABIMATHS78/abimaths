"""In-app reminders: what's waiting for the learner right now.

Computed on request from existing data, so there is nothing to schedule or keep
in sync. Email or push delivery can later be layered on the same list.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app import auth, companions, db, learning

router = APIRouter(prefix="/api")
User = Depends(auth.current_user)


def _current_plan_item(user_id: int) -> dict | None:
    """The first unfinished study-plan item up to and including the current plan week."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM study_plans WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    if row is None:
        return None
    plan = learning._plan_view(dict(row), user_id)
    week_now = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).days // 7 + 1
    for week in plan["weeks"]:
        if week["week"] > week_now:
            break
        for item in week["items"]:
            if not item["done"]:
                return item | {"week": week["week"], "overdue": week["week"] < week_now}
    return None


@router.get("/reminders")
def reminders(user: dict = User):
    uid = user["id"]
    out: list[dict] = []
    now = db.now()
    with db.connect() as conn:
        due = conn.execute("SELECT COUNT(*) FROM cards c JOIN decks d ON d.id = c.deck_id WHERE d.user_id = ? AND c.due_at <= ?",
                           (uid, now)).fetchone()[0]
        days = {r["day"] for r in conn.execute("SELECT day FROM activity WHERE user_id = ?", (uid,))}
        seen = conn.execute("SELECT feed_seen_post_id FROM users WHERE id = ?", (uid,)).fetchone()[0]
        new_posts = conn.execute(
            "SELECT COUNT(*) FROM posts p JOIN companions c ON c.id = p.companion_id "
            "WHERE p.companion_id IN (SELECT companion_id FROM follows WHERE user_id = :uid) "
            f"AND p.id > :seen AND {companions.VISIBLE_SQL}", {"uid": uid, "seen": seen}).fetchone()[0]
        unfinished = conn.execute("SELECT id, title FROM assessments WHERE user_id = ? AND submitted_at IS NULL "
                                  "ORDER BY id DESC LIMIT 1", (uid,)).fetchone()
        notices = conn.execute("SELECT id, text, link FROM notices WHERE user_id = ? AND read_at IS NULL "
                               "ORDER BY id DESC LIMIT 5", (uid,)).fetchall()
        started = conn.execute("SELECT lesson_id FROM lesson_progress WHERE user_id = ? AND status = 'started' "
                               "ORDER BY updated_at DESC", (uid,)).fetchall()

    for n in notices:
        out.append({"kind": "notice", "priority": 0, "id": n["id"], "text": n["text"], "link": n["link"] or "#/feed"})
    today = datetime.now(timezone.utc).date()
    streak = learning._streak(days)
    if streak and today.isoformat() not in days:
        out.append({"kind": "streak", "priority": 1, "text": f"Keep your {streak}-day streak going: study something today.",
                    "link": "#/progress"})
    if due:
        out.append({"kind": "cards_due", "priority": 2, "count": due,
                    "text": f"{due} flashcard{'s are' if due != 1 else ' is'} due for review.", "link": "#/review"})
    item = _current_plan_item(uid)
    if item:
        verb = {"lesson": "Lesson", "quiz": "Quiz", "flashcards": "Flashcards", "exam": "Mock exam"}[item["activity"]]
        out.append({"kind": "plan", "priority": 2 if item["overdue"] else 3,
                    "text": f"{'Catch up' if item['overdue'] else 'This week'} in your study plan: {verb} – {item['title']}.",
                    "link": "#/progress", "lesson_id": item["lesson_id"], "activity": item["activity"]})
    if unfinished:
        out.append({"kind": "unfinished_quiz", "priority": 3, "text": f"Finish your {unfinished['title'].lower()}.",
                    "link": f"#/quiz/{unfinished['id']}"})
    for r in started:
        found = learning.find_for(uid, r["lesson_id"])
        if found:
            out.append({"kind": "continue_lesson", "priority": 4, "text": f"Pick up where you left off: {found[1].title}.",
                        "link": f"#/lesson/{r['lesson_id']}"})
            break
    if new_posts:
        out.append({"kind": "new_posts", "priority": 5, "count": new_posts,
                    "text": f"{new_posts} new post{'s' if new_posts != 1 else ''} from companions you follow.", "link": "#/feed"})
    return sorted(out, key=lambda r: r["priority"])


@router.post("/reminders/notices/{notice_id}/read", status_code=204)
def read_notice(notice_id: int, user: dict = User):
    with db.connect() as conn:
        conn.execute("UPDATE notices SET read_at = ? WHERE id = ? AND user_id = ?", (db.now(), notice_id, user["id"]))


@router.post("/reminders/feed-seen", status_code=204)
def feed_seen(user: dict = User):
    with db.connect() as conn:
        mark_feed_seen(conn, user["id"])


def mark_feed_seen(conn, user_id: int) -> None:
    """Record the newest post as seen; posts after it count as new."""
    conn.execute("UPDATE users SET feed_seen_post_id = (SELECT COALESCE(MAX(id), 0) FROM posts) WHERE id = ?", (user_id,))
