"""Structured learning API: paths, lessons, quizzes, mock exams, flashcards,
summaries, study plans and progress tracking. Everything is taught by a
companion, by default the learner's primary companion."""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth, companions, content, db, tutor, usage
from app.curriculum import PATHS, PATHS_BY_ID, LearningPath, Lesson, find_lesson, lesson_brief, path_from_dict

router = APIRouter(prefix="/api")
User = Depends(auth.current_user)

QUIZ_QUESTIONS = 5
EXAM_QUESTIONS = 12
EXAM_MINUTES = 20
PASS_MARK = 70
# Leitner boxes: days until the next review after a correct answer.
BOX_INTERVAL_DAYS = {1: 0, 2: 1, 3: 3, 4: 7, 5: 16}
MASTERED_BOX = 4


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="seconds")


def local_now(user: dict, now: datetime | None = None) -> datetime:
    """The time in the learner's own time zone, or UTC until it's known."""
    now = now or _utcnow()
    try:
        return now.astimezone(ZoneInfo(user.get("timezone") or "UTC"))
    except (ZoneInfoNotFoundError, ValueError, OSError):
        return now.astimezone(timezone.utc)


def record_activity(user: dict) -> None:
    """Mark today as a study day. Days follow the learner's own calendar, so streaks match their evenings."""
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO activity (user_id, day) VALUES (?, ?)",
                     (user["id"], local_now(user).date().isoformat()))


def teacher_for(user: dict, companion_id: int | None) -> dict:
    """The companion teaching this learner: the one asked for, else their primary, else the first starter."""
    if companion_id:
        return companions.get_visible(companion_id, user["id"])
    if user.get("primary_companion_id"):
        try:
            return companions.get_visible(user["primary_companion_id"], user["id"])
        except HTTPException:
            pass  # e.g. a community companion its designer has since unpublished
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM companions WHERE is_starter = 1 ORDER BY id LIMIT 1").fetchone()
    return db.row_to_companion(row)


CUSTOM_PREFIX = "custom-"


def _custom_rows(user_id: int) -> list[dict]:
    """The learner's own custom paths, then other learners' paths that passed expert review."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT p.*, u.display_name AS owner_name FROM custom_paths p JOIN users u ON u.id = p.user_id "
            "WHERE p.user_id = ? OR p.status = 'approved' ORDER BY (p.user_id = ?) DESC, p.id", (user_id, user_id),
        ).fetchall()
    return [dict(r) for r in rows]


def custom_paths(user_id: int) -> list[LearningPath]:
    return [path_from_dict(f"{CUSTOM_PREFIX}{r['id']}", json.loads(r["data"])) for r in _custom_rows(user_id)]


def path_meta(user_id: int) -> dict[str, dict]:
    """Sharing status of each custom path the learner can see, keyed by path id."""
    return {
        f"{CUSTOM_PREFIX}{r['id']}": {
            "mine": r["user_id"] == user_id, "status": r["status"], "suggested_by": r["owner_name"],
            "review_note": r["review_note"] if r["user_id"] == user_id else "",
        }
        for r in _custom_rows(user_id)
    }


def user_paths(user_id: int) -> list[LearningPath]:
    """The curated paths, the learner's own custom paths, and community paths that passed expert review."""
    return list(PATHS) + custom_paths(user_id)


def path_for(user_id: int, path_id: str) -> LearningPath | None:
    if path_id in PATHS_BY_ID:
        return PATHS_BY_ID[path_id]
    return next((p for p in custom_paths(user_id) if p.id == path_id), None) if path_id.startswith(CUSTOM_PREFIX) else None


def find_for(user_id: int, lesson_id: str) -> tuple[LearningPath, Lesson] | None:
    found = find_lesson(lesson_id)
    if found or not lesson_id.startswith(CUSTOM_PREFIX):
        return found
    path = path_for(user_id, lesson_id.rsplit("-", 1)[0])
    lesson = next((l for l in path.lessons if l.id == lesson_id), None) if path else None
    return (path, lesson) if lesson else None


def _lesson_or_404(user_id: int, lesson_id: str) -> tuple[LearningPath, Lesson]:
    found = find_for(user_id, lesson_id)
    if not found:
        raise HTTPException(404, "Lesson not found")
    return found


def _cached_lesson(companion_id: int, lesson_id: str) -> str | None:
    with db.connect() as conn:
        row = conn.execute("SELECT body FROM lesson_content WHERE companion_id = ? AND lesson_id = ?",
                           (companion_id, lesson_id)).fetchone()
    return row["body"] if row else None


def lesson_context(user_id: int, companion_id: int, lesson_id: str) -> dict:
    """What a companion needs to tutor a learner through a lesson (used by chat)."""
    path, lesson = _lesson_or_404(user_id, lesson_id)
    body = lesson_brief(lesson)
    taught = _cached_lesson(companion_id, lesson_id)
    if taught:
        body += f"\n\nHow you taught it:\n{taught}"
    return {"kind": "lesson", "title": lesson.title, "body": f"Path: {path.title}\n{body}"}


def _material(user: dict, teacher: dict, source_type: str, source_ref: str) -> tuple[str, str, list[Lesson]]:
    """Return (title, material text, lessons covered) for a study source the learner can see."""
    if source_type == "lesson":
        path, lesson = _lesson_or_404(user["id"], source_ref)
        text = lesson_brief(lesson)
        taught = _cached_lesson(teacher["id"], lesson.id)
        return lesson.title, f"Lesson id: {lesson.id}\n{text}" + (f"\n\nLesson as taught:\n{taught}" if taught else ""), [lesson]
    if source_type == "path":
        path = path_for(user["id"], source_ref)
        if not path:
            raise HTTPException(404, "Learning path not found")
        text = "\n\n".join(f"Lesson id: {l.id}\n{lesson_brief(l)}" for l in path.lessons)
        return path.title, text, list(path.lessons)
    if source_type == "post":
        post = content.get_post(int(source_ref)) if source_ref.isdigit() else None
        if not post:
            raise HTTPException(404, "Post not found")
        companions.get_visible(post["companion_id"], user["id"])
        return post["title"], f"{post['title']}\n\n{post['body']}", []
    if source_type == "conversation":
        cid = int(source_ref) if source_ref.isdigit() else 0
        c = companions.get_visible(cid, user["id"])
        with db.connect() as conn:
            rows = conn.execute("SELECT role, content FROM messages WHERE user_id = ? AND companion_id = ? "
                                "ORDER BY id DESC LIMIT 60", (user["id"], cid)).fetchall()
        if not rows:
            raise HTTPException(400, "There's no conversation to use yet. Chat with your companion first.")
        name = c["persona"]["name"]
        text = "\n\n".join(f"{'Learner' if r['role'] == 'user' else name}: {r['content']}" for r in reversed(rows))
        return f"Your conversation with {name}", text, []
    raise HTTPException(422, "Unknown source")


# ----------------------------------------------------------------- paths & lessons


def _lesson_stats(user_id: int) -> dict[str, dict]:
    """Per lesson: status, and best/latest score from quiz and exam questions about it."""
    stats: dict[str, dict] = {}
    with db.connect() as conn:
        for r in conn.execute("SELECT lesson_id, status FROM lesson_progress WHERE user_id = ?", (user_id,)):
            stats.setdefault(r["lesson_id"], {})["status"] = r["status"]
        rows = conn.execute("SELECT questions, answers, source_type, source_ref FROM assessments "
                            "WHERE user_id = ? AND submitted_at IS NOT NULL ORDER BY id", (user_id,)).fetchall()
    for r in rows:
        per: dict[str, list[int]] = {}
        for q, a in zip(json.loads(r["questions"]), json.loads(r["answers"])):
            lid = q.get("lesson_id") or (r["source_ref"] if r["source_type"] == "lesson" else "")
            if lid:
                per.setdefault(lid, []).append(int(a == q["answer_index"]))
        for lid, marks in per.items():
            pct = round(100 * sum(marks) / len(marks))
            s = stats.setdefault(lid, {})
            s["best"] = max(s.get("best", 0), pct)
            s["latest"] = pct
    return stats


def _path_view(path: LearningPath, stats: dict[str, dict], meta: dict | None = None) -> dict:
    lessons = [{"id": l.id, "title": l.title, "minutes": l.minutes, "objectives": list(l.objectives),
                "status": stats.get(l.id, {}).get("status", "not_started"),
                "best_score": stats.get(l.id, {}).get("best")} for l in path.lessons]
    done = sum(l["status"] == "completed" for l in lessons)
    return {"id": path.id, "title": path.title, "description": path.description, "level": path.level,
            "topic": path.topic, "lessons": lessons, "completed": done, "total": len(lessons),
            "custom": path.id.startswith(CUSTOM_PREFIX)} | (meta or {}).get(path.id, {})


@router.get("/paths")
def list_paths(user: dict = User):
    stats, meta = _lesson_stats(user["id"]), path_meta(user["id"])
    return [_path_view(p, stats, meta) for p in user_paths(user["id"])]


class NewCustomPath(BaseModel):
    topic: str = Field(min_length=3, max_length=120)
    level: Literal["beginner", "intermediate", "advanced"] = "beginner"
    lessons: int = Field(5, ge=3, le=8)
    companion_id: int | None = None


MAX_CUSTOM_PATHS = 20


@router.post("/custom-paths", status_code=201)
async def create_custom_path(body: NewCustomPath, user: dict = User):
    with db.connect() as conn:
        if conn.execute("SELECT COUNT(*) FROM custom_paths WHERE user_id = ?", (user["id"],)).fetchone()[0] >= MAX_CUSTOM_PATHS:
            raise HTTPException(409, f"You can have up to {MAX_CUSTOM_PATHS} custom paths. Delete one to make another.")
    teacher = teacher_for(user, body.companion_id)
    with usage.spend(user, "path"):
        data = await tutor.design_path(companions.persona_for(teacher, user["id"]), user, body.topic.strip(), body.level, body.lessons)
    if data is None:
        raise HTTPException(502, "Your companion couldn't design that path right now. Try again or rephrase the topic.")
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO custom_paths (user_id, companion_id, data, created_at) VALUES (?, ?, ?, ?)",
                           (user["id"], teacher["id"], json.dumps(data), db.now()))
    pid = f"{CUSTOM_PREFIX}{cur.lastrowid}"
    return _path_view(path_from_dict(pid, data), {}, path_meta(user["id"]))


def _own_path(user_id: int, path_id: str) -> tuple[LearningPath, dict]:
    """A custom path the learner made (not merely one shared with them), with its sharing status."""
    path = path_for(user_id, path_id) if path_id.startswith(CUSTOM_PREFIX) else None
    meta = path_meta(user_id).get(path_id)
    if path is None or not meta or not meta["mine"]:
        raise HTTPException(404, "Custom path not found")
    return path, meta


def _path_row_id(path_id: str) -> int:
    return int(path_id[len(CUSTOM_PREFIX):])


@router.post("/custom-paths/{path_id}/submit")
def submit_for_review(path_id: str, user: dict = User):
    """Ask an expert reviewer to check this path so it can be shared with every learner."""
    _, meta = _own_path(user["id"], path_id)
    if meta["status"] not in ("private", "rejected"):
        raise HTTPException(409, f"This path is already {meta['status']}")
    with db.connect() as conn:
        conn.execute("UPDATE custom_paths SET status = 'pending', submitted_at = ?, review_note = '' WHERE id = ?",
                     (db.now(), _path_row_id(path_id)))
    return path_meta(user["id"])[path_id]


@router.post("/custom-paths/{path_id}/withdraw")
def withdraw_from_review(path_id: str, user: dict = User):
    _, meta = _own_path(user["id"], path_id)
    if meta["status"] != "pending":
        raise HTTPException(409, "Only paths waiting for review can be withdrawn")
    with db.connect() as conn:
        conn.execute("UPDATE custom_paths SET status = 'private' WHERE id = ?", (_path_row_id(path_id),))
    return path_meta(user["id"])[path_id]


@router.delete("/custom-paths/{path_id}", status_code=204)
def delete_custom_path(path_id: str, user: dict = User):
    path, meta = _own_path(user["id"], path_id)
    if meta["status"] == "approved":
        raise HTTPException(409, "This path is shared with the community. Ask a reviewer to unshare it before deleting.")
    ids = [l.id for l in path.lessons]
    marks = ",".join("?" * len(ids))
    with db.connect() as conn:
        conn.execute("DELETE FROM custom_paths WHERE id = ? AND user_id = ?", (int(path_id[len(CUSTOM_PREFIX):]), user["id"]))
        conn.execute(f"DELETE FROM lesson_progress WHERE user_id = ? AND lesson_id IN ({marks})", (user["id"], *ids))
        conn.execute(f"DELETE FROM lesson_content WHERE lesson_id IN ({marks})", ids)


@router.get("/lessons/{lesson_id}")
async def get_lesson(lesson_id: str, companion_id: int | None = None, user: dict = User):
    path, lesson = _lesson_or_404(user["id"], lesson_id)
    teacher = teacher_for(user, companion_id)
    body = _cached_lesson(teacher["id"], lesson_id)
    if body is None:
        # Cached lessons are shared by everyone this companion teaches, so no learner's custom instructions.
        # Only the learner who asks first spends a request on it.
        with usage.spend(user, "lesson"):
            body = await tutor.teach_lesson(companions.persona_for(teacher, None), path, lesson)
        if body is None:
            raise HTTPException(502, "Your companion couldn't prepare this lesson right now. Try again.")
        with db.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO lesson_content (companion_id, lesson_id, body, created_at) VALUES (?, ?, ?, ?)",
                         (teacher["id"], lesson_id, body, db.now()))
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO lesson_progress (user_id, lesson_id, status, updated_at) VALUES (?, ?, 'started', ?)",
                     (user["id"], lesson_id, db.now()))
    record_activity(user)
    idx = path.lessons.index(lesson)
    stats = _lesson_stats(user["id"]).get(lesson_id, {})
    return {
        "id": lesson.id, "title": lesson.title, "minutes": lesson.minutes, "objectives": list(lesson.objectives),
        "body": body, "status": stats.get("status", "started"), "best_score": stats.get("best"),
        "path": {"id": path.id, "title": path.title},
        "prev": path.lessons[idx - 1].id if idx > 0 else None,
        "next": path.lessons[idx + 1].id if idx + 1 < len(path.lessons) else None,
        "teacher": {"id": teacher["id"], "name": teacher["persona"]["name"], "flag": teacher["persona"]["flag"],
                    "updated_at": teacher["updated_at"]},
    }


@router.post("/lessons/{lesson_id}/complete", status_code=204)
def complete_lesson(lesson_id: str, user: dict = User):
    _lesson_or_404(user["id"], lesson_id)
    with db.connect() as conn:
        conn.execute("INSERT INTO lesson_progress (user_id, lesson_id, status, updated_at) VALUES (?, ?, 'completed', ?) "
                     "ON CONFLICT (user_id, lesson_id) DO UPDATE SET status = 'completed', updated_at = excluded.updated_at",
                     (user["id"], lesson_id, db.now()))
    record_activity(user)


# ---------------------------------------------------------------- quizzes & exams


class NewAssessment(BaseModel):
    kind: Literal["quiz", "exam"] = "quiz"
    source_type: Literal["lesson", "path", "post", "conversation"]
    source_ref: str = Field(min_length=1, max_length=80)
    companion_id: int | None = None


class Submission(BaseModel):
    answers: list[int | None] = Field(max_length=50)


def _public_assessment(row: dict, reveal: bool) -> dict:
    qs = json.loads(row["questions"])
    answers = json.loads(row["answers"]) if row["answers"] else None
    out_qs = []
    for i, q in enumerate(qs):
        item = {"question": q["question"], "options": q["options"], "lesson_id": q.get("lesson_id", "")}
        if reveal and answers is not None:
            item |= {"answer_index": q["answer_index"], "explanation": q["explanation"],
                     "your_answer": answers[i], "correct": answers[i] == q["answer_index"]}
        out_qs.append(item)
    return {
        "id": row["id"], "kind": row["kind"], "title": row["title"], "source_type": row["source_type"],
        "source_ref": row["source_ref"], "companion_id": row["companion_id"], "total": row["total"],
        "score": row["score"], "percent": round(100 * row["score"] / row["total"]) if row["score"] is not None else None,
        "time_limit_minutes": row["time_limit_minutes"], "created_at": row["created_at"],
        "submitted_at": row["submitted_at"], "questions": out_qs,
    }


def _assessment_row(user_id: int, assessment_id: int) -> dict:
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM assessments WHERE id = ? AND user_id = ?", (assessment_id, user_id)).fetchone()
    if row is None:
        raise HTTPException(404, "Quiz not found")
    return dict(row)


@router.post("/assessments", status_code=201)
async def create_assessment(body: NewAssessment, user: dict = User):
    if body.kind == "exam" and body.source_type != "path":
        raise HTTPException(422, "Mock exams cover a whole learning path")
    teacher = teacher_for(user, body.companion_id)
    title, material, lessons = _material(user, teacher, body.source_type, body.source_ref)
    exam = body.kind == "exam"
    n = EXAM_QUESTIONS if exam else QUIZ_QUESTIONS
    with usage.spend(user, body.kind):
        questions = await tutor.make_questions(companions.persona_for(teacher, user["id"]), user, material, n, lessons, exam)
    if not questions:
        raise HTTPException(502, "Your companion couldn't write questions right now. Try again.")
    with db.connect() as conn:
        cur = conn.execute(
            "INSERT INTO assessments (user_id, companion_id, kind, title, source_type, source_ref, questions, "
            "time_limit_minutes, total, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (user["id"], teacher["id"], body.kind, f"{'Mock exam' if exam else 'Quiz'}: {title}", body.source_type,
             body.source_ref, json.dumps([q.model_dump() for q in questions]), EXAM_MINUTES if exam else None,
             len(questions), db.now()),
        )
    return _public_assessment(_assessment_row(user["id"], cur.lastrowid), reveal=False)


@router.get("/assessments")
def list_assessments(user: dict = User):
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM assessments WHERE user_id = ? ORDER BY id DESC LIMIT 50", (user["id"],)).fetchall()
    return [{k: v for k, v in _public_assessment(dict(r), reveal=False).items() if k != "questions"} for r in rows]


@router.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: int, user: dict = User):
    row = _assessment_row(user["id"], assessment_id)
    return _public_assessment(row, reveal=row["submitted_at"] is not None)


@router.post("/assessments/{assessment_id}/submit")
def submit_assessment(assessment_id: int, body: Submission, user: dict = User):
    row = _assessment_row(user["id"], assessment_id)
    if row["submitted_at"]:
        raise HTTPException(409, "Already submitted")
    qs = json.loads(row["questions"])
    answers = (body.answers + [None] * len(qs))[: len(qs)]
    answers = [a if isinstance(a, int) and 0 <= a < len(q["options"]) else None for a, q in zip(answers, qs)]
    score = sum(a == q["answer_index"] for a, q in zip(answers, qs))
    with db.connect() as conn:
        conn.execute("UPDATE assessments SET answers = ?, score = ?, submitted_at = ? WHERE id = ?",
                     (json.dumps(answers), score, db.now(), assessment_id))
    record_activity(user)
    result = _public_assessment(_assessment_row(user["id"], assessment_id), reveal=True)
    if row["time_limit_minutes"]:
        elapsed = _utcnow() - datetime.fromisoformat(row["created_at"])
        result["over_time"] = elapsed > timedelta(minutes=row["time_limit_minutes"])
    result["passed"] = result["percent"] >= PASS_MARK
    if row["kind"] == "exam":
        by_lesson: dict[str, list[bool]] = {}
        for q in result["questions"]:
            if q["lesson_id"]:
                by_lesson.setdefault(q["lesson_id"], []).append(q["correct"])
        result["by_lesson"] = [
            {"lesson_id": lid, "title": find_for(user["id"], lid)[1].title, "correct": sum(m), "total": len(m)}
            for lid, m in by_lesson.items()
        ]
    return result


# ------------------------------------------------------------------- flashcards


class NewDeck(BaseModel):
    source_type: Literal["lesson", "path", "post", "conversation"]
    source_ref: str = Field(min_length=1, max_length=80)
    companion_id: int | None = None
    count: int = Field(10, ge=3, le=30)


class Review(BaseModel):
    grade: Literal["again", "good", "easy"]


def _deck_summaries(user_id: int, deck_id: int | None = None) -> list[dict]:
    now = db.now()
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT d.*, COUNT(c.id) AS total, COALESCE(SUM(c.due_at <= ?), 0) AS due, "
            f"COALESCE(SUM(c.box >= {MASTERED_BOX}), 0) AS mastered "
            "FROM decks d LEFT JOIN cards c ON c.deck_id = d.id WHERE d.user_id = ? "
            + ("AND d.id = ? " if deck_id else "") + "GROUP BY d.id ORDER BY d.id DESC",
            (now, user_id, deck_id) if deck_id else (now, user_id),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/decks", status_code=201)
async def create_deck(body: NewDeck, user: dict = User):
    teacher = teacher_for(user, body.companion_id)
    title, material, lessons = _material(user, teacher, body.source_type, body.source_ref)
    with usage.spend(user, "flashcards"):
        cards = await tutor.make_cards(companions.persona_for(teacher, user["id"]), user, material, body.count, lessons)
    if not cards:
        raise HTTPException(502, "Your companion couldn't make flashcards right now. Try again.")
    now = db.now()
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO decks (user_id, companion_id, title, source_type, source_ref, created_at) "
                           "VALUES (?, ?, ?, ?, ?, ?)", (user["id"], teacher["id"], title, body.source_type, body.source_ref, now))
        conn.executemany("INSERT INTO cards (deck_id, front, back, due_at) VALUES (?, ?, ?, ?)",
                         [(cur.lastrowid, c.front[:300], c.back[:600], now) for c in cards])
    return _deck_summaries(user["id"], cur.lastrowid)[0]


@router.get("/decks")
def list_decks(user: dict = User):
    return _deck_summaries(user["id"])


@router.get("/decks/{deck_id}")
def get_deck(deck_id: int, user: dict = User):
    found = _deck_summaries(user["id"], deck_id)
    if not found:
        raise HTTPException(404, "Deck not found")
    with db.connect() as conn:
        cards = conn.execute("SELECT * FROM cards WHERE deck_id = ? ORDER BY id", (deck_id,)).fetchall()
    return found[0] | {"cards": [dict(c) for c in cards]}


@router.delete("/decks/{deck_id}", status_code=204)
def delete_deck(deck_id: int, user: dict = User):
    with db.connect() as conn:
        cur = conn.execute("DELETE FROM decks WHERE id = ? AND user_id = ?", (deck_id, user["id"]))
    if cur.rowcount == 0:
        raise HTTPException(404, "Deck not found")


@router.get("/review")
def due_cards(deck_id: int | None = None, limit: int = 20, user: dict = User):
    sql = ("SELECT c.*, d.title AS deck_title FROM cards c JOIN decks d ON d.id = c.deck_id "
           "WHERE d.user_id = ? AND c.due_at <= ?")
    args: list = [user["id"], db.now()]
    if deck_id:
        sql += " AND d.id = ?"
        args.append(deck_id)
    sql += " ORDER BY c.box, c.due_at LIMIT ?"
    args.append(max(1, min(limit, 100)))
    with db.connect() as conn:
        return [dict(r) for r in conn.execute(sql, args).fetchall()]


@router.post("/cards/{card_id}/review")
def review_card(card_id: int, body: Review, user: dict = User):
    with db.connect() as conn:
        card = conn.execute("SELECT c.* FROM cards c JOIN decks d ON d.id = c.deck_id WHERE c.id = ? AND d.user_id = ?",
                            (card_id, user["id"])).fetchone()
        if card is None:
            raise HTTPException(404, "Card not found")
        now = _utcnow()
        if body.grade == "again":
            box, due, lapses = 1, now + timedelta(minutes=10), card["lapses"] + 1
        else:
            box = min(5, card["box"] + (2 if body.grade == "easy" else 1))
            due, lapses = now + timedelta(days=BOX_INTERVAL_DAYS[box] or 1), card["lapses"]
        conn.execute("UPDATE cards SET box = ?, due_at = ?, reviews = reviews + 1, lapses = ? WHERE id = ?",
                     (box, _iso(due), lapses, card_id))
        updated = dict(conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone())
    record_activity(user)
    return updated


# ------------------------------------------------------------------- summaries


class NewSummary(BaseModel):
    source_type: Literal["lesson", "post", "conversation"]
    source_ref: str = Field(min_length=1, max_length=80)
    companion_id: int | None = None


@router.post("/summaries")
async def make_summary(body: NewSummary, user: dict = User):
    teacher = teacher_for(user, body.companion_id)
    title, material, _ = _material(user, teacher, body.source_type, body.source_ref)
    if body.source_type == "lesson":
        taught = _cached_lesson(teacher["id"], body.source_ref)
        material = taught or material
    with usage.spend(user, "summary"):
        summary = await tutor.summarize(companions.persona_for(teacher, user["id"]), user, title, material)
    if summary is None:
        raise HTTPException(502, "Your companion couldn't summarise this right now. Try again.")
    record_activity(user)
    return {"title": title, **summary.model_dump()}


# ------------------------------------------------------------------ study plan


class NewPlan(BaseModel):
    weekly_minutes: int = Field(120, ge=30, le=1200)
    weeks: int = Field(4, ge=1, le=12)
    paths: list[str] = Field(default_factory=list, max_length=50)
    companion_id: int | None = None


def _plan_view(row: dict, user_id: int) -> dict:
    plan = json.loads(row["plan"])
    stats = _lesson_stats(user_id)
    with db.connect() as conn:
        since = row["created_at"]
        quizzed = {r["source_ref"] for r in conn.execute(
            "SELECT source_ref FROM assessments WHERE user_id = ? AND submitted_at >= ?", (user_id, since))}
        exams = {r["source_ref"] for r in conn.execute(
            "SELECT source_ref FROM assessments WHERE user_id = ? AND kind = 'exam' AND submitted_at >= ?", (user_id, since))}
        decks = {r["source_ref"] for r in conn.execute(
            "SELECT source_ref FROM decks WHERE user_id = ? AND created_at >= ?", (user_id, since))}
    for week in plan["weeks"]:
        # Drop items whose lesson has since been removed from the curriculum.
        week["items"] = [i for i in week["items"] if find_for(user_id, i["lesson_id"])]
        for item in week["items"]:
            path, lesson = find_for(user_id, item["lesson_id"])
            item |= {"title": lesson.title, "path_id": path.id, "path_title": path.title}
            item["done"] = {
                "lesson": stats.get(lesson.id, {}).get("status") == "completed",
                "quiz": lesson.id in quizzed,
                "flashcards": lesson.id in decks,
                "exam": path.id in exams,
            }[item["activity"]]
        week["done"] = all(i["done"] for i in week["items"])
    items = [i for w in plan["weeks"] for i in w["items"]]
    return {"id": row["id"], "created_at": row["created_at"], "weekly_minutes": row["weekly_minutes"],
            "companion_id": row["companion_id"], "overview": plan["overview"], "weeks": plan["weeks"],
            "done": sum(i["done"] for i in items), "total": len(items)}


@router.get("/study-plan")
def get_plan(user: dict = User):
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM study_plans WHERE user_id = ? ORDER BY id DESC LIMIT 1", (user["id"],)).fetchone()
    return _plan_view(dict(row), user["id"]) if row else None


@router.post("/study-plan", status_code=201)
async def create_plan(body: NewPlan, user: dict = User):
    available = user_paths(user["id"])
    unknown = [p for p in body.paths if p not in {a.id for a in available}]
    if unknown:
        raise HTTPException(422, f"Unknown learning path: {', '.join(unknown)}")
    teacher = teacher_for(user, body.companion_id)
    stats = _lesson_stats(user["id"])
    completed = {lid for lid, s in stats.items() if s.get("status") == "completed"}
    scores = {lid: s["best"] for lid, s in stats.items() if "best" in s}
    with db.connect() as conn:
        memories = [r["fact"] for r in conn.execute(
            "SELECT fact FROM memories WHERE user_id = ? AND companion_id = ?", (user["id"], teacher["id"]))]
    with usage.spend(user, "plan"):
        plan = await tutor.plan_study(companions.persona_for(teacher, user["id"]), user, memories, completed, scores,
                                      body.weekly_minutes, body.weeks,
                                      [p for p in available if not body.paths or p.id in body.paths])
    if plan is None or not any(w.items for w in plan.weeks):
        raise HTTPException(502, "Your companion couldn't build a plan right now. Try again.")
    plan.weeks = [w for w in plan.weeks if w.items]
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO study_plans (user_id, companion_id, weekly_minutes, plan, created_at) VALUES (?, ?, ?, ?, ?)",
                           (user["id"], teacher["id"], body.weekly_minutes, plan.model_dump_json(), db.now()))
        row = conn.execute("SELECT * FROM study_plans WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _plan_view(dict(row), user["id"])


# -------------------------------------------------------------------- progress


def _streak(days: set[str], today: date) -> int:
    # A streak survives until the end of the day after the last active day.
    d = today if today.isoformat() in days else today - timedelta(days=1)
    n = 0
    while d.isoformat() in days:
        n += 1
        d -= timedelta(days=1)
    return n


@router.get("/progress")
def progress(user: dict = User):
    uid = user["id"]
    stats = _lesson_stats(uid)
    meta = path_meta(uid)
    # Community paths count once the learner has started them; their own and curated paths always count.
    paths = [v for v in (_path_view(p, stats, meta) for p in user_paths(uid))
             if v.get("mine", True) or any(l["status"] != "not_started" for l in v["lessons"])]
    with db.connect() as conn:
        days = {r["day"] for r in conn.execute("SELECT day FROM activity WHERE user_id = ?", (uid,))}
        quiz = conn.execute("SELECT COUNT(*) AS n, AVG(100.0 * score / total) AS avg FROM assessments "
                            "WHERE user_id = ? AND kind = 'quiz' AND submitted_at IS NOT NULL", (uid,)).fetchone()
        exam = conn.execute("SELECT COUNT(*) AS n, MAX(100.0 * score / total) AS best FROM assessments "
                            "WHERE user_id = ? AND kind = 'exam' AND submitted_at IS NOT NULL", (uid,)).fetchone()
        cards = conn.execute(
            f"SELECT COUNT(*) AS total, COALESCE(SUM(c.due_at <= ?), 0) AS due, COALESCE(SUM(c.box >= {MASTERED_BOX}), 0) AS mastered "
            "FROM cards c JOIN decks d ON d.id = c.deck_id WHERE d.user_id = ?", (db.now(), uid)).fetchone()
        chats = conn.execute("SELECT COUNT(*) FROM messages WHERE user_id = ? AND role = 'user'", (uid,)).fetchone()[0]
    today = local_now(user).date()
    week = [(today - timedelta(days=i)).isoformat() for i in range(6, -1, -1)]
    needs_review = [
        {"lesson_id": lid, "title": find_for(uid, lid)[1].title, "best_score": s["best"]}
        for lid, s in stats.items() if find_for(uid, lid) and s.get("best") is not None and s["best"] < PASS_MARK
    ]
    next_lesson = next(({"lesson_id": l["id"], "title": l["title"], "path_title": p["title"]}
                        for p in paths for l in p["lessons"] if l["status"] != "completed"), None)
    return {
        "streak_days": _streak(days, today),
        "active_days": len(days),
        "last_7_days": [{"day": d, "active": d in days} for d in week],
        "lessons_completed": sum(p["completed"] for p in paths),
        "lessons_total": sum(p["total"] for p in paths),
        "quizzes_taken": quiz["n"], "quiz_average": round(quiz["avg"]) if quiz["avg"] is not None else None,
        "exams_taken": exam["n"], "best_exam": round(exam["best"]) if exam["best"] is not None else None,
        "cards_total": cards["total"], "cards_due": cards["due"], "cards_mastered": cards["mastered"],
        "questions_asked": chats,
        "paths": [{k: p[k] for k in ("id", "title", "completed", "total")} for p in paths],
        "needs_review": needs_review,
        "next_lesson": next_lesson,
    }
