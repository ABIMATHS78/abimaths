"""Staff tools: moderating reported community companions, second looks at removals the
designer appealed, expert-reviewing learner-made paths before they're shared, and
managing staff roles. Every decision is logged and the affected learner gets a notice
explaining it."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app import auth, db
from app.config import settings

router = APIRouter(prefix="/api/staff")
Moderator = Depends(auth.require("moderate"))
Reviewer = Depends(auth.require("review"))
Admin = Depends(auth.require("admin"))


def log(conn, staff_id: int, target_type: str, target_id: int, action: str, note: str = "") -> None:
    conn.execute("INSERT INTO moderation_log (staff_id, target_type, target_id, action, note, created_at) "
                 "VALUES (?, ?, ?, ?, ?, ?)", (staff_id, target_type, target_id, action, note, db.now()))


def notify(conn, user_id: int, text: str, link: str = "") -> None:
    conn.execute("INSERT INTO notices (user_id, text, link, created_at) VALUES (?, ?, ?, ?)", (user_id, text, link, db.now()))


# ------------------------------------------------------------------ companions


@router.get("/companions")
def reported_companions(user: dict = Moderator):
    """Community companions with open reports, most reported first. Hidden ones need a decision."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT c.*, u.display_name AS designer, COUNT(r.user_id) AS reports, "
            "GROUP_CONCAT(r.reason) AS reasons, MAX(r.created_at) AS last_report "
            "FROM companions c JOIN reports r ON r.companion_id = c.id LEFT JOIN users u ON u.id = c.owner_id "
            "WHERE c.removed = 0 GROUP BY c.id ORDER BY c.hidden DESC, reports DESC, last_report DESC"
        ).fetchall()
    out = []
    for r in rows:
        reasons: dict[str, int] = {}
        for reason in (r["reasons"] or "").split(","):
            reasons[reason] = reasons.get(reason, 0) + 1
        out.append(db.row_to_companion(r) | {"designer": r["designer"], "reports": r["reports"],
                                             "reasons": reasons, "last_report": r["last_report"]})
    return out


class CompanionDecision(BaseModel):
    action: Literal["restore", "dismiss", "remove"]
    note: str = Field("", max_length=500)


@router.post("/companions/{companion_id}")
def decide_companion(companion_id: int, body: CompanionDecision, user: dict = Moderator):
    """restore: unhide and clear reports. dismiss: clear reports on a still-visible companion.
    remove: take it out of the community for good (the designer keeps a private copy)."""
    with db.connect() as conn:
        c = conn.execute("SELECT * FROM companions WHERE id = ?", (companion_id,)).fetchone()
        if c is None or c["is_starter"] or c["removed"]:
            raise HTTPException(404, "Companion not found")
        if body.action == "remove" and not body.note.strip():
            raise HTTPException(422, "Explain to the designer why it was removed")
        name = json.loads(c["persona"])["name"]
        if body.action == "remove":
            conn.execute("UPDATE companions SET is_public = 0, hidden = 1, removed = 1 WHERE id = ?", (companion_id,))
            text = f"{name} was removed from the community after review: {body.note.strip()} It's still yours to use privately."
        else:
            conn.execute("UPDATE companions SET hidden = 0 WHERE id = ?", (companion_id,))
            text = (f"{name} was reviewed after reports and is back in the community." if c["hidden"]
                    else f"Reports about {name} were reviewed. No action was needed.")
        conn.execute("DELETE FROM reports WHERE companion_id = ?", (companion_id,))
        log(conn, user["id"], "companion", companion_id, body.action, body.note.strip())
        if c["owner_id"]:
            if body.action == "remove":
                text += " If you think it was a mistake, you can ask for a second look from its page."
            notify(conn, c["owner_id"], text, f"#/c/{companion_id}")
    return {"ok": True}


# --------------------------------------------------------------------- appeals


def latest_removal(conn, companion_id: int):
    return conn.execute("SELECT * FROM moderation_log WHERE target_type = 'companion' AND target_id = ? "
                        "AND action = 'remove' ORDER BY id DESC LIMIT 1", (companion_id,)).fetchone()


def moderation_for_owner(companion: dict) -> dict | None:
    """What the designer of a removed companion sees: why, and where their appeal stands."""
    if not companion["removed"]:
        return None
    with db.connect() as conn:
        removal = latest_removal(conn, companion["id"])
        appeal = conn.execute("SELECT status, message, decision_note, created_at, decided_at FROM appeals "
                              "WHERE removal_id = ?", (removal["id"],)).fetchone() if removal else None
    return {"removed_at": removal["created_at"] if removal else None, "note": removal["note"] if removal else "",
            "appeal": dict(appeal) if appeal else None, "can_appeal": removal is not None and appeal is None}


class AppealIn(BaseModel):
    message: str = Field(min_length=10, max_length=1000)


def appeal(user: dict, companion_id: int, body: AppealIn) -> dict:
    """The designer asks for a second look at a removal. One appeal per removal."""
    with db.connect() as conn:
        c = conn.execute("SELECT * FROM companions WHERE id = ?", (companion_id,)).fetchone()
        if c is None or c["owner_id"] != user["id"]:
            raise HTTPException(404, "Companion not found")
        removal = latest_removal(conn, companion_id) if c["removed"] else None
        if removal is None:
            raise HTTPException(409, "Only a companion a moderator removed from the community can be appealed")
        if conn.execute("SELECT 1 FROM appeals WHERE removal_id = ?", (removal["id"],)).fetchone():
            raise HTTPException(409, "You've already appealed this decision")
        conn.execute("INSERT INTO appeals (companion_id, user_id, removal_id, message, created_at) VALUES (?, ?, ?, ?, ?)",
                     (companion_id, user["id"], removal["id"], body.message.strip(), db.now()))
    return moderation_for_owner(db.row_to_companion(c))


@router.get("/appeals")
def appeals_queue(status: Literal["open", "upheld", "overturned"] = "open", user: dict = Moderator):
    """Appeals with the removal they're about, oldest first."""
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT a.*, l.staff_id AS removed_by, l.note AS removal_note, l.created_at AS removed_at, "
            "d.display_name AS designer, s.display_name AS removed_by_name, x.display_name AS decided_by_name "
            "FROM appeals a JOIN moderation_log l ON l.id = a.removal_id LEFT JOIN users d ON d.id = a.user_id "
            "LEFT JOIN users s ON s.id = l.staff_id LEFT JOIN users x ON x.id = a.decided_by "
            "WHERE a.status = ? ORDER BY a.id", (status,)).fetchall()
        found = {r["id"]: db.row_to_companion(r) for r in conn.execute(
            f"SELECT * FROM companions WHERE id IN ({','.join('?' * len(rows))})", [r["companion_id"] for r in rows])} if rows else {}
    return [{
        "id": r["id"], "status": r["status"], "message": r["message"], "created_at": r["created_at"],
        "decision_note": r["decision_note"], "decided_at": r["decided_at"], "decided_by": r["decided_by_name"],
        "designer": r["designer"], "companion": found[r["companion_id"]],
        "removal": {"note": r["removal_note"], "at": r["removed_at"], "by": r["removed_by_name"]},
        # A second look means someone other than the moderator who removed it.
        "can_decide": r["removed_by"] != user["id"],
    } for r in rows]


class AppealDecision(BaseModel):
    decision: Literal["overturn", "uphold"]
    note: str = Field("", max_length=500)


@router.post("/appeals/{appeal_id}")
def decide_appeal(appeal_id: int, body: AppealDecision, user: dict = Moderator):
    """overturn: put the companion back in the community. uphold: the removal stands, and the appeal is closed."""
    note = body.note.strip()
    with db.connect() as conn:
        a = conn.execute("SELECT a.*, l.staff_id AS removed_by FROM appeals a JOIN moderation_log l ON l.id = a.removal_id "
                         "WHERE a.id = ?", (appeal_id,)).fetchone()
        if a is None:
            raise HTTPException(404, "Appeal not found")
        if a["status"] != "open":
            raise HTTPException(409, f"This appeal was already decided ({a['status']})")
        if a["removed_by"] == user["id"]:
            raise HTTPException(403, "You removed this companion. Ask another moderator to take the second look.")
        if body.decision == "uphold" and not note:
            raise HTTPException(422, "Explain to the designer why the removal stands")
        c = conn.execute("SELECT * FROM companions WHERE id = ?", (a["companion_id"],)).fetchone()
        name = json.loads(c["persona"])["name"]
        if body.decision == "overturn":
            conn.execute("UPDATE companions SET removed = 0, hidden = 0, is_public = 1, published_at = ? WHERE id = ?",
                         (db.now(), c["id"]))
            text = f"Your appeal was accepted: {name} is back in the community." + (f" {note}" if note else "")
        else:
            text = f"Your appeal about {name} had a second look, and the removal stands: {note}"
        conn.execute("UPDATE appeals SET status = ?, decided_by = ?, decision_note = ?, decided_at = ? WHERE id = ?",
                     ("overturned" if body.decision == "overturn" else "upheld", user["id"], note, db.now(), appeal_id))
        log(conn, user["id"], "companion", c["id"], f"appeal:{body.decision}", note)
        notify(conn, a["user_id"], text, f"#/c/{c['id']}")
    return {"ok": True}


# ---------------------------------------------------------------------- paths


@router.get("/paths")
def review_queue(status: Literal["pending", "approved", "rejected"] = "pending", user: dict = Reviewer):
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT p.*, u.display_name AS owner_name FROM custom_paths p JOIN users u ON u.id = p.user_id "
            "WHERE p.status = ? ORDER BY COALESCE(p.submitted_at, p.created_at)", (status,)).fetchall()
    return [{"id": f"custom-{r['id']}", "owner_name": r["owner_name"], "status": r["status"],
             "submitted_at": r["submitted_at"], "reviewed_at": r["reviewed_at"], "review_note": r["review_note"],
             **json.loads(r["data"])} for r in rows]


class ReviewedLesson(BaseModel):
    title: str = Field(min_length=3, max_length=90)
    objectives: list[str] = Field(min_length=1, max_length=3)
    key_points: list[str] = Field(min_length=2, max_length=5)
    minutes: int = Field(10, ge=5, le=20)


class PathDecision(BaseModel):
    decision: Literal["approve", "reject", "unshare"]
    note: str = Field("", max_length=1000)
    title: str | None = Field(None, min_length=3, max_length=80)
    description: str | None = Field(None, max_length=400)
    # The reviewer's corrected lessons. The count must stay the same so learners' progress still lines up.
    lessons: list[ReviewedLesson] | None = None


def _path_row(conn, path_id: str):
    if not path_id.startswith("custom-") or not path_id[7:].isdigit():
        raise HTTPException(404, "Path not found")
    row = conn.execute("SELECT * FROM custom_paths WHERE id = ?", (int(path_id[7:]),)).fetchone()
    if row is None:
        raise HTTPException(404, "Path not found")
    return row


@router.post("/paths/{path_id}")
def decide_path(path_id: str, body: PathDecision, user: dict = Reviewer):
    with db.connect() as conn:
        row = _path_row(conn, path_id)
        data = json.loads(row["data"])
        expected = "approved" if body.decision == "unshare" else "pending"
        if row["status"] != expected:
            raise HTTPException(409, f"This path is {row['status']}, not {expected}")
        if body.decision != "approve" and not body.note.strip():
            raise HTTPException(422, "Explain the decision to the learner who made the path")
        if row["user_id"] == user["id"] and body.decision == "approve":
            raise HTTPException(403, "Ask another reviewer to approve your own path")
        if body.decision == "approve":
            if body.lessons is not None:
                if len(body.lessons) != len(data["lessons"]):
                    raise HTTPException(422, "Keep the same number of lessons so learners' progress still matches")
                data["lessons"] = [l.model_dump() | {"objectives": [o.strip() for o in l.objectives if o.strip()],
                                                     "key_points": [k.strip() for k in l.key_points if k.strip()]}
                                   for l in body.lessons]
            data["title"] = body.title or data["title"]
            data["description"] = body.description if body.description is not None else data["description"]
            status, text = "approved", f"Your path \"{data['title']}\" passed expert review and is now shared with every learner."
        elif body.decision == "reject":
            status, text = "rejected", f"Your path \"{data['title']}\" wasn't approved for sharing: {body.note.strip()}"
        else:
            status, text = "private", f"Your path \"{data['title']}\" is no longer shared: {body.note.strip()}"
        conn.execute("UPDATE custom_paths SET status = ?, data = ?, review_note = ?, reviewed_by = ?, reviewed_at = ? WHERE id = ?",
                     (status, json.dumps(data), body.note.strip(), user["id"], db.now(), row["id"]))
        if body.decision == "approve" and body.lessons is not None:
            # Cached lesson text was written from the old key points; let companions re-teach from the corrected ones.
            ids = [f"{path_id}-{i + 1}" for i in range(len(data["lessons"]))]
            conn.execute(f"DELETE FROM lesson_content WHERE lesson_id IN ({','.join('?' * len(ids))})", ids)
        log(conn, user["id"], "path", row["id"], body.decision, body.note.strip())
        notify(conn, row["user_id"], text, "#/learn")
    return {"ok": True, "status": status}


# ---------------------------------------------------------------------- roles


@router.get("/users")
def find_users(q: str = "", user: dict = Admin):
    like = f"%{q.strip().lower()}%"
    with db.connect() as conn:
        rows = conn.execute("SELECT * FROM users WHERE lower(email) LIKE ? OR lower(display_name) LIKE ? "
                            "ORDER BY (role != 'learner') DESC, id LIMIT 50", (like, like)).fetchall()
    return [{"id": r["id"], "email": r["email"], "display_name": r["display_name"], "role": auth.role_of(dict(r)),
             "fixed": r["email"].lower() in settings.admin_emails} for r in rows]


class RoleChange(BaseModel):
    role: Literal["learner", "reviewer", "moderator", "admin"]


@router.post("/users/{user_id}/role")
def set_role(user_id: int, body: RoleChange, user: dict = Admin):
    with db.connect() as conn:
        target = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
        if target is None:
            raise HTTPException(404, "User not found")
        if target["email"].lower() in settings.admin_emails:
            raise HTTPException(409, "This admin is set in the server configuration (AIET_ADMIN_EMAILS)")
        if user_id == user["id"]:
            raise HTTPException(409, "Ask another admin to change your own role")
        conn.execute("UPDATE users SET role = ? WHERE id = ?", (body.role, user_id))
        log(conn, user["id"], "user", user_id, f"role:{body.role}")
    return {"id": user_id, "role": body.role}


@router.get("/log")
def decisions(user: dict = Depends(auth.current_user)):
    if auth.role_of(user) == "learner":
        raise HTTPException(403, "You don't have permission to do that")
    with db.connect() as conn:
        rows = conn.execute("SELECT l.*, u.display_name AS staff_name FROM moderation_log l "
                            "LEFT JOIN users u ON u.id = l.staff_id ORDER BY l.id DESC LIMIT 100").fetchall()
    return [dict(r) for r in rows]
