"""Courses: the catalogue (app/catalog.py), enrolling, each learner's progress through a course, their labs and
capstone projects reviewed by the course's instructors, certificates (with a public page anyone can check them
on), and ratings.

A course's lessons, module quizzes, mock exam and flashcards go through the learning API (app/learning.py), since
each module is a learning path: this module adds what only courses have. A learner finishes a course by completing
every lesson and passing every module quiz (70%), every lab and the capstone project (70 of its 100 points); a
certificate is then theirs for the asking, and one for a track once they have every course's in it.
"""

from __future__ import annotations

import io
import json
import re
import secrets
import zipfile

from fastapi import APIRouter, Depends, HTTPException, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app import auth, billing, catalog, companions, db, learning, paths, tutor, usage
from app.catalog import Course, Track
from app.config import settings

router = APIRouter(prefix="/api")
links = APIRouter()  # certificates' short links, outside /api
User = Depends(auth.current_user)
Visitor = Depends(auth.optional_user)  # the catalogue, a course's page and a track's are open to everyone

PASS_MARK = learning.PASS_MARK
MAX_SUBMISSION = 40000
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"  # no 0/O or 1/I to misread


def forget_changed_lessons() -> None:
    """When the app starts: forget the instructors' written text of any course lesson whose objectives, key points
    or walkthrough changed, or that's gone, so it's written again from what the course says now."""
    now = catalog.lesson_versions()
    with db.connect() as conn:
        row = conn.execute("SELECT value FROM meta WHERE key = 'course_lessons'").fetchone()
        before = json.loads(row["value"]) if row else {}
        stale = [lid for lid, v in before.items() if now.get(lid) != v]
        for i in range(0, len(stale), 500):
            part = stale[i:i + 500]
            conn.execute(f"DELETE FROM lesson_content WHERE lesson_id IN ({','.join('?' * len(part))})", part)
        conn.execute("INSERT INTO meta (key, value) VALUES ('course_lessons', ?) "
                     "ON CONFLICT (key) DO UPDATE SET value = excluded.value", (json.dumps(now),))


def _course_or_404(slug: str) -> Course:
    c = catalog.course(slug)
    if c is None:
        raise HTTPException(404, "Course not found")
    return c


def _track_or_404(slug: str) -> Track:
    t = catalog.data().tracks_by_slug.get(slug)
    if t is None:
        raise HTTPException(404, "Track not found")
    return t


# ------------------------------------------------------------------ instructors


def _instructors(slugs: set[str]) -> dict[str, dict]:
    """The starter companions with these slugs, as course pages show them. Retired ones are left out."""
    if not slugs:
        return {}
    with db.connect() as conn:
        rows = conn.execute(f"SELECT * FROM companions WHERE is_starter = 1 AND retired = 0 AND slug IN "
                            f"({','.join('?' * len(slugs))})", sorted(slugs)).fetchall()
    out = {}
    for r in rows:
        c = db.row_to_companion(r)
        p = c["persona"]
        out[c["slug"]] = {"id": c["id"], "slug": c["slug"], "name": p["name"], "flag": p["flag"], "country": p["country"],
                          "tagline": p["tagline"], "bio": p["bio"], "expertise": p["expertise"], "updated_at": c["updated_at"]}
    return out


def _brief_instructor(i: dict) -> dict:
    return {k: i[k] for k in ("id", "slug", "name", "flag", "updated_at")}


def _teacher(c: Course, m: int) -> dict:
    """The companion who reviews a module's lab, or the project (m = 0): its instructor, while there is one."""
    found = companions.starter(c.instructor_of(m) if m else c.instructors[0])
    if found is None:
        found = next((x for x in map(companions.starter, c.instructors) if x), None)
    if found is None:
        raise HTTPException(503, "This course's instructors aren't available right now.")
    return found


# ------------------------------------------------------------- learner's state


class State:
    """What a learner has done in every course, read in a few queries."""

    def __init__(self, user_id: int | None):  # None: a visitor who isn't signed in, who has done nothing yet
        self.stats = learning._lesson_stats(user_id)
        with db.connect() as conn:
            self.enrolled = {r["course_slug"]: r["enrolled_at"] for r in conn.execute(
                "SELECT course_slug, enrolled_at FROM course_enrollments WHERE user_id = ?", (user_id,))}
            self.quizzes = {r["source_ref"]: r["best"] for r in conn.execute(
                "SELECT source_ref, MAX(100.0 * score / total) AS best FROM assessments WHERE user_id = ? AND kind = 'quiz' "
                "AND source_type = 'path' AND source_ref LIKE 'course-%' AND submitted_at IS NOT NULL GROUP BY source_ref",
                (user_id,))}
            self.work: dict[tuple[str, str], dict] = {}  # (course, item): passed ever, latest attempt, best score
            for r in conn.execute("SELECT course_slug, item, passed, score, created_at FROM course_submissions "
                                  "WHERE user_id = ? ORDER BY id", (user_id,)):
                w = self.work.setdefault((r["course_slug"], r["item"]), {"passed": False, "best": None, "attempts": 0})
                w["passed"] = w["passed"] or bool(r["passed"])
                w["attempts"] += 1
                w["latest"] = r["created_at"]
                if r["score"] is not None:
                    w["best"] = max(w["best"] or 0, r["score"])
            self.certificates = {(r["kind"], r["ref"]): dict(r) for r in conn.execute(
                "SELECT * FROM certificates WHERE user_id = ?", (user_id,))}

    def lesson(self, lesson_id: str) -> str:
        return self.stats.get(lesson_id, {}).get("status", "not_started")

    def quiz(self, module_id: str) -> tuple[str, int | None]:
        best = self.quizzes.get(module_id)
        if best is None:
            return "not_started", None
        return ("passed" if best >= PASS_MARK else "attempted"), round(best)

    def submitted(self, slug: str, item: str) -> tuple[str, int | None]:
        w = self.work.get((slug, item))
        if w is None:
            return "not_started", None
        return ("passed" if w["passed"] else "needs_work"), w["best"]

    def certificate(self, kind: str, ref: str) -> dict | None:
        return self.certificates.get((kind, ref))


def _outline(c: Course, st: State) -> tuple[list[dict], dict | None, dict, str | None]:
    """The course's modules with every item's status, its project's, what's left to finish it, and the first item
    not done."""
    modules, done, total, next_key = [], 0, 0, None
    req = {"lessons": [0, 0], "quizzes": [0, 0], "labs": [0, 0], "project": None}

    def tally(kind: str, key: str, ok: bool) -> None:
        nonlocal done, total, next_key
        total += 1
        done += ok
        req[kind][0] += ok
        req[kind][1] += 1
        if not ok and next_key is None:
            next_key = key

    for m, module in enumerate(c.modules, 1):
        items = []
        for n, lesson in enumerate(module.lessons, 1):
            lid = catalog.lesson_id(c.slug, m, n)
            status = st.lesson(lid)
            items.append({"key": f"l{m}-{n}", "type": "lesson", "id": lid, "title": lesson.title, "minutes": lesson.minutes,
                          "walkthrough": lesson.walkthrough.tool if lesson.walkthrough else None,
                          "video": bool(lesson.video), "status": status})
            tally("lessons", f"l{m}-{n}", status == "completed")
        if module.lab:
            status, _ = st.submitted(c.slug, f"lab-{m}")
            items.append({"key": f"lab{m}", "type": "lab", "id": catalog.lab_id(c.slug, m), "title": module.lab.title,
                          "minutes": module.lab.minutes, "tool": module.lab.tool, "status": status})
            tally("labs", f"lab{m}", status == "passed")
        status, best = st.quiz(catalog.module_id(c.slug, m))
        items.append({"key": f"quiz{m}", "type": "quiz", "id": catalog.module_id(c.slug, m), "title": f"Module {m} quiz",
                      "minutes": catalog.QUIZ_MINUTES, "status": status, "best_score": best,
                      "questions": learning.MODULE_QUIZ_QUESTIONS})
        tally("quizzes", f"quiz{m}", status == "passed")
        modules.append({"index": m, "title": module.title, "summary": module.summary, "minutes": module.minutes,
                        "instructor": c.instructor_of(m), "items": items})
    project = None
    if c.project:
        status, best = st.submitted(c.slug, "project")
        project = {"key": "project", "type": "project", "id": catalog.project_id(c.slug), "title": c.project.title,
                   "minutes": c.project.minutes, "status": status, "best_score": best}
        total += 1
        done += status == "passed"
        req["project"] = status == "passed"
        if status != "passed" and next_key is None:
            next_key = "project"
    requirements = {
        "lessons": {"done": req["lessons"][0], "total": req["lessons"][1]},
        "quizzes": {"done": req["quizzes"][0], "total": req["quizzes"][1]},
        "labs": {"done": req["labs"][0], "total": req["labs"][1]},
        "project": req["project"],
        "percent": round(100 * done / total) if total else 0,
        "complete": done == total,
    }
    return modules, project, requirements, next_key


def _summary(c: Course, st: State | None, people: dict[str, dict], counts: dict[str, int],
             ratings: dict[str, tuple[float, int]]) -> dict:
    """A course as the catalogue lists it."""
    rating = ratings.get(c.slug)
    out = {
        "slug": c.slug, "title": c.title, "subtitle": c.subtitle, "category": c.category, "kind": c.kind,
        "level": c.level, "hours": c.hours, "modules": len(c.modules),
        "lessons": sum(len(m.lessons) for m in c.modules), "labs": len(c.labs), "project": c.project is not None,
        "walkthroughs": sum(l.walkthrough is not None for m in c.modules for l in m.lessons),
        "skills": c.skills, "tools": c.tools, "jetbrains": c.jetbrains, "cisco": c.cisco, "aligned_with": c.aligned_with,
        "instructors": [_brief_instructor(people[s]) for s in c.instructors if s in people],
        "learners": counts.get(c.slug, 0),
        "rating": {"average": round(rating[0], 1), "count": rating[1]} if rating else None,
        "tracks": [{"slug": t.slug, "title": t.title, "kind": t.kind} for t in catalog.data().tracks_of.get(c.slug, [])],
    }
    if st is not None:
        _, _, req, next_key = _outline(c, st)
        cert = st.certificate("course", c.slug)
        out |= {"enrolled": c.slug in st.enrolled, "enrolled_at": st.enrolled.get(c.slug), "progress": req["percent"],
                "completed": req["complete"], "next": next_key, "certificate": cert["code"] if cert else None}
    return out


def _counts() -> tuple[dict[str, int], dict[str, tuple[float, int]]]:
    with db.connect() as conn:
        counts = {r[0]: r[1] for r in conn.execute("SELECT course_slug, COUNT(*) FROM course_enrollments GROUP BY course_slug")}
        ratings = {r[0]: (r[1], r[2]) for r in conn.execute(
            "SELECT course_slug, AVG(stars), COUNT(*) FROM course_ratings GROUP BY course_slug")}
    return counts, ratings


def _all_instructors() -> dict[str, dict]:
    return _instructors({s for c in catalog.data().courses for s in c.instructors})


def _track_view(t: Track, summaries: dict[str, dict], st: State) -> dict:
    courses = [summaries[s] for s in t.courses if s in summaries]
    done = sum(bool(s.get("certificate")) for s in courses)
    cert = st.certificate("track", t.slug)
    people: dict[int, dict] = {}
    for s in courses:
        for i in s["instructors"]:
            people.setdefault(i["id"], i)
    return {"slug": t.slug, "title": t.title, "kind": t.kind, "subtitle": t.subtitle, "description": t.description,
            "roles": t.roles, "courses": courses, "hours": sum(s["hours"] for s in courses), "done": done,
            "total": len(courses), "instructors": list(people.values()),
            "levels": sorted({s["level"] for s in courses}, key=catalog.LEVELS.index),
            "skills": list(dict.fromkeys(k for s in courses for k in s["skills"]))[:12],
            "certificate": cert["code"] if cert else None}


# ------------------------------------------------------------------ catalogue


@router.get("/courses")
def list_courses(user: dict | None = Visitor):
    """Every course, as the catalogue lists it, with the learner's progress in it; the subjects; and the tracks."""
    cat, st = catalog.data(), State(user["id"] if user else None)
    people, (counts, ratings) = _all_instructors(), _counts()
    summaries = {c.slug: _summary(c, st, people, counts, ratings) for c in cat.courses}
    categories = [{"name": name, "count": sum(c.category == name for c in cat.courses)} for name in catalog.CATEGORIES]
    return {"courses": list(summaries.values()), "categories": [c for c in categories if c["count"]],
            "tracks": [_track_view(t, summaries, st) for t in cat.tracks],
            "full_access": billing.unlocked(user, "courses"), "free_modules": billing.FREE_MODULES}


@router.get("/courses/{slug}")
def get_course(slug: str, user: dict | None = Visitor):
    """A course's page: what it covers, who teaches it, its syllabus with the learner's progress, and what's left.
    For a visitor who isn't signed in, as it is for a new learner."""
    c = _course_or_404(slug)
    uid = user["id"] if user else None
    st, people, (counts, ratings) = State(uid), _instructors(set(c.instructors)), _counts()
    modules, project, requirements, next_key = _outline(c, st)
    for m in modules:
        m["instructor"] = _brief_instructor(people[m["instructor"]]) if m["instructor"] in people else None
    taught = {s: sum(s in x.instructors for x in catalog.data().courses) for s in c.instructors}
    with db.connect() as conn:
        mine = conn.execute("SELECT stars FROM course_ratings WHERE user_id = ? AND course_slug = ?",
                            (uid, slug)).fetchone()
    cert = st.certificate("course", slug)
    full = billing.unlocked(user, "courses")
    for m in modules:
        m["locked"] = not full and m["index"] > billing.FREE_MODULES
    return _summary(c, st, people, counts, ratings) | {
        "access": {"full": full, "free_modules": billing.FREE_MODULES,
                   "plan": None if full else billing.plans.lowest_with("courses").name},
        "description": c.description, "outcomes": c.outcomes, "prerequisites": c.prerequisites,
        "notices": c.notices(), "syllabus": modules, "project": project, "requirements": requirements, "next": next_key,
        "study_plan": c.plan(), "mock_exam": {"questions": learning.COURSE_EXAM_QUESTIONS, "minutes": learning.COURSE_EXAM_MINUTES},
        "instructors": [people[s] | {"courses": taught[s]} for s in c.instructors if s in people],
        "certificate": cert["code"] if cert else None, "my_rating": mine["stars"] if mine else None,
    }


@router.post("/courses/{slug}/enroll", status_code=204)
def enroll(slug: str, user: dict = User):
    _course_or_404(slug)
    with db.connect() as conn:
        conn.execute("INSERT OR IGNORE INTO course_enrollments (user_id, course_slug, enrolled_at) VALUES (?, ?, ?)",
                     (user["id"], slug, db.now()))
    learning.record_activity(user)


@router.delete("/courses/{slug}/enroll", status_code=204)
def leave(slug: str, user: dict = User):
    """Take the course off the learner's list. Their progress stays, for if they come back."""
    _course_or_404(slug)
    with db.connect() as conn:
        conn.execute("DELETE FROM course_enrollments WHERE user_id = ? AND course_slug = ?", (user["id"], slug))


def _require_enrolled(user: dict, slug: str) -> None:
    with db.connect() as conn:
        if conn.execute("SELECT 1 FROM course_enrollments WHERE user_id = ? AND course_slug = ?",
                        (user["id"], slug)).fetchone() is None:
            raise HTTPException(409, "Enrol in the course first. It's free.")


class RatingIn(BaseModel):
    stars: int = Field(ge=1, le=5)


@router.put("/courses/{slug}/rating", status_code=204)
def rate(slug: str, body: RatingIn, user: dict = User):
    _course_or_404(slug)
    _require_enrolled(user, slug)
    with db.connect() as conn:
        conn.execute("INSERT INTO course_ratings (user_id, course_slug, stars, updated_at) VALUES (?, ?, ?, ?) "
                     "ON CONFLICT (user_id, course_slug) DO UPDATE SET stars = excluded.stars, updated_at = excluded.updated_at",
                     (user["id"], slug, body.stars, db.now()))


# ------------------------------------------------------------- labs & project


def _attempts(user_id: int, slug: str, item: str) -> list[dict]:
    with db.connect() as conn:
        rows = conn.execute("SELECT id, companion_id, content, passed, score, review, created_at FROM course_submissions "
                            "WHERE user_id = ? AND course_slug = ? AND item = ? ORDER BY id DESC LIMIT 20",
                            (user_id, slug, item)).fetchall()
    return [dict(r) | {"passed": bool(r["passed"]), "review": json.loads(r["review"])} for r in rows]


def _lab_or_404(c: Course, m: int):
    if not 1 <= m <= len(c.modules) or c.modules[m - 1].lab is None:
        raise HTTPException(404, "Lab not found")
    return c.modules[m - 1].lab


def _steps(steps) -> list[dict]:
    return [s.model_dump() for s in steps]


@router.get("/courses/{slug}/labs/{m}")
def get_lab(slug: str, m: int, user: dict = User):
    c = _course_or_404(slug)
    lab = _lab_or_404(c, m)
    billing.require_course_item(user, catalog.lab_id(slug, m))
    people = _instructors({c.instructor_of(m)})
    return {
        "course": {"slug": c.slug, "title": c.title}, "module": m, "module_title": c.modules[m - 1].title,
        "id": catalog.lab_id(slug, m), "title": lab.title, "tool": lab.tool, "minutes": lab.minutes, "goal": lab.goal,
        "setup": lab.setup, "steps": _steps(lab.steps), "checks": [ch.text for ch in lab.checks], "submit": lab.submit,
        "hints": lab.hints, "files": [f.path for f in lab.files],
        "instructor": _brief_instructor(people[c.instructor_of(m)]) if c.instructor_of(m) in people else None,
        "attempts": _attempts(user["id"], slug, f"lab-{m}"),
    }


class WorkIn(BaseModel):
    content: str = Field(min_length=1, max_length=MAX_SUBMISSION)


def _save(user: dict, slug: str, item: str, teacher: dict, content: str, review: dict, passed: bool, score: int | None) -> dict:
    with db.connect() as conn:
        cur = conn.execute("INSERT INTO course_submissions (user_id, course_slug, item, companion_id, content, passed, score, "
                           "review, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                           (user["id"], slug, item, teacher["id"], content, int(passed), score, json.dumps(review), db.now()))
    learning.record_activity(user)
    return _attempts(user["id"], slug, item)[0] | {"id": cur.lastrowid}


@router.post("/courses/{slug}/labs/{m}/submit", status_code=201)
async def submit_lab(slug: str, m: int, body: WorkIn, user: dict = User):
    """The learner's lab work, reviewed by the module's instructor against the lab's checks."""
    c = _course_or_404(slug)
    _lab_or_404(c, m)
    billing.require_course_item(user, catalog.lab_id(slug, m))
    _require_enrolled(user, slug)
    if not body.content.strip():
        raise HTTPException(422, "Paste in your work first.")
    teacher = _teacher(c, m)
    with usage.spend(user, "lab"):
        review = await tutor.review_lab(companions.persona_for(teacher, user["id"]), user, c, m, body.content)
    if review is None:
        raise HTTPException(502, "Your instructor couldn't review your lab right now. Try again.")
    return _save(user, slug, f"lab-{m}", teacher, body.content, review, review["passed"], None)


@router.get("/courses/{slug}/project")
def get_project(slug: str, user: dict = User):
    c = _course_or_404(slug)
    if c.project is None:
        raise HTTPException(404, "This course has no project")
    billing.require_course_item(user, catalog.project_id(slug))
    p = c.project
    people = _instructors({c.instructors[0]})
    return {
        "course": {"slug": c.slug, "title": c.title}, "id": catalog.project_id(slug), "title": p.title,
        "minutes": p.minutes, "brief": p.brief, "deliverables": p.deliverables,
        "rubric": [r.model_dump() for r in p.rubric], "submit": p.submit, "files": [f.path for f in p.files],
        "pass_mark": PASS_MARK,
        "instructor": _brief_instructor(people[c.instructors[0]]) if c.instructors[0] in people else None,
        "attempts": _attempts(user["id"], slug, "project"),
    }


@router.post("/courses/{slug}/project/submit", status_code=201)
async def submit_project(slug: str, body: WorkIn, user: dict = User):
    """The learner's capstone project, scored by the course's lead instructor against its rubric."""
    c = _course_or_404(slug)
    if c.project is None:
        raise HTTPException(404, "This course has no project")
    billing.require_course_item(user, catalog.project_id(slug))
    _require_enrolled(user, slug)
    if not body.content.strip():
        raise HTTPException(422, "Paste in your project first.")
    teacher = _teacher(c, 0)
    with usage.spend(user, "project"):
        review = await tutor.review_project(companions.persona_for(teacher, user["id"]), user, c, body.content)
    if review is None:
        raise HTTPException(502, "Your instructor couldn't review your project right now. Try again.")
    return _save(user, slug, "project", teacher, body.content, review, review["score"] >= PASS_MARK, review["score"])


def _zip(name: str, files) -> Response:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.writestr(f"{name}/{f.path}", f.content)
    return Response(buf.getvalue(), media_type="application/zip",
                    headers={"Content-Disposition": f'attachment; filename="{name}.zip"'})


@router.get("/courses/{slug}/labs/{m}/files")
def lab_files(slug: str, m: int, user: dict = User):
    lab = _lab_or_404(_course_or_404(slug), m)
    billing.require_course_item(user, catalog.lab_id(slug, m))
    if not lab.files:
        raise HTTPException(404, "This lab has no starter files")
    return _zip(f"{slug}-lab-{m}", lab.files)


@router.get("/courses/{slug}/project/files")
def project_files(slug: str, user: dict = User):
    c = _course_or_404(slug)
    if c.project is None or not c.project.files:
        raise HTTPException(404, "This project has no starter files")
    billing.require_course_item(user, catalog.project_id(slug))
    return _zip(f"{slug}-project", c.project.files)


# ---------------------------------------------------------------- certificates


class CertificateIn(BaseModel):
    name: str = Field("", max_length=80)  # the name to put on it, if not the learner's display name


def _name(user: dict, body: CertificateIn) -> str:
    name = re.sub(r"\s+", " ", body.name).strip() or user["display_name"]
    if len(name) < 2:
        raise HTTPException(422, "Put your name on the certificate.")
    return name


def _issue(user: dict, kind: str, ref: str, name: str) -> dict:
    """Make out the certificate, or change the name on one already made out (its code stays the same)."""
    with db.connect() as conn:
        row = conn.execute("SELECT code FROM certificates WHERE user_id = ? AND kind = ? AND ref = ?",
                           (user["id"], kind, ref)).fetchone()
        if row:
            conn.execute("UPDATE certificates SET learner_name = ? WHERE code = ?", (name, row["code"]))
            code = row["code"]
        else:
            code = "".join(secrets.choice(CODE_ALPHABET) for _ in range(12))
            conn.execute("INSERT INTO certificates (code, user_id, kind, ref, learner_name, issued_at) VALUES (?, ?, ?, ?, ?, ?)",
                         (code, user["id"], kind, ref, name, db.now()))
    return certificate(code)


@router.post("/courses/{slug}/certificate")
def course_certificate(slug: str, body: CertificateIn, user: dict = User):
    c = _course_or_404(slug)
    billing.require(user, "courses")
    _, _, req, _ = _outline(c, State(user["id"]))
    if not req["complete"]:
        raise HTTPException(409, "Finish every lesson and pass every quiz, lab and the project first.")
    return _issue(user, "course", slug, _name(user, body))


@router.post("/tracks/{slug}/certificate")
def track_certificate(slug: str, body: CertificateIn, user: dict = User):
    t = _track_or_404(slug)
    billing.require(user, "courses")
    st = State(user["id"])
    if any(st.certificate("course", s) is None for s in t.courses):
        raise HTTPException(409, f"Earn the certificate for every course in the {t.kind} first.")
    return _issue(user, "track", slug, _name(user, body))


@router.get("/certificates/{code}")
def certificate(code: str):
    """A certificate, for anyone with its code: whom it was made out to, for what, when, and who taught it."""
    with db.connect() as conn:
        row = conn.execute("SELECT * FROM certificates WHERE code = ?", (code.upper(),)).fetchone()
    if row is None:
        raise HTTPException(404, "There's no certificate with that code.")
    if row["kind"] == "course":
        c = catalog.course(row["ref"])
        if c is None:
            raise HTTPException(404, "There's no certificate with that code.")
        slugs, what = c.instructors, {"title": c.title, "slug": c.slug, "hours": c.hours, "level": c.level,
                                      "skills": c.skills, "kind": "Guided project" if c.kind == "guided-project" else "Course",
                                      # what finishing it took: every one of these
                                      "lessons": sum(len(m.lessons) for m in c.modules), "quizzes": len(c.modules),
                                      "labs": sum(m.lab is not None for m in c.modules), "project": c.project is not None}
    else:
        t = catalog.data().tracks_by_slug.get(row["ref"])
        if t is None:
            raise HTTPException(404, "There's no certificate with that code.")
        courses = [catalog.course(s) for s in t.courses]
        slugs = list(dict.fromkeys(s for c in courses for s in c.instructors))
        what = {"title": t.title, "slug": t.slug, "kind": t.kind, "hours": sum(c.hours for c in courses),
                "courses": [c.title for c in courses], "skills": list(dict.fromkeys(k for c in courses for k in c.skills))[:12]}
    people = _instructors(set(slugs))
    return {"code": row["code"], "kind": row["kind"], "name": row["learner_name"], "issued_at": row["issued_at"],
            "url": f"{settings.public_url}/c/{row['code']}",  # the address printed on it, for checking it
            "for": what, "instructors": [_brief_instructor(people[s]) for s in slugs if s in people][:6]}


@links.get("/c/{code}", include_in_schema=False)
def certificate_link(code: str):
    """A certificate's short link, the one printed on it: its page, whose head says who earned what, for the
    previews of shared links (LinkedIn reads them; app/pages.py)."""
    return RedirectResponse(paths.certificate(re.sub(r"[^A-Za-z0-9]", "", code)[:24].upper()), status_code=301)


# ---------------------------------------------------------------------- tracks


@router.get("/tracks/{slug}")
def get_track(slug: str, user: dict | None = Visitor):
    t = _track_or_404(slug)
    st, people, (counts, ratings) = State(user["id"] if user else None), _all_instructors(), _counts()
    summaries = {s: _summary(catalog.course(s), st, people, counts, ratings) for s in t.courses}
    return _track_view(t, summaries, st)
