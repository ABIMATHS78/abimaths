"""The course catalogue: courses in the style of Coursera's, each taught by one to three of the starter AI
Learning Companions.

Each course is a JSON file in app/course_data/, named after its slug, and tracks.json there groups courses
into Professional Certificates and Specializations. They're checked against the models below when the app
starts, and a file that doesn't fit stops it, with the reason (tests/test_courses.py checks them too).

A course has modules, taken in order. A module has lessons, may end with a hands-on lab, and always ends with
a quiz on its lessons; a course may end with a capstone project. Labs and projects are reviewed by the
course's instructors, against the lab's checks and the project's rubric.

Lessons are taught the way the learning paths' lessons are (app.curriculum): the instructor writes each one in
their own voice from its objectives and vetted key points. So each module is also a LearningPath, and its
lessons are Lessons, with ids that quizzes, mock exams, flashcards, summaries and chat already understand:

    module   course-{slug}-{m}          lesson    course-{slug}-{m}-{n}
    lab      course-{slug}-{m}-lab      project   course-{slug}-project

(m and n count from 1). A lesson can also carry a walkthrough: a tool shown step by step, with its keyboard
shortcuts, which learners see beside the lesson exactly as written here, and a recorded video.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.curriculum import LearningPath, Lesson

DATA = Path(__file__).with_name("course_data")

# Subjects, in the order the catalogue lists them.
CATEGORIES = (
    "Generative AI", "Machine Learning", "Data Science", "Software Development", "Networking", "Cybersecurity",
    "Cloud & DevOps", "AI for Business", "Responsible AI", "AI in Industry", "Robotics & Edge AI",
)
Category = Literal[
    "Generative AI", "Machine Learning", "Data Science", "Software Development", "Networking", "Cybersecurity",
    "Cloud & DevOps", "AI for Business", "Responsible AI", "AI in Industry", "Robotics & Edge AI",
]
LEVELS = ("beginner", "intermediate", "advanced")

# JetBrains' products, as a course's tools name them. A course using one is listed with the JetBrains IDE courses.
JETBRAINS = frozenset({
    "PyCharm", "IntelliJ IDEA", "WebStorm", "PhpStorm", "GoLand", "Rider", "CLion", "RustRover", "DataGrip",
    "DataSpell", "RubyMine", "Fleet", "Aqua", "JetBrains AI Assistant", "Junie", "TeamCity", "Qodana", "YouTrack",
    "Kotlin Notebook", "JetBrains Academy plugin", "IDE Features Trainer",
})

QUIZ_MINUTES = 10  # what a module's quiz adds to the time a course takes
SLUG = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
# Videos play from YouTube's privacy-enhanced address or Vimeo's player.
VIDEO = re.compile(r"^https://(?:www\.youtube-nocookie\.com/embed/|player\.vimeo\.com/video/)[\w-]+(?:\?[\w=&-]*)?$")
FILE_PATH = re.compile(r"^[\w-]+(?:\.[\w-]+)*(?:/[\w-]+(?:\.[\w-]+)*)*$")


class Content(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, frozen=True)


Text = Field(min_length=2)


def _texts(items: list[str], most: int, what: str) -> list[str]:
    for t in items:
        if not t or len(t) > most:
            raise ValueError(f"each {what} needs 1-{most} characters: {t[:60]!r}")
    return items


class Keys(Content):
    """A keyboard shortcut in the IDE's default keymaps: macOS, and Windows and Linux."""

    mac: str = Field(min_length=1, max_length=40)
    win: str = Field(min_length=1, max_length=40)


class Step(Content):
    """One thing to do: in a lesson's walkthrough, or a lab."""

    do: str = Field(min_length=3, max_length=500)
    keys: Keys | None = None  # the shortcut for it, when there is one
    code: str = Field("", max_length=6000)  # code, commands or configuration to type or run
    lang: str = Field("", max_length=20)  # what the code is (python, bash, cisco, sql…), for the label on it
    see: str = Field("", max_length=400)  # what should happen


class Walkthrough(Content):
    """A tool shown step by step beside a lesson, as a screencast would show it."""

    tool: str = Field(min_length=2, max_length=60)
    title: str = Field(min_length=3, max_length=100)
    steps: list[Step] = Field(min_length=2, max_length=14)


class CourseLesson(Content):
    title: str = Field(min_length=3, max_length=100)
    minutes: int = Field(ge=3, le=30)
    objectives: list[str] = Field(min_length=1, max_length=4)
    key_points: list[str] = Field(min_length=3, max_length=6)
    walkthrough: Walkthrough | None = None
    video: str = ""  # a recording of the lesson: a YouTube (youtube-nocookie.com/embed/…) or Vimeo player address

    @field_validator("objectives")
    @classmethod
    def _objectives(cls, v: list[str]) -> list[str]:
        return _texts(v, 220, "objective")

    @field_validator("key_points")
    @classmethod
    def _key_points(cls, v: list[str]) -> list[str]:
        return _texts(v, 360, "key point")

    @field_validator("video")
    @classmethod
    def _video(cls, v: str) -> str:
        if v and not VIDEO.match(v):
            raise ValueError("video must be a https://www.youtube-nocookie.com/embed/… or https://player.vimeo.com/video/… address")
        return v


class Check(Content):
    """Something a lab's submission must show. look_for: short pieces of text a correct submission contains,
    which decide the check in demo mode (without a Claude key) and guide the instructor otherwise."""

    text: str = Field(min_length=5, max_length=320)
    look_for: list[str] = Field(default_factory=list, max_length=5)


class StarterFile(Content):
    """A file learners download to start a lab or project from."""

    path: str = Field(min_length=1, max_length=120)
    content: str = Field(min_length=1, max_length=30000)

    @field_validator("path")
    @classmethod
    def _path(cls, v: str) -> str:
        if not FILE_PATH.match(v):
            raise ValueError(f"starter file path {v!r}: use letters, digits, - _ . and / only")
        return v


class Lab(Content):
    title: str = Field(min_length=3, max_length=100)
    tool: str = Field(min_length=2, max_length=60)
    minutes: int = Field(ge=10, le=240)
    goal: str = Field(min_length=10, max_length=700)
    setup: list[str] = Field(default_factory=list, max_length=6)  # what to have ready
    steps: list[Step] = Field(min_length=3, max_length=16)
    checks: list[Check] = Field(min_length=1, max_length=8)
    submit: str = Field(min_length=10, max_length=600)  # what to paste in for review
    files: list[StarterFile] = Field(default_factory=list, max_length=10)
    hints: list[str] = Field(default_factory=list, max_length=6)


class Criterion(Content):
    name: str = Field(min_length=3, max_length=80)
    points: int = Field(ge=5, le=50)
    description: str = Field(min_length=10, max_length=500)


class Project(Content):
    title: str = Field(min_length=3, max_length=100)
    minutes: int = Field(ge=30, le=900)
    brief: str = Field(min_length=40, max_length=2000)
    deliverables: list[str] = Field(min_length=2, max_length=8)
    rubric: list[Criterion] = Field(min_length=3, max_length=8)
    submit: str = Field(min_length=10, max_length=600)
    files: list[StarterFile] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def _hundred_points(self) -> Project:
        if sum(c.points for c in self.rubric) != 100:
            raise ValueError("a project's rubric points must add up to 100")
        return self


class Module(Content):
    title: str = Field(min_length=3, max_length=100)
    summary: str = Field(min_length=10, max_length=600)
    instructor: str = ""  # which of the course's instructors teaches it; the first one if not set
    lessons: list[CourseLesson] = Field(min_length=1, max_length=8)
    lab: Lab | None = None

    @property
    def minutes(self) -> int:
        return sum(l.minutes for l in self.lessons) + (self.lab.minutes if self.lab else 0) + QUIZ_MINUTES


class Course(Content):
    slug: str = Field(min_length=3, max_length=48)
    title: str = Field(min_length=3, max_length=100)
    subtitle: str = Field(min_length=10, max_length=240)
    category: Category
    kind: Literal["course", "guided-project"] = "course"
    level: Literal["beginner", "intermediate", "advanced"]
    instructors: list[str] = Field(min_length=1, max_length=3)  # starter companions' slugs, the lead first
    skills: list[str] = Field(min_length=3, max_length=10)
    tools: list[str] = Field(default_factory=list, max_length=8)
    outcomes: list[str] = Field(min_length=3, max_length=6)  # what you'll learn
    prerequisites: str = Field(min_length=2, max_length=500)
    description: str = Field(min_length=40, max_length=2500)
    aligned_with: str = Field("", max_length=200)  # e.g. a certification's published exam topics
    modules: list[Module] = Field(min_length=1, max_length=8)
    project: Project | None = None

    @field_validator("slug")
    @classmethod
    def _slug(cls, v: str) -> str:
        # Ending in a letter keeps every id unambiguous: course-a-1-2 is a lesson, never module 2 of "a-1".
        if not SLUG.match(v) or not v[-1].isalpha():
            raise ValueError("slug: lowercase words joined by hyphens, ending in a letter")
        return v

    @field_validator("skills", "tools")
    @classmethod
    def _short(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            raise ValueError(f"repeated: {v}")
        return _texts(v, 50, "skill or tool")

    @field_validator("outcomes")
    @classmethod
    def _outcomes(cls, v: list[str]) -> list[str]:
        return _texts(v, 220, "outcome")

    @model_validator(mode="after")
    def _shape(self) -> Course:
        if len(set(self.instructors)) != len(self.instructors):
            raise ValueError("an instructor is listed twice")
        for i, m in enumerate(self.modules, 1):
            if m.instructor and m.instructor not in self.instructors:
                raise ValueError(f"module {i}'s instructor {m.instructor!r} isn't one of the course's instructors")
        if self.kind == "guided-project" and (len(self.modules) != 1 or self.modules[0].lab is None):
            raise ValueError("a guided project has one module, which ends with its lab")
        if self.kind == "course" and len(self.modules) < 2:
            raise ValueError("a course has at least two modules")
        return self

    @property
    def minutes(self) -> int:
        return sum(m.minutes for m in self.modules) + (self.project.minutes if self.project else 0)

    @property
    def hours(self) -> float:
        """Roughly how long it takes, as Coursera shows it: whole hours, or half hours under ten."""
        h = self.minutes / 60
        return max(0.5, round(h * 2) / 2) if h < 10 else float(round(h))

    @property
    def labs(self) -> list[tuple[int, Lab]]:
        return [(i, m.lab) for i, m in enumerate(self.modules, 1) if m.lab]

    @property
    def jetbrains(self) -> list[str]:
        return [t for t in self.tools if t in JETBRAINS]

    @property
    def cisco(self) -> bool:
        return "Cisco" in self.aligned_with or any(t.startswith("Cisco") for t in self.tools)

    def instructor_of(self, m: int) -> str:
        return self.modules[m - 1].instructor or self.instructors[0]

    def notices(self) -> list[str]:
        """What the course page says about the products it names."""
        out = []
        if jb := self.jetbrains:
            names = ", ".join(jb[:-1]) + (" and " if len(jb) > 1 else "") + jb[-1]
            out.append(f"{names} {'are trademarks' if len(jb) > 1 else 'is a trademark'} of JetBrains s.r.o. "
                       "This course is independent: JetBrains hasn't made or endorsed it.")
        if self.cisco:
            out.append("Cisco, CCNA, Cisco IOS and Cisco Packet Tracer are trademarks of Cisco Systems, Inc. This course "
                       "is independent: Cisco hasn't made or endorsed it, and it doesn't give you a Cisco certification.")
        if self.tools and not out:
            out.append("The products this course names are trademarks of their owners, who haven't made or endorsed it.")
        return out


class Track(Content):
    """A series of courses: a Professional Certificate (for a job) or a Specialization (for a skill)."""

    slug: str = Field(min_length=3, max_length=48)
    title: str = Field(min_length=3, max_length=100)
    kind: Literal["Professional Certificate", "Specialization"]
    subtitle: str = Field(min_length=10, max_length=240)
    description: str = Field(min_length=40, max_length=1500)
    roles: list[str] = Field(default_factory=list, max_length=6)  # jobs it prepares you for
    courses: list[str] = Field(min_length=2, max_length=10)


# ---------------------------------------------------------------------------- loading


def load(folder: Path = DATA) -> tuple[tuple[Course, ...], tuple[Track, ...]]:
    """Every course and track in the folder, courses in catalogue order: by subject, then from beginner up,
    courses before guided projects, then by title. Raises ValueError, naming the file, on one that doesn't fit."""
    courses = []
    for f in sorted(folder.glob("*.json")):
        if f.name == "tracks.json":
            continue
        try:
            course = Course.model_validate_json(f.read_text(encoding="utf-8"))
        except ValueError as e:
            raise ValueError(f"{f.name}: {e}") from None
        if course.slug != f.stem:
            raise ValueError(f"{f.name}: its slug is {course.slug!r}; the file must be named after it")
        courses.append(course)
    courses.sort(key=lambda c: (CATEGORIES.index(c.category), LEVELS.index(c.level), c.kind != "course", c.title))
    slugs = {c.slug for c in courses}
    tracks_file = folder / "tracks.json"
    tracks = [Track.model_validate(t) for t in json.loads(tracks_file.read_text(encoding="utf-8"))] if tracks_file.exists() else []
    for t in tracks:
        missing = [s for s in t.courses if s not in slugs]
        if missing:
            raise ValueError(f"tracks.json: {t.slug} names courses that don't exist: {', '.join(missing)}")
    if len({t.slug for t in tracks}) != len(tracks):
        raise ValueError("tracks.json: two tracks share a slug")
    return tuple(courses), tuple(tracks)


def module_id(slug: str, m: int) -> str:
    return f"course-{slug}-{m}"


def lesson_id(slug: str, m: int, n: int) -> str:
    return f"course-{slug}-{m}-{n}"


def lab_id(slug: str, m: int) -> str:
    return f"course-{slug}-{m}-lab"


def project_id(slug: str) -> str:
    return f"course-{slug}-project"


@dataclass(frozen=True)
class Place:
    """Where an id points: a course, and the module and lesson in it (0 when the id isn't one)."""

    course: Course
    m: int = 0
    n: int = 0

    @property
    def module(self) -> Module | None:
        return self.course.modules[self.m - 1] if self.m else None

    @property
    def lesson(self) -> CourseLesson | None:
        return self.module.lessons[self.n - 1] if self.n else None


class Catalogue:
    """The courses and tracks, and every id in them, looked up."""

    def __init__(self, courses: tuple[Course, ...], tracks: tuple[Track, ...]):
        self.courses, self.tracks = courses, tracks
        self.by_slug = {c.slug: c for c in courses}
        self.tracks_by_slug = {t.slug: t for t in tracks}
        self.tracks_of: dict[str, list[Track]] = {}
        for t in tracks:
            for s in t.courses:
                self.tracks_of.setdefault(s, []).append(t)
        self.modules: dict[str, LearningPath] = {}
        self.lessons: dict[str, tuple[LearningPath, Lesson]] = {}
        self.places: dict[str, Place] = {}  # module, lesson, lab and project ids
        for c in courses:
            for m, module in enumerate(c.modules, 1):
                lessons = tuple(
                    Lesson(lesson_id(c.slug, m, n), l.title, tuple(l.objectives), tuple(l.key_points), l.minutes)
                    for n, l in enumerate(module.lessons, 1)
                )
                path = LearningPath(module_id(c.slug, m), f"{c.title} · Module {m}: {module.title}", module.summary,
                                    c.level, c.category, lessons)
                self.modules[path.id] = path
                self.places[path.id] = Place(c, m)
                for n, lesson in enumerate(lessons, 1):
                    self.lessons[lesson.id] = (path, lesson)
                    self.places[lesson.id] = Place(c, m, n)
                if module.lab:
                    self.places[lab_id(c.slug, m)] = Place(c, m)
            self.places[project_id(c.slug)] = Place(c)


_catalogue: Catalogue | None = None


def data() -> Catalogue:
    """The catalogue, read from app/course_data/ the first time it's needed (the app asks when it starts)."""
    global _catalogue
    if _catalogue is None:
        _catalogue = Catalogue(*load())
    return _catalogue


def course(slug: str) -> Course | None:
    return data().by_slug.get(slug)


def taught_by(slug: str) -> list[Course]:
    """The courses a starter companion teaches, in catalogue order."""
    return [c for c in data().courses if slug in c.instructors]


def find_lesson(lesson_id: str) -> tuple[LearningPath, Lesson] | None:
    return data().lessons.get(lesson_id) if lesson_id.startswith("course-") else None


def find_module(path_id: str) -> LearningPath | None:
    return data().modules.get(path_id) if path_id.startswith("course-") else None


def module_paths(c: Course) -> list[LearningPath]:
    return [data().modules[module_id(c.slug, m)] for m in range(1, len(c.modules) + 1)]


def place(item_id: str) -> Place | None:
    """The course (and module and lesson) behind a module, lesson, lab or project id, or a course's slug."""
    cat = data()
    if item_id in cat.by_slug:
        return Place(cat.by_slug[item_id])
    return cat.places.get(item_id)


def instructor_for(item_id: str) -> str | None:
    """The slug of the instructor who teaches an item: its module's, or for a project or a whole course, the lead."""
    p = place(item_id)
    if p is None:
        return None
    return p.course.instructor_of(p.m) if p.m else p.course.instructors[0]


def _step_text(s: Step) -> str:
    keys = f" (shortcut: {s.keys.mac} on macOS, {s.keys.win} on Windows and Linux)" if s.keys else ""
    code = f"\n    {s.code.replace(chr(10), chr(10) + '    ')}" if s.code else ""
    see = f"\n    You should see: {s.see}" if s.see else ""
    return f"{s.do}{keys}{code}{see}"


def walkthrough_text(w: Walkthrough) -> str:
    return f"Walkthrough in {w.tool}: {w.title}\n" + "\n".join(f"{i}. {_step_text(s)}" for i, s in enumerate(w.steps, 1))


def lesson_extra(lesson_id: str) -> str:
    """What a course lesson adds to its brief (app.curriculum.lesson_brief): its course, and its walkthrough."""
    p = data().places.get(lesson_id)
    if p is None or not p.n:
        return ""
    out = f"Course: {p.course.title} ({p.course.level}), module {p.m}: {p.module.title}"
    if p.lesson.walkthrough:
        out += "\n" + walkthrough_text(p.lesson.walkthrough)
    return out


def lesson_versions() -> dict[str, str]:
    """A fingerprint of what each course lesson is taught from (its brief and walkthrough), by lesson id, so
    lessons already written can be written again when that changes."""
    import hashlib

    from app.curriculum import lesson_brief

    out = {}
    for lid, (_, lesson) in data().lessons.items():
        text = lesson_brief(lesson) + "\n" + lesson_extra(lid)
        out[lid] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return out


def lab_brief(c: Course, m: int) -> str:
    lab = c.modules[m - 1].lab
    parts = [f"Lab: {lab.title} (in {lab.tool}, about {lab.minutes} minutes)", f"Goal: {lab.goal}"]
    if lab.setup:
        parts.append("Setup:\n" + "\n".join(f"- {s}" for s in lab.setup))
    parts.append("Steps:\n" + "\n".join(f"{i}. {_step_text(s)}" for i, s in enumerate(lab.steps, 1)))
    parts.append("It passes when:\n" + "\n".join(f"{i}. {ch.text}" for i, ch in enumerate(lab.checks, 1)))
    parts.append(f"Learners submit: {lab.submit}")
    if lab.hints:
        parts.append("Hints:\n" + "\n".join(f"- {h}" for h in lab.hints))
    return "\n\n".join(parts)


def project_brief(c: Course) -> str:
    p = c.project
    return "\n\n".join([
        f"Capstone project: {p.title} (about {p.minutes} minutes)", p.brief,
        "Deliverables:\n" + "\n".join(f"- {d}" for d in p.deliverables),
        "Rubric:\n" + "\n".join(f"{i}. {r.name} ({r.points} points): {r.description}" for i, r in enumerate(p.rubric, 1)),
        f"Learners submit: {p.submit}",
    ])


def item_context(item_id: str) -> dict | None:
    """What an instructor needs to help with a lab or a project in chat (lessons use the lesson's own context)."""
    p = data().places.get(item_id)
    if p is None or p.n:
        return None
    c = p.course
    if item_id == project_id(c.slug) and c.project:
        return {"kind": "project", "title": c.project.title, "body": f"Course: {c.title}\n\n{project_brief(c)}"}
    if p.m and item_id == lab_id(c.slug, p.m):
        return {"kind": "lab", "title": p.module.lab.title,
                "body": f"Course: {c.title}, module {p.m}: {p.module.title}\n\n{lab_brief(c, p.m)}"}
    return None
