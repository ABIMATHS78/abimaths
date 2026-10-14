"""Claude-backed learning materials: lessons, quizzes, flashcards, summaries, study plans.

Every generator is grounded in the curriculum's vetted key points (or a post or
conversation the learner chose) and written in the teaching companion's voice.
Offline mode builds the same structures from the key points directly.
"""

from __future__ import annotations

import logging
import random
from typing import TypeVar

import anthropic
from pydantic import BaseModel, Field

from app import llm
from app.config import settings
from app.curriculum import PATHS, LearningPath, Lesson, lesson_brief
from app.persona import Persona

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


class Question(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options.")
    answer_index: int = Field(description="0-based index of the single correct option.")
    explanation: str = Field(description="Why the answer is right, in the companion's voice, 1-3 sentences.")
    lesson_id: str = Field("", description="The lesson id this question tests, when one was given.")


class QuestionSet(BaseModel):
    questions: list[Question]


class Card(BaseModel):
    front: str = Field(description="A question or term, under 120 characters.")
    back: str = Field(description="The answer or definition, under 300 characters.")


class CardSet(BaseModel):
    cards: list[Card]


class Summary(BaseModel):
    summary: str = Field(description="A 2-4 sentence summary in the companion's voice.")
    key_takeaways: list[str] = Field(description="3-6 short takeaways.")
    next_step: str = Field(description="One concrete suggestion for what to study or practise next.")


class PlanItem(BaseModel):
    lesson_id: str = Field(description="A lesson id from the catalogue.")
    activity: str = Field(description="One of: lesson, quiz, flashcards, exam.")
    note: str = Field(description="Why this item, in one short sentence.")


class PlanWeek(BaseModel):
    week: int
    focus: str
    items: list[PlanItem]


class StudyPlan(BaseModel):
    overview: str = Field(description="2-3 sentences, in the companion's voice, explaining the plan.")
    weeks: list[PlanWeek]


async def _parse(persona: Persona, prompt: str, schema: type[T], effort: str, max_tokens: int = 16000) -> T | None:
    try:
        resp = await llm.client().beta.messages.parse(
            model=settings.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": llm.PLATFORM_RULES + "\n\n" + llm.persona_prompt(persona),
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": prompt}],
            output_format=schema,
            output_config={"effort": effort},
            betas=[llm.FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.APIError:
        log.exception("%s generation failed", schema.__name__)
        return None
    if resp.stop_reason in ("refusal", "max_tokens"):
        log.warning("%s generation stopped: %s", schema.__name__, resp.stop_reason)
        return None
    return resp.parsed_output


def _learner_line(learner: dict) -> str:
    return f"The learner is {learner['display_name']} ({learner['level']} level). Goals: {learner['goals'] or 'not shared'}."


# ------------------------------------------------------------------- lessons


class LessonText(BaseModel):
    body: str = Field(description="The lesson in markdown.")


async def teach_lesson(persona: Persona, path: LearningPath, lesson: Lesson) -> str | None:
    if settings.offline:
        return _offline_lesson(persona, lesson)
    prompt = (
        f"Teach this lesson from the '{path.title}' learning path to your followers.\n\n{lesson_brief(lesson)}\n\n"
        f"Write it in markdown, about {lesson.minutes * 60} words at most, in your own voice and teaching style. "
        "Cover every key point accurately, use a concrete example or analogy, and finish with a short "
        "'Check yourself' section of two questions (no answers). Don't add facts you aren't sure of."
    )
    out = await _parse(persona, prompt, LessonText, effort="high")
    return out.body if out else None


def _offline_lesson(persona: Persona, lesson: Lesson) -> str:
    points = "\n".join(f"- {k}" for k in lesson.key_points)
    goals = "\n".join(f"- {o}" for o in lesson.objectives)
    return (
        f"#### What you'll learn\n{goals}\n\n#### The key ideas\n{points}\n\n"
        f"{persona.writing_style.sample_line or ''}\n\n"
        f"#### Check yourself\n1. Explain \"{lesson.title.lower()}\" in one sentence.\n"
        f"2. Which key idea above surprised you most, and why?\n\n{persona.writing_style.signature_phrase}"
    ).strip()


# ---------------------------------------------------------------- questions


def _valid_questions(qs: list[Question], lesson_ids: set[str]) -> list[Question]:
    good = []
    for q in qs:
        if len(q.options) == 4 and 0 <= q.answer_index < 4 and len(set(q.options)) == 4 and q.question.strip():
            if q.lesson_id not in lesson_ids:
                q.lesson_id = ""
            good.append(q)
    return good


async def make_questions(persona: Persona, learner: dict, material: str, n: int,
                         lessons: list[Lesson], exam: bool = False) -> list[Question]:
    ids = {l.id for l in lessons}
    if settings.offline:
        return _offline_questions(lessons, material, n)
    kind = "a mock exam" if exam else "a short quiz"
    prompt = (
        f"Write {kind} of exactly {n} multiple-choice questions based only on the material below. "
        f"{_learner_line(learner)}\n"
        "Each question has exactly 4 plausible options and one correct answer. Vary the position of the correct "
        "answer. Test understanding and application, not trivia or wording. "
        + ("Spread questions across all lessons and set lesson_id for each. " if exam else
           "Set lesson_id when a lesson id is given. ")
        + f"\n\n{material}"
    )
    out = await _parse(persona, prompt, QuestionSet, effort="high")
    return _valid_questions(out.questions, ids)[:n] if out else []


def _offline_questions(lessons: list[Lesson], material: str, n: int) -> list[Question]:
    pool = [(l, k) for l in lessons for k in l.key_points]
    others = [k for p in PATHS for l in p.lessons if l not in lessons for k in l.key_points]
    random.shuffle(pool)
    out = []
    for lesson, point in pool[:n]:
        options = random.sample(others, 3) + [point]
        random.shuffle(options)
        out.append(Question(
            question=f"Which statement belongs to the lesson \"{lesson.title}\"?",
            options=options, answer_index=options.index(point),
            explanation=f"{point} (Demo mode: real quizzes are written by your companion.)",
            lesson_id=lesson.id,
        ))
    if not out:  # Post or conversation material without lessons.
        sentences = [s.strip() for s in material.replace("\n", " ").split(". ") if len(s.split()) > 5][:n]
        for s in sentences:
            options = [s] + random.sample(others, 3)
            random.shuffle(options)
            out.append(Question(question="Which of these was in the material you studied?", options=options,
                                answer_index=options.index(s), explanation=s))
    return out


# --------------------------------------------------------------- flashcards


async def make_cards(persona: Persona, learner: dict, material: str, n: int, lessons: list[Lesson]) -> list[Card]:
    if settings.offline:
        cards = [Card(front=f"{l.title}: key idea {i + 1}?", back=k) for l in lessons for i, k in enumerate(l.key_points)]
        if not cards:
            cards = [Card(front="Main idea?", back=s.strip()[:300]) for s in material.split("\n") if len(s.split()) > 5]
        return cards[:n]
    prompt = (
        f"Make {n} flashcards from the material below. {_learner_line(learner)} One idea per card; fronts are "
        "questions or terms, backs are crisp answers. Cover the most important ideas first.\n\n" + material
    )
    out = await _parse(persona, prompt, CardSet, effort="medium")
    return [c for c in out.cards if c.front.strip() and c.back.strip()][:n] if out else []


# ---------------------------------------------------------------- summaries


async def summarize(persona: Persona, learner: dict, what: str, material: str) -> Summary | None:
    if settings.offline:
        lines = [s.strip(" -#*") for s in material.split("\n") if len(s.split()) > 4]
        return Summary(summary=f"Here's the short version of {what}. (Demo mode summary.)",
                       key_takeaways=lines[:5] or ["Nothing to summarise yet."],
                       next_step="Take a quick quiz to lock it in.")
    prompt = (f"Summarise {what} for the learner. {_learner_line(learner)} Be faithful to the material; "
              f"don't add new facts.\n\n{material}")
    return await _parse(persona, prompt, Summary, effort="low")


# --------------------------------------------------------------- study plan


async def plan_study(persona: Persona, learner: dict, memories: list[str], completed: set[str],
                     quiz_scores: dict[str, int], weekly_minutes: int, weeks: int,
                     paths: list[LearningPath]) -> StudyPlan | None:
    valid = {l.id for p in paths for l in p.lessons}
    if settings.offline:
        plan = _offline_plan(paths, completed, weekly_minutes, weeks)
    else:
        catalogue = "\n".join(
            f"Path '{p.title}' ({p.level}):\n" + "\n".join(
                f"  - {l.id}: {l.title} ({l.minutes} min)"
                + (" [completed]" if l.id in completed else "")
                + (f" [best quiz {quiz_scores[l.id]}%]" if l.id in quiz_scores else "")
                for l in p.lessons)
            for p in paths
        )
        prompt = (
            f"Create a personalised {weeks}-week study plan. {_learner_line(learner)}\n"
            f"They can study about {weekly_minutes} minutes per week; a lesson takes its listed minutes, a quiz about 5, "
            "flashcards about 5, an exam about 20. Don't exceed the weekly time.\n"
            f"What you remember about them: {'; '.join(memories) or 'nothing yet'}\n"
            "Skip completed lessons unless their quiz score is under 70%, in which case schedule a review quiz. "
            "Follow each path's lesson order, mix in quizzes and flashcards for retention, and end a path with an exam. "
            "Use only lesson ids from this catalogue:\n" + catalogue
        )
        plan = await _parse(persona, prompt, StudyPlan, effort="high")
    if plan is None:
        return None
    for w in plan.weeks:
        w.items = [i for i in w.items if i.lesson_id in valid and i.activity in ("lesson", "quiz", "flashcards", "exam")]
    return plan


def _offline_plan(paths: list[LearningPath], completed: set[str], weekly_minutes: int, weeks: int) -> StudyPlan:
    todo = [(p, l) for p in paths for l in p.lessons if l.id not in completed]
    out, i = [], 0
    for w in range(1, weeks + 1):
        budget, items, focus = weekly_minutes, [], []
        while i < len(todo) and budget >= todo[i][1].minutes + 5:
            path, lesson = todo[i]
            items += [PlanItem(lesson_id=lesson.id, activity="lesson", note=f"Learn {lesson.title.lower()}."),
                      PlanItem(lesson_id=lesson.id, activity="quiz", note="Check your understanding.")]
            if path.title not in focus:
                focus.append(path.title)
            budget -= lesson.minutes + 5
            i += 1
        if items:
            out.append(PlanWeek(week=w, focus=" and ".join(focus), items=items))
    return StudyPlan(overview="A steady plan that works through your paths in order with a quiz after each lesson. "
                              "(Demo mode plan.)", weeks=out)


# -------------------------------------------------------------- custom paths


class DesignedLesson(BaseModel):
    title: str = Field(description="Specific lesson title, under 70 characters.")
    objectives: list[str] = Field(description="2-3 learning objectives starting with a verb.")
    key_points: list[str] = Field(description="3-5 accurate, well-established facts the lesson must teach, one sentence each.")
    minutes: int = Field(description="Estimated minutes, 5-20.")


class DesignedPath(BaseModel):
    title: str = Field(description="Path title, under 60 characters.")
    description: str = Field(description="One or two sentences on what the learner will be able to do.")
    lessons: list[DesignedLesson]


async def design_path(persona: Persona, learner: dict, topic: str, level: str, n_lessons: int) -> dict | None:
    """Design a learning path on a topic the learner chose. Returns the stored JSON shape."""
    if settings.offline:
        designed = _offline_path(topic, n_lessons)
    else:
        prompt = (
            f"Design a {n_lessons}-lesson learning path on this topic: {topic!r}, for a {level} learner. "
            f"{_learner_line(learner)}\n"
            "Order lessons from foundations to application, each building on the last. Key points must be accurate "
            "and well established; leave out anything contested, fast-changing or that you are unsure of. If the topic "
            "is not about AI or technology, design the path around how AI relates to it."
        )
        designed = await _parse(persona, prompt, DesignedPath, effort="high")
    if designed is None:
        return None
    lessons = [
        {"title": l.title.strip()[:90], "objectives": [o.strip()[:200] for o in l.objectives if o.strip()][:3],
         "key_points": [k.strip()[:300] for k in l.key_points if k.strip()][:5],
         "minutes": max(5, min(20, l.minutes))}
        for l in designed.lessons
    ]
    lessons = [l for l in lessons if l["title"] and l["objectives"] and len(l["key_points"]) >= 2][:n_lessons]
    if not lessons:
        return None
    return {"title": designed.title.strip()[:80], "description": designed.description.strip()[:400],
            "level": level, "topic": topic, "lessons": lessons}


def _offline_path(topic: str, n: int) -> DesignedPath:
    stages = [
        ("What {t} is", "Explain what {t} means", "{T} is the topic of this demo path; real paths are designed by your companion."),
        ("Why {t} matters", "Describe why {t} is useful", "Understanding {t} helps you judge where AI fits."),
        ("Key ideas in {t}", "Name the core ideas of {t}", "Every field has a few core ideas worth learning first."),
        ("{T} in practice", "Apply {t} to a real example", "Practising on real examples makes ideas stick."),
        ("Risks and limits of {t}", "Recognise the limits of {t}", "Knowing the limits of a tool is part of using it well."),
        ("Next steps with {t}", "Plan what to learn next about {t}", "Learning is a loop: try, reflect, and go deeper."),
    ]
    t, T = topic.strip(), topic.strip()[:1].upper() + topic.strip()[1:]
    lessons = [
        DesignedLesson(title=a.format(t=t, T=T), objectives=[b.format(t=t, T=T)],
                       key_points=[c.format(t=t, T=T), "(Demo mode: set ANTHROPIC_API_KEY for a real path.)"], minutes=10)
        for a, b, c in (stages * 2)[:n]
    ]
    return DesignedPath(title=f"{T}: a learning path", description=f"A demo path about {t}.", lessons=lessons)
