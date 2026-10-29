"""What the companions know about the courses in chat: every course with the address of its page, the ones they teach
themselves, what the learner's plan opens, and, in demo mode, answers to course questions from the catalogue."""

import re

from tests.test_billing import paywall, pays_for  # noqa: F401  (fixtures)
from tests.test_learning import onboard


def user_row(client) -> dict:
    from app import db

    with db.connect() as conn:
        return dict(conn.execute("SELECT * FROM users WHERE id = ?", (client.get("/api/me").json()["id"],)).fetchone())


def test_every_companion_knows_every_course_and_the_ones_it_teaches():
    from app import catalog, course_guide
    from app.llm import build_system
    from app.starter_companions import STARTERS

    data = catalog.data()
    slug = data.courses[0].instructors[0]
    persona = STARTERS[slug].persona
    taught = course_guide.taught({"id": 5, "slug": slug, "is_starter": 1})
    learner = {"display_name": "Ada", "level": "beginner", "goals": ""}
    shared, own, about = (b["text"] for b in build_system(persona, learner, [], None, taught=taught,
                                                          access="Their plan: Free."))
    # Every course and series, linked, in the part every companion shares.
    for c in data.courses:
        assert f"[{c.title}](/courses/{c.slug})" in shared
    for t in data.tracks:
        assert f"[{t.title}](/tracks/{t.slug})" in shared
    assert "Never say you can't see the courses" in shared and "# Your persona" not in shared
    # Its own courses, and how it links a path for the learner to design, in its own part.
    assert catalog.taught_by(slug) and all(f"(/courses/{c.slug})" in own for c in catalog.taught_by(slug))
    assert f"/learn?design=<topic-words>&with={slug}" in own and persona.name in own
    assert "Their plan: Free." in about
    # A learner's own companion teaches none of them, but knows them all just the same.
    assert "None: you don't teach" in course_guide.taught({"id": 9, "slug": "maya-1", "is_starter": 0})


def test_companions_know_what_the_learners_plan_opens(client, paywall):
    from app import course_guide

    onboard(client)
    free = course_guide.access(user_row(client))
    assert "Their plan: Free." in free and "first module of any course free" in free
    assert "with the Plus plan and above" in free and "with the Pro plan and above, so say so" in free
    pays_for(client, "plus")
    plus = course_guide.access(user_row(client))
    assert "Every course is included in full." in plus and "with the Pro plan and above" in plus
    pays_for(client, "pro")
    assert course_guide.access(user_row(client)).startswith("What their plan opens: every course in full")


def test_without_the_paywall_everything_is_open_to_them(client):
    from app import course_guide

    onboard(client)
    assert course_guide.access(user_row(client)).startswith("What their plan opens: every course in full")


def test_search_finds_courses_by_what_they_teach():
    from app import course_guide

    found = course_guide.search("Do you have a course on regression?")
    assert found and all("regress" in " ".join(c.skills + [c.title]).lower() for c in found)
    assert course_guide.search("Is there one for that?") == []
    assert course_guide.topic("Is there a course on Underwater Basket-Weaving?") == "underwater-basket-weaving"


def test_in_demo_mode_companions_answer_course_questions_from_the_catalogue(client):
    from app import catalog

    starter = onboard(client)
    r = client.post(f"/api/companions/{starter['id']}/chat", json={
        "message": "I want to learn regression. Do you have any course related to machine learning or data analytics?"})
    assert r.status_code == 200
    slugs = re.findall(r"\(/courses/([a-z0-9-]+)\)", r.text)
    assert slugs and all(catalog.course(s) for s in slugs)
    assert any("regress" in " ".join(catalog.course(s).skills).lower() for s in slugs)
    # Nothing on it: the companion offers to design a path on it, which it would design itself.
    r = client.post(f"/api/companions/{starter['id']}/chat",
                    json={"message": "Is there a course on underwater basket weaving?"})
    assert f"(/learn?design=underwater-basket-weaving&with={starter['slug']})" in r.text
    # Not a question about courses: the usual reply.
    assert "](/" not in client.post(f"/api/companions/{starter['id']}/chat", json={"message": "What is a neuron?"}).text
