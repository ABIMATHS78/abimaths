import csv
import io
import json
import zipfile

import pytest

from tests.test_learning import onboard

SLUG = "python-programming-pycharm"  # the example course, which the tests rely on


# ------------------------------------------------------------------ the content


def test_every_course_file_fits_the_format_and_names_real_instructors():
    """What scripts/courses/check.py checks, for every course: the format, starter companions as instructors (none
    modelled on a real person), and no Claude model IDs, which go out of date."""
    from app import catalog
    from app.starter_companions import CSV_PATH, REAL_PEOPLE
    from scripts.courses.check import check

    starters = {row["slug"] for row in csv.DictReader(CSV_PATH.open(encoding="utf-8"))}
    files = sorted(p for p in catalog.DATA.glob("*.json") if p.name != "tracks.json")
    assert files
    for f in files:
        errors, _ = check(f)
        assert not errors, f"{f.name}: {errors}"
    courses, tracks = catalog.load()
    for c in courses:
        assert set(c.instructors) <= starters and not set(c.instructors) & REAL_PEOPLE
    ids = [i for c in courses for i in [catalog.project_id(c.slug)] + [
        catalog.lesson_id(c.slug, m, n) for m, mod in enumerate(c.modules, 1) for n in range(1, len(mod.lessons) + 1)]]
    assert len(ids) == len(set(ids))
    for t in tracks:
        assert all(catalog.course(s) for s in t.courses)


def test_a_broken_course_file_is_reported_by_name(tmp_path):
    from app import catalog

    good = json.loads((catalog.DATA / f"{SLUG}.json").read_text(encoding="utf-8"))
    (tmp_path / "wrong-name.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="wrong-name.json"):
        catalog.load(tmp_path)
    (tmp_path / "wrong-name.json").unlink()
    good["project"]["rubric"][0]["points"] += 1
    (tmp_path / f"{SLUG}.json").write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="add up to 100"):
        catalog.load(tmp_path)


def test_ids_are_unambiguous():
    from app import catalog

    path, lesson = catalog.find_lesson(catalog.lesson_id(SLUG, 2, 3))
    assert path.id == catalog.module_id(SLUG, 2) and lesson.title == catalog.course(SLUG).modules[1].lessons[2].title
    assert catalog.find_lesson(catalog.module_id(SLUG, 2)) is None and catalog.find_module("tokens") is None
    assert catalog.instructor_for(catalog.lesson_id(SLUG, 1, 1)) == "donte-jacksonee"  # module 1's own instructor
    assert catalog.instructor_for(catalog.project_id(SLUG)) == "vladimir-novak"  # the lead
    assert catalog.item_context(catalog.lab_id(SLUG, 4))["kind"] == "lab"
    assert catalog.item_context(catalog.lesson_id(SLUG, 1, 1)) is None


# ------------------------------------------------------------------ catalogue


def test_catalogue_and_course_page(client):
    onboard(client)
    cat = client.get("/api/courses").json()
    course = next(c for c in cat["courses"] if c["slug"] == SLUG)
    assert course["jetbrains"] == ["PyCharm"] and course["labs"] == 5 and course["project"]
    assert [i["slug"] for i in course["instructors"]] == ["vladimir-novak", "donte-jacksonee"]
    assert course["enrolled"] is False and course["progress"] == 0 and course["next"] == "l1-1"
    assert any(c["name"] == "Software Development" and c["count"] >= 1 for c in cat["categories"])

    page = client.get(f"/api/courses/{SLUG}").json()
    assert page["outcomes"] and page["instructors"][0]["bio"] and page["instructors"][0]["courses"] >= 1
    assert any("JetBrains" in n for n in page["notices"])
    first = page["syllabus"][0]
    assert first["instructor"]["slug"] == "donte-jacksonee"
    assert [i["type"] for i in first["items"]] == ["lesson", "lesson", "lesson", "lab", "quiz"]
    assert first["items"][0]["walkthrough"] == "PyCharm"
    assert page["project"]["status"] == "not_started" and page["requirements"]["complete"] is False
    assert client.get("/api/courses/no-such-course").status_code == 404


def test_course_lessons_are_taught_by_their_module_instructor_with_the_walkthrough(client):
    onboard(client)  # whose primary companion is someone else
    from app import catalog, db

    lesson = client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 4, 1)}").json()
    assert lesson["teacher"]["name"] == "Dontè Jacksoneè"
    assert lesson["course"] == {"slug": SLUG, "title": "Python Programming with PyCharm", "module": 4, "lesson": 1,
                                "module_title": "Debugging like a professional"}
    assert lesson["walkthrough"]["tool"] == "PyCharm" and lesson["walkthrough"]["steps"][0]["keys"]["win"] == "Ctrl+F8"
    assert client.get(f"/api/lessons/{catalog.lesson_id(SLUG, 2, 1)}").json()["teacher"]["name"] == "Vladimìr Novàk"
    with db.connect() as conn:  # cached for the instructor, like any lesson
        assert conn.execute("SELECT COUNT(*) FROM lesson_content WHERE lesson_id LIKE 'course-%'").fetchone()[0] == 2


def test_lab_review_needs_enrolment_and_follows_the_checks(client):
    onboard(client)
    work = "Python 3.12.1\nRunning from /home/ada/hello-pycharm/.venv\nrequests 2.32.3"
    assert client.post(f"/api/courses/{SLUG}/labs/1/submit", json={"content": work}).status_code == 409
    assert client.post(f"/api/courses/{SLUG}/enroll").status_code == 204

    missing = client.post(f"/api/courses/{SLUG}/labs/1/submit", json={"content": "Python 3.12.1"}).json()
    assert missing["passed"] is False and [c["met"] for c in missing["review"]["checks"]] == [True, False, False]

    done = client.post(f"/api/courses/{SLUG}/labs/1/submit", json={"content": work}).json()
    assert done["passed"] is True and done["content"] == work
    lab = client.get(f"/api/courses/{SLUG}/labs/1").json()
    assert lab["instructor"]["slug"] == "donte-jacksonee" and len(lab["attempts"]) == 2 and lab["checks"]
    page = client.get(f"/api/courses/{SLUG}").json()
    assert page["syllabus"][0]["items"][3]["status"] == "passed" and page["requirements"]["labs"] == {"done": 1, "total": 5}
    assert client.get(f"/api/courses/{SLUG}/labs/9").status_code == 404


def test_starter_files_download_as_a_zip(client):
    onboard(client)
    r = client.get(f"/api/courses/{SLUG}/labs/3/files")
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    names = zipfile.ZipFile(io.BytesIO(r.content)).namelist()
    assert sorted(names) == [f"{SLUG}-lab-3/expenses.csv", f"{SLUG}-lab-3/tracker.py"]
    assert client.get(f"/api/courses/{SLUG}/labs/1/files").status_code == 404  # lab 1 has none


def test_chat_about_a_lab_knows_the_lab(client):
    onboard(client)
    from app import catalog

    teacher = client.get(f"/api/courses/{SLUG}").json()["instructors"][1]
    r = client.post(f"/api/companions/{teacher['id']}/chat",
                    json={"message": "My cart total is wrong", "lesson_id": catalog.lab_id(SLUG, 4)})
    assert r.status_code == 200 and "Fix the buggy shopping cart" in r.text


def _answers(assessment_id: int) -> list[int]:
    from app import db

    with db.connect() as conn:
        qs = json.loads(conn.execute("SELECT questions FROM assessments WHERE id = ?", (assessment_id,)).fetchone()[0])
    return [q["answer_index"] for q in qs]


def test_module_quiz_and_course_exam(client):
    onboard(client)
    from app import catalog

    quiz = client.post("/api/assessments", json={"kind": "quiz", "source_type": "path",
                                                 "source_ref": catalog.module_id(SLUG, 1)}).json()
    assert len(quiz["questions"]) == 8 and quiz["companion_id"] == client.get(f"/api/courses/{SLUG}").json()["instructors"][1]["id"]
    assert all(q["lesson_id"].startswith(f"course-{SLUG}-1-") for q in quiz["questions"])
    result = client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": _answers(quiz["id"])}).json()
    assert result["passed"]
    items = client.get(f"/api/courses/{SLUG}").json()["syllabus"][0]["items"]
    assert items[-1]["status"] == "passed" and items[-1]["best_score"] == 100

    exam = client.post("/api/assessments", json={"kind": "exam", "source_type": "course", "source_ref": SLUG}).json()
    assert len(exam["questions"]) == 15 and exam["time_limit_minutes"] == 25
    assert client.post("/api/assessments", json={"kind": "exam", "source_type": "course", "source_ref": "nope"}).status_code == 404


def _finish(client, upto_project=True):
    """Do everything the example course asks."""
    from app import catalog

    c = catalog.course(SLUG)
    client.post(f"/api/courses/{SLUG}/enroll")
    for m, module in enumerate(c.modules, 1):
        for n in range(1, len(module.lessons) + 1):
            assert client.post(f"/api/lessons/{catalog.lesson_id(SLUG, m, n)}/complete").status_code == 204
        quiz = client.post("/api/assessments", json={"kind": "quiz", "source_type": "path",
                                                     "source_ref": catalog.module_id(SLUG, m)}).json()
        client.post(f"/api/assessments/{quiz['id']}/submit", json={"answers": _answers(quiz["id"])})
        if module.lab:
            work = " ".join(p for ch in module.lab.checks for p in ch.look_for)
            assert client.post(f"/api/courses/{SLUG}/labs/{m}/submit", json={"content": work}).json()["passed"]
    if upto_project:
        project = client.post(f"/api/courses/{SLUG}/project/submit", json={"content": "word " * 200}).json()
        assert project["passed"] and project["score"] >= 70


def test_certificate_once_the_course_is_finished(client):
    onboard(client)
    _finish(client, upto_project=False)
    assert client.post(f"/api/courses/{SLUG}/certificate", json={}).status_code == 409
    short = client.post(f"/api/courses/{SLUG}/project/submit", json={"content": "Here it is."}).json()
    assert short["passed"] is False and short["score"] < 70
    client.post(f"/api/courses/{SLUG}/project/submit", json={"content": "word " * 200})

    page = client.get(f"/api/courses/{SLUG}").json()
    assert page["requirements"]["complete"] and page["progress"] == 100 and page["next"] is None
    cert = client.post(f"/api/courses/{SLUG}/certificate", json={"name": "  Ada   Lovelace "}).json()
    assert cert["name"] == "Ada Lovelace" and cert["for"]["title"] == "Python Programming with PyCharm"
    assert [i["slug"] for i in cert["instructors"]] == ["vladimir-novak", "donte-jacksonee"]
    again = client.post(f"/api/courses/{SLUG}/certificate", json={"name": "Ada King"}).json()
    assert again["code"] == cert["code"] and again["name"] == "Ada King"
    assert client.get(f"/api/courses/{SLUG}").json()["certificate"] == cert["code"]

    client.post("/api/auth/logout")  # anyone with the code can check it
    public = client.get(f"/api/certificates/{cert['code'].lower()}").json()
    assert public["name"] == "Ada King" and public["kind"] == "course"
    assert client.get("/api/certificates/NOPE").status_code == 404


def test_rating_and_leaving_a_course(client):
    onboard(client)
    assert client.put(f"/api/courses/{SLUG}/rating", json={"stars": 5}).status_code == 409
    client.post(f"/api/courses/{SLUG}/enroll")
    assert client.put(f"/api/courses/{SLUG}/rating", json={"stars": 6}).status_code == 422
    assert client.put(f"/api/courses/{SLUG}/rating", json={"stars": 4}).status_code == 204
    page = client.get(f"/api/courses/{SLUG}").json()
    assert page["rating"] == {"average": 4.0, "count": 1} and page["my_rating"] == 4 and page["learners"] == 1
    assert client.delete(f"/api/courses/{SLUG}/enroll").status_code == 204
    assert client.get(f"/api/courses/{SLUG}").json()["enrolled"] is False


def test_tracks_need_every_course(client):
    onboard(client)
    from app import catalog

    tracks = [t for t in catalog.data().tracks if SLUG in t.courses]
    if not tracks:
        pytest.skip("no track includes the example course yet")
    t = tracks[0]
    view = client.get(f"/api/tracks/{t.slug}").json()
    assert view["total"] == len(t.courses) and view["done"] == 0
    assert client.post(f"/api/tracks/{t.slug}/certificate", json={}).status_code == 409


def test_courses_in_the_data_export(client):
    onboard(client)
    _finish(client)
    client.post(f"/api/courses/{SLUG}/certificate", json={})
    data = client.get("/api/me/export").json()["courses"]
    assert data["enrolled"][0]["course_slug"] == SLUG and len(data["labs_and_projects"]) == 6
    assert data["certificates"][0]["kind"] == "course"
