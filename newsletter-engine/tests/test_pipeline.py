"""Claude calls end to end through the real SDK over a mock transport: no key, no network."""

import json

from nle import pipeline
from nle.llm import Claude
from nle.paths import ENGINE_DIR
from nle.prompts import write_schema


def answer(body: str) -> dict:
    return {"archetype": "setup_steps", "slug": "test-draft-issue", "title": "Test draft.", "subtitle": "For the tests.",
            "description": "A draft written by the tests.", "tags": ["test"], "body": body, "fact_checks": []}


GOOD = "\n\n".join(f"Line {i} of a clear draft about skills." for i in range(30)) + "\n\n---\n\n## Do it\n\nOne step."


def test_write_saves_a_draft_and_repairs_errors(fake_api, monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ISSUES_DIR", tmp_path)
    fake_api.responder = lambda body, n: answer(GOOD + ("\n\nJoin AI Every Time." if n == 1 else ""))
    result = pipeline.write({"topic": "Skills", "notes": ["Skills are folders."]}, "issue",
                            claude=Claude(client=fake_api.client()), log=lambda m: None)
    assert len(fake_api.requests) == 2          # the first draft named the old brand, so one repair
    first = fake_api.requests[0]
    assert first["model"] == "claude-opus-5-5" and first["thinking"] == {"type": "adaptive"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"]["format"]["type"] == "json_schema" and first["fallbacks"] == "default"
    assert [m["role"] for m in fake_api.requests[1]["messages"]] == ["user", "assistant", "user"]
    assert "AI Every" in fake_api.requests[1]["messages"][2]["content"]
    saved = (tmp_path / "test-draft-issue.md").read_text(encoding="utf-8")
    assert 'status = "draft"' in saved and "AI Every Time" not in saved and result["ok"]


def test_every_schema_property_is_required():
    for kind in ("issue", "guide"):
        schema = write_schema(kind)
        assert set(schema["required"]) == set(schema["properties"]) and schema["additionalProperties"] is False


def test_examples_are_picked_only_from_complete_allowed_issues():
    corpus = [{"key": "a", "text": "skills folder claude", "complete": True, "use_as_example": True},
              {"key": "b", "text": "skills folder claude", "complete": False, "use_as_example": True},
              {"key": "c", "text": "skills folder claude", "complete": True, "use_as_example": False}]
    assert [e["key"] for e in pipeline.pick_examples(corpus, {"topic": "claude skills folder"})] == ["a"]


def test_sample_brief_is_valid():
    brief = json.loads((ENGINE_DIR / "briefs" / "claude-skills.json").read_text(encoding="utf-8"))
    assert brief["topic"] and brief["notes"] and "https://mojeedabisiga.com/" in brief["links"]
