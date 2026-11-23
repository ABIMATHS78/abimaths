"""The Claude calls, end to end through the real SDK over a mock transport: no key, no network."""

import json

import pytest

from lpe.cli import main
from lpe.dataset import Example
from lpe.llm import Claude, LLMError, annotate_example
from lpe.pipeline import generate
from lpe.schemas import optional_count
from lpe.templates import TEMPLATES

GOOD_POST = "Stop guessing your prompts.\n\nTry this instead:\n\n" + "\n".join(f"{i}. A short, real step." for i in range(1, 12)) + "\n\nSave this for later."


def plan(post, template="prompt_ladder"):
    return {"angle": "Prompts need goals.", "hook_type": "biggest_mistake", "structure": "tip_list", "post": post,
            "alternative_hooks": ["One", "Two"], "template": template, "motion": "png",
            "graphic_brief": "Bad, better and best prompts.", "alt_text": "Three prompts.", "fact_checks": []}


def responder_for(first_post):
    def respond(body, n):
        props = body["output_config"]["format"]["schema"]["properties"]
        if "post" in props:
            return plan(first_post if n == 1 else GOOD_POST)
        return TEMPLATES["prompt_ladder"].sample
    return respond


def test_generate_builds_correct_requests_and_repairs(fake_api, tmp_path):
    from lpe.profile import load_profile

    marker = (load_profile()["guardrails"]["creator_markers"] or ["how-to-ai.guide"])[0]
    fake_api.responder = responder_for(GOOD_POST + f"\n\nThanks to {marker}.")
    claude = Claude(client=fake_api.client())
    result = generate({"topic": "Better prompts", "notes": "Goal, context, questions."}, claude=claude,
                      out_dir=tmp_path, render_graphic=False, log=lambda m: None)

    assert len(fake_api.requests) == 3  # write, repair (the first draft named the training creator), design
    first = fake_api.requests[0]
    assert first["model"] == "claude-opus-5-5"
    assert first["fallbacks"] == "default" and first["thinking"] == {"type": "adaptive"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"]["format"]["type"] == "json_schema"
    assert "server-side-fallback-2026-07-01" in fake_api.headers[0]["anthropic-beta"]
    assert [m["role"] for m in fake_api.requests[1]["messages"]] == ["user", "assistant", "user"]
    assert all(optional_count(r["output_config"]["format"]["schema"]) == 0 for r in fake_api.requests)
    assert marker not in result["spec"]["post"]
    assert result["spec"]["template"] == "prompt_ladder"
    assert json.loads((tmp_path / "spec.json").read_text())["data"]["rungs"]
    assert result["cost"]["calls"] == 3 and result["cost"]["usd_estimate"] > 0


def test_refusal_is_reported(fake_api):
    fake_api.responder = lambda body, n: ("{}", "refusal")
    claude = Claude(client=fake_api.client())
    with pytest.raises(LLMError, match="declined"):
        claude.json_call("system", [{"role": "user", "content": "hi"}], TEMPLATES["badge"].schema, label="test")


def test_annotate_sends_the_image(fake_api, tmp_path):
    from PIL import Image

    Image.new("RGB", (60, 75), "white").save(tmp_path / "A Image.png")
    fake_api.responder = lambda body, n: {"archetype": "badge", "template": "badge", "theme": "terracotta", "hook_type": "custom",
                                          "structure": "custom", "cta_type": "save_this", "graphic_notes": "A shape.", "content_warning": ""}
    example = Example(key="A", text="A post.", media_file=tmp_path / "A Image.png", media_kind="image")
    note = annotate_example(example, Claude(client=fake_api.client()))
    content = fake_api.requests[0]["messages"][0]["content"]
    assert content[0]["type"] == "image" and content[0]["source"]["media_type"] == "image/jpeg"
    assert note["template"] == "badge" and "content_warning" not in note


def test_cli_lists_templates_and_lints(tmp_path, capsys):
    assert main(["templates"]) == 0
    assert "cheatsheet" in capsys.readouterr().out
    post = tmp_path / "post.txt"
    post.write_text("Too short with **markdown**.", encoding="utf-8")
    assert main(["lint", str(post)]) == 2
    assert "markdown" in capsys.readouterr().out.lower()


def test_prompts_carry_the_website_and_the_names_to_avoid():
    from lpe.brand import load_brand
    from lpe.profile import load_profile
    from lpe.prompts import design_system, write_system

    brand, profile = load_brand(), load_profile()
    system = write_system(profile, brand)
    assert "https://mojeedabisiga.com/" in system and "Mojeed Abisiga" in system
    assert "Never mention, in the post or the graphic: AI Every Time" in system
    assert "Publication:" not in system
    design = design_system(profile, brand)
    assert "from Mojeed Abisiga" in design and "Never mention: AI Every Time" in design
