import pytest

from nle.config import load_site
from nle.content import Item
from nle.guardrails import brand_mentions, has_errors, lint_item, longest_shared_run
from nle.md import render

SITE = load_site()
PROFILE = {"guardrails": {"creator_markers": ["Jane Creator"], "training_domains": ["janes.guide"]},
           "avoid_phrases": ["delve", "game-changer"]}
BODY = "\n\n".join(f"Line {i} of a useful issue about prompts and how to use them well." for i in range(40))


def lint(body: str, status: str = "published", brief: str = "", corpus=None, **meta):
    item = Item(kind="issue", slug="t", title=meta.get("title", "A title."), subtitle=meta.get("subtitle", "A subtitle."),
                body=body, status=status)
    return lint_item(item, render(body), SITE, PROFILE, corpus or [], brief)


def codes(issues):
    return {i.code for i in issues}


def test_a_clean_issue_passes():
    assert not has_errors(lint(BODY))


@pytest.mark.parametrize("text", ["Join AI Every Time.", "AI Everytime", "AI-Every-Time", "aieverytime.org/guides", "A.I. Every Time"])
def test_never_mention_names_in_any_spelling(text):
    assert "brand_avoid" in codes(lint(BODY + "\n\n" + text))


def test_ordinary_words_only_warn():
    issues = lint(BODY + "\n\nAsk AI every time you start a draft.")
    assert "brand_like" in codes(issues) and not has_errors(issues)
    assert [level for _, _, level in brand_mentions("aieverytime.org", ["AI Every Time", "aieverytime.org"])] == ["error"]


def test_creator_identity_cheating_and_copying_are_errors():
    assert "third_party" in codes(lint(BODY + "\n\nThanks to Jane Creator."))
    assert "integrity" in codes(lint(BODY + "\n\nLet AI take the exam for you."))
    assert "integrity" in codes(lint(BODY + "\n\nHere's how to bypass AI detectors."))
    assert "integrity" not in codes(lint(BODY + "\n\nUse it with examples and testing notes."))
    original = "Write three requests that should use the skill and two that should not and fix the description first please."
    assert "copied" in codes(lint(BODY + "\n\n" + original, corpus=[{"key": "Ex 1", "text": original}]))
    assert longest_shared_run("a b c d e f g h i", "x a b c d e f g h i y") == 9


def test_published_pieces_cannot_have_holes():
    holes = BODY + "\n\n::screenshot[The settings page]\n\nWe saved [CHECK: hours] hours."
    assert {"screenshots", "needs_input"} <= codes(lint(holes)) and has_errors(lint(holes))
    assert not has_errors(lint(holes, status="draft"))


def test_links_must_come_from_the_brief():
    issues = lint(BODY + "\n\nSee [this](https://example.org/x) and https://lnkd.in/abc")
    assert {i.level for i in issues if i.code == "unknown_link"} == {"warning", "error"}
    assert "unknown_link" not in codes(lint(BODY + "\n\nSee [this](https://example.org/x)", brief="https://example.org/x"))
    assert "unknown_link" not in codes(lint(BODY + "\n\nOne file: SKILL.md. More at [my site](https://mojeedabisiga.com/about/)."))


def test_style_warnings():
    found = codes(lint(BODY + "\n\nLet's delve in — it's not a prompt, it's a plan.", title="A very long title that keeps going on and on and on and on and on"))
    assert {"ai_phrases", "em_dash", "not_x_but_y", "long_title"} <= found
