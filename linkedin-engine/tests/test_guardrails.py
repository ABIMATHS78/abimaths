import pytest

from lpe.guardrails import (brand_mentions, fix_post, has_errors, lint_extras, lint_graphic, lint_post,
                            longest_shared_run, plan_extras)

PROFILE = {
    "voice": {"avoid_phrases": ["delve", "game-changer", "here's the thing"]},
    "guardrails": {"creator_markers": ["Jane Creator", "janes-letter.guide"], "training_domains": ["janes-letter.guide", "janeskills.de"]},
}
BRAND = {"site": "mojeedabisiga.com", "url": "https://mojeedabisiga.com/", "never_mention": ["AI Every Time", "aieverytime.org"]}
BODY = "\n".join(f"Line {i} of a useful post about prompts." for i in range(12))


def codes(issues):
    return {i.code for i in issues}


def test_clean_post_passes():
    text = "Stop guessing your prompts.\n\n" + BODY + "\n\nMore about my work: https://mojeedabisiga.com"
    issues = lint_post(text, PROFILE, [], "", BRAND)
    assert not has_errors(issues), issues


def test_limits_and_formatting():
    assert "too_long" in codes(lint_post("x" * 3100, PROFILE))
    assert "too_short" in codes(lint_post("Tiny post.", PROFILE))
    assert "markdown" in codes(lint_post("Use **bold** here.\n" + BODY, PROFILE))
    assert "hook_past_fold" in codes(lint_post("word " * 60 + "\n\n" + BODY, PROFILE))


def test_ai_giveaways():
    found = codes(lint_post("Let's delve in — here's the thing.\n" + BODY + "\nIt's not a prompt, it's a plan.", PROFILE))
    assert {"ai_phrases", "em_dash", "not_x_but_y"} <= found


def test_training_creator_identity_is_an_error():
    issues = lint_post("Thanks to Jane Creator for this.\n" + BODY + "\nRead janes-letter.guide", PROFILE)
    assert {"third_party", "training_link"} <= codes(issues) and has_errors(issues)


def test_links_must_come_from_the_brief_or_the_brand():
    text = BODY + "\nGuide: https://lnkd.in/abc123\nAlso see example.com/page\nAnd https://www.mojeedabisiga.com/about/"
    issues = [i for i in lint_post(text, PROFILE, [], "", BRAND) if i.code == "unknown_link"]
    assert {i.level for i in issues} == {"error", "warning"}  # lnkd.in is an error, other unknown links a warning
    assert not [i for i in lint_post(BODY + "\nexample.com/page", PROFILE, [], "Link: example.com/page", BRAND) if i.code == "unknown_link"]


def test_copying_from_training_posts_is_caught():
    original = "Write three requests that should use the skill and two that should not and fix the description first."
    corpus = [{"key": "Post 9", "text": original}]
    issues = lint_post(BODY + "\n" + original, PROFILE, corpus)
    assert "copied" in codes(issues)
    assert longest_shared_run("a b c d e f g", "x a b c d e f g y") == 7


def test_integrity_bait_placeholders_and_numbers():
    assert "integrity" in codes(lint_post(BODY + "\nLet ChatGPT finish the quizzes for your certificate.", PROFILE))
    assert "integrity" in codes(lint_post(BODY + "\nThen add the badges, fooling recruiters.", PROFILE))
    assert "engagement_bait" in codes(lint_post(BODY + "\nComment YES below for the guide.", PROFILE))
    assert "needs_input" in codes(lint_post(BODY + "\nWe saved [CHECK: hours saved] hours.", PROFILE))
    numbers = [i for i in lint_post(BODY + "\n1. First\n2. Second\nIt saved 37% of the time.", PROFILE) if i.code == "check_numbers"]
    assert numbers and "37%" in numbers[0].message and "1" not in numbers[0].message.split(":")[1].split(",")[0].strip(" .")


def test_fix_post():
    assert fix_post("One — two and **three**") == "One - two and three"


def test_lint_graphic():
    assert "no_source" in codes(lint_graphic({"template": "big_number", "data": {"source": ""}}))
    assert "third_party" in codes(lint_graphic({"template": "badge", "data": {"title": "By Jane Creator"}}, PROFILE))


def test_never_mention_names_are_errors_in_posts_and_graphics():
    issues = lint_post(BODY + "\nFree guides at AI Every Time: aieverytime.org", PROFILE, [], "", BRAND)
    assert "brand_avoid" in codes(issues) and has_errors(issues)
    assert "brand_avoid" in codes(lint_graphic({"template": "badge", "data": {"title": "Join AI Every Time"}}, PROFILE, [], BRAND))
    assert "brand_avoid" in codes(lint_graphic({"template": "badge", "data": {"title": "Hi"}, "alt_text": "aieverytime.org"}, PROFILE, [], BRAND))
    assert "brand_avoid" not in codes(lint_post(BODY + "\nMore about my work: https://mojeedabisiga.com", PROFILE, [], "", BRAND))


NAMES = ["AI Every Time", "aieverytime.org", "aieverytime"]


@pytest.mark.parametrize("text", ["Free guides at AI Every Time.", "AI Everytime", "AI-Every-Time", "A.I. Every Time",
                                  "AI Every\nTime", "AIEveryTime", "AI​Every​Time", "AI EVERY TIME",
                                  "aieverytime.org/guides", "@aieverytime"])
def test_any_spelling_of_a_never_mention_name_is_an_error(text):
    assert [level for _, _, level in brand_mentions(text, NAMES)] == ["error"]  # one finding, even for overlapping names
    assert "brand_avoid" in codes(lint_post(BODY + "\n" + text, PROFILE, [], "", BRAND))


@pytest.mark.parametrize("text", ["Ask AI every time you start a draft.", "I use AI. Every time it helps.", "ai every time"])
def test_the_same_words_as_ordinary_prose_are_a_warning(text):
    issues = lint_post(BODY + "\n" + text, PROFILE, [], "", BRAND)
    assert "brand_like" in codes(issues) and "brand_avoid" not in codes(issues)


def test_every_field_a_reader_sees_is_checked():
    plan = {"alternative_hooks": ["Join AI Every Time today", "Two"], "alt_text": "A card by Jane Creator",
            "graphic_brief": "Link it: https://lnkd.in/xyz"}
    issues = lint_extras(plan_extras(plan), PROFILE, BRAND, "")
    assert {"brand_avoid", "third_party", "unknown_link"} <= codes(issues) and has_errors(issues)
    assert {i.message.split(":")[0] for i in issues} == {"In the other openings", "In the alt text", "In the graphic brief"}
    assert "the alt text" not in plan_extras(plan, alt_text=False)


def test_links_to_the_authors_site_must_be_known_pages():
    brand = {**BRAND, "pages": ["https://mojeedabisiga.com/about/"]}
    fine = lint_post(BODY + "\nhttps://mojeedabisiga.com/ and https://www.mojeedabisiga.com/about", PROFILE, [], "", brand)
    assert "unknown_page" not in codes(fine)
    made_up = lint_post(BODY + "\nGet the kit: https://mojeedabisiga.com/starter-kit", PROFILE, [], "", brand)
    assert [i.level for i in made_up if i.code == "unknown_page"] == ["warning"]


def test_links_on_graphics_are_checked_and_file_names_are_not_links():
    assert has_errors(lint_graphic({"template": "badge", "data": {"title": "Guide: lnkd.in/abc"}}, PROFILE, [], BRAND))
    assert "unknown_link" in codes(lint_graphic({"template": "badge", "data": {"title": "example.org"}}, PROFILE, [], BRAND))
    assert not codes(lint_graphic({"template": "badge", "data": {"title": "mojeedabisiga.com"}}, PROFILE, [], BRAND))
    assert "unknown_link" not in codes(lint_post(BODY + "\nOne required file: SKILL.md, plus notes.txt", PROFILE, [], "", BRAND))
