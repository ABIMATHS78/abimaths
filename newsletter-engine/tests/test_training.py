from nle.analyze import aggregate, analyze_issue, archive_stats, blocks
from nle.paths import STYLE_GUIDE_PATH
from nle.profile import load_profile, render_style_guide

POST = {
    "title": "Tiny.", "subtitle": "A test issue:", "audience": "everyone", "wordcount": 40,
    "body_html": '<p>Hello there.</p><p>Second line.</p><ol><li>a</li><li>b</li></ol>'
                 '<div class="button-wrapper"><a class="button primary" href="#">Go</a></div><hr>'
                 '<h2>First part</h2><p>Text <img src="x.png"> more.</p><div class="captioned-image-container"><figure><img src="y.png"></figure></div>'
                 '<pre><code>a prompt</code></pre><hr><h2>End</h2><p>Bye.</p>',
}


def test_blocks_handle_void_tags_and_special_containers():
    kinds = [b["kind"] for b in blocks(POST["body_html"])]
    assert kinds == ["paragraph", "paragraph", "list", "button", "divider", "heading", "paragraph", "image", "prompt",
                     "divider", "heading", "paragraph"]


def test_analyze_and_aggregate():
    a = analyze_issue(POST)
    assert a["sections"] == 3 and a["images"] == 1 and a["prompts"] == 1 and a["buttons"] == 1
    assert a["opening_has_list"] and a["opening_has_button"] and a["title_period"] and a["subtitle_colon"]
    stats = aggregate([a])
    assert stats["complete_issues"] == 1 and stats["sections"]["median"] == 3
    arch = archive_stats([{"post_date": "2026-01-01T00:00:00Z", "title": "One.", "subtitle": "x:", "audience": "everyone"},
                          {"post_date": "2026-01-08T00:00:00Z", "title": "Two 2", "subtitle": "y", "audience": "only_paid"}])
    assert arch["issues_per_week"] == 1.0 and arch["free_share"] == 0.5 and arch["title_period_share"] == 0.5


def test_saved_style_guide_matches_knowledge_py():
    """The engine reads knowledge.py on every run; the committed guide must say the same (python -m nle guide --write)."""
    assert STYLE_GUIDE_PATH.read_text(encoding="utf-8") == render_style_guide(load_profile())
