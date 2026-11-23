import json

from lpe import drive
from lpe.analyze import aggregate, analyze_text, hook_of
from lpe.dataset import load_examples, stem_key
from lpe.profile import build_profile, render_voice_guide, training_domains
from lpe.retrieval import similar_examples

SAMPLE = """Give me 5 minutes, and I'll make your mornings easier (for free):

Step 1. Open your notes.
Step 2. Paste the transcript.
Step 3. Ask for owners & dates.

✦ Bonus
→ Settings → Capabilities → turn it on.

"Goal: [what you want]. Context: [who it's for]. Ask me questions first."

Save this for later.
♻️ Repost it to your team."""


def test_stem_key_pairs_text_and_graphic_names():
    assert stem_key("Post 3 Text.txt") == "Post 3"
    assert stem_key("Post 3 Image.jpg") == "Post 3"
    assert stem_key("Post 20.gif") == "Post 20"
    assert stem_key("launch_caption.md") == "launch"
    assert stem_key("launch-graphic.png") == "launch"


def test_load_examples_pairs_and_reports_orphans(tmp_path):
    from PIL import Image

    (tmp_path / "A Text.txt").write_text("Post A", encoding="utf-8")
    Image.new("RGB", (40, 50), "white").save(tmp_path / "A Image.png")
    (tmp_path / "B Text.txt").write_text("Post B", encoding="utf-8")
    Image.new("RGB", (40, 50), "white").save(tmp_path / "C.gif")
    examples, problems = load_examples(tmp_path)
    assert [e.key for e in examples] == ["A", "B"]
    assert examples[0].media_kind == "image" and examples[1].media_file is None
    assert any("C" in p and "without a text" in p for p in problems)


def test_analyze_text_measures_the_style():
    a = analyze_text(SAMPLE)
    assert a["hook_ends_with_colon"] and a["hook_has_number"] and a["hook_has_aside"]
    assert a["labelled_lines"] == 3
    assert a["glyph_lines"].get("✦") == 1 and a["arrows"] >= 2
    assert a["ampersands"] == 1 and a["quoted_prompts"] == 1
    assert "save" in a["ctas"] and "repost_or_share" in a["ctas"]
    assert a["emoji"] == 1 and a["hashtags"] == 0
    assert hook_of("First line\nsecond line\nthird\n\nNext") == "First line\nsecond line"


def test_aggregate_reports_distributions():
    stats = aggregate([analyze_text(SAMPLE), analyze_text("Short post.\n\nWith a line.")])
    assert stats["posts"] == 2
    assert stats["chars"]["p10"] <= stats["chars"]["median"] <= stats["chars"]["p90"]
    assert 0 <= stats["share_with_numbered_list"] <= 1


def test_parse_folder_id_accepts_links_and_ids():
    fid = "1UKZY_aUShSc2LYfIva_L_xG79VxAXChA"
    assert drive.parse_folder_id(f"https://drive.google.com/drive/folders/{fid}?usp=sharing") == fid
    assert drive.parse_folder_id(f"https://drive.google.com/open?id={fid}") == fid
    assert drive.parse_folder_id(fid) == fid


def test_parse_listing_reads_files_and_folders():
    page = (
        '<div class="flip-entry" id="entry-AAAAAAAAAAAA" tabindex="0"><div class="flip-entry-info">'
        '<a href="https://drive.google.com/file/d/AAAAAAAAAAAA/view"><div class="flip-entry-title">Post 1 Text.txt</div></a></div></div>'
        '<div class="flip-entry" id="entry-BBBBBBBBBBBB" tabindex="0"><div class="flip-entry-info">'
        '<a href="https://drive.google.com/drive/folders/BBBBBBBBBBBB"><div class="flip-entry-title">More &amp; more</div></a></div></div>'
    )
    files = drive.parse_listing(page)
    assert [(f.id, f.name, f.is_folder) for f in files] == [("AAAAAAAAAAAA", "Post 1 Text.txt", False), ("BBBBBBBBBBBB", "More & more", True)]


def test_profile_keeps_patterns_not_example_text(tmp_path):
    (tmp_path / "A Text.txt").write_text(SAMPLE + "\nVisit someones-newsletter.guide", encoding="utf-8")
    examples, _ = load_examples(tmp_path)
    analyses = [analyze_text(e.text) for e in examples]
    annotations = {"creator_markers": ["Someone Else"], "posts": {"A": {"template": "cheatsheet", "hook_type": "time_box_promise"}}}
    profile = build_profile(examples, analyses, [], annotations)
    assert profile["guardrails"]["creator_markers"] == ["Someone Else"]
    assert "someones-newsletter.guide" in profile["guardrails"]["training_domains"]
    guide = render_voice_guide(profile)
    assert "## Hooks that worked" in guide and "time_box_promise" in guide
    assert "Paste the transcript" not in guide and "someones-newsletter" not in guide
    json.dumps(profile)  # serialisable


def test_training_domains_skip_everyday_platforms():
    assert training_domains(["see github.com/x and claude.ai and my-site.guide/path"]) == ["my-site.guide"]


def test_similar_examples_ranks_relevance_and_respects_exclusions():
    corpus = [
        {"key": "skills", "text": "Install these Claude skills: upload the zip in Customize, Skills.", "template": "card_grid"},
        {"key": "models", "text": "Pick the right model: Opus for hard work, Haiku for speed.", "template": "decision_tree"},
        {"key": "hidden", "text": "Claude skills skills skills upload zip", "template": "card_grid", "use_as_example": False},
    ]
    best = similar_examples("how to upload claude skills", corpus, k=2)
    assert best[0]["key"] == "skills"
    assert all(r["key"] != "hidden" for r in best)
    boosted = similar_examples("anything", corpus, k=1, template="decision_tree")
    assert boosted[0]["key"] == "models"


def test_brand_file_signs_as_mojeed_and_links_his_site():
    from lpe.brand import load_brand, own_hosts, site_host

    brand = load_brand()
    assert brand["name"] == "Mojeed Abisiga"
    assert brand["url"] == "https://mojeedabisiga.com/" and own_hosts(brand) == {"mojeedabisiga.com"}
    text = " ".join([brand["footer"]["left"], brand["footer"]["right"], brand["headline"], *brand["cta"].values()]).lower()
    assert "mojeedabisiga.com" in text and "ai every time" not in text and "aieverytime" not in text
    assert "AI Every Time" in brand["never_mention"]
    assert site_host("https://www.Example.com/page") == "example.com"


def test_brand_defaults_fill_footer_and_url(tmp_path):
    from lpe.brand import load_brand

    path = tmp_path / "brand.json"
    path.write_text('{"name": "Ada", "url": "https://ada.dev/"}', encoding="utf-8")
    brand = load_brand(path)
    assert brand["site"] == "ada.dev" and brand["footer"] == {"left": "from Ada", "right": "ada.dev"}
