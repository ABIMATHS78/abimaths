import pytest

from lpe.brand import load_brand
from lpe.markup import plain, rich, words
from lpe.pipeline import resolve_formats
from lpe.prompts import ANNOTATE_SCHEMA, write_schema
from lpe.render import build_html
from lpe.schemas import api_schema, clean_data, obj, optional_count, string
from lpe.templates import TEMPLATES
from lpe.themes import BASE, THEMES, theme_vars

UNSUPPORTED = {"minLength", "maxLength", "minimum", "maximum", "multipleOf", "oneOf", "not", "maxItems"}


def walk(node):
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from walk(value)


@pytest.mark.parametrize("name", list(TEMPLATES))
def test_sample_data_is_valid(name):
    template = TEMPLATES[name]
    data, warnings = clean_data(template.sample, template.schema)
    assert warnings == []
    assert (template.name + ".html.j2") == template.file


@pytest.mark.parametrize("schema", [t.schema for t in TEMPLATES.values()] + [write_schema(), write_schema("cycle"), ANNOTATE_SCHEMA])
def test_api_schemas_fit_structured_output_limits(schema):
    sent = api_schema(schema)
    assert optional_count(sent) == 0
    for node in walk(sent):
        assert not (UNSUPPORTED & set(node)), node
        assert "default" not in node
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False


def test_clean_data_is_forgiving():
    schema = obj({"title": string(), "style": string(enum=["cards", "facets"], default="cards"), "n": {"type": "integer"}}, ["title"])
    data, warnings = clean_data({"style": "FACETS", "n": "3", "extra": 1}, schema)
    assert data == {"title": "", "style": "facets", "n": 3}
    assert any("missing" in w for w in warnings) and any("extra" in w for w in warnings)
    data, warnings = clean_data({"title": "x", "style": "grid"}, schema)
    assert data["style"] == "cards" and warnings


def test_markup_is_escaped_and_styled():
    html = str(rich("Use *this* **now** ==here== [[pill]] `code` <script>alert(1)</script>"))
    assert "<script>" not in html and "&lt;script&gt;" in html
    for fragment in ('<em class="accent">this</em>', "<strong>now</strong>", "<mark>here</mark>", '<span class="pill">pill</span>', '<code class="chip">code</code>'):
        assert fragment in html
    assert plain("**Bold** and *accent* and `code`") == "Bold and accent and code"
    wrapped = str(words("Say *it twice* <b>", "pop", 0.4))
    assert wrapped.count('class="w"') == 2 and wrapped.count('class="w accent"') == 2
    assert 'data-dur="0.4"' in wrapped and "&lt;b&gt;" in wrapped


def test_resolve_formats():
    assert resolve_formats(None, None, "mindmap") == ["png", "gif"]
    assert resolve_formats("auto", "mp4", "cheatsheet") == ["png", "mp4"]
    assert resolve_formats("png,mp4", None, "badge") == ["png", "mp4"]
    assert resolve_formats(["all"], None, "badge") == ["png", "gif", "mp4"]


def test_themes_define_every_token():
    for name in THEMES:
        assert set(BASE) <= set(theme_vars(name))
    with pytest.raises(ValueError):
        theme_vars("nope")


def test_learn_theme_keeps_green_out_of_the_funnel_and_tree():
    funnel, _ = build_html({"template": "funnel", "data": TEMPLATES["funnel"].sample, "theme": "learn"}, load_brand())
    assert "#25591d" not in funnel and "#2f2d2d" in funnel       # "do this" bars run from beige to charcoal
    tree, _ = build_html({"template": "decision_tree", "data": TEMPLATES["decision_tree"].sample, "theme": "learn"}, load_brand())
    assert "--yes:#2F2D2D" in tree and "--step-bg:#F6EED2" in tree
    default, _ = build_html({"template": "funnel", "data": TEMPLATES["funnel"].sample, "theme": "cream"}, load_brand())
    assert "#25591d" in default                                   # other themes keep their greens


@pytest.mark.parametrize("name", list(TEMPLATES))
@pytest.mark.parametrize("motion", [False, True])
def test_every_template_builds_html(name, motion):
    html, meta = build_html({"template": name, "data": TEMPLATES[name].sample}, load_brand(), motion=motion)
    assert "@font-face" in html and "window.__boot" in html and 'id="canvas"' in html
    assert ("html lang=\"en\" class=\"motion\"" in html) == motion
    assert meta["width"] in (1080, 1200) and meta["warnings"] == []


def test_commands_and_menu_paths_do_not_break_badly():
    html = str(rich("Type /learn-anything, or go to Customize → Skills → +"))
    assert '<span class="nw">/learn-anything</span>' in html
    assert "Customize\u00a0→ Skills\u00a0→\u00a0+" in html
    assert "nw" not in str(rich("See example.com/first-prompt, ~/.claude/skills/ and https://a.com/x-y"))
    assert plain("Customize → Skills → +") == "Customize → Skills → +"
