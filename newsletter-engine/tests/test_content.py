import datetime as dt

import pytest

from nle.content import ContentError, Item, dump, load_library, parse
from nle.md import render

GOOD = '+++\ntitle = "Hello."\nsubtitle = "A test."\ndate = 2026-11-23\ntags = ["a"]\n+++\nBody text.\n'


def test_parse_and_dump_round_trip():
    item = parse(GOOD, "issue", "hello")
    assert item.title == "Hello." and item.date == dt.date(2026, 11, 23) and item.tags == ["a"] and item.published
    again = parse(dump(item), "issue", "hello")
    assert (again.title, again.subtitle, again.date, again.tags, again.body) == (item.title, item.subtitle, item.date, item.tags, item.body)


@pytest.mark.parametrize("text, message", [
    ("no header", "must start"),
    ('+++\ntitle = "x"\nfoo = 1\ndate = 2026-01-01\n+++\nb', "unknown header"),
    ('+++\nsubtitle = "x"\n+++\nb', "needs a title"),
    ('+++\ntitle = "x"\n+++\nb', "needs a date"),
    ('+++\ntitle = "x"\nstatus = "live"\ndate = 2026-01-01\n+++\nb', "status"),
])
def test_bad_files_are_explained(text, message):
    with pytest.raises(ContentError, match=message):
        parse(text, "issue", "x")


def test_slug_must_be_an_address():
    with pytest.raises(ContentError, match="lowercase"):
        parse(GOOD, "issue", "Hello World")


def test_the_sample_library_loads():
    library = load_library()
    assert library.issues and library.guides and library.path.get("modules") and library.library.get("items")
    assert [i.sort_date for i in library.issues] == sorted((i.sort_date for i in library.issues), reverse=True)
    assert all(i.published for i in library.visible().all())


def test_marks_render_safely():
    body = """Hi **there** [guide](guide:x) [bad](javascript:alert(1)) <script>x</script>

> [!WARNING]
> Careful.

```prompt Try this
Do the thing.
```

::button[Go](page:subscribe)
::screenshot[The settings page]
::card[issue:missing]

- [ ] tick me
"""
    out = render(body, resolve=lambda ref: {"guide:x": "/g/x/", "page:subscribe": "/s/"}.get(ref))
    html = out.html
    assert '<a href="/g/x/">guide</a>' in html and "<script>" not in html and "javascript:" not in html.split("[bad]")[0]
    assert 'class="callout callout-warning"' in html and "Careful." in html and "[!WARNING]" not in html
    assert 'class="prompt"' in html and "Try this" in html and "data-copy" in html
    assert 'href="/s/"' in html and "figure-missing" in html and 'type="checkbox"' in html
    assert out.screenshots == ["The settings page"] and any("issue:missing" in p for p in out.problems)
    assert "\n" in out.text   # one line per block, so checks never read across blocks


def test_item_defaults():
    item = Item(kind="guide", slug="g", title="T", body="b\n")
    assert item.ref == "guide:g" and item.published and item.sort_date == dt.date.min
