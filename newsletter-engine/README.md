# Newsletter Engine

Writes Mojeed Abisiga's own guides and newsletter issues, and builds them into a website that works like a
newsletter publication with a guide library. It learned the **structure** of a successful AI newsletter (how issues
open, how sections are cut, where prompts, screenshots and buttons go) from example issues, never their words.

The site lives at **mojeedabisiga.com/learn/** once published: cream paper, red accents and charcoal ink, tall
uppercase headings (Anton) with red italic accents (Playfair Display), Open Sans for reading and Oswald for labels, in
light and dark themes. The colours and fonts are tokens at the top of `static/css/site.css`. In `site.json`, `*words*`
in the tagline become the red italic accent on the home page, `wordmark` sets the AI WITH MOJEED ABISIGA name in the
header and footer (Archivo, like the logo files in `brand-logos/`), and `hero_image` the photo beside the headline:

| Page | Address | What it does |
|---|---|---|
| Home | `/learn/` | Subscribe box, "start here" picks, latest issues, guides |
| Newsletter archive | `/learn/archive/` | Every issue, with full-text search and topic filters |
| Issue | `/learn/p/<slug>/` | Byline, share buttons, the issue, subscribe box, previous/next |
| Guides | `/learn/guides/` | Evergreen guides, searchable, filtered by level |
| Guide | `/learn/guides/<slug>/` | Contents that follow you down the page, reading progress, checklists that remember ticks |
| Start here | `/learn/start/` | A learning path in modules, with "mark as done" progress (saved in the reader's browser) |
| Library | `/learn/library/` | Prompts and templates with Copy buttons, searchable by category |
| About, Subscribe | `/learn/about/`, `/learn/subscribe/` | The author, and a welcome page with the subscribe form |
| Feeds | `/learn/feed.xml`, `/learn/sitemap.xml`, `/learn/search.json` | RSS with full text, sitemap, search index |

Every page has its canonical address, description, Open Graph tags, a social image (a title card drawn in the site's
style) and structured data that points to the Person on mojeedabisiga.com. An email-ready copy of each issue is
written to `dist/email/` for pasting into a newsletter tool.

## What it learned

`python -m nle train --fetch` followed the links in the LinkedIn engine's example posts (lnkd.in short links
included) to 30 issues of the example newsletter, read its last 135 issues' metadata through Substack's public API,
and kept a few of its pages. That material stays on this machine (`training/data/`, `model/corpus.jsonl` and
`training/sources.local.json` are git-ignored). What's committed is only:

- **Measurements** (`model/style_profile.json`): a complete issue is about 2,250 words in about 87 very short
  paragraphs (half of them 12 words or fewer), cut into about 9 sections by dividers, with about 18 screenshots,
  2 prompt boxes, 7 lists and 7 buttons. Titles are about 2 words and 92% end with a full stop; subtitles about
  8 words, 37% ending with a colon that leads into the issue.
- **Patterns** (`nle/knowledge.py`, written by hand after reading): how an issue opens and closes, eight kinds of
  issue (from-zero walkthrough, set-up steps, one tool many jobs, prompt guide, tips, comparison, safety warning,
  essay) with their section plans, the guide blueprint, title formulas with `[placeholders]`, and the rules.
- **The style guide** Claude writes from: [`model/style_guide.md`](model/style_guide.md).

The engine borrows shape, never wording: its checks block the example creator's names and sites and any run of 12 or
more words copied from an example issue. One example issue teaches getting past AI detectors; it's excluded, and the
checks reject that kind of advice. The example creator's email-gated download page and certificate site were studied
but not imitated: the library's downloads are free, and nothing offers a credential.

## Set up

Python 3.11 or later. Run every command from this folder.

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium    # optional: social images
```

## Write an issue or a guide

Content lives in `content/`: issues in `content/issues/`, guides in `content/guides/`, each a Markdown file whose
name is its address. Three ways to write one:

- **By hand:** `python -m nle new issue "Your title"` (or `new guide`) starts a draft laid out the learned way.
- **With Claude Code:** ask for a newsletter issue or a guide. The `newsletter-engine` skill
  ([`.claude/skills/newsletter-engine/SKILL.md`](.claude/skills/newsletter-engine/SKILL.md)) has Claude write it from
  the style guide and your brief, then check it.
- **With a Claude API key:** `python -m nle write issue --brief briefs/claude-skills.json` writes a draft, runs the
  checks and has Claude fix any errors (up to two repairs). It prints the cost.

Drafts have `status = "draft"`. When it's ready (screenshots added, `[CHECK: ...]` notes filled in), change it to
`"published"`.

### The file format

```markdown
+++
title = "Make your first skill."
subtitle = "Turn the instructions you paste most into a Claude skill, in five steps:"
date = 2026-11-23              # issues; guides use updated = ...
status = "published"           # or "draft"
tags = ["claude", "skills"]
featured = true                # optional: shown first and marked "Popular"
level = "beginner"             # guides: beginner, intermediate or advanced
linkedin_url = ""              # optional: the LinkedIn post where readers can discuss it
brief = "briefs/claude-skills.json"   # the facts it was written from (the checks use it)
+++
The body, in Markdown.
```

Besides ordinary Markdown (headings, lists, `> quotes`, `---` dividers, tables, `- [ ]` checklists, images from
`content/media/`), the body can use:

| Mark | Becomes |
|---|---|
| ```` ```prompt Label ```` … ```` ``` ```` | A prompt box with a Copy button |
| `> [!TIP]` / `[!NOTE]` / `[!WARNING]` / `[!KEY]` | A coloured callout |
| `::button[Label](guide:slug)` | A button (to a guide, an issue, a page or a web address) |
| `::card[issue:slug]` | A card linking to another issue or guide |
| `::subscribe` | The subscribe box |
| `::screenshot[What it must show]` | A slot for a screenshot you still need to take (blocks publishing until filled) |
| `::download[Label](file)` | A download button for a file in `content/media/` |
| `[text](issue:slug)`, `(guide:slug)`, `(page:subscribe)` | Links that always point to the right page |

`content/path.toml` is the "Start here" path, `content/library.toml` the library, `content/about.md` the About page.

## Check, preview, publish

```bash
python -m nle check                 # every check on every issue and guide
python -m nle serve --drafts        # preview at http://localhost:8000/learn/
python -m nle build                 # the site in dist/learn/, email copies in dist/email/
python -m nle publish --into ../mojeedabisiga.com/site
```

`publish` refuses to run while a published piece has an error, then copies the site into `mojeedabisiga.com/site/learn/`.
Commit and merge as usual: the main site's checks run on the pull request, and merging publishes it.
The tests run those same checks with the publication inside the main site, so a publish can't break them.

**Errors** (block publishing): any spelling of a name in `site.json`'s `never_mention` (AI Every Time,
aieverytime.org), the example creator's names or sites, 12 or more words copied from an example issue, advice to cheat
on tests or get past AI detectors, a `lnkd.in` link not in the brief, an empty screenshot slot, a `[CHECK: ...]` note,
a link to a page that doesn't exist or isn't published, an image missing from `content/media/`.
**Warnings:** AI giveaways ("delve", em dashes, "It's not X, it's Y"), engagement bait, links not in the brief,
paragraphs over 70 words, long titles, no subtitle. **Notes:** numbers that aren't in the brief.

## Subscriptions

A static site can't send email, so the subscribe box connects to a newsletter service. Set it in `site.json`:

| `newsletter.provider` | Also set | What readers see |
|---|---|---|
| `none` (now) | | "Follow on LinkedIn" and "RSS feed" buttons |
| `buttondown` | `username` | An email form that subscribes them on Buttondown |
| `kit` | `form_action` (the form's address from Kit) | An email form |
| `substack` | `username` (your-name for your-name.substack.com) | Substack's own subscribe box |
| `beehiiv` | `embed_url` | Beehiiv's subscribe box |
| `form` | `form_action` | An email form that posts to any service |

To send an issue, open its copy in `dist/email/<slug>.html` in a browser, then copy and paste it into your service,
or paste its HTML. The service adds the unsubscribe link.

## The folder

```
newsletter-engine/
  site.json            the publication: name, author, address, subscribe service, names never to use
  content/             issues/, guides/, media/, about.md, path.toml, library.toml
  briefs/              the facts each piece is written from
  nle/                 the engine (python -m nle)
    content.py md.py   the file format and the Markdown marks
    build.py           the site: pages, feed, sitemap, search, social images (og.py), email copies (email.py)
    guardrails.py      the checks
    train.py           sources.py (download), analyze.py (measure), profile.py + knowledge.py (style guide)
    prompts.py llm.py pipeline.py   writing with Claude
  templates/ static/   the pages (Jinja) and their CSS, JavaScript, fonts (SIL Open Font License) and images
  model/               style_profile.json, style_guide.md (committed); corpus.jsonl (local)
  training/            annotations.json (committed); data/ and sources.local.json (local)
  tests/               python -m pytest
```
