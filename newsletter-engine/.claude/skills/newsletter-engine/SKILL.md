---
name: newsletter-engine
description: Writes Mojeed Abisiga's newsletter issues and guides in the structure learned from successful AI newsletters, and builds them into his publication at mojeedabisiga.com/learn/ (archive, guides, start-here path, library). Use when asked for a newsletter issue, a guide, a Substack-style post, or to build, check, preview or publish the newsletter site.
---

# Newsletter engine

The engine is the Python package `nle`. Run every command from the folder that contains `nle/` and `site.json`
(`newsletter-engine/` inside the `abimaths` repository).

## 1. Set up (once per machine)

```bash
python -m pip install -r requirements.txt
```

## 2. Get the brief

You need the topic, who it's for, what the reader should be able to do, and the real material: facts, steps, menu
paths, numbers, links, the author's own experience. Save it as `briefs/<name>.json` (see `briefs/claude-skills.json`).
Ask for what's missing in one message. Never invent facts, numbers, stories, quotes, links or screenshots.

## 3. Write it

1. Run `python -m nle guide` and read the whole style guide: the structure of an issue, the kinds of issue, the
   guide blueprint, the marks and the rules.
2. Read one published issue or guide in `content/` to see the marks in use.
3. Start the file: `python -m nle new issue "Title"` (or `new guide`), then replace its body. Fill the header:
   title, subtitle, description, tags, archetype, `brief = "briefs/<name>.json"`, `fact_checks`.
4. Write in Mojeed's voice: plain, warm, direct; very short paragraphs; the exact steps; prompts in prompt boxes;
   a `::screenshot[...]` slot wherever the reader must find something on screen. Link to published pieces with
   `issue:<slug>` and `guide:<slug>`.
5. Never mention AI Every Time or aieverytime.org, in any spelling, and never the example creator or his sites.

## 4. Check and preview

```bash
python -m nle check
python -m nle serve --drafts      # http://localhost:8000/learn/
```

Fix every `[error]` and look at the warnings. A draft may keep screenshot slots and `[CHECK: ...]` notes; a published
piece may not.

## 5. Hand over

Tell the user the file path, the screenshots still to take (the check lists them) and the fact checks. Only the
author changes `status` to `"published"` and runs `python -m nle publish --into ../mojeedabisiga.com/site`.

## With a Claude API key

`python -m nle write issue --brief briefs/<name>.json` does step 3 and the checks in one go, with repairs.
