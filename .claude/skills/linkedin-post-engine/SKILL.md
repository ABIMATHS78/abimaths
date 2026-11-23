---
name: linkedin-post-engine
description: Writes a LinkedIn post and renders its graphic as a PNG, GIF or MP4 in the house style the engine learned from the training posts, for Mojeed Abisiga and AI Every Time. Use when asked for a LinkedIn post, a post graphic or infographic, an animated GIF or short video for LinkedIn, or to retrain the style from the Google Drive folder.
---

# LinkedIn post engine

The engine is in `linkedin-engine/` (Python). It learned a writing style and 17 graphic templates from example
posts, and renders graphics with headless Chromium and ffmpeg. With an API key it can do the whole job itself
(`python -m lpe generate`); without one, you write the post and the graphic data, and the engine checks and renders them.

Run every command from `linkedin-engine/`.

## 1. Set up (once per machine)

```bash
python -m pip install -r requirements.txt
python -m playwright install chromium   # not needed in Claude Code cloud sessions: the shared Chromium is used
```

GIF and MP4 also need `ffmpeg` on the PATH. If `model/corpus.jsonl` is missing, the example posts haven't been
downloaded on this machine: run `python -m lpe train` (it reuses the folder link saved in
`training/sources.local.json`) or `python -m lpe train --drive "<folder link>"` and ask the user for the link.

## 2. Get the brief

You need: the topic, who it's for, what the reader should do, and the real material: facts, numbers, steps,
links, the author's own experience. Ask for what's missing in one message. Never invent facts, numbers, stories,
quotes or links: write `[CHECK: what's needed]` instead.

## 3. Write the post

1. Read `model/voice_guide.md` (the trained style) and `brand.json` (who is writing and where readers go).
2. Run `python -m lpe examples "<topic>" --k 3 --full` and read the closest example posts for shape only.
   They belong to another creator: never reuse their wording, facts, names, links or sign-offs.
3. Write the post: a hook formula from the guide filled with the brief's specifics, one body structure, one
   closing that points to the author's site. Plain text, short lines, no markdown, no hashtags.

## 4. Design the graphic

1. Run `python -m lpe templates` and pick the template that turns the post into a save-worthy image
   (the guide's "Which template fits" section helps). Then `python -m lpe templates <name> --sample` shows its
   fields and an example.
2. Create `output/<YYYY-MM-DD>-<slug>/spec.json`:

```json
{
  "template": "card_grid",
  "data": { "...": "fields from the template's schema" },
  "post": "the full post text",
  "alt_text": "one or two sentences describing the graphic",
  "formats": ["png", "gif"],
  "alternative_hooks": ["...", "..."],
  "fact_checks": ["anything the author must verify"],
  "brief": { "topic": "...", "notes": "the facts you were given", "links": [] }
}
```

`formats`: `png` is always made; add `gif` (drawn build-up, best for `mindmap` whiteboard) or `mp4`
(`kinetic`, `decision_tree`, `big_number`). Text markup on graphics: `*accent*`, `**bold**`, `==highlight==`,
`[[pill]]`, `__underline__`, `` `code` ``. Optional `"theme"`: run `python -m lpe themes`.

## 5. Render and check

```bash
python -m lpe render output/<YYYY-MM-DD>-<slug>/spec.json
```

It writes `graphic.png` (plus GIF/MP4), `post.txt`, `post.md`, `preview.html` (LinkedIn-style preview) and
`report.json`. Fix every `[error]` it prints (third-party names or links, copied wording, length, markdown) and
look at the warnings, then render again. Open `graphic.png` and check it reads well: if the text is tiny, cut
items; if the layout looks empty, the template may not fit the material.

## 6. Hand over

Give the user the post text ready to paste, the graphic file paths, the two other openings, and the fact checks.

## With a Claude API key

`python -m lpe generate "<topic>" --notes @notes.txt --format auto` does steps 3 to 5 in one go, with a repair
pass when the checks find errors. `--template`, `--theme` and `--format png|gif|mp4|all` override its choices.
