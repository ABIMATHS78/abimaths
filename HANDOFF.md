# Handoff: mojeedabisiga.com/learn, the LinkedIn post engine and the brand files

This guide is for a teammate taking over Mojeed Abisiga's content work from their own Claude account. Start a new Claude Code session (claude.ai/code) with the GitHub repository **ABIMATHS78/abimaths** attached, then paste this whole file into the first message. It explains what exists, how to run it, what's unfinished, and the rules to keep.

Other projects in the same repository have their own guides:

- `mojeedabisiga.com/HANDOFF.md`: Mojeed's personal website and Google knowledge panel.
- `ai-every-time/HANDOFF.md`: the AI Every Time website.

## 1. What this project is

Three connected parts, all signed as **Mojeed Abisiga** (Data & AI Leader · Educator · Speaker):

| Part | Folder | What it does |
|---|---|---|
| **The /learn publication** | `newsletter-engine/` (package `nle`) | Writes and builds Mojeed's free guides and newsletter issues into a website at **https://mojeedabisiga.com/learn/**: home, archive, guides, a "Start here" path, a prompt library, About and Subscribe. Subscriptions go through his Substack, **mojeedabisiga.substack.com**. |
| **The LinkedIn post engine** | `linkedin-engine/` (package `lpe`) | Writes LinkedIn posts in a style learned from 26 example posts, and draws their graphics (PNG, GIF or MP4). Posts send readers to the exact /learn page that goes deeper. |
| **Brand files** | `brand-logos/` | Logos (AI WITH MOJEED ABISIGA in red, LEARN AI *With* MOJEED in cream, wide and round versions) and LinkedIn banners (profile and company page), with the scripts that drew them. |

The /learn site is published inside the main website: `nle publish` copies it into `mojeedabisiga.com/site/learn/`, and the GitHub Pages workflow deploys `mojeedabisiga.com/site/` when a pull request is merged into `master`.

Each engine has a full manual, `newsletter-engine/README.md` and `linkedin-engine/README.md`, and a Claude Code skill (`.claude/skills/newsletter-engine/` and `.claude/skills/linkedin-post-engine/`) that loads automatically. Read the manual before changing anything in an engine.

## 2. Where things stand (26 Nov 2026)

- **Everything is merged and live.** The last pull request was #70. Nothing is waiting for review.
- **On /learn:** 4 guides and 4 issues.
  - Guides: Claude skills from zero; Skill, project, instructions or connector?; Ask Claude about your data; Brief Claude like a new colleague.
  - Issues: Make your first skill; Ten skills worth writing; The one line Claude reads; Check a skill before you trust it.
- **Look:** cream, red (`#D02E2E`) and charcoal, modelled on learnaiwithmariah.com/challenge. Headings use Anton, accents Playfair Display italic, body text Open Sans, labels Oswald, and the AI WITH MOJEED ABISIGA wordmark Archivo. All fonts are served from the site.
- **Every guide and issue ends with a paywall box** ("The rest of this guide is for paid subscribers"), like a paid Substack post. Its words, prices and links are in `newsletter-engine/site.json` under `paywall`.
- **Substack paid subscriptions are not switched on yet.** See section 7.
- **LinkedIn posts:** 11 have been drafted. Their text and graphic specs are in `linkedin-engine/posts/`. Section 6 lists which ones Mojeed liked.

## 3. Accounts you need

Ask Mojeed to invite **your own email address** to each one. Never share a login.

| Service | What it's for | How Mojeed invites you |
|---|---|---|
| **GitHub**, repo `ABIMATHS78/abimaths` | All code and pull requests | Repo **Settings → Collaborators → Add people** (Write access) |
| **Claude Code** (claude.ai/code) | Working with Claude | Sign in with your own account and connect GitHub once you have repo access |
| **Substack**, publication mojeedabisiga | Subscribers, paid plans, emails | Substack dashboard **Settings → Team** (or **Contributors**): add you as an admin |
| **Google Drive** training folder | The 26 example posts the LinkedIn engine learned from | Mojeed shares the folder with you (read-only is enough) |
| **Anthropic API key** (optional) | Only for `lpe generate` and `nle write`, which call Claude directly | From the Claude Console. Without a key, write posts and guides in Claude Code instead. |

LinkedIn posting stays with Mojeed: hand him the text and the graphic, and he posts them.

## 4. Set up (once)

Python 3.11 or later, Chromium (via Playwright), and ffmpeg for GIF/MP4.

```bash
cd linkedin-engine && python -m pip install -r requirements.txt && cd ..
cd newsletter-engine && python -m pip install -r requirements.txt && cd ..
python -m playwright install chromium   # not needed in Claude Code cloud sessions
```

**Training data is not in the repository on purpose.** The example posts belong to another creator, so they're git-ignored (`training/data/`, `model/corpus.jsonl`, `training/sources.local.json`). Get the Google Drive folder link from Mojeed, then:

```bash
cd linkedin-engine && python -m lpe train --drive "<the folder link>"
```

Without it, `lpe render` and `lpe lint` still work, but `lpe examples` and the copied-wording check have nothing to compare against. The newsletter engine's learned style files are committed, so it needs nothing extra.

Run the tests to confirm the setup works:

```bash
cd newsletter-engine && python -m pytest -q
cd ../linkedin-engine && python -m pytest -q
python3 mojeedabisiga.com/tools/check_site.py mojeedabisiga.com/site     # from the repo root
```

## 5. How to do the common jobs

### Add a guide or an issue to /learn

1. `cd newsletter-engine && python -m nle new guide "Title"` (or `new issue`). Write it in `content/guides/` or `content/issues/`. Markdown with `+++` front matter; the manual lists the extras: prompts to copy, callouts, buttons, cards, `::subscribe`.
2. `python -m nle check`: fix every error and read the warnings.
3. `python -m nle publish --into ../mojeedabisiga.com/site`, then `python -m nle catalog`. The catalog tells the LinkedIn engine the new page exists.
4. From the repo root: `python3 mojeedabisiga.com/tools/check_site.py mojeedabisiga.com/site`. It must say OK.
5. Optionally add the page to `content/path.toml` (Start here) and its prompts to `content/library.toml`.
6. Commit, push your branch and open a pull request. **Merging to `master` deploys it**, in about a minute.

### Make a LinkedIn post

Ask Claude Code for a LinkedIn post; the `linkedin-post-engine` skill walks through the steps. In short:

- **Write the brief:** save one in `linkedin-engine/briefs/` with only real facts, numbers and links.
- **Read the style:** `python -m lpe guide` and `python -m lpe examples "<topic>" --k 3 --full`.
- **Point to /learn:** link the exact /learn pages from `learn_catalog.json`. If the guide the post needs doesn't exist yet, write and publish it first.
- **Render and check:** create `output/<date>-<slug>/spec.json`, then `python -m lpe render <spec>`. Fix every error from the checks. The post must pass `python -m lpe lint`.
- **Save it:** copy `spec.json` and `post.txt` into `linkedin-engine/posts/<date>-<slug>/`, so the next person can see it (`output/` isn't saved in Git).

Graphic templates: `python -m lpe templates`. Two were built to match samples Mojeed liked:

- **`cheatsheet`** with `"headline_in_column": true`, `"number_panels": false` and a `"column"` on each panel: the "CLAUDE, explained like you're 5" look.
- **`puzzle`**: five features as jigsaw pieces, the "… Explained" look.

Themes: `learn` is the /learn site's cream and red, and `cream` is terracotta (the samples' colour).

### Prepare Substack posts

`cd newsletter-engine && python -m nle substack` writes every issue and guide as paste-ready HTML, plus a kit page with copy buttons, into `dist/substack/`. Images load from the live site, so publish the site first.

### Change the look

- **Colours and fonts:** tokens at the top of `newsletter-engine/static/css/site.css` (light and dark themes).
- **`site.json` settings:**
  - `tagline`: `*words*` become the red italic accent;
  - `hero_image`: the home and About portrait;
  - `wordmark`: the header and footer name;
  - `subscribe_title`: the subscribe page headline;
  - `paywall`: the paid-subscription ending;
  - `never_mention`: names the site must never show.
- **Social images** (`nle/og.py`) and **email copies** (`nle/email.py`) use the same palette.

## 6. The LinkedIn posts so far (`linkedin-engine/posts/`)

| Folder | Topic | Mojeed's reaction |
|---|---|---|
| 2026-11-24-which-claude-tool | Skill vs project vs instructions vs connector | Early post |
| 2026-11-24-check-a-skill | Check a skill before you install it | Early post |
| 2026-11-24-skill-description | The one line Claude reads | Early post |
| 2026-11-24-skills-masterclass | Skills from zero, 8 lessons, 9 /learn links | Asked for after "not detailed enough" feedback |
| 2026-11-25-ten-skills | Ten skills worth writing | No reaction recorded |
| 2026-11-25-skill-fails | Why your first skill fails | **Rejected** (didn't like the hook or graphic) |
| 2026-11-25-data-answers | Trust Claude's answers about your data | No reaction recorded |
| 2026-11-25-brief-claude | Brief Claude like a colleague | **Rejected** (topic, graphic and hook) |
| 2026-11-25-eight-guides | 8 free guides in 3 levels (like sample #25) | No reaction recorded |
| 2026-11-25-claude-explained | Five features as jigsaw pieces (like sample #11) | Not what he meant (he wanted sample #3) |
| 2026-11-26-claude-eli5 | "CLAUDE, explained like you're 5" cheat sheet (like sample #3) | Asked for more detail, now 9 panels |

**What Mojeed wants from posts:**

- He likes posts that follow one of the samples closely in shape and graphic. Ask which sample when unsure; the samples are "Post 1" to "Post 26" in the Drive folder.
- He wants graphics as dense and detailed as the samples.
- He wants strong, specific hooks.
- He wants every post to point to the exact /learn guide that goes deeper.

To redraw any post: `cd linkedin-engine && python -m lpe render posts/<folder>/spec.json`.

## 7. Unfinished work

1. **Turn on Substack paid subscriptions.** Mojeed was in the middle of this.
   - In the publication dashboard, go to **Settings → Payments** and connect Stripe.
   - Set the prices to **$15 a month and $150 a year**. These must match the paywall text on /learn.
   - Add a **$300 Founding member** plan, then turn on **Enable payments**.
   - Write the paid-subscriber benefits and the two welcome emails; drafts were given to Mojeed in chat.
   - Stripe must support the country of his bank account.
2. **Create the paid content the paywall promises.** Make three paid-only Substack posts titled exactly:
   - "Next live session";
   - "Template and prompt packs";
   - "Tool deals".
   Then switch on Substack Chat for paid subscribers. The paid welcome email links to `mojeedabisiga.substack.com/p/next-live-session`, `/p/template-and-prompt-packs` and `/p/tool-deals`, so the titles must match.
3. **Post the issues to Substack** with the kit from `nle substack`. This was paused by Mojeed.
4. **Ask before acting on these:**
   - **A separate repository for the LinkedIn engine:** on hold, by Mojeed's choice.
   - **The Substack accent colour:** set it to `#D02E2E` so the subscribe form's button matches the site. Mojeed has to do this in Substack.
   - **The newsletter paywall wording:** the box says "The rest of this guide is for paid subscribers". Mojeed may want the exact Substack wording, "This post is for paid subscribers".

## 8. Rules to keep

- **Never name "AI Every Time" (aieverytime.org)** in LinkedIn posts, their graphics, or anything on /learn. Both engines check for it.
- **Never copy the example posts' wording.** Follow their shape only. The LinkedIn engine flags shared runs of words, so fix every flag.
- **Never invent facts, numbers, quotes, stories or links.** Use only what's in a brief or a /learn guide, and write `[CHECK: …]` when something's missing. For example, Mojeed has no real "I taught my mom" story, so don't write one.
- **Don't change Mojeed's prices or promises** without asking him. The paywall's price and perks came from him: live sessions, template and prompt packs, a private community, early access, and free or discounted tools and events.
- **Keep the checks green before every pull request:** both test suites, `nle check`, `lpe lint` and `check_site.py`.
- **Work on a branch and open a pull request.** Mojeed reviews and merges, and merging publishes.
- **No secrets in the repository:** no API keys, passwords or personal data, in code or in skills.
