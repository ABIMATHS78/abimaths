+++
title = "Opus 5.5, in plain words."
subtitle = "What changed, which Claude and effort to pick, and a six-step way to make it check its own work:"
description = "Opus 5.5 in plain words: what changed, how it compares with Sonnet, Haiku and Fable, how to set effort, and a six-step self-check with prompts."
date = 2026-11-28
status = "published"
tags = ["claude", "models", "opus"]
archetype = "comparison"
cover = "opus-5-5/cheat-sheet.webp"
cover_alt = "A one-page Opus 5.5 cheat sheet: what changed, which model, how to switch, which effort, the six-step self-check loop, when to stop checking, what to check yourself and a starter prompt"
brief = "briefs/opus-5-5.json"
fact_checks = ["Check the model menu next to the send button in your own Claude app: Opus 5.5, Sonnet 5.5, Haiku 5.5, 'More models' and 'Effort', with the levels Low, Medium, High, Extra high and Max.", "Check that Medium is still the level marked 'Default' for Opus 5.5 in the Effort menu, as Anthropic's docs and the Sonnet 5.5 announcement say.", "Check that Sonnet 4.6 is still due to leave the model selector on December 8, 2026, as the help centre says.", "Check the help centre's effort descriptions (Low and Medium for routine tasks, High as the best balance, Extra high for long-running coding and agentic tasks, Max for the deepest reasoning) against the current article.", "Check that fast mode for Opus 5.5 is still listed for Claude Code and the API at up to 2.5x speed and $8 / $40 per million tokens."]
+++
There's a new Claude in your model menu.

It's called Opus 5.5.

You've seen the headlines and the benchmark charts.

But you have three simple questions. Should I use it? For what? And how do I get the best work out of it?

This issue answers all three, in plain words.

I teach people to use AI at work, and these are the questions I hear after every new model.

So every fact here comes from Anthropic's own announcement, help centre and documentation. No hype, and no made-up tests.

**Inside:**

1. What changed, in six lines
2. What Anthropic tested, and under which conditions
3. The five models side by side
4. Which Claude for which job
5. How to switch, in three clicks
6. How to set the effort
7. A six-step loop that makes Claude check its own work
8. When to stop checking
9. The mistakes to avoid
10. What stays the same
11. The new words, in plain English

![A one-page Opus 5.5 cheat sheet with eight panels: what changed, which model, switching in three clicks, which effort, the self-check loop, when to stop, what to check yourself and a starter prompt](opus-5-5/cheat-sheet.webp "The whole issue on one page. Save it, print it, and keep it next to you the first week.")

::subscribe

---

## What changed, in six lines

Anthropic announced Claude Opus 5.5 on September 22, 2026, the first model in the Claude 5.5 family.

Here's what Anthropic says is new, compared with Opus 5:

- **Smarter.** It performs at the level of Claude Fable 5.1 on most work.
- **Cheaper.** It costs 40% less to run than Opus 5 on typical workloads.
- **Faster.** Its answers come more than 30% faster.
- **Easier to read.** Main point first, less jargon, and it follows your writing rules.
- **More careful with facts.** Anthropic's docs say it slips less often on figures and sources.
- **More room.** At launch, Anthropic raised five-hour limits on Pro, Max, Team and seat-based Enterprise, and gave subscribers one limit reset.

| | Opus 5 | Opus 5.5 |
|---|---|---|
| **API price per million tokens (in / out)** | $5 / $25 | $4 / $20 |
| **Default effort on the API** | High | Medium |
| **Thinking (API)** | On by default, but could be switched off | Always on, at every effort level |
| **Knows the world up to** | May 2026 | June 2026 |

**The lower default isn't a downgrade.** In Anthropic's tests, Opus 5.5 at Medium matches or beats Opus 5 at High on coding and knowledge work.

**Clearer writing makes checking easier.** Anthropic says it "has made Opus 5.5's work easier to follow and check". That idea runs through this issue.

> [!KEY]
> If you only remember one thing: Anthropic's docs say that if you're unsure which model to use, start with Opus 5.5.

It follows writing rules closely, so give it yours at the start of a chat.

```prompt My writing rules
For everything you write in this chat, follow these rules:
- Put the main point in the first line.
- Short sentences. Plain words. No jargon.
- [Add your own: British spelling, no bullet points in emails, a word limit]
If a rule clashes with something I ask, tell me which rule you broke and why.
```

**Use it when:** you start a chat that will produce text other people read.

::button[Read Anthropic's announcement](https://www.anthropic.com/claude-opus-5-5)

---

## What Anthropic tested

These are Anthropic's own numbers, not independent tests.

### The conditions

Unless noted, Anthropic ran these at max effort, with its production safety checks on, which it says likely reduce some scores. At Medium in the app, your results will differ.

### Real-world work

GDPval-AA v2.1 tests real-world work across 44 occupations. Scores are Elo ratings: higher is better.

![Bar chart of GDPval-AA v2.1 scores as Anthropic reports them: GPT-6 Astra 1542, GPT-5.6 Sol 1588, Opus 5 1708, Fable 5.1 1735 and Opus 5.5 1846](opus-5-5/scores.webp "Anthropic's numbers, at max effort with its safety checks on. Opus 5.5 leads, but Anthropic itself says the real-world gap with Fable 5.1 is narrower.")

### The full table

| Test | Area | Opus 5.5 | Compared with |
|---|---|---|---|
| GDPval-AA v2.1 | Knowledge work, 44 occupations | 1846 Elo | Fable 5.1: 1735. Opus 5: 1708 |
| Humanity's Last Exam, with tools | Knowledge work | 67.7% | |
| Terminal-Bench 4.0, at Extra high | Coding | 66.4% | Opus 5: 52.3% |
| CursorBench 4.0 | Coding | 57.8% | 52.5% at Medium |
| OSWorld 2.1 (partial) | Using a computer | 81.8% | |
| Chartography, with tools | Reading charts | 89.0% | |

At Medium, the default, Anthropic reports 54.6% on FrontierCode. It says that beats GPT-6 Astra's top score of 53.3%, for about a fifth of the cost per task.

### Research without made-up numbers

Anthropic asked Opus 5.5, Fable 5.1 and Opus 5 to write a report on a company's quarterly results, using only what they could find on a copy of the web.

An automated grader checked every figure and every quote against the sources. One invented figure or quote meant a fail.

Across effort settings, 16 of 18 Opus 5.5 reports passed.

Neither Fable 5.1 nor Opus 5 passed in any attempt.

> [!NOTE]
> Anthropic itself says benchmark margins have become a less reliable guide to real-world differences. In its own use, the gap between Opus 5.5 and Fable 5.1 is narrower than the scores suggest. Read the scores as direction, not as a promise for your work.

Send this to one colleague who keeps asking "which Claude should I use?"

---

## The five models side by side

The current lineup, from Anthropic's own pages.

![Five cards comparing Claude models: Opus 5.5 to start with, Sonnet 5.5 for everyday work, Haiku 5.5 for high volume, Fable 5.2 to step up and Opus 5 as the previous model, each with what it's for, its speed and its API price](opus-5-5/which-model.webp "One card per model: what it's for, how fast it is, and what developers pay per million tokens.")

| | Opus 5 | Opus 5.5 | Sonnet 5.5 | Haiku 5.5 | Fable 5.2 |
|---|---|---|---|---|---|
| **What it's for** | The previous Opus | Complex work that needs careful judgment | Well-scoped everyday tasks, polished documents, slides and spreadsheets | High-volume, quick tasks | Demanding reasoning and long agentic work |
| **Speed (Anthropic's label)** | Not rated. Opus 5.5 is 30%+ faster | Moderate | Fast | Fastest | Moderate |
| **Knows the world up to** | May 2026 | June 2026 | June 2026 | June 2026 | June 2026 |
| **API price (in / out)** | $5 / $25 | $4 / $20 | $2 / $10 | from $0.10 / $0.50 | $8 / $40 |
| **On the Free plan?** | No | No | Sonnet: yes | Haiku: yes | No |

Three notes:

- **API prices** are what developers pay. In the app, you pay your plan price, and the model you pick changes how fast you use your limits.
- **Free plan:** Anthropic's pricing table shows Sonnet and Haiku on Free. Opus is on Pro, Max and Ultra.
- **Fable 5.2** is on paid plans only. On Max and Ultra you can spend up to half your weekly limit on it. On Pro it runs on usage credits.

::card[issue:right-ai-for-the-job]

---

## Which Claude for which job

Anthropic calls Opus 5.5 its "strongest daily driver".

Its advice: start most work with Opus 5.5. Move up to Fable when the job needs more. Move down to Sonnet 5.5 or Haiku 5.5 when it needs less.

Two questions get you to the right model.

![A decision tree: does the job need careful judgment? If not, small and repeated jobs go to Haiku 5.5 and clear everyday work to Sonnet 5.5. If it does, start with Opus 5.5 and step up to Fable 5.2 only if Opus falls short. Each answer has a suggested effort and a prompt to try](opus-5-5/which-model-tree.webp "Two questions, four answers. Each one comes with a starting effort and a prompt to try.")

"Careful judgment" means a recommendation, a risk, a trade-off, a long or messy file, or work going to someone who matters.

Not sure where your own work fits? Let Claude sort your week, using the same rules.

```prompt Sort my week by model
Here are the tasks on my list this week: [paste the list].

Sort them into four groups, using these rules:
- Opus 5.5: needs judgment, long files, or goes to someone important
- Sonnet 5.5: clear, well defined, and I want it quickly
- Haiku 5.5: small, quick and repeated
- Fable 5.2: the hardest reasoning, only if Opus already fell short

Give me a table: task, group, one-line reason. Flag any task you're unsure about.
```

**Use it when:** you plan your week, or you keep hitting your limits. Keep the table and reuse it.

> [!TIP]
> Anthropic lists model choice and effort level among the things that use up your limits faster. Keep Opus 5.5 for the jobs that matter, and use Sonnet 5.5 or Haiku 5.5 for the rest.

::button[How Claude usage limits work](guide:claude-usage-limits)

---

## Switch to Opus 5.5 in three clicks

The path, from Anthropic's help centre:

1. Find the model name. On the web and the desktop app, it's next to the send button, below where you type. On mobile, it's at the top of the screen.
2. Click it, then pick **Opus 5.5**. Click **More models** to see the rest.
3. Click **Effort** and choose a level. Start with the one marked **Default**.

![Labelled drawing of the Claude model menu: click the model name, pick Opus 5.5, then click Effort](opus-5-5/model-menu.webp "Where the model and effort live. A drawing, not a screenshot: your screen may look a little different.")

**What you'll see:** the model name changes to Opus 5.5. The change applies from Claude's next response, and you can change it again at any point.

### Try it on one real task

Pick a task you did last week with another model. Run it again with Opus 5.5 at the default effort.

```prompt First test of Opus 5.5
I'm trying a new model on a task I do often.

Here's the task: [describe the task and paste the material].
Here's who will read the result: [your manager, a client, your team].
Here are my writing rules: short sentences, the main point first, no jargon.

Do the task. Then, in three lines, tell me what you were least sure about.
```

**Use it when:** you want to know if Opus 5.5 earns its place on a task you repeat.

**What you'll get back:** the task, plus three lines on its weakest points. Compare it with last week's answer yourself. That's the test that counts.

---

## Set the effort for the job

Effort controls how much thinking Claude puts into a response. Higher is more thorough, but slower, and uses more of your limits.

The path: model name, then **Effort**, then a level. The recommended one is marked **Default**.

![Labelled drawing of the Effort menu with five levels: Low for routine tasks, Medium as Opus 5.5's default, High as the best balance, Extra high for long coding and agent tasks, and Max for the deepest reasoning](opus-5-5/effort-menu.webp "Five levels, one rule: start at the default and go up only when the job is hard. A drawing, not a screenshot.")

| Level | What Anthropic says | Use it for |
|---|---|---|
| **Low** | Works well for routine tasks and stretches your usage further | Rewording, short replies, quick lookups in a file |
| **Medium** | Opus 5.5's default | Most everyday work: drafts, summaries, a first analysis |
| **High** | The best overall balance of quality and speed | A long document, a decision, a tricky analysis |
| **Extra high** | For long-running coding and agentic tasks, with deeper reasoning than High | Long coding or multi-step agent jobs |
| **Max** | The most thorough option, for the deepest reasoning | The hardest problem of your month, when you can wait |

Three rules from Anthropic's docs:

- **Want a lighter answer? Lower the effort first.** It cuts thinking, cost and waiting more reliably than typing "be quick".
- **Save Extra high and Max** for work where you've seen them make a real difference.
- **Don't copy your old setting across.** At the same level, Opus 5.5 tends to think more per turn than Opus 5, especially at Extra high and Max.

### Plan high, write at the default

For big work, try two passes. Anthropic's Claude Code guide suggests a similar split: plan with Opus, then do the work with Sonnet.

Set effort to **High** and send this:

```prompt Plan it first
Before writing anything, plan this with me.
The task: [describe it]. The reader: [who]. The deadline: [when].
Give me the 3 to 5 sections you'd write, the facts each section needs, and anything you need from me.
Don't write the draft yet.
```

**Use it when:** the work is long, or a wrong start would waste an hour.

**What you'll get back:** a short outline with the gaps named. Then set effort back to **Medium** and ask for the draft, section by section.

---

## Make Claude check its own work, in six steps

A smarter model still makes mistakes.

Anthropic's help centre says Claude can "hallucinate": give incorrect or misleading information, even about its own abilities, like claiming it sent an email. It says not to rely on Claude as your only source of truth.

Anthropic says Opus 5.5 slips less often on figures. Less often isn't never.

So here's a loop for anything with your name on it: six steps, one prompt each, in the same chat as the work.

![A roadmap of the self-check loop in six stops: set the finish line, allow I don't know, trace every claim, grade against the list, attack the weak spots, and fix, log and hand over, ending with your own check](opus-5-5/self-check-loop.webp "The whole loop on one page. Each stop gets its own card below.")

### Step 1: Set the finish line

Before Claude writes a word, decide what "done and right" means.

Anthropic's prompting guide says to treat Claude like a brilliant new employee who lacks context, and to be specific about the format and limits you want. A finish line does both, as tests.

![Step 1 card: what to do, the prompt to copy, examples of good tests such as one page or less and every figure from my file, a weak versus strong example, and why it works](opus-5-5/step-1.webp "Good tests are things you could check in seconds: a length, a position, a source.")

```prompt Step 1: Set the finish line
Here's what I need: [the task].
It's for: [who reads it, and what they'll do with it].

Before you start, write the finish line: 3 to 7 tests this must pass to be done and right.
Make each test something I could check quickly, like a length, a position or a source.
Add any test you think I've missed. Then wait for my OK.
```

**Use it when:** someone else will read the result.

**What you'll get back:** a numbered list, such as "1. One page or less. 2. The recommendation comes first. 3. Every figure comes from the attached file." Edit it, then say OK.

### Step 2: Allow "I don't know"

Now ask for the work, with one extra permission.

Anthropic says giving Claude explicit permission to say "I don't know" can drastically reduce false information.

![Step 2 card: paste your material, allow I don't know, tag guesses as GUESS, and get questions back, with a weak versus strong example](opus-5-5/step-2.webp "A tagged guess is easy to fix. A smooth guess goes out under your name.")

```prompt Step 2: Do the work, flag the gaps
Do the task now, using only the material I've given you.
If something isn't in my material, or you're not sure, say "I don't know" or tag it [GUESS].
Don't fill gaps to make it read smoothly.
At the end, list the questions you'd want me to answer.
```

**Use it when:** the finish line is agreed. Attach your notes or files.

**What you'll get back:** a draft with a few [GUESS] tags and some questions. That's a good sign.

### Step 3: Trace every claim

Next, make every fact show where it came from.

![Step 3 card: list every number, name, date and quote with its source, mark untraceable ones unverified, with an example table and a tip for long files](opus-5-5/step-3.webp "The row marked 'own memory' is the one to check first.")

```prompt Step 3: Trace every claim
Before I use this, check your answer.
List every number, name, date and quote you used.
For each one, say where it came from: the file I gave you, a source you found, or your own knowledge.
Mark anything you can't trace to a source as "unverified".
Don't fix anything yet. Just show me the list.
```

**Use it when:** an answer has numbers, names or dates in it.

For long reports, contracts and threads, Anthropic suggests a stronger version: quotes first, answer second.

```prompt Step 3, for long files: Quotes first
Before you answer, find the word-for-word quotes in the document that matter for my question. Number them.
Then answer using only those quotes, citing them by number.
If a claim has no quote behind it, take it out and tell me you did.
```

For decks and spreadsheets, use what Anthropic says Opus 5.5 is now good at: catching details that are easy to miss in large files.

```prompt Step 3, for decks and sheets: Find the mismatches
Check this file for things that don't agree with each other.
For example: a chart that doesn't match the figures in the table, a total that doesn't add up, a date that falls on the wrong weekday, or a name spelled two ways.
List each mismatch with where you found it. Don't change anything yet.
```

**Use it when:** a deck or workbook is about to go to anyone else.

### Step 4: Grade against the finish line

Now hold the draft up against the tests from Step 1.

![Step 4 card: paste the finish line, ask for pass, partly or fail per test with evidence, and no fixes yet, with an example grading table](opus-5-5/step-4.webp "Grade first, fix later. Mixing the two hides what changed.")

```prompt Step 4: Grade against the finish line
Read the finish line we agreed again. Then read your draft.
Make a table with three columns: the test, the result (pass, partly or fail), and the line in the draft that proves it.
Be strict. If you can't point to a line, it's not a pass.
Don't fix anything yet.
```

**Use it when:** the draft is done, before you ask for any edits.

**What you'll get back:** rows like "Three risks: partly. Only two are named." The gaps are visible at a glance.

### Step 5: Attack the weak spots

A draft can pass every test and still be wrong. So ask Claude to argue against itself, as your toughest reader would.

![Step 5 card: ask for the three strongest reasons it's wrong, pick a critic for each kind of work, and what good attacks look like](opus-5-5/step-5.webp "Pick the critic who will actually read it: the client, the finance lead, the person doing the work.")

```prompt Step 5: Argue against yourself
Now take the other side, as [my reader: the client, our finance lead, a new starter] would.
Give me the three strongest reasons your draft could be wrong or fail with them, worst first.
For each one, quote the line it's about and say what I'd need to check to find out.
Then tell me if you'd still make the same recommendation, and why.
```

**Use it when:** the work recommends something, or someone senior will read it.

**What you'll get back:** three ranked objections, each tied to a line, and a clear "yes, still" or "no, change it".

### Step 6: Fix, log and hand over

Only now does Claude fix anything. And it shows you what it changed.

![Step 6 card: fix only what failed, get a change log and a list of what to check yourself, with an example change log and your last-mile checks](opus-5-5/step-6.webp "The change log is your shortcut: check what changed, not the whole thing again.")

```prompt Step 6: Fix, log and hand over
Fix only the tests that failed or were partly met, and the objections you agreed were fair.
Then give me:
1. The new version.
2. A change log: what you changed, and why, one line each.
3. A list of what I must still check myself, such as unverified claims, sources to open and figures to compare.
```

**Use it when:** Steps 4 and 5 are done, never before.

If Claude says it did something, one more check.

```prompt Step 6: Check the actions
List every action you said you took in this chat: files made, searches run, messages sent.
For each one, tell me whether you actually did it with a tool, or only described it.
Give me the file name or link for anything you made.
```

**Use it when:** Claude says it saved, sent, searched or created something.

### The whole loop in one prompt

**Use it when:** the job is small, and six separate messages would be overkill.

```prompt The whole loop, in one go
Do the task below in six steps, and show each step under its own heading:
1. Write the finish line: 3 to 5 tests this must pass.
2. Do the task. Tag anything you're unsure of as [GUESS].
3. List every number, name and quote with its source, or mark it "unverified".
4. Grade the draft against the finish line: pass, partly or fail.
5. Give the two strongest reasons it could be wrong.
6. Fix what failed, and list what I must still check myself.

The task: [describe it and paste the material].
```

> [!WARNING]
> A self-check makes Claude's work better. It doesn't replace yours. For anything with your name on it, open the sources and check the numbers yourself.

::card[issue:check-a-skill-before-you-trust-it]

---

## When to stop checking

Checking has a cost. Every round uses your limits and your time.

So decide your exits before you start.

![Two columns: stop when every test passes, a round finds nothing new, it needs a fact only you have, or after two or three rounds on one problem; keep going when a test still fails, a key claim is unverified, items are still open or it offers to continue instead of continuing](opus-5-5/stop-early.webp "Know your exits. After two or three rounds on the same problem, look at your brief instead.")

Anthropic gives similar advice for long, unattended agent work: stop after two or three automatic nudges on the same task, so a stuck job ends and can be reviewed.

It also says to treat a "done" as a report, not as proof. Check the list.

**Use this when:** Claude stops, or says it's finished, with work still open.

```prompt Finish the open items
Your checklist still has open items: [list them].
Carry on with them now.
If one is blocked, tell me exactly what's blocking it and what you need from me.
```

---

## The mistakes to avoid

![Eight mistakes to stop, such as trusting benchmark charts over your own test and using Max effort on every job, above eight fixes to start, such as running one real task and starting at Medium](opus-5-5/mistakes.webp "Stop the habits at the top. Start the ones at the bottom.")

The three that cost the most:

- **Asking "is this right?"** A yes tells you nothing. Ask for a graded table with evidence (Step 4).
- **Max effort on everything.** It's slower and uses more of your limits. Start at the default.
- **Staying on Sonnet 4.6.** It leaves the model selector on Free, Pro, Max and Ultra on December 8, 2026. Anthropic recommends Sonnet 5.5 instead.

---

## What stays the same

You don't need to relearn Claude. Here's what carries over, from Anthropic's docs:

- **The model menu.** It's in the same place, next to the send button.
- **The effort levels.** Low, Medium, High, Extra high and Max, with Medium as the default in the Claude apps.
- **Thinking stays on.** As on Opus 5, Sonnet 5.5 and Haiku 5.5, you can't turn it off in the app. Lower the effort instead.
- **The amount of text it takes in.** Up to 1M tokens in chat on paid plans, as with Opus 5. For scale, Anthropic says 200K tokens is about 500 pages.
- **Zero data retention.** Like previous Opus models, Opus 5.5 is available with it.
- **Your prompts.** A good brief is still a good brief.

### Two things that might surprise you

**Claude may switch models for some requests.** Opus 5 and Opus 5.5 run safety checks on every request, and on what Claude reads, such as files and search results. Routine bug fixing is fine, but most cybersecurity tasks may be answered by Opus 4.8, and some biology requests by Opus 5. That's expected behaviour.

**Research has its own model.** Claude's Research feature runs on Opus 5.5, whatever model you picked for the chat.

### For developers

The model ID is `claude-opus-5-5`, on the Claude API, Amazon Web Services, Google Cloud and Microsoft Azure. Fast mode runs it at up to 2.5x speed in Claude Code and on the Claude Platform, at $8 / $40. Requests that switch thinking off now return an error: set an effort level instead.

::button[Brief Claude like a colleague](guide:brief-claude-like-a-colleague)

---

## The new words, in plain English

Model launches come with their own vocabulary. Here's every term from this issue, in one line each.

![A glossary of 18 terms in plain English, from adaptive thinking, context window and effort to fallback, fast mode, safeguards, tokens and usage limits](opus-5-5/glossary.webp "Eighteen terms, one line each. Come back to it when an announcement loses you.")

If you learn one, make it **effort**: the setting you'll touch most.

::card[issue:claude-set-up-right]

---

## What to do today

Open Claude.

Click the model name next to the send button and pick Opus 5.5.

Leave effort on the default.

Paste "My writing rules" at the top of a new chat.

Run one real task from your week.

Then run the six-step loop on it, starting with the finish line.

You'll learn more about your own work than any benchmark can tell you.

::subscribe

::card[guide:claude-from-zero]

Send this to one person on your team who still uses the default model without knowing why.

Mojeed
