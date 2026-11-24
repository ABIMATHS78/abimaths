+++
title = "Ten skills worth writing."
subtitle = "The ten Claude skills I wrote for the people I teach: what each one does, why it works, and a request to try:"
description = "Ten Claude skill ideas for learning, research, your career, everyday work and data, each with what makes it work and a request to try."
date = 2026-11-23
status = "published"
tags = ["claude", "skills"]
archetype = "tips_list"
featured = true
cover = "skills/ten-skills.webp"
cover_alt = "A grid of ten cards, one for each Claude skill"
brief = "briefs/claude-skills.json"
+++
People ask me the same question every week.

"I get what a skill is. But what should I actually make one for?"

So here's my answer, in full.

These are the ten skills I wrote for the people I teach.

Each one does one job, the way I'd want it done.

I'll show you what each does, the rules that make it work, and a request to try.

Steal the ideas. Then write your own versions, for the way you work.

What's inside:

1. Two skills for learning
2. Two for research
3. Two for your career
4. Two for everyday work
5. One for data
6. One that writes the others
7. How to choose your first

::subscribe

---

## The ten, on one page

![Ten cards, one per skill, each with what it does and an example request](skills/ten-skills.webp "Save this one. The details for each skill are below.")

Every skill below follows the same pattern.

A clear job. A few firm rules. One thing it never does.

That last part matters most. A good skill isn't only what Claude does. It's what Claude refuses to do.

---

## Learn and remember

### learn-anything

A patient tutor for any topic.

It doesn't start explaining right away. First it asks what you already know.

Then it makes a short plan, and explains **one idea at a time**, each with an analogy and a small example. Before moving on, it checks you understood.

**Why it works:** most explanations start in the wrong place. Too basic, and you're bored. Too advanced, and you're lost. Asking first fixes both.

```prompt Try
Teach me how neural networks learn. I know basic algebra.
```

> [!TIP]
> Always say what you already know. "I know basic algebra" changes the whole explanation.

### quiz-me

Turns your notes, a document or a topic into practice questions.

It asks them **one at a time**. It explains every answer, keeps track of what you miss, and comes back to it.

**Why it works:** reading notes feels like learning. Answering questions is learning. And coming back to the ones you missed is where it sticks.

```prompt Try
Quiz me on these lecture notes. Ten questions, mixed.
```

::button[Make your own version: the five steps](issue:make-your-first-skill)

---

## Research and explain

### research-brief

Answers a question with a **one-page, sourced brief**.

It separates what's known from what's only claimed. It gives each finding a confidence level. And it **never invents a source**.

**Why it works:** the danger with AI research isn't a wrong answer. It's a wrong answer that sounds certain. Confidence levels and real sources make the uncertainty visible.

### news-explainer

Turns AI and tech news into a plain explainer, in four parts:

1. What happened
2. Why it matters
3. Who it affects
4. What's still unknown

Every claim is attributed. No hype.

**Why it works:** the fourth part. Most news coverage skips what isn't known yet. Naming it keeps you honest with your readers.

```prompt Try
Explain this announcement for my newsletter readers.
```

Paste the link or the text under it.

---

## Grow your career

### linkedin-post

Writes LinkedIn posts in **your own voice**, from an idea, a lesson or a piece of work.

You get an opening worth reading and one clear point.

What you don't get: invented stories or engagement bait.

**Why it works:** it learns your voice from your own old posts. So the post sounds like you on a good day, not like a template.

```prompt Try
Turn this project I finished into a post. Here are two of my old posts.
```

### job-application

Tailors your CV and cover letter to **one specific job ad**.

It matches your real experience to what the employer asks for. And it tells you honestly where the gaps are.

**Why it works:** honesty. A tailored CV that invents experience falls apart in the interview. One that tells you where you're weak gives you time to prepare.

```prompt Try
Here's my CV and the job ad. Tailor both.
```

::subscribe[Get the next issue by email]

---

## Get work done

### meeting-to-actions

Turns meeting notes or a transcript into:

- the decisions
- the actions, with owners and deadlines
- the open questions
- a follow-up email, ready to send

It **never invents an owner or a date**.

**Why it works:** a summary nobody acts on is wasted. Owners and deadlines turn a meeting into work that gets done. And refusing to invent them means you find the gaps before they cost you.

```prompt Try
Here's the transcript of today's team call.
```

### presentation-planner

Plans a talk **before you open slide software**:

- the story
- one message per slide
- titles that state the point
- the visuals
- the speaker notes

**Why it works:** most bad decks are a planning problem, not a design problem. Fix the story first, and the slides almost write themselves.

```prompt Try
Plan a ten-minute talk on our quarterly results for the board.
```

---

## Work with data

### check-the-numbers

Answers questions about spreadsheets, CSV files and SQL results, and **proves the answer first**.

A script profiles each file before Claude answers:

- missing values
- numbers stored as text
- repeated rows and keys
- gaps in the dates

Every join is checked for multiplied rows. And the answer comes with its caveats.

**Why it works:** this is the skill where a script earns its place. Checking data must come out the same every time, and code does that better than words.

```prompt Try
Which products drove last quarter's revenue? Here's the export.
```

> [!KEY]
> When a step must be exact (a calculation, a format check, a conversion), put it in a script, not in the instructions.

---

## Make your own

### skill-maker

The one that writes the others.

It interviews you about a task you repeat. Then it writes the SKILL.md, checks it against the rules, and packages the folder as a zip, ready to upload.

**Why it works:** the hardest part of a skill is describing your own job. Being interviewed is much easier than starting from a blank page.

```prompt Try
Help me turn the way I write our weekly report into a skill.
```

::button[The full walkthrough: make your first skill](issue:make-your-first-skill)

---

## What a good skill looks like inside

Every skill above is the same shape inside: a name, a description, and short, firm instructions.

Here's how learn-anything could look, so you can see the pattern:

```markdown
---
name: learn-anything
description: Teaches any topic step by step, starting from what the learner already knows. Use when someone asks to learn, understand or be taught a topic.
---

## Steps
1. Ask what I already know about the topic, and why I want to learn it.
2. Make a short plan: three to five ideas, in order.
3. Explain one idea at a time, with an analogy and a small example.
4. Ask one question to check I understood before moving on.

## Mistakes to avoid
- Don't explain everything in one long answer.
- Don't move on if my answer shows I'm confused.

## Example
(one short exchange you liked)
```

Notice three things.

The **description** says what **and** when. That's how Claude knows to reach for it.

The **steps** are numbered and short. Claude follows them in order.

The **mistakes** come from real failures. Every line there is something that went wrong before the skill existed.

---

## Make them yours

These ten are starting points, not finished products.

The best skills capture the tasks you repeat every week, **the way your organisation does them**.

So before you write your version, ask:

### For learning skills

What do you study most? A new tool, a field, a language? Put your level and your goal into the instructions.

### For research skills

Which sources do you trust? Which ones should never be cited? Write that down.

### For career skills

Paste two of your best posts, or the CV you're proudest of, as the example. That's what teaches Claude your voice.

### For work skills

What does your team's format look like? Headings, order, length, tone. The more exact, the better.

### For data skills

What goes wrong with your data most often? Missing dates, duplicate rows, text where numbers should be? Make the skill check for exactly that.

> [!TIP]
> Open a skill, change its steps to match how you work, and upload your version. That's how a generic skill becomes yours.

---

## How to choose your first

Don't write ten.

Look at your last week of chats with Claude. Find the job you explained more than once.

That's your first skill.

Then ask three questions before you start:

- [ ] Is it **one** job, not three?
- [ ] Have I explained it to Claude **three times** already?
- [ ] Can I describe **when** Claude should use it, in one line?

Three ticks? Start writing.

> [!WARNING]
> Found a ready-made skill online instead? Read every file before you switch it on. A skill can contain code that runs when Claude uses it.

::card[issue:check-a-skill-before-you-trust-it]

---

## What to do today

Pick one of the ten.

Try its request in a normal chat, without a skill.

Notice what you had to explain, and what Claude got wrong.

That list is the first draft of your skill.

::card[issue:make-your-first-skill]

If this helped, send it to one person who uses Claude every day.

Mojeed
