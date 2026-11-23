+++
title = "Make your first skill."
subtitle = "Turn the instructions you paste most into a Claude skill, in five steps:"
description = "Write a SKILL.md, follow the rules that matter, zip it the right way and test it with five requests."
date = 2026-11-23
status = "published"
tags = ["claude", "skills"]
archetype = "setup_steps"
brief = "briefs/claude-skills.json"
fact_checks = ["Check the name and description limits (64 and 200 characters) against Anthropic's current documentation."]
+++
You've explained the same thing to Claude three times.

That's your sign.

It's time to make it a skill.

A skill is a folder with one file in it. You write it once. Claude uses it every time a request needs it.

Here's the whole process:

1. Pick the task
2. Write the SKILL.md
3. Write the instructions
4. Zip the folder
5. Upload it and test it

::button[New to skills? Start with the guide](guide:claude-skills-from-zero)

---

## Step 1: Pick the task

Pick the job you explain most.

The weekly report. The meeting summary. The client email in your team's format.

One skill, one job. If you need two jobs, make two skills.

I'll use a weekly report as the example below.

---

## Step 2: Write the SKILL.md

Make a folder named after the skill: `weekly-report`.

Inside it, make a file called `SKILL.md`. It starts with the name and the description, between two lines of dashes:

```yaml
---
name: weekly-report
description: Turns the week's notes and numbers into our team's weekly report format. Use when someone asks for the weekly report or pastes weekly updates.
---
```

The rules that matter:

| Part | Rule |
|---|---|
| Name | Lowercase letters, numbers and hyphens. 64 characters at most. The same as the folder's name. Can't contain "claude" or "anthropic". |
| Description | What it does **and** when to use it, in the third person. 200 characters at most in the Claude apps. No tags in angle brackets. |

> [!KEY]
> Claude decides when to use your skill from the description alone. Spend your time there.

---

## Step 3: Write the instructions

Below the dashes, write how you want the job done.

Keep it under 500 lines. Include:

- numbered steps
- the mistakes to avoid
- one short example
- a checklist, if the job is long

Need a template or reference notes? Put them in the folder, link each one from SKILL.md, and say when to read it.

Don't want to write it alone? Ask Claude to draft it with you:

```prompt Draft it with Claude
Help me turn the way I write our weekly report into a skill. Ask me questions first, then write the SKILL.md.
```

::subscribe[Get the next guide by email]

---

## Step 4: Zip the folder

This is where most uploads fail.

Zip the **folder**, not the files inside it.

```text
weekly-report/
├── SKILL.md        required: name, description and instructions
├── reference.md    optional: detail Claude reads only when it needs it
├── templates/      optional: files to fill in
└── scripts/        optional: code for steps that must be exact
```

> [!WARNING]
> The zip must contain `weekly-report/SKILL.md`. If the files sit at the top of the zip without their folder, the upload won't work.

---

## Step 5: Upload it and test it

Upload it: **Customize → Skills → + → Create skill → Upload a skill**. Check its switch is on.

Then test it with five requests:

1. Write three requests that **should** use the skill.
2. Write two that **shouldn't**.
3. Try all five.

Did Claude miss one? Fix the description first. It's what Claude uses to choose.

- [ ] Three requests that should use it
- [ ] Two that shouldn't
- [ ] All five behave as expected

---

## What to do today

Open the last chat where you pasted your usual instructions.

Copy them into a SKILL.md. Add a name and a description.

That's your first skill.

::card[guide:claude-skills-from-zero]

Send this to one colleague who still pastes the same prompt every morning.

Mojeed
