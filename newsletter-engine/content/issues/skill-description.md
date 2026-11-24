+++
title = "The one line Claude reads."
subtitle = "Your Claude skill isn't broken. Its description is. Here's how to write the line Claude uses to choose:"
description = "Claude picks a skill from its description alone. What a strong description says, the rules, before-and-after examples and a prompt that rewrites yours."
date = 2026-11-24
status = "published"
tags = ["claude", "skills"]
archetype = "prompt_guide"
featured = true
cover = "skills/description-split.webp"
cover_alt = "Weak and strong skill descriptions side by side"
brief = "briefs/claude-skills.json"
+++
You made a Claude skill.

You uploaded it. You switched it on.

And Claude never uses it.

Or worse: it uses it when you didn't ask.

You open the instructions. You rewrite the steps. You add examples.

Nothing changes.

Because the problem isn't in the steps. It's in **one line** at the top of the file.

What's inside:

1. Why one line decides everything
2. What Claude actually sees
3. The two jobs of a description
4. The rules
5. Before and after: three rewrites
6. A prompt that rewrites yours
7. The five-request test

::subscribe

---

## Why one line decides everything

Claude decides when to use a skill from its **description alone**.

Not the steps.

Not the examples.

Not the long instructions you wrote below it.

That's because skills load in layers:

1. **Always:** Claude sees just the name and description of every skill you've switched on. A line or two each.
2. **When a request matches:** it reads that skill's instructions.
3. **Only if needed:** it opens the extra files.

So most of the time, the description is the **only** part of your skill Claude sees.

> [!KEY]
> A great skill with a weak description is a skill Claude never opens.

---

## What Claude actually sees

Picture your skills the way Claude meets them: a short list of names and descriptions, one or two lines each.

Something like this:

```text
weekly-report: Turns the week's notes and numbers into our team's weekly report format. Use when someone asks for the weekly report or pastes weekly updates.
meeting-notes: Helps with meetings.
```

A request comes in: "Here are this week's updates. Can you write the report?"

The first line tells Claude exactly when to step in. The second could mean anything.

Your description is competing for attention in that list. Make it impossible to misread.

---

## The two jobs of a description

A strong description does **two** things:

1. Says **what** the skill does.
2. Says **when** to use it.

Most weak descriptions do only the first.

![Weak descriptions versus strong ones, side by side](skills/description-split.webp "What it does, and when to use it. Both, every time.")

The "when" is where you name the **trigger**: the words or situations that should make Claude reach for it. "When someone asks for the weekly report." "When someone pastes a meeting transcript."

---

## The rules

| Part | Rule |
|---|---|
| Voice | The third person: "Turns the week's notes...", not "I help you..." |
| Length | 200 characters at most in the Claude apps (the API allows 1,024) |
| Format | No tags in angle brackets |
| Content | What it does **and** when to use it |

And one rule from experience: **one skill, one job.** A description that has to explain two jobs is a sign you need two skills.

::button[New to skills? Start with the guide](guide:claude-skills-from-zero)

---

## Before and after: three rewrites

### The weekly report

```prompt Weak
Helps with reports.
```

```prompt Strong
Turns the week's notes and numbers into our team's weekly report format. Use when someone asks for the weekly report or pastes weekly updates.
```

### Meeting notes

```prompt Weak
I help you with your meetings.
```

```prompt Strong
Turns meeting notes or a transcript into decisions, actions with owners and deadlines, and a follow-up email. Use when someone pastes meeting notes or a transcript.
```

### Learning a topic

```prompt Weak
A tutor skill for learning things and also making quizzes.
```

```prompt Strong
Teaches any topic step by step, starting from what the learner already knows. Use when someone asks to learn, understand or be taught a topic.
```

Notice what changed each time. Third person. One job. A clear trigger.

And the quiz part moved out: that's a second skill.

::subscribe[Get the next issue by email]

---

## A prompt that rewrites yours

Have a skill already? Paste its SKILL.md into a normal chat with this:

```prompt Rewrite my skill's description
Here is my skill's SKILL.md. Rewrite its description so it says what the skill does and when to use it, in the third person, in 200 characters or fewer, with no tags in angle brackets. Then give me three requests that should use the skill and two that shouldn't.

[paste your SKILL.md here]
```

Read what comes back. Keep it only if it's true to how you work.

You'll find more prompts like this in [the library](page:library).

---

## The five-request test

A new description isn't done until it passes this:

1. Write **three** requests that should use the skill.
2. Write **two** that shouldn't.
3. Upload the skill, switch it on, and try all five.

- [ ] All three "should" requests used the skill
- [ ] Neither "shouldn't" request did

Claude missed one?

Don't touch the steps. **Rewrite the description first.**

> [!TIP]
> Still ignored? Check that the skill's switch is on in Customize → Skills, and name it in your request: "Use my weekly-report skill on these notes."

---

## What to do today

Open your skills.

Read each description out loud.

Does it say what **and** when? If not, run the prompt above, then the five-request test.

Ten minutes per skill. Then Claude uses them when it should, and only then.

::card[issue:make-your-first-skill]

Send this to one person whose skill never gets used.

Mojeed
