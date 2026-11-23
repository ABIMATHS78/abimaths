+++
title = "Check a skill before you trust it."
subtitle = "A skill can run code on your behalf. Here's the five-minute check:"
description = "What can go wrong with a Claude skill, and the five-minute check to run before you switch one on."
date = 2026-11-30
status = "draft"
tags = ["claude", "skills", "safety"]
archetype = "safety_warning"
brief = "briefs/claude-skills.json"
fact_checks = ["Add the two screenshots.", "Fill in the [CHECK] note with a real example from your work."]
+++
A colleague sends you a skill.

It looks useful. You upload it.

But a skill can contain code that runs when Claude uses it.

So treat an unknown skill like an unknown app.

Here's what can go wrong, and the check to run first:

1. The two risks
2. The five-minute check
3. What to do if something looks wrong

::subscribe

---

## The two risks

**Prompt injection:** hidden instructions that steer Claude into something you didn't ask for.

**Data leaks:** a script that sends your information to a website.

[CHECK: one real example of a skill you checked, and what you found]

---

## The five-minute check

Open the skill's folder before you upload it.

::screenshot[A skill's folder open on the desktop, showing SKILL.md and a scripts folder]

- [ ] It comes from a source I trust
- [ ] I read SKILL.md from top to bottom
- [ ] I read every script
- [ ] Nothing downloads software
- [ ] Nothing sends information to a website
- [ ] No passwords, API keys or personal data inside

> [!WARNING]
> Even a skill a colleague shares deserves a look first.

---

## If something looks wrong

Don't switch it on.

Already did? Switch it off in **Customize → Skills**.

::screenshot[The Customize → Skills page with one skill's switch turned off]

On Enterprise plans, ask an owner about skill scanning. It checks uploaded skills for malicious content.

---

## What to do today

Open the skills you've switched on. Run the check on each one.

::card[guide:claude-skills-from-zero]

Mojeed
