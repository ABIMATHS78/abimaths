---
name: app-spec
description: Turns an app idea into a build-ready spec for an AI coding tool, with users, features, screens, data, rules, what's out of scope and a step-by-step plan. Use before asking an AI to build an app.
---

# App spec

Give an AI coding tool a clear plan, so it builds the app you mean, in small steps you can check.

## Before you start

Ask for: the problem the app solves and who will use it; the three to five things users must be able to do; apps they like; where it should run (web, phone or desktop); any tools or languages they want; whether it needs accounts, payments or personal data; and how much coding experience they have. Ask in one short message.

## Steps

1. **Write the one-sentence pitch**: who it's for, and what it lets them do.
2. **Define the first version.** List the must-have features as user stories ("As a…, I want… so that…"), and list what's out of scope for now. The smaller the first version, the better it will work.
3. **Describe each screen**: what it shows, what users can do there, and where they go next, including its empty, loading and error states.
4. **Define the data**: each kind of record, its fields, and how records relate, such as "a user has many projects".
5. **Write the rules**: who can see and change what, what's checked when people enter data, the limits, and what happens when something goes wrong.
6. **Choose a simple stack** that suits the person's experience and where the app will run, and explain the choice in a sentence. Prefer popular, well-documented tools.
7. **Plan the build in small steps**, each one working and testable before the next: set-up, the data, one screen and one feature at a time, then accounts, then going live.
8. **Write a check for each step**: what to click or run to see that it works.

## Output

```
# <App name>: spec
Pitch: <one sentence>

## Users and goals
## First version: must-haves
## Out of scope for now
## Screens
## Data
## Rules and permissions
## Stack, and why
## Build plan
| Step | What to build | How to check it works |
|---|---|---|
```

## Rules

- Never put passwords, API keys or other secrets in the spec or the code. Plan to keep them in environment variables or the host's secret settings.
- If the app handles personal data, payments or health information, say so plainly, and plan for security and the laws that apply. Suggest a review by someone experienced before launch.
- Don't add features the person didn't ask for. Suggest them, and ask first.
- Mark assumptions "to confirm".

## Checklist before you finish

- [ ] The first version is small, and what's out of scope is listed.
- [ ] Every screen has its empty and error states.
- [ ] The permissions say who can see and change what.
- [ ] Each build step has a way to check it works.
