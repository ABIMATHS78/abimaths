---
name: sop-writer
description: Writes a standard operating procedure from a walkthrough or notes, with numbered steps, roles, checks and what to do when something goes wrong. Use when a task must be done the same way every time.
---

# SOP writer

Capture how a task is done well, so anyone trained for it can do it the same way, every time.

## Ask first

If any of this is missing, ask for it in one short message:
- the walkthrough: notes, a transcript, screenshots or a description of each step;
- who does the task, how often, and what triggers it;
- the tools and systems used;
- the organisation's SOP template, if it has one.

## Steps

1. **Define the scope**: the purpose, when the procedure starts and ends, and who it's for.
2. **List what's needed first**: access, tools, materials, information.
3. **Write the steps** in order, numbered, one action each, starting with a verb: "Open the order screen." Add the expected result where it helps: "The status changes to Paid."
4. **Put checks and warnings in the right place**, before the step they apply to.
5. **Cover the exceptions**: what to do if a step fails, who to contact, and when to stop.
6. **Add the record**: what to log, where, and how long to keep it.
7. **Test the draft** by reading it as a new team member. Mark every gap or assumption "to confirm".

## Output

```
# SOP: <task>
Owner: <role> · Version: <n> · Last reviewed: <date> · Next review: <date>

## Purpose and scope
## Who does it
## Before you start
## Steps
1. <verb + action>. Expected: <result>
## If something goes wrong
| Problem | What to do | Contact |
|---|---|---|
## Records
```

## Rules

- **Write only what the walkthrough shows.** Never invent steps, settings or approvals; mark gaps "to confirm".
- One action per step. Split long steps.
- Use the exact names of screens, buttons and fields.
- Include safety, privacy and legal checks where the task needs them, and keep the owner's review and sign-off in the process.
