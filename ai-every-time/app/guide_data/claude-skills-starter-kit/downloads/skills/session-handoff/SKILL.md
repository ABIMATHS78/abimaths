---
name: session-handoff
description: Writes a handoff note that lets a fresh chat or a colleague pick up long work where it stopped, with the goal, decisions, current state and next steps. Use when a chat gets long or work changes hands.
---

# Session handoff

Long conversations get slow and start to lose detail. A good handoff note lets a new chat, or a person, continue without rereading everything.

## When to offer it

- The conversation is long, or the person says it's getting slow or confused.
- The work will continue tomorrow, in another tool, or with someone else.
- The person asks for a summary to "start fresh".

## What goes in

Write the note in this order. Leave out any section with nothing in it.

```
# Handoff: <project name>, <date>

## Goal
<What we're trying to achieve, in one or two sentences, and how we'll know it's done.>

## Background the next session needs
<Facts, constraints and preferences that still matter. Who it's for.>

## Decisions made
- <Decision>, because <reason>.

## Where things stand
- Done: <items>
- In progress: <item, and exactly where it stopped>
- Not started: <items>

## Files, links and names
<Exact file names, paths, URLs, names of people, versions. Copy them; don't paraphrase.>

## What didn't work
- <Approach>, because <what went wrong>. Don't try it again unless <condition>.

## Open questions
- <Question>, and who can answer it.

## Next steps
1. <The very next action, specific enough to start at once.>
2. <Then this.>
```

## Rules

- **Compress, don't lose**: keep numbers, names, file paths, versions, exact wording that was agreed, and short code snippets that matter. Drop greetings, back-and-forth and dead ends (except under "What didn't work").
- Keep it under one page unless the person asks for more.
- Write it so someone who wasn't in the conversation understands every line.
- Never include passwords, API keys or other secrets. Write "<stored in …>" instead.
- Put the note in one code block so it's easy to copy.

## Finish with

"Paste this into a new chat and say: *Continue from this handoff, starting with step 1.*"
