+++
title = "Brief Claude like a new colleague."
subtitle = "Short prompts get average answers. A two-minute brief gets the finished thing, first time:"
description = "The five parts of a good brief: the goal, the background, the material, the shape of the answer and the finish line. With examples and prompts to copy."
updated = 2026-11-25
status = "published"
tags = ["claude", "prompts"]
archetype = "guide"
featured = true
level = "beginner"
topic = "Prompts"
+++
Imagine your manager walks past your desk and says:

"Write me a report on sales."

Then walks off.

You'd have questions. Which sales? For whom? How long? By when? What's it for?

Claude has the same questions. It just doesn't ask them. It guesses, and gives you something average.

The fix isn't a magic phrase. It's a **brief**: the same two minutes of context you'd give a sharp new colleague on their first day.

**By the end of this guide you'll have:**

- the five parts every good brief has
- a way to get Claude to ask *you* the questions first
- a habit that turns three rounds of fixes into one

You need: Claude, and one task you'd normally describe in a single line.

---

## 1. Know why short prompts get average answers

A one-line prompt leaves Claude to fill in everything you didn't say.

So it picks the most likely answer. The safe, middle-of-the-road version that would suit anyone.

That's why the result often feels generic. It **is** generic. It was written for nobody in particular, because Claude didn't know who it was for.

Every detail you add replaces a guess with a fact.

> [!KEY]
> Claude doesn't need clever wording. It needs the information a colleague would need.

---

## 2. Use the five parts

A good brief answers five questions. You don't need headings, but you do need the answers.

### The goal: what is this for?

Not the task. The reason behind it.

"Write a report on sales" is a task. "Help the board decide whether to keep selling in Ghana" is a goal. The second tells Claude what matters and what to leave out.

### The background: what does Claude need to know?

Who's it for, and what do they already know? What happened before? What's been tried?

"The board meets on Friday. They've seen last quarter's numbers. Two members think we should pull out."

### The material: what should it work from?

Paste or attach it: the notes, the numbers, the draft, the email thread.

Claude can't use what it can't see. If it has to invent the facts, the answer is only as good as its guesses.

### The shape: what should the answer look like?

Length, format, tone and order.

"One page. Three options, each with the main risk. Plain English, no jargon. Recommendation at the top."

### The finish line: when is it done?

Say what "done" means, and what Claude should check before it stops.

"Done when every number comes from the attached sheet and each option has a cost. If something's missing, list it at the end instead of guessing."

---

## 3. Put it together

Compare these two:

```prompt One line
Write me a report on our Ghana sales.
```

```prompt A brief
Help the board decide whether to keep selling in Ghana. They meet on Friday, have seen last quarter's numbers, and two members want to pull out.

Work only from the attached sales sheet and the notes below.

Write one page: three options (stay, shrink, pull out), each with its cost and main risk, and your recommendation at the top. Plain English.

You're done when every number comes from the sheet. If anything's missing, list it at the end instead of guessing.

[notes]
```

The second one takes two minutes longer to write. It saves the three rounds of "no, not like that" that come after the first one.

---

## 4. Let Claude ask you first

Don't know what to put in the brief? Make Claude interview you.

```prompt Ask me first
Before you start, ask me the five questions you most need answered to do this well. Wait for my answers.
```

Add it to the end of any request. You'll often find the questions are better than the brief you would have written.

> [!TIP]
> This works especially well for big jobs: a strategy, a proposal, a hard email. The questions make *you* think about what you actually want.

---

## 5. Show it what good looks like

The fastest way to describe a style is to show it.

- Paste a past report you were happy with: "Match this format."
- Paste an email in your tone: "Write like this."
- Paste a bad example too: "Not like this, it's too formal."

One example beats a paragraph of adjectives like "professional but friendly".

---

## 6. Correct it like feedback, not a fresh start

The first answer is a draft. Treat it like a colleague's draft.

Don't start a new chat and rewrite the prompt. Tell Claude what to change, and why:

- "Shorter. The board reads this on their phones."
- "Option two is too vague. Give it a real cost."
- "Keep the structure, but lead with the risk."

The why matters. "Shorter" fixes this answer. "Shorter, because they read it on their phones" fixes every answer after it.

---

## 7. Save the briefs you reuse

Writing the same brief every week? Don't.

- **Context you always need** (who you are, how you like things) goes into custom instructions.
- **Background for one piece of work** goes into a project.
- **A job you do the same way every time** becomes a skill.

Not sure which one? This guide picks the right one in two questions:

::card[guide:skill-or-something-else]

---

## Before you finish

- [ ] Your brief says what the work is **for**, not just what it is.
- [ ] It gives the background a colleague would need.
- [ ] The material is pasted or attached.
- [ ] It describes the shape: length, format, tone.
- [ ] It says when the work is done, and what to do if something's missing.

---

## Questions

### Isn't this a lot of typing?

Two minutes. Less than the time you'd spend fixing a generic answer. And once you've written a brief you reuse, you can save it.

### Do I still need phrases like "act as an expert"?

Rarely. A role-play line adds little on its own. The goal, the background and the material do far more.

### Does this work with other AI tools?

Yes. The five parts work with any AI tool, and with any new colleague too.
