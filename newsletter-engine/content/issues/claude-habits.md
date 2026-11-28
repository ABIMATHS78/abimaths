+++
title = "Claude, past the basics."
subtitle = "Twenty-six small habits that save time, usage and mistakes:"
description = "Twenty-six Claude habits for set-up, asking, files, checking answers, usage limits and Cowork, each with the clicks, a prompt to copy and a graphic."
date = 2026-11-28
status = "published"
cover = "claude-habits/cheat-sheet.webp"
cover_alt = "A one-page cheat sheet of twenty-six Claude habits in six groups: set it up once, ask so the first answer lands, projects and files, check before you trust, make usage last and hand bigger jobs to Cowork, with the five to start with"
tags = ["claude", "productivity", "cowork"]
archetype = "tips_list"
brief = "briefs/claude-habits.json"
fact_checks = ["Check Settings → Your Claude in your own Claude app: Instructions, Manage next to Your memory (and the Topics list behind it), Start import, Generate memory from chats, and Search and reference chats.", "Check that + in the message box of a new chat shows Memory (to turn it off for that chat), Add files or photos, Research and Connectors, and that Connectors offers Tool access with Auto and On demand.", "Check a project's page: + New Project, the + button in the project knowledge, Set project instructions and Save instructions, and Use account memory under Memory.", "Check Settings → Usage on a paid plan: Current session with the time remaining, when the weekly limit resets, and Usage credits.", "Check whether your sidebar says Customize or Plugins (Anthropic is renaming Customize to Plugins), and the skill upload path + → + Create skill → Upload a skill.", "In Cowork, check the mode selector in the chat box (Manually approve, Automatically approve, Skip all approvals), Scheduled in the left sidebar with New task, and Enable computer use under Settings → Capabilities in the desktop app."]
+++
Getting one good answer out of Claude is the easy part.

The rest of the week is where things slip.

The chat that runs so long it eats into your limits.

The number that went into a report without a check.

The background you explain again, and again.

The limit that arrives halfway through a task.

A better prompt won't fix those. Small habits will.

I teach people to use AI at work. These are the habits I'd want every one of them to have.

Each one rests on a documented Claude feature or on Anthropic's own advice, linked where it's Anthropic's.

Most take one click or one sentence.

**Here are twenty-six, in six groups:**

1. **Set it up once** (habits 1 to 3)
2. **Ask so the first answer lands** (4 to 9)
3. **Projects, files and artifacts** (10 to 14)
4. **Check before you trust** (15 to 18)
5. **Make your usage last** (19 to 22)
6. **Hand bigger jobs to Cowork** (23 to 26)

Each habit has the clicks, the reason, a prompt to copy and a graphic.

New to Claude? Start with [Claude, from the first click](guide:claude-from-zero), then come back.

![All twenty-six habits on a one-page cheat sheet, in six groups, with the five to start with and a handover prompt](claude-habits/cheat-sheet.webp "All twenty-six habits on one page. Save it; the details are below.")

::subscribe

---

## Set it up once

Three settings that shape every chat after them.

### 1. Write your Instructions once

Caught yourself typing "shorter" or "no jargon" again?

A correction you make twice belongs in **Instructions**.

**Where:** click your initials in the lower left corner, then **Settings → Your Claude → Instructions**. In the older Settings layout, that page is called **Account**.

**Why it works:** anything you add there [applies to all of your conversations](https://support.claude.com/en/articles/10185728-understanding-claude-s-personalization-features). You stop paying for the same correction in every chat.

Anthropic suggests four things to put there:

- the approaches or methods you prefer
- the terms you use
- the situations you often face
- how you want Claude to communicate

In the new Claude experience, Cowork's Global instructions moved into this box too.

**When to use it:** once, then whenever you catch yourself repeating a correction.

```prompt Instructions to adapt
I work as [your role] in [your field].
Write in plain English: short paragraphs, no jargon, no filler.
Start long answers with a three-line summary.
When I say "[a term you use]", I mean [what it means].
If something important is missing, ask me before you start.
```

Not sure what to write? Let Claude draft it from a chat where it got you right.

**When to use it:** at the end of a chat that went well.

```prompt Draft my Instructions
Based on this chat, draft Instructions for Claude that I can paste into Settings. Keep them under 120 words, in four short parts: how I like to work, the terms I use, the situations I often face, and how you should talk to me. Leave out anything that only applies to this one task.
```

Read what comes back, cut anything too specific, then paste it in.

![A labelled drawing of Settings, Your Claude, with the Instructions box open and a sample instruction typed in, and the three steps numbered](claude-habits/instructions.webp "Instructions apply to every chat. Keep them general. A labelled drawing, not a screenshot.")

> [!TIP]
> Keep it general. One project's details go in its project instructions (habit 10).

### 2. Read what Claude remembers, and fix it

Claude keeps a memory of you.

On the Free, Pro, Max and Ultra plans, it's on by default. On Team and Enterprise, your organisation's owners decide.

It saves what it learns as **topics** while you chat. Mention that a deadline moved, and your next conversation already knows.

Useful, until something it remembers goes out of date.

**Where:**

1. Click your initials, then **Settings → Your Claude**.
2. Click **Manage** next to **Your memory**.
3. Everything is listed under **Topics**. Open one, then use the edit icon to fix it, or **Delete** to remove it.

**Why it works:** fix a topic once, and [every conversation from then on](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context) uses the fixed version.

Each project keeps its own memory too.

Moving from another AI assistant? On the same page, **Start import** shows a prompt to run in your old service. Paste what it gives you, then click **Add to memory**.

Anthropic calls import experimental. Only memory comes across, [not your old chats](https://support.claude.com/en/articles/12123587-import-and-export-your-memory-from-claude).

**When to use it:** when something about you changes, or Claude gets a detail about you wrong twice.

```prompt Update your memory
Please update your memory: I no longer [the old detail]. I now [the new detail]. Forget anything that says otherwise.
```

The change applies from your next conversation.

![A labelled drawing of Settings, Your Claude: Instructions, Your memory with a Manage button, and the memory switches](claude-habits/memory-settings.webp "Your memory and your Instructions live on the same page. A labelled drawing, not a screenshot.")

### 3. Switch memory off before a one-off chat

Helping a friend with their CV, or asking about something that shouldn't shape later answers?

Turn memory off for that one chat. Do it **before** your first message: you can't change it later.

1. Start a new chat.
2. Click **+** in the message box.
3. Click **Memory** to turn it off, then send your message.

A crossed-out memory icon then appears next to the chat title. Claude won't use its memory there, and saves nothing from it.

Want it out of your chat history too? Use an [incognito chat](https://support.claude.com/en/articles/12260368-use-incognito-chats): the ghost icon, upper right, in a new chat outside a project.

| | Memory off for one chat | Incognito chat |
|---|---|---|
| Uses or adds to your memory | No | No |
| Stays in your chat history | Yes | No |
| Found when Claude searches past chats | Yes | No |
| Can make files and run code | As in any chat | Not in the new Claude experience |
| How to start it | **+** → **Memory**, before you send | Ghost icon, upper right |

Incognito chats aren't used for training. They're kept for 30 days by default.

**When to use it:** when you shared something in a normal chat that Claude shouldn't keep.

```prompt Forget this
Please forget what I told you about [the topic]. Don't use it in future conversations.
```

![A labelled drawing of the + menu in a new chat: Memory, Research and Connectors](claude-habits/one-chat.webp "Set these before your first message. A labelled drawing, not a screenshot.")

::button[Set up Claude properly: the toolkit issue](issue:claude-set-up-right)

---

## Ask so the first answer lands

Six habits for the minute before you press send, each from Anthropic's own guidance.

### 4. Plan the chat, then send one full message

Anthropic's first tip for getting more from your plan: [plan the conversation](https://support.claude.com/en/articles/9797557-usage-limit-best-practices) before you begin.

- Know what you need.
- Group related questions into one message.
- Give the background up front.
- Paste the whole text you want edited, not one piece at a time.

Then read your message once, and fix it before you send.

**Why it works:** a clear first message saves a round of follow-ups. And Anthropic lists the length of the current chat among the things that use your limits faster.

**When to use it:** any time you have more than one question about the same piece of work.

```prompt One message, everything in it
I'm preparing [the piece of work] for [who it's for].
Here's what I have: [paste it, or attach the file].
Please answer these together:
1. [first question]
2. [second question]
3. [third question]
Put each answer under its own heading. If you need anything else from me, ask before you start.
```

Expect one reply with three headed answers, instead of three separate rounds.

![The same request written three ways: a vague one-liner, a clearer request, and one full message with the background, three numbered questions and the format](claude-habits/one-message.webp "From a vague one-liner to one full message. The third version gets the best first answer.")

### 5. Brief it like a capable new colleague

Anthropic's prompting guide says to think of Claude as [a brilliant but new employee](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) who doesn't know your norms or how you work.

The more precisely you explain what you want, the better the result.

Its **key test**: show your prompt to a colleague with little context on the task. If they'd be confused, Claude will be too.

Three more lines from the same guide:

- A role, even a single sentence, focuses Claude's tone.
- If you want more than the minimum, ask for it.
- Steps go in a numbered list when order matters.

And from Anthropic's article for app users: [don't over-engineer](https://claude.com/resources/articles/best-practices-for-prompt-engineering). Longer isn't always better. Test, then refine.

**When to use it:** for any task you'd brief a person on.

```prompt The colleague brief
You're helping me as [a role, e.g. an editor for a finance team].
The goal: [the outcome you need].
It's for [who will read it], who already knows [what they know].
Work from: [your notes, file or link].
Give it to me as [the format and length].
Go beyond the minimum: [the extra you want, e.g. flag risks].
Before you start, ask me up to three questions if anything is unclear.
```

Expect Claude to come back with a few questions first. Answer them, and the draft that follows fits.

![One prompt split into coloured parts: the vague request to delete, the task to keep, and the role, audience, inputs, format and questions to add, with a reason beside each](claude-habits/brief-anatomy.webp "The anatomy of a colleague brief. Keep the task; add who it's for, what to use and what to hand back.")

### 6. Say what to do, and why

Two small changes from Anthropic's guide.

**Say what you want, not what you don't.** Its example: ask for "smoothly flowing prose paragraphs" instead of "Do not use markdown".

Your own style rubs off too. A prompt with no markdown tends to get an answer with less of it.

**Give the reason.** Its example swaps "NEVER use ellipses" for the reason behind it: the reply will be read aloud by a text-to-speech engine.

**Why it works:** Claude can generalise from a reason. It handles cases your rule never mentioned.

**When to use it:** on any prompt or Instructions you reuse.

```prompt Fix my rules
Here's a prompt I reuse: [paste it].
Rewrite every "don't" or "never" as a positive instruction, and add a one-line reason after each rule. Keep my meaning. Show me only the new version.
```

Expect your rules to come back as "Write..." and "Use..." lines, each with a short "because".

![Two columns side by side: rules written as don'ts on the left, and the same rules said positively with a reason on the right](claude-habits/say-why.webp "Turn each 'don't' into a 'do', and add the reason. Claude can apply the reason to cases you didn't list.")

### 7. Show one example of good

Anthropic calls examples one of the most reliable ways to steer format, tone and structure.

Its developer docs suggest three to five. Its article for app users says [start with one](https://claude.com/resources/articles/best-practices-for-prompt-engineering), and add more only if the output still doesn't match.

One warning from the same article: recent models pay very close attention to the details in an example.

So pick an example you'd be happy to see copied. Its length, its tone and its quirks will all come back.

**When to use it:** when the format matters more than the content: subject lines, product blurbs, status updates.

```prompt Match this example
Write [how many] [things] for [the purpose].
Match the format, length and tone of this example, but not its content:
<example>
[paste one you like]
</example>
```

The tags are optional. They just make clear where the example starts and stops.

![A habit card for examples: what to do, why it works, the prompt to copy and what to watch out for](claude-habits/one-example.webp "One example teaches the shape. Choose it carefully: Claude copies the details too.")

### 8. Put long documents first, and your question last

Anthropic's guide says to put long documents at the top and your question at the end.

In its tests, questions at the end [can improve response quality by up to 30%](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices), especially with several documents at once.

A second tip, from Anthropic's guide to [reducing hallucinations](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations): ask for word-for-word quotes first, then the answer built on them.

Anthropic says these techniques reduce mistakes, but don't remove them.

**When to use it:** contracts, policies, long reports, interview transcripts.

```prompt Quotes first
<document>
[paste the document]
</document>

First, pull out the word-for-word quotes from the document that matter for my question, inside <quotes> tags. Then answer using only those quotes. If no quote supports a point, leave the point out.

My question: [your question]
```

Expect a short list of quotes, then an answer you can check line by line.

![A habit card for long documents: the order to paste things in, the quotes-first prompt and what it doesn't fix](claude-habits/long-docs.webp "Document at the top, question at the bottom, quotes in between.")

### 9. Tell it when to search, and when to research

In the new Claude experience, there's no web search switch: Claude searches when it helps. Otherwise, it's **+ → Web search**.

Either way, steer it in [plain words](https://support.claude.com/en/articles/11095361-when-should-i-use-web-search-extended-thinking-and-research):

| You need | Do this | Why |
|---|---|---|
| A quick, current fact | Ask it to search the web | One or two searches, with citations |
| A full report from many sources | **+** → **Research** (paid plans) | Five or more searches, over 1 to 3 minutes |
| Work on what you've given it | Say "Don't search the web for this" | Searches add to your usage |
| Anything after June 2026 | Ask it to search | The newest models learned from data up to June 2026 |

A link to a long article pulls [the whole article](https://support.claude.com/en/articles/10684626-enable-and-use-web-search) into the chat, which uses far more than a normal search. And Research started from the phone app skips your connectors.

**When to use it:** for a quick fact that may have changed.

```prompt Search, with sources
Search the web for [your question]. Prefer official or primary sources. Give me the answer in three lines, then list the links you used, with the date on each page if it shows one.
```

**When to use it:** when the answer is in the material you gave it.

```prompt No search
Don't search the web for this. Work only from what I've given you below, and tell me if something you'd need is missing.
```

![A labelled drawing of the + menu in the message box: Add files or photos, Web search, Research and Connectors, with the steps numbered](claude-habits/plus-menu.webp "Research, files and connectors all start from the + button. A labelled drawing, not a screenshot.")

![A decision guide: when to ask for a quick search, when to start Research, when to say no search, and what each one costs in usage](claude-habits/search-or-research.webp "Search for a fact, Research for a report, no search for your own material.")

::button[Brief Claude like a colleague: the guide](guide:brief-claude-like-a-colleague)

---

## Projects, files and artifacts

Five habits for work that runs for days, and for the files that go with it.

### 10. Give each ongoing job its own project

A project is a workspace with its own chats, files and instructions.

Every plan has projects. Free accounts can make up to five.

**Why it works:** your files and instructions are in every chat, with no re-uploading. And project files are cached, so they [count less against your limits](https://support.claude.com/en/articles/9797557-usage-limit-best-practices) when reused.

**Step 1: create it.**

1. Click **Projects** on the left, or go to claude.ai/projects.
2. Click **+ New Project** in the upper right corner.
3. Give it a name and a description.

![A labelled drawing of the Projects page with the + New Project button and the name and description boxes, the steps numbered](claude-habits/new-project.webp "Step 1: create the project. A labelled drawing, not a screenshot.")

**Step 2: set it up.**

1. Click **+** in the project knowledge and add your core files.
2. Click **Set project instructions**, write them, then **Save instructions**.
3. Want the project kept fully separate? Turn off **Use account memory** under **Memory**.

Claude can't see your project's name or description. Anything it needs goes in the [instructions or the files](https://support.claude.com/en/articles/9519177-how-can-i-create-and-manage-projects).

**When to use it:** when you write the project instructions.

```prompt Project instructions to adapt
This project is [what it is] for [who it's for].
Your role: [the role, in one line].
Always: [two or three rules that matter here].
The files in the project knowledge are [what they are]. Treat [file name] as the latest version.
```

![A labelled drawing of a project page: Set project instructions, the + in project knowledge, and Use account memory](claude-habits/project.webp "Step 2: files, instructions and memory. A labelled drawing, not a screenshot.")

> [!TIP]
> The cache expires after a while without use. After a long break, your first message counts the project's files in full again.

### 11. Keep project instructions short, and save decisions there

Anthropic's advice: use project instructions for the big picture.

That means the context, the key guidelines and Claude's role. One task's instructions stay in that task's chat. And clear out files you've stopped using.

**Why it works:** shorter instructions and fewer files help your [context window and your usage](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work).

One more thing to know: chats in the same project don't share context on their own. Only what's in the project knowledge reaches every chat.

So when a chat settles something, save it.

**When to use it:** at the end of any chat that made a decision.

```prompt Decision note
Write a short note on what we settled in this chat: the decisions, the reasons, the open questions and the next step. Plain text, no more than 10 lines, ready to add to the project knowledge.
```

Save the note as a text file. Then add it with **+** in the project knowledge.

Started the chat outside the project? Click the dropdown arrow next to the chat name, then **Add to project**.

![A habit card for project upkeep: what belongs in project instructions, what to clear out, the decision-note prompt and how to add it to the project](claude-habits/project-notes.webp "Big picture in the instructions, decisions in the knowledge, old files out.")

### 12. Know what Claude reads in your files

Claude doesn't read every file the same way.

| What you upload | What Claude reads |
|---|---|
| PDF of 100 pages or fewer | The text, and the charts and images in it |
| PDF of 101 to 1,000 pages | The text only |
| Word, CSV and other non-PDF documents | The text only |
| An image: JPEG, PNG, GIF or WebP | The image itself |

So a chart inside a Word document, or deep in a long PDF, isn't seen as a chart.

**The fix:** if a chart matters, upload it on its own as an image.

Upload with **+ → Add files or photos**, or drag files into the chat. You can add up to 20 files per chat, each up to 500MB.

**When to use it:** whenever a chart or table image carries your point.

```prompt Read the chart
This image is a chart from [the report]. First describe what it shows: the axes, the units and the time period. Then list the three biggest changes, with the values you read off the chart. Mark any value you're not sure of.
```

Check any marked value against the original.

![A table of what Claude reads in each kind of file, the upload limits, and the fix for charts it can't see](claude-habits/file-reading.webp "What Claude sees in each file type. When the chart matters, send it as an image.")

### 13. Ask for the finished file, by name

Claude can make the file itself: Word, Excel, PowerPoint and PDF.

It can also make **artifacts**: anything you'd put in front of someone, like a document, a deck, a dashboard or a small tool. An artifact opens beside the chat, and you can edit and share it.

**What's on which plan:** Docs, Slides and Design artifacts are on every plan. Data is on paid plans. Sheets and Whiteboards are in beta on paid plans. Slides export to PowerPoint or PDF.

**Check the setting once:** files and artifacts need **Code execution and file creation** turned on in **Settings → Capabilities**. In the updated layout it's called **Create and edit files**. It's on by default on Free, Pro, Max and Ultra.

It won't make photos or illustrations, but it does make [diagrams and charts](https://support.claude.com/en/articles/9002504-can-claude-produce-images).

**When to use it:** whenever the answer is going into a file anyway.

```prompt Make the file
Turn this into an Excel file: one sheet per [month], a totals row with working formulas, and a "Notes" sheet that explains each column in one line. Give me the .xlsx to download.
```

Expect a file to download, not a table to copy. Open it and click a total to see the formula.

![A grid of cards, one for each thing Claude can hand back: Word, Excel, PowerPoint and PDF files, and Docs, Slides, Design, Data, Sheets and Whiteboards artifacts, with the plan each needs and an example request](claude-habits/file-formats.webp "Name the format and you get the file. Plans are marked on each card.")

![A labelled drawing of Settings, Capabilities, with the Code execution and file creation switch turned on](claude-habits/capabilities.webp "Files, artifacts and skills all need this switch on. A labelled drawing, not a screenshot.")

### 14. Turn a brief you keep pasting into a skill

Pasting the same long brief for one kind of job?

That's what **skills** are for: folders of instructions, scripts and resources that Claude loads [when a task calls for them](https://support.claude.com/en/articles/12512176-what-are-skills).

Skills are on every plan. They need code execution on (habit 13).

Anthropic's built-in skills already handle Excel, Word, PowerPoint and PDF files, without you asking.

**To add your own:**

1. Put your skill in a folder and zip the folder.
2. Go to **Customize → Skills** and click **+**.
3. Click **+ Create skill**, then **Upload a skill**.

Anthropic is renaming **Customize** to **Plugins**, with [one Plugins page](https://support.claude.com/en/articles/17740565-manage-plugins-skills-and-connectors-from-one-plugins-page) in place of the separate tabs. Free, Pro, Max and Ultra accounts have it now.

**When to use it:** after you've explained the same job to Claude three times.

```prompt Turn this into a skill
Turn the way we did [the task] in this chat into instructions for a skill. Include: when to use it, the steps in order, the mistakes to avoid, and one short example of a good result. Write it so someone who wasn't in this chat could follow it.
```

Expect a draft set of instructions. The full walk-through, from folder to upload, is in [Make your first skill](issue:make-your-first-skill).

![A labelled drawing of the Skills page with the + button, + Create skill and Upload a skill, the steps numbered](claude-habits/skill-upload.webp "Where a skill goes in. Your sidebar may say Plugins instead of Customize. A labelled drawing, not a screenshot.")

::button[Give every job a home: the projects guide](guide:cowork-projects)

---

## Check before you trust

Anthropic is clear: don't treat Claude as your [only source of truth](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on), and look hard at any advice with high stakes.

Four checks turn that into a habit.

### 15. Let Claude say "I don't know"

Claude can write a quote that sounds authoritative and isn't grounded in fact.

Anthropic's guide to reducing hallucinations has a simple fix: give Claude explicit permission to admit it isn't sure.

Anthropic says this can [drastically reduce false information](https://platform.claude.com/docs/en/test-and-evaluate/strengthen-guardrails/reduce-hallucinations).

**When to use it:** add it to any question where a confident guess would hurt. Or put it in your Instructions once.

```prompt Permission not to know
If you're not sure of something, say "I don't know" or "I'm not sure" instead of guessing. I'd rather have a gap I can fill than a confident mistake. If the answer isn't in what I gave you, say so.
```

Expect the occasional "I don't know". That's the habit working.

![The same question asked three ways: a bare question, the question with permission to say I don't know, and the question tied to one source with a fallback answer](claude-habits/dont-know.webp "Give Claude a way out, and it stops filling gaps with guesses.")

### 16. Trace every claim to its source

Answers that use web search come with citations.

Click the ones your point depends on. The original page may hold context that Claude's summary left out.

Numbers, names and dates can slip too. So before anything goes into a report, ask where each one came from.

Anthropic suggests a stricter version too: have Claude find a supporting quote for each claim, and drop the claim if it can't.

**When to use it:** on anything you'll put your name to.

```prompt Trace every claim
Before I use this, list every number, name, date and quote in your answer. For each one, say where it came from: my file, a web source (with the link) or your general knowledge. Mark anything you're not sure of.
```

Then check everything marked "general knowledge" yourself.

![A table of where a claim can come from, your file, a web link, general knowledge or unsure, and what to check for each one](claude-habits/trace-claims.webp "Every claim has a source. Check the ones marked general knowledge first.")

### 17. Open the Thinking on hard questions

On the newest models, like Opus 5.5 and Sonnet 5.5, thinking can't be switched off.

A **Thinking** section sits above the reply. Click it to read a summary of how Claude worked the problem.

Anthropic says this can help you [check how Claude reached its conclusion](https://support.claude.com/en/articles/8664678-change-the-model-effort-and-thinking-settings).

If the reasoning rests on an assumption you'd never make, you've found the problem.

Want Claude to think less on a simple job? Choose a lower effort (habit 19).

**When to use it:** before you act on advice about money, people or plans.

```prompt Show me the assumptions
Before your final answer, list the assumptions you're making about [my situation] as numbered points, so I can correct any that are wrong. Then give the answer.
```

![A labelled drawing of a reply with the Thinking section above it, expanded to show a summary of how Claude worked it out](claude-habits/thinking.webp "The Thinking section sits above the reply. Click it before you act. A labelled drawing, not a screenshot.")

### 18. Treat "sent" as a claim

Claude can claim it sent an email or made a document somewhere else. It can also give you links that don't work.

But Claude [can't use tools](https://support.claude.com/en/articles/8241188-claude-is-producing-links-that-don-t-work-and-falsely-claiming-that-it-has-sent-emails-or-produced-external-documents-what-s-going-on) that aren't connected to it. That includes your email and your word processor, unless you've connected them.

So when it says "sent", look in your sent folder. Click any link before you forward it.

**When to use it:** at the start of any chat that ends with something going out.

```prompt What can you actually do here?
Before we start: which tools and connected apps can you use in this chat? List them. For anything outside that list, give me the text or the file, and I'll do the last step myself.
```

![Two columns side by side: what Claude might say it did, and what you check yourself before you trust it](claude-habits/sent-check.webp "A 'done' from Claude is a claim. Check the outbox, the file or the page.")

::button[Ask Claude about your data, the careful way](guide:ask-claude-about-your-data)

---

## Make your usage last

Most accounts have a limit that resets on a rolling five-hour window. Paid plans add a weekly limit.

There's no fixed message count.

Anthropic lists what [uses your limits faster](https://support.claude.com/en/articles/9797557-usage-limit-best-practices): long messages, big attachments, long chats, tools like Research and web search, the model, the effort level, artifacts, and multi-step tasks.

Four habits handle most of that.

![What uses your limits faster, from long chats to high effort, narrowing into the habits that stretch them](claude-habits/usage-drains.webp "What drains your usage, and the habit that answers each one.")

### 19. Match the effort and the model to the job

Effort sets how much thinking Claude puts into each reply. The default in the apps is **Medium**. Higher effort is more thorough, but slower, and uses more of your limits.

**Where:** click the model name next to the send button, click **Effort**, and choose:

- **Low** or **Medium** for routine work. They stretch your usage furthest.
- **High** for the best balance of quality and speed.
- **Extra high** for long coding and agent tasks.
- **Max** for the hardest problems, when you can wait.

The model matters too. Anthropic calls Opus 5.5 its [strongest daily driver](https://www.anthropic.com/claude-fable-and-mythos-5-2), yet for chat users it suggests [the latest Sonnet or Haiku for most tasks](https://support.claude.com/en/articles/12429409-manage-usage-credits-for-paid-claude-plans), because they cost fewer tokens than Opus.

On the Free plan, you get Sonnet and Haiku.

Changes apply from the next reply. Raise effort for the one hard question, then bring it back down.

**When to use it:** when you need a fast answer, not a deliverable. Saying so is one of Anthropic's own usage tips.

```prompt Quick answer only
Quick answer only: two or three sentences, no file, no research. [your question]
```

![A labelled drawing of the model menu next to the send button: Opus 5.5, Sonnet 5.5, Haiku 5.5, More models and Effort with its five levels](claude-habits/model-effort.webp "Model and effort live in one menu, next to send. A labelled drawing, not a screenshot.")

### 20. Start a fresh chat, and bring the thread with you

When a long chat nears its limit, Claude can summarise the earlier messages to keep going. You may see it "organizing its thoughts".

Handy, but those long chats use more of your limits.

Anthropic suggests a new chat when you're close to your limit in a long one. And a new chat for unrelated work, so Claude isn't carrying extra context.

Take the useful part with you.

**When to use it:** when a chat has run long, or the topic has changed.

```prompt Hand over to a new chat
Write a handover note for a fresh chat: the goal, what we've decided, the facts and files that matter, the open questions and the next step. Write it so I can paste it as the first message of a new chat.
```

On paid plans, there's a second way back. Claude can [search your past chats](https://support.claude.com/en/articles/11817273-use-claude-s-chat-search-and-memory-to-build-on-previous-context), and you'll see the search in the chat as a tool call. Inside a project, it searches only that project's chats.

**When to use it:** when you've lost the thread and the old chat had it.

```prompt Pick up where we left off
Find our conversation about [the topic]. Remind me what we decided, then let's continue from there.
```

To stop Claude searching old chats, switch off **Search and reference chats** in **Settings → Your Claude**.

![A five-stop path from a long chat to a fresh one: notice the signs, ask for the handover note, open a new chat, paste it in, and carry on](claude-habits/handover.webp "Five steps from a heavy chat to a light one, with nothing lost.")

### 21. Switch off the connectors a chat doesn't need

Connectors let Claude reach your apps and services, read from them and take actions. They use your own permissions in each app.

Anthropic's note: tools and connectors are [token-intensive](https://support.claude.com/en/articles/11647753-how-do-usage-and-length-limits-work). It suggests turning off the ones a chat doesn't need.

**Where:**

1. Click **+** in the message box, or type **/**.
2. Hover over **Connectors**.
3. Switch on only the apps this chat needs.

Lots of apps connected? Under **Connectors → Tool access**, the default is **Auto**. With 10 or more connectors active, Anthropic suggests [On demand](https://support.claude.com/en/articles/11176164-use-connectors-to-extend-claude-s-capabilities), to give your chats more room.

Want Claude to ask before it uses one tool? **Customize → Connectors**, pick the connector, then **Permissions → Tool permissions → Always ask**. Anthropic is rolling this out gradually.

**When to use it:** at the start of a chat, when you're not sure which apps it needs.

```prompt Which apps does this need?
For this task: [describe it]. Which of my connected apps do you actually need? Name them, and tell me which ones I can switch off for this chat.
```

![A labelled drawing of the + menu with Connectors open, a switch for each app, and Tool access set to Auto or On demand](claude-habits/connectors.webp "Only the apps this chat needs. A labelled drawing, not a screenshot.")

### 22. Check Settings → Usage before a big job

Your five-hour session starts with your first message and resets in full five hours later. The weekly limit on paid plans resets at a fixed day and time.

On Pro, Max, Ultra, Team and seat-based Enterprise plans, **Settings → Usage** shows:

- **Current session:** how much you've used, and the time left
- **Weekly limits:** when your week resets
- **Usage credits** (Pro, Max and Ultra): an optional switch to keep working at API rates, with a monthly spend limit you set

If credits take over, the warning changes to "Now using credits". You haven't been cut off.

Been given a limit reset? It shows there as **Reset for free**. A reset can't be undone, so keep it for a real deadline.

**When to use it:** before a Research report, a big file or a Cowork task.

```prompt Plan around my limit
I have a big job: [describe it]. Before you start, split it into steps I can run in separate, smaller chats. Mark which steps need Research, a big file or a long output, so I can run those when my usage has reset.
```

![A labelled drawing of Settings, Usage: Current session, Weekly limits and Usage credits, the steps numbered](claude-habits/usage.webp "Look here before you start something long. A labelled drawing, not a screenshot.")

::button[Stop running out of Claude: the usage guide](guide:claude-usage-limits)

---

## Hand bigger jobs to Cowork

Cowork brings Claude Code's way of working to everyday knowledge work. You describe an outcome, step away, and [come back to finished work](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork).

It's on paid plans: Pro, Max, Ultra, Team and Enterprise.

To start one, select **Cowork** in the message box, describe the task, review Claude's approach, then let it run.

In the new Claude experience, there's no Cowork option to pick. You just describe the task in any conversation.

Tasks run in the cloud and keep going if you close your laptop. Local files, the browser and computer use need Claude Desktop open.

Cowork uses more of your limit than a chat. Four habits keep it safe and worth it.

### 23. Hand over the outcome, and one folder

Cowork works best when you describe the finished result, not the clicks.

Claude can only read and write files in folders you've connected. Anthropic's advice: make a [dedicated working folder](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) for Claude rather than giving it broad access, and keep backups of important files.

Claude must ask (**Allow**) before it permanently deletes a file.

Anthropic's usage tip: group related work into one task instead of several.

Standing preferences go in **Settings → Cowork → Edit** next to **Global instructions**, then **Save**. In the new Claude experience, they live in your Instructions (habit 1).

**When to use it:** for any job with files going in and a file coming out.

```prompt A Cowork brief
In the folder [folder name], read every [invoice PDF]. Make one Excel file with a row per invoice: date, supplier, amount and due date, with a total at the bottom. Save it in the same folder as [file name]. Don't move, rename or delete anything. Show me your plan before you start.
```

Expect a short plan first. Read it before you let it run.

![A habit card for Cowork: describe the outcome, use one dedicated folder, the brief to copy and what to watch out for](claude-habits/cowork-outcome.webp "One outcome, one folder, a plan you read first.")

### 24. Match the approval mode to the risk

Cowork has three approval modes. You can change the mode at any time from the mode selector in the chat box.

| Mode | What happens | Use it when |
|---|---|---|
| **Manually approve** | Claude pauses for **Allow** or **Deny** on actions | Sensitive files, accounts or sites; a new tool, plugin or site; anything hard to undo |
| **Automatically approve** | An automated check reviews each action and blocks unsafe ones | Everyday tasks you can easily check and undo |
| **Skip all approvals** | Nothing checks Claude's actions | Only when you completely trust every action, connector, file and app involved |

Auto is rolling out as the default on Pro, Max, Ultra and Team. Manual is the default on Enterprise.

Anthropic says the automatic check [can miss risky actions](https://support.claude.com/en/articles/13345190-get-started-with-claude-cowork). For money, messages sent as you or important files, stay close or switch to Manual.

And you remain responsible for everything Claude does on your behalf.

**When to use it:** at the top of any task that could send, delete or buy.

```prompt Ask before anything permanent
Before you send, delete, move or buy anything, stop and ask me. Show me exactly what you're about to do, and wait for my yes.
```

![A three-row guide to the approval modes: what each one does, when to use it, and the default on each plan](claude-habits/approval-modes.webp "Pick the mode by what could go wrong, not by what's fastest.")

### 25. Schedule the jobs you repeat

A weekly summary, a Monday report, a daily inbox digest?

Scheduled tasks run a saved prompt on a regular basis, or when you ask. They're on [all paid plans](https://support.claude.com/en/articles/13854387-schedule-recurring-tasks-in-claude-cowork).

**Where:**

1. Click **Scheduled** in the left sidebar, or type **/schedule** in a Cowork task.
2. Click **New task** in the upper right.
3. Choose **Create with Claude** (it asks you questions, then you click **Schedule**) or **Set up manually**.

Set up manually asks for a name, the prompt, the approval mode, how often (hourly, daily, weekly, on weekdays, or manually), and an optional model and folder.

Pick a folder, and **Require this computer** turns on: the task runs only while your computer is awake with Claude Desktop open. Without one, it runs remotely, even while your computer sleeps.

In the new Claude experience, describe the task and how often in any chat, then click **Schedule** on Claude's proposal.

Anthropic's safety advice: start simple. Don't schedule tasks that touch sensitive files, send messages as you or buy things. Review the output after each run, and pause tasks you're not using.

**When to use it:** for any job you've done the same way three weeks running.

```prompt A weekly digest to schedule
Every weekday morning, read the files added to [folder] since the last run. Write a one-page summary: what's new, what needs a decision from me, and what can wait. Save it as a document named with today's date. Don't send it anywhere.
```

![A labelled drawing of the Scheduled page in Cowork with the New task button, Create with Claude and Set up manually, the steps numbered](claude-habits/scheduled.webp "Scheduled is in the left sidebar. A labelled drawing, not a screenshot.")

### 26. Keep computer use for last

Computer use lets Claude work your screen: clicking, typing and opening apps.

Claude only reaches for it when there's no connector or tool for the job. It tries connectors first, then the browser, then your screen.

It's on Pro, Max and Ultra, in Cowork and Claude Code, in Claude Desktop for Mac and Windows. It's in beta on Team and Enterprise, and not available on Linux.

**Where:** in the desktop app, **Settings → Capabilities → Enable computer use**. No section there? Try **Settings → System**, under **This computer**.

Claude asks your permission before it uses each app.

Limits come built in: browsers and many trading apps are view-only, terminals and code editors get clicks but no typing, and you can block apps yourself.

Computer use isn't limited to your connected folders, and the delete protection from habit 23 [doesn't cover it](https://support.claude.com/en/articles/14128542-let-claude-use-your-computer-in-cowork).

Anthropic's advice: don't give it banking, healthcare or government apps, and start with simple tasks. It's slower than a connector, and your computer must stay awake with Claude Desktop open.

**When to use it:** for your first try, on a harmless, easy-to-check task.

```prompt A safe first try
Open [an everyday app, like Notes] and [one simple, harmless action]. Use only that app. Ask before you open any other app, and stop if anything asks for a password or a payment.
```

![A labelled drawing of the desktop app's Settings, Capabilities, with Enable computer use switched on and the app permission steps numbered](claude-habits/computer-use.webp "Turn it on in the desktop app, then approve one app at a time. A labelled drawing, not a screenshot.")

::button[Hand Claude the mouse: the computer use issue](issue:claude-uses-your-computer)

---

## Start with these five

Twenty-six is a lot. Start with the five that pay off in your next chats:

1. **Write your Instructions** (habit 1). Set once, and every chat starts closer.
2. **Send one full message** (habit 4). Fewer round trips, and a better first answer.
3. **Trace every claim** (habit 16). Run it on anything you'll put your name to.
4. **Hand over to a fresh chat** (habit 20). Long work keeps its thread, without a long chat's cost.
5. **Check Settings → Usage** (habit 22). Before anything big.

Then add one more habit each week.

![A five-stop path through the habits to start with: Instructions, one full message, trace every claim, hand over to a fresh chat and check your usage](claude-habits/start-five.webp "Your first five habits, in order. Add one a week after that.")

::subscribe[Get the next issue by email]

---

## What to do today

Open **Settings → Your Claude** and read your memory topics.

Fix one thing that's out of date.

Then add one correction you keep making to your Instructions.

Two habits, done before your next chat.

::card[issue:one-good-prompt]

Send this to one colleague whose Claude chats run for days.

Mojeed
