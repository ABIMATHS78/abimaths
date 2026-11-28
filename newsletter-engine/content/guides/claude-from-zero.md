+++
title = "Claude, from the first click."
subtitle = "Never opened it? Everything you need for your first week, in plain words:"
description = "What Claude is, the plans, the apps, how to ask, what to trust and check, the features that matter, a ten-minute trial and eight ways professionals use it."
updated = 2026-11-28
status = "published"
cover = "claude-from-zero/cheat-sheet.webp"
cover_alt = "A one-page Claude cheat sheet: what it is, where to get it, the plans, the apps, how to ask, what to check, the features that matter and your first week"
tags = ["claude", "beginners"]
archetype = "guide"
level = "beginner"
topic = "Claude"
brief = "briefs/claude-from-zero.json"
fact_checks = ["Check the Settings names in your own Claude app: Settings → Your Claude → Instructions, Manage next to Your memory, and Start import (the help pages say the older layout calls Your Claude 'Account').", "Check the + menu in the message box: Add files or photos, Web search, Research and Connectors (in the new Claude experience there is no Web search switch).", "Check the name of the training switch in Settings → Privacy: one help page calls it 'Help improve our AI models', the privacy centre calls it 'Model Improvement'.", "Check that Free accounts can still create projects: the help centre says up to five, while the pricing page lists Projects under Pro.", "Check the sign-in screen still offers 'Continue with Google' and 'Continue with email', and that Settings → Usage still shows 'Plan usage limits', 'Current session', 'Weekly limits' and 'Usage credits'.", "Check the model menu next to the send button: 'More models', 'Effort', and the levels Low, Medium, High, Extra high and Max with one marked 'Default'.", "Check which plans Claude for PowerPoint lists: its documentation names Pro, Max, Team and Enterprise, while claude.com says the Microsoft 365 add-ins are on all paid plans.", "Check whether Free accounts can share artifacts by link: the help-centre table says no, claude.com's Artifacts page says yes."]
+++
Everyone seems to be talking about Claude.

You've never opened it.

Or you opened it once, typed a question, and weren't sure what to do next.

That's fine. This guide starts at zero and goes one step at a time.

I teach people to use AI at work, and my mission is to make data and AI easy for everyone. So: plain words, real steps, and a picture wherever you need to find something on screen.

**By the end of this guide you'll have:**

- Claude on your phone and your computer, on the right plan
- the few settings that matter, set once
- a simple way to ask that gets good answers
- a clear sense of what to trust and what to check
- one real task done, and eight ideas borrowed from professionals

**You need:** a phone or a computer, an email address, and to be 18 or older.

**How to read it:** sections 1 to 4 get you set up. Sections 5 to 9 teach you to use it well. Sections 10 to 13 are practice.

![A one-page Claude cheat sheet with nine panels: what it is, where to get it, the plans, the apps, how to ask, what to check, the features that matter, your first week and where to click](claude-from-zero/cheat-sheet.webp "The whole guide on one page. Save it, or keep it open beside Claude in your first week.")

::subscribe

---

## 1. Meet Claude

Claude is an AI assistant made by a company called Anthropic.

Anthropic's help centre describes it as a large language model, trained to be helpful, honest and harmless, with a conversational tone.

In plain words: you write to it the way you'd write to a person, and it writes back.

It helps in two ways.

**It answers.** Ask a quick question, get a quick answer, then ask a follow-up.

**It does the work.** Describe the result you want, and Claude works through the steps: it searches the web, reads your files, runs code and hands back finished files. On paid plans, longer tasks keep running in the cloud if you close your laptop.

Anthropic calls itself a public benefit corporation. Its stated mission is the responsible development of advanced AI for the long-term benefit of humanity.

![Four cards explaining Claude: who makes it, what it is, how you talk to it, and the two ways it helps, each with a plain-English line and an example](claude-from-zero/what-it-is.webp "Claude in four facts. The examples show the kind of thing you can type on day one.")

### What it can make for you

Most beginners undersell this. Claude doesn't only write replies. It makes things you can use:

- **Files:** Word, Excel, PowerPoint and PDF, to download or save to Google Drive. Spreadsheets come with working formulas.
- **Artifacts:** a design, a deck, a document, a dashboard or a small interactive tool, opened beside the chat.
- **Diagrams and charts:** drawn in the chat on the web and desktop when you say "draw this as a diagram".

![A grid of twelve cards, each one a thing Claude can hand back, from a Word report and an Excel budget to a slide deck, a dashboard, a diagram and a sourced answer, with an example request on each](claude-from-zero/what-it-makes.webp "Twelve kinds of thing Claude can hand back. Name the format in your request and you'll get it.")

Not sure where to start? Anthropic's own advice is to ask Claude what it can help with. Make it personal:

```prompt What can you do for me?
I work as [your job]. A normal week for me includes [three or four tasks, for example writing reports, answering customer emails, planning events]. List ten specific ways you could help me this week. For each one, give me the exact first message I should send you. Start with the ones that would save me the most time.
```

**Use it** in your first five minutes. **You'll get** ten jobs, each with a first message you can copy straight back in.

> [!KEY]
> Treat Claude like a capable new colleague. Give it a proper brief, and check its work before anything goes out under your name.

---

## 2. See how it differs from ChatGPT

It's the first question most people ask.

Here's a fair answer, built only from what each company publishes about its own product.

| | Claude | ChatGPT |
|---|---|---|
| **Free plan** | Yes, with limited usage every 5 hours and every week | Yes, advertised as unlimited text chats with GPT-6, within abuse guardrails |
| **First paid plan** | Pro: $20 a month, or $17 a month paid yearly | Plus: $20 a month |
| **Bigger plans** | Max at $100 and Ultra at $200 a month | Pro 100, Pro 200 and Pro 500, at $100, $200 and $500 a month |
| **Ads** | None: Anthropic has committed to keeping Claude ad-free | May appear on the Free and Go plans |
| **Making pictures** | No photos or illustrations; it makes diagrams, charts and visuals | Image generation on every plan (limited on Free) |
| **Text it takes in at once** | Up to 1M tokens in chat on paid plans, with the newest models | 27K to 128K tokens (GPT Instant) or 256K to 400K (GPT Reasoning), by plan |

A token is a small piece of text. For scale, Anthropic says 200K tokens is about 500 pages.

![A side-by-side comparison of Claude and ChatGPT across the free plan, paid plans, ads, image generation and how much text each takes in at once](claude-from-zero/vs-chatgpt.webp "Claude and ChatGPT side by side, using only what each company publishes about itself.")

What that means for you:

- **You want pictures made?** ChatGPT makes images. Claude doesn't.
- **You work with long documents?** Claude's paid plans take in more text at once.
- **You'd rather not see ads?** Anthropic has promised no ads and no sponsored links in Claude. ([Its announcement](https://www.anthropic.com/news/claude-is-a-space-to-think), and [OpenAI's page on ads](https://help.openai.com/en/articles/20001047-ads-in-chatgpt).)
- **Price?** The first paid step costs the same: $20 a month.

> [!TIP]
> Coming from ChatGPT? Your old chats can't move over, but its memory of you can. Go to **Settings → Your Claude** and select **Start import**. Claude shows you a prompt to use in your old assistant. Paste the answer back and click **Add to memory**. Anthropic calls this experimental. [Its guide is here](https://support.claude.com/en/articles/12123587-import-and-export-your-memory-from-claude).

::card[issue:right-ai-for-the-job]

---

## 3. Get Claude and pick a plan

### Download and open it

One account, three places:

- **In a browser:** go to **claude.ai**.
- **On your computer:** Claude Desktop, from [claude.com/download](https://claude.com/download).
- **On your phone:** search "Claude by Anthropic" in the App Store or the Play Store.

Your conversations, projects, memory and preferences sync between them.

| Device | Needs |
|---|---|
| Mac | macOS 13 (Ventura) or newer |
| Windows | Windows 10 or newer |
| Linux (beta) | Ubuntu 22.04 or newer, or Debian 12 or newer |
| iPhone and iPad | iOS or iPadOS 18.0 or newer |
| Android | Android 8.0 (Oreo) or newer |

You must be **18 or older** and in a **supported location**. Nigeria, the United Kingdom and the United States are on Anthropic's list, with many others. [Check yours here](https://support.claude.com/en/articles/8461763-where-can-i-access-claude). A new account also needs a phone number from a supported location.

### Sign in

There's no password. You get two ways in:

1. **Continue with Google:** sign in with your Google account and click **Next**.
2. **Continue with email:** type your email and click the button. Claude emails you a link titled "Secure link to log in to Claude.ai".

Opened that email on another device? The link gives you a code instead. Type it on the device where you started.

No email? Check your spam folder for a message from @mail.anthropic.com.

![A labelled drawing of the Claude sign-in screen with Continue with Google, Continue with email and the verification code step numbered](claude-from-zero/sign-in.webp "Two ways in, and no password to remember. A labelled drawing, not a screenshot.")

### Pick a plan

Start on Free. It costs nothing and covers a lot.

These are US prices: Anthropic shows local currency where it supports it. Prices exclude tax and can change.

| Plan | Price | What you get |
|---|---|---|
| **Free** | $0 | Chat on web, desktop and phone, web search, file creation, memory, connectors and artifacts |
| **Pro** | $20 a month, or $17 a month billed yearly ($200 up front) | More usage than Free, plus Cowork, Claude Code, Research, more models, and Claude in Chrome and Microsoft 365 |
| **Max** | $100 a month | 5x Pro's usage, early access to features, priority at busy times |
| **Ultra** | $200 a month | 10x Pro's weekly usage, and no 5-hour limit for verified accounts |
| **Team** | $20 a seat a month billed yearly, or $25 monthly (Premium seats: $100 or $125) | Teams of 2 to 150, with admin controls |
| **Enterprise** | $20 a seat a month billed yearly, plus usage at API rates | Larger organisations, from 20 seats |

[See the full plan details](https://claude.com/pricing).

- **Your first week:** Free.
- **You keep hitting the limit, or want Cowork or Claude Code:** Pro.
- **Pro's limits still stop you most days:** Max or Ultra.

**To upgrade:** click your initials (bottom left), then **Settings → Billing → Upgrade plan → Get Pro plan**. Choose monthly or yearly and click **Subscribe**. Max and Ultra are monthly only.

![A plan sheet with the six Claude plans, their prices, what each adds, how billing works and which plan fits which person](claude-from-zero/plans.webp "Six plans on one sheet. Most people start on Free and move to Pro when the limit gets in the way.")

### How the limits work

There's no fixed number of messages. Instead:

- Your limit runs on a **five-hour window**, starting with your first message.
- Paid plans add a **weekly limit**, which resets at a fixed day and time.
- On Free, how much you get also varies with demand.
- Web, desktop, phone and Claude Code all draw from the **same pool**.

To see where you stand, open **Settings → Usage**. **Current session** shows how much of your five-hour limit you've used and when it resets. On paid plans, **Weekly limits** shows your weekly reset.

Paid plans also see **Usage credits**: a pay-as-you-go top-up, with a monthly cap you set. Leave it off in week one.

![A labelled drawing of Settings, Usage, with Current session, Weekly limits and Usage credits numbered](claude-from-zero/usage.webp "Settings → Usage tells you what's left right now. A labelled drawing, not a screenshot.")

> [!WARNING]
> Paying monthly? To stop, cancel in **Settings → Billing** at least 24 hours before your renewal date, or the next charge goes through. Your plan stays active until the end of the period you paid for.

::button[How Claude's usage limits work](guide:claude-usage-limits)

---

## 4. Find your way around

"Claude" can mean an app, a way of working or an add-in. Here's the map.

![A grid of nine cards: Claude on the web, on desktop and on your phone, plus chat, Cowork, Claude Code, Claude in Chrome, Claude for Microsoft 365 and voice mode, each with its plans and a first thing to try](claude-from-zero/surfaces.webp "Every place you'll meet Claude, with the plan each one needs.")

### Chat, Cowork and Claude Code

| | Chat | Cowork | Claude Code |
|---|---|---|---|
| **You** | Talk it through | Hand over a task | Build software |
| **Plans** | All, including Free | Paid | Paid |
| **A good first try** | A question or a draft | A report from your files | A small change to your code |

**Cowork** brings Claude Code's way of working to everyday work, with no command line. Claude makes a plan, does the steps and hands back finished files. It runs in the cloud, so the work carries on if you close your laptop.

**Claude Code** reads a codebase, edits files and runs commands. Don't write code? Skip it.

Here's what a Cowork-sized request looks like:

```prompt A whole task (paid plans)
Research [five suppliers, tools or options] for [your situation and budget]. For each one, find the price, the main features and one weakness, and note your sources. Give me an Excel file with one row per option, and a one-page Word summary of which you'd pick and why. Ask me what you need to know before you start.
```

**You'll get** a plan first, then two files to download, with sources you can open.

**Which do you have?** Look at your message box.

- **It shows Chat and Cowork:** select **Cowork**, bottom left, to hand over a task. Select **Chat** to go back.
- **It shows neither:** you have the new Claude experience. Chat and Cowork are one conversation. Just describe what you need. It's rolling out to Pro, Max and Ultra, and once your account moves, it can't switch back.

> [!WARNING]
> Cowork can take actions, like changing files. Anthropic warns that tools that change things carry more risk than tools that only read. For anything with real consequences, such as sending messages or changing important files, stay close and review what Claude does.

### The message box and the + button

Everything starts in the **message box**. Type, then click send.

**The + button**, lower left, holds the extras:

- **Add files or photos:** attach a document or an image, or drag it in.
- **Web search:** answers with sources you can click.
- **Research** (paid plans): many searches that build on each other.
- **Connectors:** link the apps you use.

Type **/** to see more options and commands.

![A labelled drawing of Claude's message box with the + menu open, showing Add files or photos, Connectors, Web search and Research](claude-from-zero/message-box.webp "Where things are in the message box. A labelled drawing, not a screenshot.")

> [!NOTE]
> Have the new Claude experience? There's no Web search switch: Claude searches when it helps. **Research** is still under **+**, or type **/deep-research**.

### The model menu

**The model name**, next to the send button, shows which Claude you're talking to. On your phone, it's at the top of the screen. Click it to switch, or click **More models**.

| Model | Anthropic says it's for | Speed | Plans |
|---|---|---|---|
| **Fable 5.2** | Demanding reasoning and long, complex work | Moderate | Paid |
| **Opus 5.5** | Complex work that needs careful judgment | Moderate | Pro and up |
| **Sonnet 5.5** | Everyday tasks, documents, slides and spreadsheets | Fast | All, including Free |
| **Haiku 5.5** | Quick, high-volume jobs | Fastest | All, including Free |

Not sure? Anthropic says start with **Opus 5.5** ([more on Opus 5.5](issue:opus-5-5)). For everyday chat, it also suggests Sonnet or Haiku, which use less of your limit.

The same menu holds **Effort**: Low, Medium, High, Extra high and Max. Higher effort thinks longer and uses more of your limit. The recommended level is marked **Default**. For everyday tasks, leave it.

![A labelled drawing of the model menu open above the message box, with Opus 5.5, Sonnet 5.5, Haiku 5.5, More models and Effort, and the steps numbered](claude-from-zero/model-picker.webp "The model menu, next to the send button. A labelled drawing, not a screenshot.")

Two more things on screen. **The ghost icon**, top right when you start a new chat outside a project, opens an **incognito chat**: not saved to your history or memory, not used for training, though Anthropic keeps it for 30 days. **Your initials**, bottom left, open **Settings**.

---

## 5. Talk to it like a colleague

Anthropic's own advice: talk to Claude the way you'd talk to a coworker or a friend.

You don't need magic words. You need context. Give it three things:

1. **The outcome:** what you want to end up with. "A one-page summary."
2. **The format:** a Word document, a slide deck, a message you can paste.
3. **The inputs:** paste the text, attach the file, or name the app.

![One prompt split into coloured parts: the vague opening to delete, the request to keep, and the outcome, format and inputs to add, with a reason beside each](claude-from-zero/prompt-anatomy.webp "The anatomy of a good brief: keep the request, add the outcome, the format and the inputs.")

See the difference:

```prompt Too thin
Write an email about the meeting.
```

```prompt A real brief
Write a short, friendly email to my team about Thursday's planning meeting.
Outcome: everyone knows the three decisions and who does what by Friday.
Format: under 150 words, one opening line, then bullet points.
Inputs: my notes are below.

[paste your notes]
```

![The same request written three ways in chat boxes: a bad one-line prompt, a better one with the audience, and a best one with outcome, format and inputs highlighted](claude-from-zero/prompt-ladder.webp "Bad, better, best. Each step up adds one thing Claude can't guess.")

Keep this fill-in template somewhere handy:

```prompt The brief template
I need [the outcome, for example a one-page summary or a table of options].
It's for [who will read or use it].
Work from [paste the text, attach the file, or name the source].
Give it to me as [the format: a Word document, bullet points, an email I can paste].
Keep it [length and tone].
Before you start, ask me anything you need to know.
```

**Use it** for any request that matters. **You'll get** a few questions first, then a draft that fits.

Not sure what Claude needs? Hand the questions over:

```prompt Let it ask first
I want to [the job]. Before you start, ask me the questions you need answered to do this well. One question at a time. When you have enough, tell me and start.
```

### Five habits from Anthropic's own guides

1. **The colleague test.** If a colleague with no background would be confused by your request, Claude will be too.
2. **Give the reason.** "Under 100 words, because it'll be read on a phone" beats "under 100 words".
3. **Say what to do.** "Write in flowing paragraphs" beats "don't use bullet points".
4. **Document first, question last.** With a long document, put it first and your question at the end.
5. **Show one example.** If the format matters, paste one you like.

The first answer is a draft. Refine it: "Shorter, and warmer." "Turn that into a table." "What's missing?"

Before a request that matters, let Claude check your brief:

```prompt Review my prompt
Here's a request I'm about to give you. Don't answer it yet. Tell me what's unclear or missing and what you would have to guess. Then rewrite it as a better brief I can approve.

[paste your request]
```

> [!TIP]
> New topic? New chat. Long conversations use up your limit faster.

::card[guide:brief-claude-like-a-colleague]

---

## 6. Four jobs it does well

Four jobs where beginners get value fastest, each with a prompt.

### Explaining anything at your level

Claude explains at whatever level you ask for, and it doesn't mind the fifth follow-up.

![A handout on learning with Claude: what to give it, a prompt to copy, what a good answer looks like and how to check you've understood](claude-from-zero/good-explain.webp "Learning with Claude: say your level, ask for an everyday comparison, and let it quiz you.")

```prompt Explain it at my level
Explain how [a topic you've always half-understood] works. I'm a complete beginner. Use one everyday comparison and one short example, then ask me one question to check I've understood. If I get it wrong, explain it a different way.
```

**You'll get** a few short paragraphs, one comparison, one example and a question for you.

### Writing and rewriting

Emails, posts, reports and hard messages, in the tone you ask for. Paste your own rough draft for the best results.

![A handout on writing with Claude: paste your draft, say who it's for and the tone, ask for two versions, then refine with one follow-up](claude-from-zero/good-writing.webp "Writing with Claude: your draft plus the reader and the tone, then one round of follow-ups.")

```prompt Rewrite my draft
Here's a message I need to send to [who]. Make it clear, polite and short. Keep my meaning and any facts exactly as they are. Give me two versions: one warmer, one more formal. Under each, tell me in one line what you changed.

[paste your draft]
```

**You'll get** two versions and a note on what changed, so you stay in control of the meaning.

### Reading your files

Attach a file with **+ → Add files or photos**, or drag it in.

Claude reads PDF, Word, Excel, CSV, text and several other document types, plus JPEG, PNG, GIF and WebP images. Up to 20 files per chat, and PDFs of up to 1,000 pages.

One catch: it looks at images and charts only in PDFs of 100 pages or fewer. In longer PDFs, it reads the text only.

![A handout on reading files with Claude: which files it reads, the limits, a prompt to copy and what to check before you rely on the answer](claude-from-zero/good-files.webp "Reading files: attach, ask for decisions and actions, and ask where in the document each one is.")

```prompt Read a document
Read this document. Summarise it in five bullet points. Then list anything that needs a decision or an action from me, with the page or section where it appears. If something is unclear in the document, say so instead of guessing.
```

**You'll get** five bullets, then an action list with page numbers you can check.

### Making real files

Ask for the file by name: "as an Excel file", "as a Word document", "as a PowerPoint deck", "as a PDF". Then download it or save it to Google Drive.

Won't make files? Check **Settings → Capabilities → Create and edit files** is on (**Code execution and file creation** in the older layout).

![A handout on getting files from Claude: the four file types, how to ask, the setting to check and a spreadsheet prompt with working formulas](claude-from-zero/good-making.webp "Making files: name the format, describe the columns or slides, and ask for formulas that work.")

```prompt Get a spreadsheet back
Make me a monthly budget as an Excel file. Columns for income, fixed costs, spending and savings, one row per month for a year. Use formulas so the totals update when I change a number, and add a summary row at the bottom. Then tell me which cells I should fill in first.
```

**You'll get** a file to download, and a short note on where to type.

---

## 7. Know where it slips, and what to do instead

Here are the four ways Claude lets beginners down, and the fix for each.

### It can be confidently wrong

Anthropic calls this hallucination: a confident answer that's wrong or misleading. Its advice: don't treat Claude as your only source of truth, scrutinise high-stakes advice, and check the sources it cites.

It can also claim things it didn't do, like sending an email. It can only use tools that are actually connected.

![Two columns: what can go wrong, such as invented facts, wrong numbers, broken links and claims it sent an email, and what to do instead for each](claude-from-zero/bad-made-up.webp "When Claude is confidently wrong, and the habit that catches each case.")

```prompt Check your own answer
Go back through your answer. List every fact, number, name and date you used, where each one came from, and how sure you are. Flag anything I should check myself. If you don't know something, say "I don't know" rather than guessing.
```

For long documents, Anthropic's guidance says asking for exact quotes first reduces made-up answers, though it doesn't remove them:

```prompt Quotes first
Before you answer, find the exact sentences in the document that relate to my question and quote them word for word. Then answer using only those quotes. If the document doesn't answer it, say so.

My question: [your question]
```

### It doesn't know recent news

Claude's newest models were trained on data up to June 2026. They may not know what happened after that.

The fix: let it search. Web search brings in current pages, with citations, on every plan.

![A handout on Claude's training cutoff: what it means, the cutoff month for each model, when to use web search or Research, and a prompt to copy](claude-from-zero/bad-cutoff.webp "Training cutoffs, and how to get current answers anyway: search for facts, Research for big questions.")

```prompt Get current facts
Search the web for the latest on [topic] in [your country or industry]. Give me the five most important points, each with its date and a link to the source. Tell me if sources disagree.
```

Anthropic suggests web search for questions that need one or two searches, and **Research** (paid plans) for ones that need five or more.

### It won't make photos

Claude doesn't generate photos or illustrations. It makes diagrams, charts and interactive visuals with code, and it can read images you upload.

![Two columns: what Claude won't make, such as photos and illustrations, and what it will make instead, such as diagrams, charts, flowcharts and dashboards](claude-from-zero/bad-images.webp "No photos or illustrations. Ask for a diagram, a chart or a visual instead.")

```prompt Draw it as a diagram
Draw this process as a simple diagram: [describe the steps, for example how a customer order moves from website to delivery]. Label each step in a few words and show who is responsible for each one.
```

To keep a diagram, use **Copy as image**, **Download** or **Save as artifact**. Need a real photo? Use an image tool for that part.

### It runs out

Anthropic lists what uses your limit faster: long messages, big attachments, long conversations, Research and web search, bigger models, higher effort, artifacts and multi-step tasks.

![A funnel from habits that burn through your limit, like one endless chat and big files you don't need, to habits that make it last, like new chats for new topics and batched questions](claude-from-zero/bad-limits.webp "What drains your limit, and the habits that make it last.")

Its fix: batch related questions into one message, with the background up front.

```prompt Batch it
I have four related questions about [topic]. Background: [two lines]. Answer each one in under 100 words, numbered, in one reply.
1. [question]
2. [question]
3. [question]
4. [question]
```

> [!WARNING]
> Check every number, name, date, quote and link before you use it. Anthropic's own advice is to scrutinise high-stakes answers and check the sources Claude cites. [Its note on wrong answers](https://support.claude.com/en/articles/8525154-claude-is-providing-incorrect-or-misleading-responses-what-s-going-on).

::card[guide:ask-claude-about-your-data]

---

## 8. Meet the five features worth knowing

Five words make Claude far more useful. Learn these, skip the rest.

### Projects

A workspace with its own chats, files and instructions. Use one per ongoing piece of work: a client, a course, a job search.

1. Click **Projects** on the left, or go to **claude.ai/projects**.
2. Click **+ New Project**, upper right.
3. Add a name and description. They're for you: Claude doesn't see them.
4. On the right of the project page, click **+** to add files.
5. Click **Set project instructions**, write them, then **Save instructions**.

Chats in a project don't share what's said in them. Only the project's files and instructions are shared, so put anything important there. The help centre says Free accounts can make up to five projects.

![A labelled drawing of the Projects page with + New Project, the knowledge + button and Set project instructions numbered](claude-from-zero/new-project.webp "Make a project in five clicks. A labelled drawing, not a screenshot.")

```prompt Project instructions
In this project I'm [the ongoing work, for example preparing for job interviews in marketing]. The files here are [what you uploaded]. Use them as your main source. Write in plain English, keep answers short, and end each answer with one suggested next step. If something isn't in the files, say so before using general knowledge.
```

### Artifacts

Anything Claude makes that you'd put in front of someone: a design, a deck, a document, a dashboard or a small tool. It opens beside the chat. To share it, click **Share** and choose who can open it under **Who has access**.

Docs, Slides and Design templates are on every plan. Slides can be exported to PowerPoint or PDF.

### Memory and Instructions

**Memory** is what Claude remembers about you between chats. It's on by default for Free, Pro, Max and Ultra, and each project keeps its own.

**Instructions** are standing rules for every conversation: your job, your tone, how long answers should be.

Both live in **Settings → Your Claude**. Select **Manage** next to **Your memory** to see and edit what Claude remembers.

![A labelled drawing of Settings, Your Claude, with Instructions, Your memory with its Manage button, and Start import numbered](claude-from-zero/settings.webp "Settings → Your Claude: your Instructions and your memory. A labelled drawing, not a screenshot.")

Paste this into **Instructions** and make it yours:

```prompt Your Instructions
I work as [your job] in [your country]. Write in plain English, with [British or American] spelling. Keep answers short unless I ask for more. Use numbered steps for instructions. If my request is unclear, ask me one question before you start. If you're not sure about a fact, say so.
```

You can also add to memory any time:

```prompt Remember this
Remember this for future chats: [the fact, for example "I manage a team of six in customer support, and my reports go to the operations director every Friday"].
```

### Connectors

Connectors link Claude to the apps you use, so it can find information and take actions there, with your own permissions. Find them with **+ → Connectors → Manage connectors**, or in **Customize → Connectors**. Start with one app you use daily.

### Skills

Folders of instructions and resources that teach Claude a specialised job, on every plan. You already use some: built-in skills make the Excel, Word, PowerPoint and PDF files. Later, you can add your own in **Customize → Skills**.

> [!NOTE]
> Whether your chats help train Claude is a switch in **Settings → Privacy**. On Free, Pro, Max and Ultra you can opt out. On Team and Enterprise, content isn't used for training by default. Incognito chats are never used.

::card[guide:claude-skills-from-zero]

---

## 9. Features to leave until later

You'll see these in announcements and on social media. None of them matters in your first week.

![A glossary of sixteen terms you can ignore for now, from API and benchmarks to tokens and usage credits, each with a one-line meaning](claude-from-zero/skip-words.webp "Sixteen words you'll hear and can safely park. Come back to this list in a month.")

The two you'll meet first: **tokens** (Claude tells you if a message is too long) and **effort** (leave it on the default).

::button[Claude in plain English](issue:plain-english-claude)

---

## 10. Ten small experiments for week one

One a day, or two if you have time.

![A circle of ten first-week tries around a big number ten, from learning something at your level to talking instead of typing](claude-from-zero/first-week.webp "Your first week, in ten tries. Tick them off in any order.")

Four of them already have prompts in section 6: learn something, fix a message, question a PDF, get a spreadsheet. Two are settings from section 8: your Instructions and a project. Here are the other four.

```prompt Plan something real
Help me plan [a trip, an event, a move or a project]. Ask me up to five questions first, one at a time. Then give me a simple plan as a table with dates, tasks and who does each one, followed by a checklist I can tick off.
```

```prompt Build an artifact
Make me a simple checklist for [my weekly shopping, my packing list or my Monday routine] that I can tick off as I go. Group the items into sections and add a button that clears all the ticks.
```

It opens beside the chat. Click into it and use it.

```prompt Search with sources
What changed this month in [your field or your city]? Search the web, give me the three most important updates with dates, and link the source for each one. Then open two of the links yourself.
```

```prompt Talk it through (voice mode)
I'm stuck on a decision: [describe it]. Ask me questions one at a time to help me think it through, then sum up the options and what you'd weigh most.
```

Voice mode is in beta on every plan. It works best on a phone, and it counts toward your usage limit.

---

## 11. Run one ten-minute trial

Now put it together on one real task from your week. Three steps.

### Step 1: Give it a real brief

Open a new chat. Paste the prompt below and fill in the brackets.

![A labelled drawing of a new chat with the trial prompt typed in the message box, and the steps to fill in the brackets and send](claude-from-zero/drive-1.webp "Step 1: one real task, with the outcome, the reader and the format. A labelled drawing, not a screenshot.")

```prompt Trial
I need [the thing you want, for example a one-page plan for next month]. It's for [who will read it]. Here's the background: [two or three lines]. Give it to me as [a table, a short document or bullet points]. Before you start, ask me up to three questions.
```

### Step 2: Answer, then sharpen

Answer Claude's questions in one message. Read the draft. Then fix one thing: "Shorter", "More formal", "Add a timeline", "Put the costs in a table".

![A labelled drawing of a chat where you answer Claude's three questions in one message, then send a one-line follow-up to sharpen the draft](claude-from-zero/drive-2.webp "Step 2: answer its questions in one go, then ask for one change at a time. A labelled drawing, not a screenshot.")

### Step 3: Get the file, and check it

Ask for a file: "Now make this a Word document." Download it. Then ask what to verify:

```prompt What should I check?
What in this should I double-check before I use it? List each fact, number or assumption, and tell me where I can confirm it.
```

Last, open **Settings → Usage** and look at **Current session**. Now you know what a task like this costs you.

![A labelled drawing of asking Claude for a Word document, asking what to double-check, and opening Settings, Usage, with each step numbered](claude-from-zero/drive-3.webp "Step 3: turn the draft into a file, ask what to verify, and see what it used. A labelled drawing, not a screenshot.")

**How you know it worked:** you have a file you'd actually use, and a short list of things to verify before you do.

::button[Write one good prompt](issue:one-good-prompt)

---

## 12. Watch eight professionals at work

Once the basics feel easy, borrow from people who use Claude for real work.

Eight roles, one job each, each built on a feature Anthropic documents. Swap in your own details.

### The team lead: follow-ups in Outlook

**Feature:** Claude for Microsoft 365 (paid plans). Install **Claude for Microsoft 365** from Microsoft AppSource with **Get it now**, then sign in with your Claude account. In Outlook, its drafts and invites open in the normal compose window and wait for you to click send.

![A role card for a team lead: the job, the Outlook add-in, the prompt and what comes back, with the draft waiting for you to send](claude-from-zero/pro-team-lead.webp "The team lead: Claude drafts, Outlook holds it, you press send.")

```prompt Meeting follow-up
Here are my notes from today's meeting. Draft a follow-up email to everyone who attended: the decisions, each action with an owner and a date, and the open questions. Keep it under 200 words. Then draft an invite for a 30-minute follow-up meeting next week.

[paste your notes]
```

### The analyst: answers from a workbook

**Feature:** Claude for Excel (paid plans). It answers questions with cell-level citations, changes assumptions while keeping formulas intact, and debugs errors.

![A role card for an analyst: the job, Claude for Excel, the prompt and what comes back, with cell references for every answer](claude-from-zero/pro-analyst.webp "The analyst: every answer points to the cells it came from.")

```prompt Understand this workbook
Explain how this workbook works: what each sheet does and where the key numbers come from, with cell references. Then show me what happens to the final total if [the assumption] changes from [x] to [y]. Keep the existing formulas intact and tell me which cells you changed.
```

### The researcher: a sourced briefing

**Feature:** Research (paid plans): many searches that build on each other. Turn it on with **+ → Research**, or type **/deep-research** in the new Claude experience.

![A role card for a researcher: the job, the Research feature, the prompt and what comes back, a report with citations](claude-from-zero/pro-researcher.webp "The researcher: one question, many searches, every claim with a source.")

```prompt Research briefing
Research [the question, for example how small retailers in my country use online marketplaces]. I need a two-page briefing for [who]. Cover the main trends, three examples with sources, the key numbers with dates, and where sources disagree. Finish with three questions I should ask next.
```

### The presenter: slides in your own template

**Feature:** Claude for PowerPoint (paid plans). It reads your slide master, fonts and colours, builds slides in your template, and turns bullets into editable diagrams and charts. On Free, ask in chat: "Convert this document into a PowerPoint presentation."

![A role card for a presenter: the job, Claude for PowerPoint, the prompt and what comes back, an editable deck in the company template](claude-from-zero/pro-presenter.webp "The presenter: a first draft of the deck in your own template, with charts you can still edit.")

```prompt Deck from a document
Turn the attached document into an 8-slide deck for [audience]. One idea per slide, a short headline on each, no more than four bullets. Turn the numbers on page [x] into a chart. Use the layouts in this template. End with a slide of next steps.
```

### The operations lead: a weekly report on a schedule

**Feature:** scheduled tasks (paid plans). Describe the task and how often, review Claude's proposal and click **Schedule**. Tasks run in the cloud, even when your computer is asleep.

![A role card for an operations lead: the job, scheduled tasks, the prompt and what comes back every week, with the safety rules](claude-from-zero/pro-operations.webp "The operations lead: one prompt, run every Monday. Start simple and review every run.")

```prompt Weekly digest
Every Monday morning, search the web for news about [your industry and region] from the past week. Give me the five items that matter most for [your role], each with one line on why it matters and a link to the source. Keep it to one screen.
```

Anthropic's advice: start simple, avoid sensitive data, review each run, and pause tasks you don't use.

### The reviewer: long documents, with quotes

**Feature:** file uploads (every plan), plus the quotes-first habit from section 7.

![A role card for a document reviewer: the job, file uploads, the prompt and what comes back, a table of answers with the exact quote and page for each](claude-from-zero/pro-reviewer.webp "The reviewer: every answer comes with the exact wording and the page.")

```prompt Review against my questions
Read the attached [policy, contract or tender]. Answer each of my questions in a table with four columns: the question, the answer, the exact quote that supports it, and the page. If the document doesn't answer a question, write "Not covered". Don't use outside knowledge.

My questions:
1. [question]
2. [question]
3. [question]
```

Treat the table as a reading aid. Get professional advice where the stakes are high.

### The trainer: a course project and a quiz

**Feature:** a project for the course materials, and an artifact for the quiz, which you can share.

![A role card for a trainer: the job, a project plus an artifact, the prompt and what comes back, a shareable quiz](claude-from-zero/pro-trainer.webp "The trainer: the course lives in a project, the quiz is an artifact you can share.")

```prompt Make a quiz
Using the session notes in this project, make a 10-question multiple-choice quiz as an interactive artifact. One correct answer per question, a one-line explanation after each answer, and a score at the end. Pitch it at [level].
```

### The small-business owner: a whole task handed over

**Feature:** Cowork, or any conversation in the new Claude experience (paid plans).

![A role card for a small-business owner: the job, Cowork, the prompt and what comes back, a spreadsheet with formulas and a short slide deck](claude-from-zero/pro-founder.webp "The small-business owner: describe the outcome, step away, come back to files.")

```prompt Compare and recommend
I run [your business] in [your city]. Compare five [suppliers, tools or options] for [the need], with a budget of [amount] a month. Build an Excel file with one row each: price, key features, contract terms, one weakness and a source link, plus a formula that ranks them. Then make a three-slide PowerPoint summary with your recommendation. Ask me anything you need first.
```

Check prices at the source before you sign anything.

> [!TIP]
> Pick the one role closest to yours and try it this week. One workflow you use every week beats eight you tried once.

---

## 13. Choose your first step

Here's the whole guide as a path, one stop at a time.

![A winding roadmap from day one to week four: install and sign in, set your Instructions, brief it well, use your files, make a project, hand over a task, and adopt one professional workflow](claude-from-zero/roadmap.webp "Your first month with Claude, one stop at a time. The first four stops work on Free.")

## Before you finish

- [ ] Claude is on my computer and my phone
- [ ] I know which plan I'm on, and when I'd move up
- [ ] My Instructions and my privacy choice are set
- [ ] I know where Settings → Usage is
- [ ] I've used the brief template at least once
- [ ] I've asked Claude to check its own answer
- [ ] I've tried at least three of the ten
- [ ] I've done the ten-minute trial
- [ ] I've picked one professional workflow to try

## Questions

### Does Claude cost anything?

The Free plan costs $0, with limited usage every five hours and every week.

### Why did Claude stop answering?

You've most likely reached your usage limit. Claude tells you when you do. **Settings → Usage** shows when it resets.

### Does Claude work in my language?

It's trained extensively in English and performs well in many other common languages.

### Which model should I use?

On Free, Sonnet 5.5 for most things and Haiku 5.5 for quick jobs. On paid plans, Anthropic says start with Opus 5.5 if you're not sure.

## What to do after week one

**Hand over whole tasks.** Cowork does the steps for you, from start to finished file.

::card[guide:cowork-from-zero]

**Build the daily habits** that make Claude more useful every week.

::card[issue:claude-habits]

**Take Anthropic's own course.** Claude 101 in [Claude Academy](https://academy.claude.com/courses) has 13 lessons and 1 quiz, takes about 2.5 hours and ends with a completion badge.

Open Claude today and give it one real job. That's day one done.

Mojeed
