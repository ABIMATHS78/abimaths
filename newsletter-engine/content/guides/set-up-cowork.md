+++
title = "Set up Cowork once."
subtitle = "Tried Cowork a few times? Set it up properly in one sitting, step by step:"
description = "One working folder, three context files, global instructions, templates, approval modes, whole-job briefs, six usage habits and your first 20 minutes."
updated = 2026-11-28
status = "published"
cover = "set-up-cowork/cheat-sheet.webp"
cover_alt = "A one-page Cowork set-up cheat sheet: what you need, the working folder, three context files, connecting it, global instructions, approval modes, whole-job briefs, usage habits and the first 20 minutes"
tags = ["claude", "cowork", "productivity"]
archetype = "guide"
level = "intermediate"
topic = "Claude"
brief = "briefs/set-up-cowork.json"
fact_checks = ["Check where global instructions live in your own Claude app: Settings → Cowork → Edit next to Global instructions (classic layout), or Instructions for Claude in Settings → Your Claude in the new Claude experience (Settings → Account in the older Settings layout).", "In Claude Desktop, check the Projects + menu: Start from scratch, Import from project and Use an existing folder, then Create.", "Check the approval mode selector in the chat box: Manually approve, Automatically approve and Skip all approvals (Auto and Manual in the new experience), and which one is the default on your plan.", "Check Settings → Usage: Current session, Weekly limits with the reset time, the Usage credits toggle, Adjust limit and Buy usage credits.", "Check voice input: dictation in a Cowork task, and the sound wave symbol for voice mode in the new Claude experience.", "In the new Claude experience, check that Settings shows 'Storage folder' and 'Trusted folders'."]
+++
You've tried Cowork a few times.

It did something useful. Then it asked a string of questions, saved files somewhere odd, and used more of your allowance than you expected.

Most of that comes from one thing: every task started knowing nothing about you.

This guide fixes that, once.

I teach people to use AI at work, and my mission is to make data and AI easy for everyone. This is the set-up I'd want every professional to do before their tenth Cowork task.

It takes one sitting. Short of time? Step 9 is a minute-by-minute plan for your first 20 minutes.

**By the end of this guide you'll have:**

1. one working folder, laid out so you and Claude always know where things go
2. three short context files: who you are, who reads your work, how you write
3. the folder connected, with ground rules Claude reads first
4. global instructions, written from a template, that every task starts with
5. a Templates folder, so Claude matches formats you already like
6. an approval mode that fits each kind of work
7. a way of briefing whole jobs, so you review instead of steering
8. six habits that make your usage allowance last
9. one real task done

**You need:**

- a paid Claude plan: Pro, Max, Ultra, Team or Enterprise (Cowork isn't on Free)
- Claude Desktop on your computer, so Claude can work with your own files
- one small, real job you'd happily hand to a capable colleague

New to Cowork? Start with [Cowork, from zero](guide:cowork-from-zero) and come back here.

![A one-page Cowork set-up cheat sheet: what you need, the working folder, three context files, connecting it, global instructions, approval modes, whole-job briefs, usage habits and the first 20 minutes](set-up-cowork/cheat-sheet.webp "The whole set-up on one page. Keep it open while you work through the ten steps.")

::subscribe

---

## 1. See why a set-up pays off

Cowork is built for a different kind of work than chat.

Anthropic describes it as bringing Claude Code's agentic abilities to knowledge work beyond coding. It takes on multi-step tasks instead of answering one prompt at a time.

The intended way to use it: describe an outcome, step away, and come back to finished work.

When a task starts, Claude makes a plan, breaks the work into smaller tasks, runs them in an isolated environment on Anthropic's servers, may run some in parallel, and hands you the results.

It can make Excel spreadsheets with working formulas, PowerPoint presentations and formatted documents.

Here's the catch. A task can only use what you've given it.

So the quality of what you get depends on four things you control:

- **what Claude can reach:** one tidy folder
- **what Claude knows about you:** context files and instructions
- **what good looks like:** templates of work you liked
- **how you hand work over:** whole jobs with a clear finish line

![Two columns side by side: how Cowork behaves without a set-up, and the same habits after a one-sitting set-up](set-up-cowork/before-after.webp "Same Cowork, two very different weeks. Every row on the right is one step of this guide.")

Set those up once and every task after that starts from a better place. You also spend less of your allowance explaining yourself again.

### Check which version you have

Some menu names depend on it. Open Claude and look at the message box.

- **You see 'Chat' and 'Cowork' options.** That's the classic layout. Select **Cowork**, describe the task, review Claude's approach and let it run.
- **You don't see them.** You have the new Claude experience. Chat and Cowork are one conversation, and Claude decides which tools to use. Just describe the task.

The new experience is rolling out to Pro, Max and Ultra, and to Team and Enterprise where 'Chat and Cowork unified' is on. Once your account has it, it can't switch back.

Where a menu differs, this guide gives you both paths.

### Install Claude Desktop

Cowork runs in the cloud. Your sessions and files are saved to your Claude account, so work carries on if you close your laptop.

But Claude can only reach files on your computer through Claude Desktop. Download it from [claude.com/download](https://claude.com/download).

It needs macOS 13 or later, Windows 10 or later, or Ubuntu 22.04 or Debian 12 or later (Linux is in beta).

Two rules to remember:

- a task that uses your local files needs the desktop app open on that computer
- a task that doesn't touch your files keeps running when the app is closed or the computer is asleep

**Check it worked:** open Claude Desktop, sign in and start a new conversation. If you see the 'Cowork' option, or an empty message box ready for a task, you're set.

Keep this prompt for later. Run it once you've finished step 3, and any time a task surprises you:

```prompt What can you see right now?
Before we start, tell me in a short list:
1. which folders you can read and write in this conversation
2. which connected apps you can use
3. what you already know about me and how I like to work
Don't open or change any files. Just report.
```

**Use it** at the start of a session when you're not sure what Claude can reach. **You'll get** a short inventory. If a folder or app is missing, fix that before you hand over work.

---

## 2. Make one working folder

This is the step most people skip. It saves the most time.

Anthropic's own advice: create a dedicated working folder for Claude instead of giving it broad access to your computer, and keep backups of important files.

Claude can only read and write in folders you connect. One tidy folder means Claude always knows where to look, and you always know where to find the result.

### The layout

Make one folder called **Claude work**. Inside it go four folders and one file.

![A folder tree of Claude work: Read me first.md at the top, then Context with three files, Inbox, Outputs and Templates, each with a note on what it holds, and who writes where](set-up-cowork/folder-layout.webp "One folder, four rooms. Claude reads from Context and Inbox, writes to Outputs, and copies the shape of Templates.")

| Inside Claude work | What goes in it | Who writes there |
|---|---|---|
| **Read me first.md** | The ground rules for this folder, in about ten lines | Claude drafts, you approve |
| **Context** | Three short files: about you, your readers, your style | You, with Claude's help |
| **Inbox** | Source files for this week's tasks: notes, data, drafts | You |
| **Outputs** | Everything Claude makes | Claude |
| **Templates** | Finished work you liked, kept as patterns to copy | You |

Why this layout works:

- **Inbox and Outputs are separate,** so Claude never overwrites the files you gave it.
- **Context is small and stable,** so Claude reads the same few facts about you every time.
- **Templates hold your best examples,** so you can say "match the weekly update template" instead of explaining a format again.
- **Read me first.md sits at the top,** so the rules are the first thing Claude finds.

Move finished work you want to keep out of **Outputs**, for example into an **Archive** folder outside **Claude work**. Then Claude isn't reading last month's drafts by mistake.

### Let Claude build it

Make an empty folder called **Claude work**, connect it (step 3 shows you how), then paste this.

```prompt Build my working folder
In the folder I've connected, create four folders: Context, Inbox, Outputs and Templates.

Then create a file called "Read me first.md" in the main folder, with these rules in plain words:

- Context: short files about me. Read them before any writing task. Never change them unless I ask.
- Inbox: the source files I want you to work on.
- Outputs: save everything you make here, named YYYY-MM-DD topic v1.
- Templates: finished work I liked. Match one when I name it. Never change them.
- Never delete anything. If something should go, tell me and I'll move it.

Show me the folder list and the text of Read me first.md when you're done.
```

**Use it** once, on an empty folder. **You'll get** four folders, the rules file and a short summary. Deleting is guarded anyway: Claude must ask with an **Allow** prompt before it permanently deletes any file.

### The three context files

Context files are short notes Claude reads before it writes anything for you.

Think of them as the briefing you'd give a new colleague on day one. Anthropic's prompting guide uses the same picture: treat Claude like a brilliant but new employee who doesn't know your norms or how you work.

You need three. Each has a card below and a prompt that has Claude interview you, so you never start from a blank page.

### Context file 1: About me

What goes in: your role and type of organisation, the main kinds of work you do, your priorities this quarter, the people you work with most, and the terms and acronyms you use.

![A card for the About me context file: what goes in it, example lines with placeholders, a prompt that has Claude interview you, and how to keep it useful](set-up-cowork/context-about-me.webp "Context file 1. Five headings, one screen. Claude reads it before it writes anything for you.")

```prompt Interview me for About me.md
I want to write a short file called "About me.md" for my Context folder, so you know who I am before any task.

Interview me. Ask one question at a time, no more than eight in total, about:
my role and organisation, what my work is mostly about, my main projects this quarter, the people I work with most, and the terms or acronyms I use.

Then write the file as short bullet points under five headings: Role, Main work, Current priorities, People, Terms.
Keep it to one screen. Save it in Context and show it to me.
```

**Use it** today. **You'll get** one question at a time, then a page of bullets like "Role: [job title] in a [type of] team, reporting to [who]". Correct anything wrong before you move on.

### Context file 2: My readers

What goes in: the three to five people or groups who read your work most, what each one needs from you, what they usually push back on, and the format and length they prefer.

![A card for the My readers context file: what goes in it, an example reader entry with Always include and Avoid lines, the interview prompt and a privacy tip](set-up-cowork/context-readers.webp "Context file 2. Describe each reader's role and needs, so Claude writes for the person who will actually read it.")

```prompt Interview me for My readers.md
Help me write "My readers.md" for my Context folder.

Ask me, one at a time, about the three to five people or groups who read my work most: their role, what they need from me, what they usually push back on, and the format and length they prefer.

Write one short section per reader, each ending with a line called "Always include" and a line called "Avoid".
Describe roles, not private details about people. Save it in Context and show it to me.
```

**You'll get** a section per reader, such as "Finance director: reads on her phone between meetings. Always include: the decision you need. Avoid: background before the answer."

### Context file 3: My style

What goes in: your tone, spelling, sentence length, words you avoid, how you structure documents, your default file types, and one short sample of your own writing.

![A card for the My style context file: what goes in it, a vague rule rewritten as a rule with a reason, the interview prompt and a tip on samples](set-up-cowork/context-style.webp "Context file 3. Rules said positively, each with a reason, plus one sample paragraph in your voice.")

```prompt Interview me for My style.md
Help me write "My style.md" for my Context folder.

First, ask me to paste two or three short pieces I've written that I'm happy with.
Then describe my style back to me: tone, sentence length, structure, and the words I use or avoid.
Ask me what I'd change. Then write the file as a short list of rules, each with a reason, plus one sample paragraph in my voice.
Save it in Context and show it to me.
```

**You'll get** a mirror first ("you write short sentences, lead with the answer, and avoid jargon"), then the rules. The mirror is the useful part: it shows you what Claude will copy.

### Two rules for all three files

- **Keep them short.** One screen each is plenty. Anthropic's usage advice includes keeping project instructions short and removing files you no longer use. Every file Claude reads uses some of your allowance.
- **Write facts and reasons, not wishes.** "My manager reads on her phone, so the answer goes first" beats "be concise". Anthropic's prompting guide says to give the reason behind an instruction, because Claude can apply it to cases you didn't list.

**Check it worked:** open **Context** on your computer. You should see three files, each short enough to read in a minute.

> [!WARNING]
> If you rename or move a connected folder, Claude may lose it. On macOS 14 or later with Claude Desktop 6.6395.0 or later, Claude can find it and ask you to **Keep connected**. On Windows, Linux or older macOS, you connect it again. Name the folder once and leave it where it is.

---

## 3. Connect the folder and set its ground rules

Connecting is what lets Claude read and write in **Claude work**.

What that means in practice:

- Your folders stay on your computer. When a cloud task needs a file, Claude fetches a copy of just that file, and those copies are deleted when you delete the session.
- Claude reaches the folder through Claude Desktop, and only while the app is open.
- Claude reads single files up to 50 MB.
- Claude must ask with an **Allow** prompt before it permanently deletes any file.

### The steady route: make it a project

If you'll use the folder for most of your work, turn it into a Cowork project.

Every session you start inside the project opens with the folder connected and the project's instructions applied. Files Claude creates land in the project's folder.

1. In Claude Desktop, find **Projects** in the left navigation and click **+**.
2. Choose **Use an existing folder** and pick **Claude work**.
3. Name the project, paste the text of **Read me first.md** as its instructions, and click **Create**.

![A labelled drawing of Claude Desktop: Projects in the left navigation, the + menu open with Start from scratch, Import from project and Use an existing folder, and the parts of a project, with three numbered steps](set-up-cowork/connect-folder.webp "Make your folder a project once, and every session in it starts with the folder connected and the instructions applied. A labelled drawing, not a screenshot.")

Each project also has its own scheduled tasks and its own memory. What Claude learns in one project doesn't carry over to other projects. Projects in Cowork need the latest Claude Desktop.

For more, read [Cowork projects](guide:cowork-projects).

### The quick route: one task

For a one-off job, select **Claude work** as the local folder when you start the task in Claude Desktop.

In the new Claude experience, **Settings** lists the folders you've given Cowork access to under **Trusted folders**, and shows where Cowork saved its own work as **Storage folder**. Check there if you're not sure what Claude can reach.

### Folder instructions, CLAUDE.md and the 10 November change

When you select a local folder on desktop, Cowork can use folder instructions: extra context for that folder only.

Claude can update them itself during a session. That's handy: it can note how you name files or which template you prefer.

It used to have a side effect. Since Claude Desktop v5.1231.0 (10 November 2026), edits Claude makes to a session's CLAUDE.md no longer update your global instructions. Anthropic's changelog says to use Settings to change those.

In plain words: Claude can keep notes about a folder, but only you change the rules that apply everywhere.

Here's how the layers fit together:

| Layer | Where it lives | What goes in | Who changes it |
|---|---|---|---|
| **Global instructions** | Settings (step 4) | How you work, everywhere | You |
| **Project instructions** | The project's page | Rules for this area of work | You |
| **Folder instructions** | The folder you select on desktop | Notes for this folder | Claude can update them |
| **Read me first.md and Context** | Files in your folder | Rules and facts about you | You, with Claude drafting |
| **The brief** | Your message | This one job | You |

Before you move on, let Claude check the rules for gaps:

```prompt Check my folder rules
Read "Read me first.md" and the three files in Context.
Tell me:
1. any rule that's unclear or contradicts another
2. anything a new colleague would still need to know before working for me
3. anything that's too long and could be cut
Suggest the edits, but don't change the files until I say yes.
```

**Check it worked:** start a new conversation in the project, or with the folder selected, and run "What can you see right now?" from step 1. **Claude work** should be on the list.

---

## 4. Give Claude standing orders

Global instructions are standing instructions that apply to every Cowork session.

Anthropic suggests three things for them: your preferred tone, your output format, and background on your role.

They're the layer that follows you everywhere, whichever folder or project you're in.

### Where to put them: classic layout

If your message box shows 'Chat' and 'Cowork':

1. Open **Settings → Cowork**.
2. Click **Edit** next to **Global instructions**.
3. Type or paste your instructions and click **Save**.

![A labelled drawing of Settings in the classic layout: the Cowork section, the Edit button next to Global instructions, and sample instructions being typed, with three numbered steps](set-up-cowork/instructions.webp "Classic layout: Settings, Cowork, Edit, then Save. A labelled drawing, not a screenshot.")

### Where to put them: new Claude experience

In the new experience, Cowork's global instructions became part of **Instructions for Claude**, in **Settings → Your Claude** (**Settings → Account** in the older Settings layout). They apply to every conversation, chat and task alike.

1. Click your initials in the lower left corner and select **Settings**.
2. Open **Your Claude**.
3. Under **Instructions**, paste your instructions.

![A labelled drawing of Settings in the new Claude experience: Your Claude in the sidebar, the Instructions row, and sample instructions, with three numbered steps](set-up-cowork/instructions-new.webp "New Claude experience: your Cowork global instructions now live in Instructions for Claude, and apply to every chat too. A labelled drawing, not a screenshot.")

Because these now apply to quick chats too, word folder rules so they make sense anywhere: "when my working folder is connected, save files in Outputs".

### What good instructions look like

Good instructions are short, specific and say what to do.

![Global instructions split into coloured parts: vague wishes to delete, who you are and who reads your work to keep, and rules with reasons plus safety and reporting rules to add](set-up-cowork/instructions-anatomy.webp "Cut the wishes Claude follows anyway. Keep who you are. Add rules with reasons and a clear way to report back.")

Three things from Anthropic's own guidance:

- **Say what to do, not what to avoid.** "Write in short paragraphs" works better than "don't write long paragraphs".
- **Give the reason.** "Answer first, because my manager reads on her phone" lets Claude handle cases you didn't list.
- **Give permission to say "I don't know".** Anthropic says this can drastically reduce false information.

### The template

Copy this, fill in the brackets, and delete any line that doesn't fit you.

```prompt Global instructions template
About me

- I work as [your role] at [type of organisation]. When my working folder is connected, more context is in Context/About me.md.
- My work is read mostly by [who], who need [what].

How to work

- For a task with more than three steps, show me a short plan first and wait for my go-ahead.
- If something important is unclear, ask me up to three questions at the start, then work without stopping.
- Use only the files and sources I point you to. If you need something else, ask.
- Never send, post, buy or delete anything without asking me first.
- If you don't know or can't check something, say so instead of guessing.

Files

- Read from Inbox and Context. Save everything you make in Outputs.
- Name files: YYYY-MM-DD topic v1.
- When I name a template, match its sections, length and tone.

How to write

- [British or American] spelling. Plain English and short sentences, because [your reason].
- Answer or recommendation first, then the detail.
- Documents: Word. Numbers: Excel with working formulas. Slides: PowerPoint.

When you finish

- Tell me in five lines or fewer: what you made, where you saved it, and what you weren't sure about.
- List any number, name or claim I should check before I share it.
```

Keep it to about a page. No source documents a length limit for Cowork global instructions, and Anthropic warns that longer prompts aren't always better.

### Or let Claude draft them

```prompt Draft my global instructions
I want to write global instructions for Cowork: standing instructions you'll read at the start of every task.

Interview me, one question at a time, no more than eight questions, about: my role, who reads my work, the file types I use, how I like things written, what you should never do without asking, and how you should report back when you finish.

Then write my instructions as a short list under these headings: About me, How to work, Files, How to write, When you finish.
Say each rule positively and add the reason where it helps. Keep the whole thing to one page.
```

**Use it** if the template feels like homework. **You'll get** eight short questions, then a draft in the same five sections as the template.

**Check it worked:** save, start a new task and paste this.

```prompt Test my instructions
Without opening any files: in five bullet points, what do you know about how I like you to work, write and report back?
Then tell me one thing my instructions don't cover that would help you.
```

**You'll get** your own rules back in Claude's words. If one is missing, it wasn't saved or wasn't clear. Fix it now, not after ten tasks.

---

## 5. Save templates of work you like

Explaining a format takes paragraphs. Showing one takes a file.

Anthropic calls examples one of the most reliable ways to steer format, tone and structure. For app users, it advises starting with one example and adding more only if the output still doesn't match.

The same advice carries a warning: Claude pays very close attention to details in examples. So a template should show the shape you want, without the names and numbers of one old job.

### How to make one

1. When Claude makes something you like, keep the file in **Outputs**.
2. Ask Claude to turn it into a template. It swaps names, numbers and dates for [placeholders] and adds a short note on how to use it.
3. Save it in **Templates** with a plain name, like **Weekly update.docx** or **Client proposal.pptx**.
4. Next time, name it in your brief: "Match the Weekly update template."

![A card on templates: why one example beats a paragraph, four steps to make a template, a prompt to copy, plain names for common templates, and a warning about copied details](set-up-cowork/templates.webp "Show, don't describe. One good template replaces a paragraph of formatting rules.")

```prompt Turn this into a template
Take [file name] in Outputs and make a template from it in Templates, called [Template name].
- Keep the structure, headings, length, tone and formatting.
- Replace every name, number, date and client detail with a [placeholder] that says what goes there.
- Add a short note at the top: what this template is for, who reads it, and two rules for filling it in.
Don't change the original file.
```

**Use it** the moment you're happy with a piece of work. **You'll get** a clean copy in **Templates**, with a note at the top such as "For: the Monday team update. Reader: the whole team. Rule: three wins, three risks, one ask."

```prompt Use a template
Make [what] for [who], using the files in Inbox called [names].
Match the template Templates/[Template name]: same sections, same length, same tone.
If something in my files doesn't fit a section, put it under "Open questions" at the end instead of forcing it in.
Save it in Outputs.
```

**You'll get** work that already looks like yours, so your review is about the content, not the layout.

Start with one template. The weekly or monthly document you dread most is a good first choice.

---

## 6. Choose an approval mode

The approval mode decides how often Claude stops to ask you before it acts, for example before it uses one of your connectors.

You can change it at any time from the mode selector in the chat box.

![A labelled drawing of the chat box with the mode selector open: Manually approve, Automatically approve and Skip all approvals, with three numbered steps](set-up-cowork/mode-selector.webp "The mode selector sits in the chat box. Change it as the work changes. A labelled drawing, not a screenshot.")

There are three modes:

| Mode | What happens | Older name |
|---|---|---|
| **Manually approve** (Manual) | Claude pauses on actions and you choose **Allow** or **Deny** | Ask before acting |
| **Automatically approve** (Auto) | Claude keeps working. An automated check reviews each action for safety, such as data exfiltration or prompt injection, and blocks unsafe ones. If Claude keeps getting blocked, it goes back to asking you | None |
| **Skip all approvals** (Skip) | Claude doesn't pause, and nothing checks its actions | Act without asking |

In the new Claude experience, the permission setting in the message box offers **Auto** and **Manual**. Auto is rolling out as the default on Pro, Max, Ultra and Team; Manual is the default on Enterprise.

If you last chose Manual in a browser or app, new conversations there keep starting in Manual.

On Team and Enterprise, your admin decides whether Auto is available. If it's turned off, it won't appear in your selector.

### Which mode, when

Anthropic's advice is to switch to Manual when a task touches sensitive files, accounts or sites, when you're using a new tool, plugin or site, or when mistakes would be hard to undo.

![A guide to the three approval modes: what each does, the jobs each one suits, the default by plan, how to make one connector tool always ask, and a prompt to copy](set-up-cowork/which-mode.webp "Match the mode to the job. Auto for everyday work in your folder, Manual for anything hard to undo.")

A simple rule that follows from it:

- **Auto** for everyday work inside **Claude work**: summaries, drafts, spreadsheets, tidying files.
- **Manual** for anything involving money, messages sent as you, or important files outside your working folder.
- **Skip** only when you completely trust every action, connector, file and app involved. Most people never need it.

### Make one tool always ask

Some tools deserve a pause in every mode, like one that sends email.

Go to **Customize → Connectors**, select the connector, open the **Permissions** tab and set that tool to **Always ask**. Claude is built to ask before each use of it, whatever the mode.

'Always ask' is rolling out gradually. On Team and Enterprise, your organisation's settings come first.

Want a belt-and-braces line in your brief too?

```prompt Pause before anything final
For this task: before you send, post, buy, move or delete anything, stop.
Show me exactly what you're about to do and why, and wait for my "yes".
Everything else inside my working folder, just do.
```

**Use it** in Auto mode for a task that ends with something going out. **You'll get** uninterrupted work, then one clear pause at the moment that matters.

> [!WARNING]
> Anthropic says the Auto check can miss risky actions. For work with real consequences, such as money, messages sent as you or important files, stay close or switch to Manual. You remain responsible for everything Claude does on your behalf.

Read Anthropic's page on [using Cowork safely](https://support.claude.com/en/articles/13364135-use-claude-cowork-safely) for the full picture.

::card[issue:claude-uses-your-computer]

---

## 7. Stop being the bottleneck

Once the set-up is done, the slowest part of most Cowork tasks is the person.

Chat taught everyone a habit: one small prompt, read the answer, one more small prompt.

In Cowork that habit wastes what it does best. Claude can plan, split the work, run parts in parallel and keep going while you do something else. It can't do any of that if you hand it one step at a time.

![The same job asked three ways: one step at a time, one task, and a whole job with a finish line, a template, a plan check and a save location highlighted](set-up-cowork/delegation-ladder.webp "Climb the ladder. Each rung hands Claude more of the job and leaves you more of the review.")

Four moves fix it.

### Describe the outcome, not the steps

Say what finished looks like: the file, its length, who reads it and what they need to decide. Leave the route to Claude.

Anthropic's test for any prompt: show it to a colleague with little context on the task. If they'd be confused, Claude will be too.

![A whole-job brief split into coloured parts: the vague ask to delete, the outcome to keep, and the reader, inputs, finish line and reporting rules to add](set-up-cowork/brief-anatomy.webp "A brief Claude can run with answers five questions: what, for whom, from what, what finished looks like, and how to report back.")

```prompt Brief a whole job
Task: [the outcome in one sentence, for example "a two-page summary of last month's customer feedback for my manager"].

Use: the files in Inbox called [names]. Background is in Context.

Finished looks like:
- [file type, for example a Word document]
- [shape, for example two pages, a table of the top five issues, three recommendations]
- [who it's for and what they need to decide]
Match the template Templates/[name] if it fits.

Rules:
- [anything to avoid, for example no customer names]
- If you hit a small decision, choose, note it and carry on.

Save it in Outputs. When you're done, tell me in five lines what you made, the choices you made and what I should check.
```

**Use it** for any job you'd normally do in five or more chat messages. **You'll get** a plan, then one finished file and a five-line report.

### Let Claude ask first

You don't have to think of everything. Give Claude the job of finding the gaps.

```prompt Ask before you start
Here's what I need: [one or two sentences].
Before you start, ask me the questions you need answered to do this well, all in one message, numbered, no more than five.
After my answers, show me your plan, then run it without stopping.
```

**You'll get** a short numbered list. Answer it in one reply. One extra turn now saves several later.

### Review the plan, not every step

When a task starts, Claude analyses the request and makes a plan. Read it. That's the cheapest moment to change direction.

```prompt Plan first, then go
Before you do anything, show me your plan as numbered steps: what you'll read, what you'll make and where you'll save it.
Flag any step where you'd be guessing.
Wait for my "go". After that, don't stop to check in unless you hit something risky or truly unclear.
```

While it runs, progress indicators show what Claude is doing. You can jump in to correct course, and you can open the same session on another device to check progress or answer Claude's questions.

Anthropic's safety advice is to watch for patterns rather than every command: files or sites you didn't mention, or a task quietly growing bigger than you asked. If something feels off, stop the task.

### Hand over the whole job, then review it

When the result lands, review it like a manager would.

```prompt Review before I use it
Before I share this, check your own work.
1. List every number, name, date and claim in the document, and the file it came from.
2. Mark anything you inferred rather than read.
3. Tell me the three places a careful reader is most likely to push back.
Don't change the document yet. Just give me the list.
```

**You'll get** a short audit trail, with the inferred items marked. Check those first.

Claude can still get facts wrong. Anthropic's advice is to check high-stakes work and the sources it cites. That check is your job, and a far better use of your time than steering.

### Brief faster with your voice

Typing a full brief is the part people skip. Saying it out loud is quicker.

- **In Cowork (classic layout):** dictation is available; voice mode isn't. Dictation turns your speech into text, so you type the brief by talking.
- **In the new Claude experience:** voice mode works in any conversation, including one where Claude is working on a task. On the web or in Claude Desktop, click the sound wave symbol in the lower right corner of the chat window. Click **Stop** when you're done.

Voice mode is in beta, counts toward your usage limits and works best in a quiet room. You can switch between voice and typing in the same conversation. On Linux, dictation isn't available in the current Claude Desktop release.

Spoken briefs ramble. Tidy them before Claude starts:

```prompt Tidy my spoken brief
I've just talked through a task, so it's messy. Don't start yet.
Turn what I said into a clear brief with: the outcome, the files to use, the rules, what finished looks like, and where to save it.
List anything I didn't say that you need, as numbered questions.
When I've answered, start.
```

For more on writing briefs, read [Brief Claude like a colleague](guide:brief-claude-like-a-colleague).

::card[guide:brief-claude-like-a-colleague]

---

## 8. Make your usage last

Cowork uses more of your allowance than a quick chat.

Every step of a multi-step task uses tokens: running code, creating files, using connected apps and the browser. Anthropic's help page says multi-step tasks use more of your usage than a quick question, and that Claude may coordinate several sub-agents at once. No official page says how much more.

How the allowance works:

- Chat, Claude Code, Claude Desktop and Cowork all draw on **the same** usage limit.
- Paid plans have a **five-hour session limit** and **weekly limits**.
- The five-hour window starts with your first message and resets in full five hours later.
- Weekly limits reset at a fixed day and time set for your account.

Anthropic's Cowork help page lists four habits. Two more come from its other usage pages. Here are all six.

### Tip 1: Group related work into one task

One task that reads three files and makes one report beats three tasks that each read a file. Anthropic's wider usage advice says the same: batch similar requests and give background up front.

![A tip card: group related work into one task, why it saves usage, a before and after example and a prompt to copy](set-up-cowork/tip-group.webp "Usage tip 1. One task that reads three files beats three tasks that each read one.")

```prompt One task, not three
In one task, please:
1. read [file A], [file B] and [file C] in Inbox
2. make [the one deliverable] that combines them
3. add a short list of where the three files disagree
Save it in Outputs.
```

### Tip 2: Start a new conversation for unrelated work

Anthropic's reason: so Claude isn't carrying extra context. Long conversations cost more, and when one nears its limit, Claude summarises the earlier part automatically, which uses more of your allowance too.

![A tip card: start a new conversation for unrelated work, why long conversations cost more, signs it's time, and a hand-off prompt](set-up-cowork/tip-new-chat.webp "Usage tip 2. New topic, new conversation. Carry over a ten-line summary, not the whole history.")

Before you switch, take what you need with you:

```prompt Hand me a fresh start
Summarise this conversation in ten lines or fewer so I can start a fresh one: what we decided, which files matter, and what's still open.
Write it so I can paste it as the first message of a new conversation.
```

### Tip 3: Say when you only need a quick answer

Tell Claude when you want a quick answer, not a file or a finished deliverable. Otherwise it may build something you didn't ask for.

Anthropic also suggests turning off connected apps a chat doesn't need, and asking Claude not to search the web when you don't need current information.

![A tip card: say when you only need a quick answer, with a before and after example, other usage savers and a prompt to copy](set-up-cowork/tip-quick-answer.webp "Usage tip 3. 'Quick answer only' stops Claude from building a file you didn't ask for.")

```prompt Quick answer only
Quick answer only: no files, no web search, three sentences at most.
[your question]
```

### Tip 4: Keep your context files short

Anthropic suggests short project instructions and removing project files you no longer use. The same logic applies to **Context** and **Templates**.

![A tip card: keep context files short, with the official advice, a one-screen rule, a monthly trim prompt and what to archive](set-up-cowork/tip-short-context.webp "Usage tip 4. Every file Claude reads costs usage. Short context files pay you back on every task.")

```prompt Trim my context
Read the files in Context and Read me first.md.
Suggest cuts that keep every rule and fact you actually use, so each file fits on one screen.
List anything that's out of date or repeated in more than one file.
Show me the shorter versions. Don't save them until I say yes.
```

**Use it** once a month, or whenever your role or priorities change.

### Tip 5: Check Settings → Usage before a big task

Open **Settings → Usage**. You'll see a bar for your **Current session** (the five-hour window) and your **Weekly limits**, with the time of your next weekly reset.

![A labelled drawing of Settings, Usage: the Current session bar, Weekly limits with the reset time, and the Usage credits section, with three numbered steps](set-up-cowork/usage.webp "Usage tip 5. Read the Current session bar before a long task. A labelled drawing, not a screenshot.")

If the session bar is nearly full, do the small jobs first and start the big task after the reset.

### Tip 6: Know your options at the limit

When you hit a limit, you can wait for the reset, upgrade, or, on Pro, Max and Ultra, turn on **usage credits** to keep working at standard API rates.

![A card on what to do at a usage limit: wait, upgrade or turn on usage credits, the four steps in Settings, Usage, what Now using credits means, and the rules for Reset for free](set-up-cowork/tip-credits.webp "Usage tip 6. Know your three options before you need them, and set a spend limit you're comfortable with.")

1. Open **Settings → Usage**.
2. Click the toggle in the **Usage credits** section.
3. Use **Adjust limit** to set a monthly spend limit.
4. Click **Buy usage credits**, choose an amount and click **Continue**.

Usage credits are charged separately from your subscription. When your work switches to credits, the limit warning changes to **Now using credits**: you haven't been cut off.

If you see **Reset for free** in **Settings → Usage**, it restores a five-hour, weekly or Fable limit. Use it with care: a reset can't be undone, and an unused one is lost if you downgrade or cancel.

### Two more levers: effort and model

Effort runs from Low, Medium, High and Extra high to Max. The default in the Claude apps is Medium. To change it, click the model name next to the send button, then **Effort**. Low and Medium suit routine tasks and stretch your usage further.

For the model, Anthropic positions Sonnet 5.5 for well-scoped everyday work and polished documents, slides and spreadsheets, Opus 5.5 for complex work that needs careful judgment, and Haiku 5.5 for high-volume work.

Anthropic's [usage limit best practices](https://support.claude.com/en/articles/9797557-usage-limit-best-practices) page has the full list.

::card[guide:claude-usage-limits]

---

## 9. Twenty minutes, start to finish

Here's the whole set-up in four blocks. Set a timer if it helps.

![A winding roadmap of the first 20 minutes: the folder, instructions and mode, one context file and a template, and the first real task, ending with the review](set-up-cowork/first-20.webp "Twenty minutes, four blocks. Do them in order and you finish with one real piece of work done.")

### Minutes 0 to 5: the folder

- Open Claude Desktop, or install it from [claude.com/download](https://claude.com/download).
- Make the **Claude work** folder and connect it: **Projects → + → Use an existing folder**.
- Run "Build my working folder".
- Drop one or two real files into **Inbox**.

![A card for minutes 0 to 5: make and connect the Claude work folder, the menu path, the build prompt and how to know it's done](set-up-cowork/min-0-5.webp "Minutes 0 to 5. One folder, connected, with four folders and a rules file inside.")

### Minutes 5 to 8: instructions and mode

- Copy the global instructions template. Fill in the brackets you can and delete the rest.
- Save it in **Settings → Cowork → Global instructions** (classic) or **Settings → Your Claude → Instructions** (new experience).
- Check the mode selector: Auto for a first task inside your folder, Manual if it touches anything outside it.

![A card for minutes 5 to 8: paste global instructions in either layout and check the approval mode, with where to click and a check prompt](set-up-cowork/min-5-8.webp "Minutes 5 to 8. Instructions saved, mode checked. Three minutes is enough when you start from the template.")

### Minutes 8 to 15: one context file and one template

- Run "Interview me for About me.md". Answer briefly; you can improve it later.
- Have a past document you liked? Drop it into **Templates** and run "Turn this into a template". No such document? Skip it: your first good output becomes your first template.

![A card for minutes 8 to 15: interview for About me.md and add one template, with the prompt and a done check](set-up-cowork/min-8-15.webp "Minutes 8 to 15. One context file now; the other two can wait until tomorrow.")

### Minutes 15 to 20: the first real task

- Glance at **Settings → Usage**.
- Hand over one small, real job in one message.

```prompt Your first real task
Read the files in Inbox and Context/About me.md.
Make one Word document that summarises the Inbox files for [who it's for]: the five most important points, any decisions or open questions, and a suggested next step for each.
If there's a file in Templates that fits, match its format.
Show me your plan first, then run it without stopping.
Save it in Outputs and tell me in five lines what you made and what I should check.
```

![A card for minutes 15 to 20: check usage, hand over the first real task in one message, and review it](set-up-cowork/min-15-20.webp "Minutes 15 to 20. Hand over the whole job, step away, then review instead of watching.")

While it runs, do something else. Come back, run "Review before I use it", and fix what needs fixing.

Tomorrow, write **My readers.md** and **My style.md**. Each takes about ten minutes and improves every task after it.

Know someone who says Cowork "asks too many questions"? Send them this guide.

---

## 10. Checklist and questions

### Your set-up checklist

- [ ] I'm on a paid plan and have Claude Desktop installed
- [ ] I have one **Claude work** folder with Context, Inbox, Outputs, Templates and Read me first.md
- [ ] I connected only that folder, and I keep backups of important files
- [ ] **About me.md** is written; **My readers.md** and **My style.md** are on my list
- [ ] My global instructions are saved, and the test prompt echoes them back
- [ ] I have at least one file in **Templates**, or I'll save my first good output there
- [ ] I've picked my approval mode, and I know when to switch to Manual
- [ ] I brief whole jobs with a clear finish line, and review the plan before Claude runs it
- [ ] I know where **Settings → Usage** is, and I check it before a big task
- [ ] I've finished one real task and reviewed it

### Can I use Cowork on the Free plan?

No. Cowork is on paid plans only: Pro, Max, Ultra, Team and Enterprise.

### Does a task keep running if I close my laptop?

Yes, as long as it doesn't need your local files. Tasks run on Anthropic's servers. A task that reads or writes your files needs Claude Desktop open on that computer.

### Can Claude delete my files?

Not without asking. Claude must get your permission through an **Allow** prompt before it permanently deletes any file. Keep backups of anything important anyway.

### Global instructions, folder instructions, context files: which is which?

Global instructions live in Settings and apply everywhere. Folder instructions belong to a folder you select on desktop, and Claude can update them. Context files are ordinary files in your folder that you own. The table in step 3 puts them side by side.

### Will Claude remember my set-up between tasks?

Partly. In cloud sessions, Claude starts from what it remembers from your chats, and what comes up in a Cowork task carries back to chat. A project keeps its own memory. Your folder, context files and instructions are what make the set-up reliable, because they don't depend on memory.

To run a task without memory, turn **Memory** off in the **+** menu before you send the first message. You can't change it later in that task.

### Can I share a Cowork session with a colleague?

Not the session itself. You can share individual artifacts Claude makes. On Team and Enterprise, you can share a Cowork project with 'Can view' or 'Can edit' access.

### How long can my instructions be?

No source documents a limit. Keep them to about a page, so they stay easy to update and cheap to read.

### What's next after the set-up?

Put the jobs you repeat on a schedule: type **/schedule** in a task, or click **Scheduled** in the left sidebar. Start simple, avoid hard-to-undo actions, and review each run.

---

## What to do today

1. Make the folder and connect it.
2. Save your global instructions.
3. Hand over one whole task, and review it instead of watching it.

The first whole task is the one that changes how you use Cowork.

::button[Next: Cowork projects](guide:cowork-projects)

::button[Subscribe for the next guide](page:subscribe)

::card[issue:claude-set-up-right]

Set it up once, then let it work. Mojeed
