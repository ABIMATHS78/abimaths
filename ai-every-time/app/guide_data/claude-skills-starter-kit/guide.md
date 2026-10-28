## Teach Claude a job once, and it does it your way every time

Most people use Claude the same way every day: they paste in the same long instructions, explain the same format again, and correct the same mistakes. A **skill** ends that. It is a small folder of instructions, and sometimes scripts and templates, that teaches Claude one job the way you want it done. You install it once. From then on, Claude picks it up whenever a request needs it.

This guide gives you a starter kit of ten skills to download, written for AI Every Time. It shows you how to install them in a few minutes and which skills are worth having for twenty different roles. It ends with how to make your own, and how to stay safe while you do.

## What a skill is

Every skill is a folder with one required file, `SKILL.md`. Its first lines hold two pieces of information:

- a **name**, which is also the command you can type, such as `/learn-anything`;
- a **description** of what the skill does and when to use it.

Below that come the instructions: steps, rules, examples and checklists. A skill can also carry extra files: reference notes, templates to fill in, and scripts for steps that must come out exactly right every time.

Skills load in layers, which is why you can install many of them without slowing Claude down:

1. **Always**: just the name and description of every skill you've switched on, a line or two each.
2. **When a request matches**: the skill's instructions.
3. **Only if needed**: its extra files. Scripts run, and only their results come back to Claude.

### Skills compared with the other ways to shape Claude

| Feature | What it gives Claude | When it applies |
|---|---|---|
| Custom instructions | Your general preferences | Every conversation |
| Projects | Background documents and knowledge | Chats inside that project |
| Connectors (MCP) | Access to other apps and data, such as your drive or calendar | When Claude needs that app |
| **Skills** | **A way of doing a particular job** | **Only when a request needs it, in any chat** |

They work well together. A connector lets Claude reach your project tracker; a skill teaches it how your team writes tickets.

Skills also follow an open standard, the [Agent Skills specification](https://agentskills.io), so a skill you write isn't locked to one product.

## Where skills work

- **Claude on the web and in the desktop app**, on every plan from Free to Enterprise, once code execution is turned on.
- **Claude Code**, from folders on your computer. Skills you switch on in your Claude account also sync to Claude Code when you sign in with the same account.
- **Claude for Excel, PowerPoint, Word and Outlook**: skills you've switched on work in the add-ins too.
- **Cowork**, which uses the skills switched on for your account.
- **The Claude API**, in beta, through the code execution tool. Skills there run in a sandbox with no internet access, so everything a skill needs must be inside it.

## Install a skill in four steps

1. **Download the skill's zip file** from the Downloads section at the end of this guide. Don't unzip it.
2. In Claude, open **Settings → Capabilities** and turn on **Code execution and file creation**. In the newer Settings layout, it's called **Create and edit files**.
3. Open **Customize → Skills**, click **+**, then **Create skill → Upload a skill**, and choose the zip.
4. Check that the skill's switch is **on**. That's it: ask for something the skill covers.

On Team plans, skills are on by default. On Enterprise plans, an owner must first turn them on in **Organization settings → Plugins & skills**, on the **Policy** tab.

### In Claude Code

- For every project on your computer, unzip the skill into `~/.claude/skills/`, so you end up with `~/.claude/skills/learn-anything/SKILL.md`.
- For one project, and everyone who works on it, unzip it into the project's `.claude/skills/` folder and commit it.
- Run it by typing a slash and its name, such as `/learn-anything`, or simply ask and let Claude choose. `/skills` lists what's installed.

### Using a skill

You rarely need to do anything special. Claude reads the descriptions of your skills and uses one by itself when your request matches. To be sure, name it: *"Use my check-the-numbers skill on these sales figures."* In Claude Code and in the Microsoft 365 add-ins, you can also type `/` and pick a skill from the list.

If Claude doesn't use a skill you expected, check that its switch is on, name the skill in your request, or make its description clearer about when it applies.

## The starter kit: ten skills to download

I wrote these ten for what AI Every Time learners do most: learning something new, researching and explaining, growing a career, getting work done and working with data. Each one is a folder you can open and read before you install it, and a short file you can change to suit you.

### Learn and remember

**learn-anything** · `/learn-anything`
A patient tutor for any topic. It asks what you already know, makes a short plan, then explains one idea at a time with an analogy and a small example, and checks your understanding before moving on.
*Try:* "Teach me how neural networks learn. I know basic algebra."

**quiz-me** · `/quiz-me`
Turns your notes, a document or a topic into practice questions and asks them one at a time. It explains every answer, keeps track of what you miss and comes back to it.
*Try:* "Quiz me on these lecture notes. Ten questions, mixed."

### Research and explain

**research-brief** · `/research-brief`
Answers a question with a one-page, sourced brief. It separates what's known from what's only claimed, gives each finding a confidence level and never invents a source.
*Try:* "How many small businesses in Nigeria use AI? Give me a brief."

**news-explainer** · `/news-explainer`
Turns AI and tech news into a plain explainer: what happened, why it matters, who it affects and what's still unknown. Every claim is attributed, and there's no hype. It's how I like the news explained.
*Try:* "Explain this announcement for my newsletter readers." (Paste the link or the text.)

### Grow your career

**linkedin-post** · `/linkedin-post`
Writes LinkedIn posts in your own voice from an idea, a lesson or a piece of work. You get an opening worth reading, one clear point, and no invented stories or engagement bait.
*Try:* "Turn this project I finished into a post. Here are two of my old posts."

**job-application** · `/job-application`
Tailors your CV and cover letter to a specific job ad. It matches your real experience to what the employer asks for, and tells you honestly where the gaps are.
*Try:* "Here's my CV and the job ad. Tailor both."

### Get work done

**meeting-to-actions** · `/meeting-to-actions`
Turns meeting notes or a transcript into decisions, actions with owners and deadlines, and open questions, with a follow-up email ready to send. It never invents an owner or a date.
*Try:* "Here's the transcript of today's team call."

**presentation-planner** · `/presentation-planner`
Plans a talk before you open slide software: the story, one message per slide, titles that state the point, visuals and speaker notes. With code execution on, Claude can then build the deck with its built-in PowerPoint skill: two skills working together.
*Try:* "Plan a ten-minute talk on our quarterly results for the board."

### Work with data

**check-the-numbers** · `/check-the-numbers`
Answers questions about spreadsheets, CSV files and SQL results, and proves the answer first. A script profiles each file: missing values, numbers stored as text, repeated rows and keys, and gaps in the dates. Every join is checked for multiplied rows, and the answer comes with its caveats.
*Try:* "Which products drove last quarter's revenue? Here's the export."

### Make your own

**skill-maker** · `/skill-maker`
Interviews you about a task you repeat, writes the `SKILL.md`, checks it against the rules, and packages the folder as a zip ready to upload. It includes the checker, the packager and a template.
*Try:* "Help me turn the way I write our weekly report into a skill."

## Skills worth having for your role

The best skills capture the tasks you repeat every week, the way your organisation does them. Here are ideas for twenty roles. Find them in Claude's directory, or make them with **skill-maker**.

### Learning and teaching

| Role | Skills worth having |
|---|---|
| Students | A study plan built from your syllabus and exam dates, **quiz-me** on past papers, and feedback on an essay against its marking rubric that tells you what to improve without rewriting it |
| Teachers and trainers | Lesson plans from a learning objective, worksheets at three levels of difficulty, and feedback drafts against your rubric for you to review |
| Researchers | Literature review tables, **research-brief** for quick evidence summaries, and a check of your methods section against your field's reporting checklist |

### Data and technology

| Role | Skills worth having |
|---|---|
| Data analysts | **check-the-numbers**, a sheet that defines every KPI once, and commentary that explains what changed on a dashboard and why |
| Software developers | Code reviews in your team's conventions, pull request descriptions, and bug reports turned into fix plans |
| IT and cybersecurity | Incident reports in your template, sorting suspected phishing emails, and access review checklists |
| Product managers | Product requirement documents, summaries of user interviews, and release notes |

### Business

| Role | Skills worth having |
|---|---|
| Founders and small businesses | Investor and customer updates, price comparisons with competitors, and an FAQ written from your support emails |
| Marketing | Content calendars, campaign briefs, and one article turned into posts for each channel |
| Sales | Research on a prospect before a call, call notes into a follow-up email, and proposals in your house format |
| Customer support | Replies in your brand's tone, ticket sorting and tagging, and help-centre articles from solved tickets |
| Operations | Standard operating procedures, **meeting-to-actions**, and process audit checklists |
| Finance | Budget-versus-actual explanations, expense checks against policy, and month-end summaries |
| HR and recruiting | Job descriptions, interview question sets, and onboarding plans for a new starter's first 30 days, with a person making every hiring decision |
| Legal and compliance | Contract review against your own clause checklist and plain-language policy summaries, always with a lawyer's final review |
| Leaders and managers | Leadership updates, decision memos with options and risks, and preparation for one-to-one meetings |

### Media, creative work and public service

| Role | Skills worth having |
|---|---|
| Journalists and editors | **news-explainer**, a fact-check list to run before publishing, and interview transcripts turned into quotes and a story outline |
| Content creators | Video scripts with strong openings, one talk turned into a newsletter and posts, and **linkedin-post** |
| Designers | Design briefs, your brand guidelines applied to every asset, and accessibility checks for colour contrast and alt text |
| Healthcare and nonprofits | Plain-language information for patients or communities, grant applications that follow the funder's questions, and impact reports; never diagnosis, always within privacy rules and checked by a professional |

### Where to find more

- **Claude's own directory**: in **Customize → Skills**, click **+**, then **Browse skills**. It includes skills from partners such as Notion, Figma and Atlassian, built to work with their connectors.
- **Anthropic's examples on GitHub**, at [github.com/anthropics/skills](https://github.com/anthropics/skills), are good to learn from and adapt.

## Make your own skill

A skill is worth making once you've explained the same thing to Claude three times.

### The parts of a skill

```
weekly-report/
├── SKILL.md        required: name, description and instructions
├── reference.md    optional: detail Claude reads only when it needs it
├── templates/      optional: files to fill in
└── scripts/        optional: code for steps that must be exact
```

`SKILL.md` starts with its name and description between two lines of dashes:

```yaml
---
name: weekly-report
description: Turns the week's notes and numbers into our team's weekly report format. Use when someone asks for the weekly report or pastes weekly updates.
---
```

### The rules that matter

| Part | Rule |
|---|---|
| Name | Lowercase letters, numbers and hyphens only; 64 characters at most; can't contain "claude" or "anthropic"; the same as the folder's name |
| Description | Say what the skill does **and** when to use it, in the third person. Claude decides when to use the skill from this line alone. 200 characters at most in the Claude apps (the API allows 1,024). No tags in angle brackets |
| Instructions | Under 500 lines. Numbered steps, the mistakes to avoid, one short example, and a checklist for long jobs |
| Extra files | Link each one directly from `SKILL.md`, and say when to read it |
| Scripts | For steps that must come out the same every time: calculations, format checks, conversions |

### Package it

Zip the **folder**, not the files inside it. The zip must contain `weekly-report/SKILL.md`. If the files sit at the top of the zip, without their folder, the upload won't work.

### Test it

1. Write three requests that should use the skill and two that shouldn't.
2. Upload the skill, switch it on, and try all five.
3. If Claude misses one, improve the description first: it's what Claude uses to choose.

**Shortcuts:** install **skill-maker** and say *"Help me make a skill for…"*, or start from the **SKILL.md template** in the Downloads. On a Mac, Cowork can even record you doing a task and propose a skill from it (on Pro, Max and Team plans).

## Stay safe with skills

A skill can contain code that runs when Claude uses it, so treat an unknown skill like an unknown app.

- **Install only from sources you trust.** Even a skill a colleague shares deserves a look first.
- **Read every file before you switch a skill on**, especially scripts. Look for anything that downloads software or sends information to a website.
- **Never put passwords, API keys or personal data in a skill.**
- The main risks are *prompt injection*, where hidden instructions steer Claude into something you didn't ask for, and *data leaks*. Anthropic's advice is the same as above: check before you trust.
- On Enterprise plans, owners can turn on skill scanning, which checks uploaded skills for malicious content.

Every skill in this starter kit is plain text you can read in a minute. The three scripts (one in **check-the-numbers**, two in **skill-maker**) use only Python's standard library and never connect to the internet.

## Downloads

Everything below is ready to use:

- **Each skill as its own zip**, ready to upload in Customize → Skills.
- **All ten in one zip** for Claude Code: unzip it into `~/.claude/skills/`. For the Claude apps, upload the skills one at a time.
- **The SKILL.md template**, to start a skill of your own.
- **A one-page cheat sheet** of installing, using and making skills.

*How skills work, where they run and how to install them is from Anthropic's documentation. The ten skills are my own, written for AI Every Time learners. Ask me about any of them, or about the skill you want to make next.*
