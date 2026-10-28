## Teach Claude a job once, and it does it your way every time

Most people use Claude the same way every day: they paste in the same long instructions, explain the same format again, and correct the same mistakes. A **skill** ends that. It is a small folder of instructions, and sometimes scripts and templates, that teaches Claude one job the way you want it done. You install it once. From then on, Claude picks it up whenever a request needs it.

This guide gives you a starter kit of ten skills to download, written for AI Every Time. It shows you how to install them in a few minutes and which skills are worth having for eighteen kinds of work. It ends with how to make your own, and how to stay safe while you do.

## What a skill is

Every skill is a folder with one required file, `SKILL.md`. Its first lines hold two pieces of information:

- a **name**, which is also the command you can type, such as `/clarify-first`;
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

- For every project on your computer, unzip the skill into `~/.claude/skills/`, so you end up with `~/.claude/skills/clarify-first/SKILL.md`.
- For one project, and everyone who works on it, unzip it into the project's `.claude/skills/` folder and commit it.
- Run it by typing a slash and its name, such as `/clarify-first`, or simply ask and let Claude choose. `/skills` lists what's installed.

### Using a skill

You rarely need to do anything special. Claude reads the descriptions of your skills and uses one by itself when your request matches. To be sure, name it: *"Use my analyze-data skill to check these sales figures."* In Claude Code and in the Microsoft 365 add-ins, you can also type `/` and pick a skill from the list.

If Claude doesn't use a skill you expected, check that its switch is on, name the skill in your request, or make its description clearer about when it applies.

## The starter kit: ten skills to download

I wrote these ten for the jobs nearly everyone does: getting clear on the task, writing, keeping long work on track, building and analysing. Each one is a folder you can open and read before you install it, and each is a short file you can change to suit you.

### Think before you start

**clarify-first** · `/clarify-first`
Asks you one focused question at a time, with suggested answers, until the goal, audience, limits and finish line are clear. Then it writes a short brief and waits for your go-ahead.
*Try:* "I need a website for my bakery."

**brief-builder** · `/brief-builder`
Turns a brain dump, voice-note transcript or pile of notes into a clear task brief: goal, deliverable, requirements, limits and what "done" means. It asks no more than three questions and confirms the brief before any work starts.
*Try:* "Here are my notes from this morning's call. Make them into a brief."

### Write better

**natural-voice** · `/natural-voice`
Edits a draft so it reads like a thoughtful person wrote it. It cuts stock phrases, filler and unnecessary formatting, adds specifics, and keeps every fact. Share a sample of your own writing and it matches your voice.
*Try:* "Make this LinkedIn post sound like me. Here's one I wrote last month."

**persuasive-copy** · `/persuasive-copy`
Writes headlines, landing pages, emails, ads and calls to action from a short brief. It uses proven structures such as problem, agitate, solve, and gives you two variants to test. It never invents numbers, reviews or deadlines.
*Try:* "Write a landing page for my Excel course for small-business owners."

**co-write** · `/co-write`
Writes a longer document with you in five stages: context, outline, section-by-section drafts, a read-through as a newcomer would see it, then polish. Missing facts are marked, never made up.
*Try:* "Help me write a funding proposal for our coding club."

**plain-technical-english** · `/plain-technical-english`
Rewrites manuals, procedures and safety text in the style of Simplified Technical English: short sentences, one instruction per step, active voice, the same word for the same thing. It includes a script that checks the result for long sentences, passive voice and wordy phrases.
*Try:* "Rewrite these machine set-up instructions in plain technical English."

### Keep long work moving

**session-handoff** · `/session-handoff`
When a chat gets long, it writes a one-page handoff note: the goal, decisions and their reasons, where things stand, exact file names, what didn't work, and the next steps. Paste it into a new chat and carry on.
*Try:* "This chat is getting slow. Write a handoff so I can start fresh."

### Build and analyse

**idea-to-website** · `/idea-to-website`
Takes you from an idea to a live website in five phases: plan, design, build, put the code on GitHub, and deploy to a host such as Vercel, Netlify or GitHub Pages. It ends with a launch checklist. In Claude Code it can run the steps; in chat it gives you the files and the exact steps.
*Try:* "I want a one-page site for my photography, live this week."

**analyze-data** · `/analyze-data`
Answers questions about spreadsheets, CSV files and SQL results, from quick lookups to full analyses. Before trusting a number, it defines each metric and checks the data for duplicates and missing values. It also checks whether a join multiplied the rows. Then it reports the answer first, with its caveats.
*Try:* "Which products drove last quarter's revenue? Here's the export."

### Make your own

**skill-maker** · `/skill-maker`
Interviews you about a task you repeat, writes the `SKILL.md`, checks it against the rules, and packages the folder as a zip ready to upload. It includes the checker, the packager and a template.
*Try:* "Help me turn the way I write our weekly report into a skill."

## Skills worth having for every job

The best skills capture the tasks you repeat every week, in the way your organisation does them. Here are ideas for eighteen kinds of work. Find them in the directories listed below, or make them with **skill-maker**.

### Growing the business

| Job | Skills worth having |
|---|---|
| Organic marketing | A content calendar planner, an SEO brief writer, and one that turns a single article into five posts for different channels |
| Paid marketing | Ad copy variants within each platform's length limits, a campaign results summary, and audience and keyword research |
| Sales | Research on a prospect before a call, discovery-call notes into a follow-up email, and proposals in your house format |
| Fundraising | Grant application drafts that follow the funder's questions, donor updates, and impact reports from programme data |

### Running the organisation

| Job | Skills worth having |
|---|---|
| C-suite | Board memos, one-page strategy summaries, and a decision log with options, risks and reasons |
| Operations | Standard operating procedures, process audit checklists, and meeting notes into owners and deadlines |
| Finance and accounting | Explanations of budget-versus-actual variances, invoice and expense checks against policy, and month-end summaries |
| Legal and compliance | Contract review against your own clause checklist, plain-language policy summaries, and compliance evidence tracking, always with a lawyer's final review |

### People and learning

| Job | Skills worth having |
|---|---|
| HR and people | Job descriptions, onboarding plans for a new starter's first 30 days, and answers to policy questions in your company's words |
| Recruiting | Interview question sets for a role, candidate update emails, and CV screening against agreed criteria, with a person making every decision |
| Education and training | Lesson plans, quizzes with answer keys, and feedback on student work against a rubric |

### Customers, products and research

| Job | Skills worth having |
|---|---|
| Customer support | Replies in your brand's tone, ticket sorting and tagging, and help-centre articles from solved tickets |
| Product | Product requirement documents, summaries of user interviews, and release notes |
| Research | Literature review tables, source credibility checks, and analysis of interviews and surveys |
| Healthcare | Plain-language patient information and clinic procedures, never diagnosis, with privacy rules followed and a clinician checking every output |

### Creative work and building

| Job | Skills worth having |
|---|---|
| Writing and content | Your style guide applied to every draft, newsletters, and an editing checklist |
| Design and brand | Brand guidelines (colours, fonts and logo use), design briefs, and accessibility checks for contrast and alt text |
| Building with AI ("vibe coding") | **idea-to-website**, code review checklists, and bug reports turned into fix plans |

### Where to find thousands more

- **Claude's own directory**: in **Customize → Skills**, click **+**, then **Browse skills**. It includes skills from partners such as Notion, Figma and Atlassian, built to work with their connectors.
- **Anthropic's examples on GitHub**, at [github.com/anthropics/skills](https://github.com/anthropics/skills), are good to learn from and adapt.
- **Ruben Hassid's Claude Skill Library**, at [skillsclau.de](https://skillsclau.de/), sorts more than 33,000 skills by job. It was the inspiration for this guide.

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

Every skill in this starter kit is plain text you can read in a minute. The three scripts (one in **plain-technical-english**, two in **skill-maker**) use only Python's standard library and never connect to the internet.

## Downloads

Everything below is ready to use:

- **Each skill as its own zip**, ready to upload in Customize → Skills.
- **All ten in one zip** for Claude Code: unzip it into `~/.claude/skills/`. For the Claude apps, upload the skills one at a time.
- **The SKILL.md template**, to start a skill of your own.
- **A one-page cheat sheet** of installing, using and making skills.

*How skills work, where they run and how to install them is from Anthropic's documentation. The ten skills are my own, written for AI Every Time learners. Ask me about any of them, or about the skill you want to make next.*
