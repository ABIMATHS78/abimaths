---
name: skill-maker
description: Helps you make a new Claude skill by learning the task, drafting SKILL.md with a strong name and description, checking it, and packaging the folder as a zip to upload. Use when someone wants a skill.
---

# Skill maker

Turn something the person does often into a skill Claude can use every time.

## 1. Learn the task

Ask, a few at a time:
- What task should the skill handle? Get one real example of the input and of a great result.
- What would someone type when they need it? Collect three requests that **should** use the skill and two that **shouldn't**.
- What steps does a good result need? Which mistakes happen most often?
- Does any step need to be exact every time (a calculation, a format check, a file conversion)? That step belongs in a script.
- Are there reference files: templates, brand guides, glossaries, examples?

## 2. Write the frontmatter

Start from `templates/SKILL-template.md`.

- **name**: lowercase letters, numbers and hyphens only; 64 characters at most; must not contain "claude" or "anthropic". The skill's folder gets the same name.
- **description**: say what the skill does **and** when to use it, in the third person ("Writes…", "Checks…"). Keep it to 200 characters, the limit in the Claude apps. Claude decides when to use the skill from this line alone, so name the situations and words people use. If it contains a colon followed by a space, wrap it in double quotes.

## 3. Write the instructions

- Open with one or two sentences on the goal.
- Number the steps. Say what to do, and what good looks like.
- Add rules for the mistakes you heard about in step 1.
- Include one short example of input and output.
- For long workflows, add a checklist Claude can copy and tick off.
- Keep SKILL.md under 500 lines. Move long reference material into separate files (for example `reference.md`), and link each one directly from SKILL.md.
- Put exact, repeatable steps in `scripts/`, and say when to run each one and what it prints.

## 4. Check it

Run the checker on the skill's folder:

```bash
python scripts/check_skill.py path/to/my-skill
```

It checks the name, the description, the folder name and SKILL.md's length, and that the files SKILL.md links to exist. Then test the description against the requests from step 1: would each one clearly match or clearly not?

## 5. Package it

```bash
python scripts/package_skill.py path/to/my-skill
```

This makes `my-skill.zip` with the folder at the top of the zip, which is how the Claude apps expect it. Then tell the person how to add it:

1. In Claude, open **Settings → Capabilities** and turn on **Code execution and file creation**.
2. Open **Customize → Skills**, click **+**, then **Create skill → Upload a skill**, and choose the zip.
3. Turn the skill on, then try one of the requests from step 1.

In Claude Code, copy the folder into `~/.claude/skills/` instead, and run it by typing `/my-skill`.

If you can't run scripts here, do the checks by hand and explain how to zip the folder: compress the folder itself, not the files inside it.

## Rules

- Don't put passwords, API keys or personal data in a skill.
- One skill, one job. Suggest two skills rather than one that does everything.
- Read any script you add, and keep it to the standard library where you can.
