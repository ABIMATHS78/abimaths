+++
title = "Infographics that stay sharp."
subtitle = "Clear, editable one-page visuals with AI, and a template your team can reuse:"
description = "Build infographics as code with Claude so text stays sharp and editable, add pictures from an image tool, and save the method as a project or skill."
updated = 2026-11-27
status = "published"
cover = "infographics-with-ai/cheat-sheet.webp"
cover_alt = "A one-page cheat sheet in eight panels: pick the right tool, switch on file creation, write the words first, ask for code, fix it in words, add a picture, export and share, make it repeatable"
tags = ["claude", "infographics", "productivity"]
archetype = "guide"
level = "intermediate"
topic = "Claude"
brief = "briefs/infographics-with-ai.json"
fact_checks = ["Check that Settings → Capabilities still shows 'Code execution and file creation' (the help page says the updated layout calls it 'Create and edit files').", "Ask Claude for a custom visual and check the options under it: Copy as image, Download (.svg or .html) and Save as artifact.", "Check the skill upload path: Customize → Skills → + → + Create skill → Upload a skill.", "Open a Design artifact and check that Export offers .zip, PDF, PowerPoint and standalone HTML."]
+++
You need a one-page visual for a report, a post or a slide.

You ask an AI image tool for it.

It looks great at first glance. Then you spot a wrong word, or a number you need to change.

And there's no way to fix one word in a picture. You make the whole thing again.

There's a better way for most infographics: build them as code.

An infographic built in HTML and SVG is a web page. Every word is real text. Every colour is a setting. You fix it by asking for the change in plain words.

I teach people to use AI at work, and my mission is to make data and AI easy for everyone. This is the method I'd show a team that makes visuals every week.

**By the end of this guide you'll have:**

1. a clear rule for when to use Claude and when to use an image generator
2. your first infographic, built as code, with text you can edit
3. a way to add an illustration without losing sharp text
4. a project or a skill so your team gets the same look every time

**You need:**

- a Claude account (artifacts are on every plan, Free included)
- the words and numbers your infographic should show
- for illustrations only: an account with an image generator, such as ChatGPT or Gemini

![A one-page cheat sheet of this guide in eight panels](infographics-with-ai/cheat-sheet.webp "The whole guide on one page. Keep it open while you build your first one.")

::subscribe

---

## 1. Pick the right tool for each part

Most infographics are words, numbers and simple shapes.

Some also need a picture: a person, a scene, an object.

Those are two different jobs, and they suit two different tools.

Anthropic is clear about what Claude does here. It doesn't generate photos or illustrations the way image tools do. It builds diagrams, charts and interactive visuals with HTML and SVG. ([Anthropic's page](https://support.claude.com/en/articles/9002504-can-claude-produce-images).)

Image generators do the opposite job well. ChatGPT lists image generation on every plan, limited on Free. In the Gemini app, the Nano Banana 2 image model is free.

| Part of the infographic | Best tool | Why |
|---|---|---|
| **Headings, labels, numbers** | Claude, as code | Real text: change one word without redrawing anything |
| **Charts, timelines, flows** | Claude, as code | Shapes follow your numbers, and you can check them |
| **Brand colours and fonts** | Claude, as code | One setting changes every block |
| **Photos, scenes, characters** | An image generator | Claude doesn't make photos or illustrations |
| **A mix of both** | Picture first, then Claude | The picture sits inside a layout of live text |

A simple rule: if you'd ever need to edit it, build it as code.

> [!KEY]
> Words and numbers belong in code. Pictures belong in an image generator. Keep them apart and your infographic stays easy to fix.

---

## 2. Switch on file creation

Claude's artifacts need one setting turned on.

Here's where it is:

1. Open **Settings**.
2. Go to **Capabilities**.
3. Switch on **Code execution and file creation**.

Anthropic's help page says the updated layout calls this setting **Create and edit files**. If you see that name, it's the same switch.

![A labelled drawing of Settings → Capabilities with the Code execution and file creation switch turned on](infographics-with-ai/capabilities.webp "Settings → Capabilities. An illustration, not a screenshot: your screen may look a little different.")

**How to check it worked:** in a new chat, ask Claude to "draw this as a diagram" with three short lines of text. You should see a visual appear in the conversation.

One more thing to know: custom visuals in chat work on the web and the desktop app. They don't show in the iOS or Android apps. Build your infographics on a computer.

::button[New to Claude? Start here](guide:claude-from-zero)

---

## 3. Write the words before the design

This is the step most people skip.

They ask for "an infographic about our results", and the AI fills the gaps with filler.

An infographic is only as good as its content. So write the content first, as plain text.

Keep it tight:

- **One message.** What should the reader remember after five seconds?
- **One reader.** Who is it for, and where will they see it?
- **Five to seven blocks.** Each one is a heading plus one or two lines.
- **Every number with its source.** If you can't say where it came from, leave it out.

If your notes are messy, let Claude help you shape them before any design.

```prompt Turn rough notes into infographic content
Here are my notes for a one-page infographic:

[paste your notes, figures and sources]

It's for [who will see it], who will see it on [LinkedIn / a slide / a printed report].

Before any design, write the content only:
1. A title of no more than eight words.
2. A one-line subtitle.
3. Five to seven blocks, each with a heading of three to five words and one or two short lines.
4. A source line for every number.

Use only the facts in my notes. If something is missing or unclear, ask me instead of guessing.
```

Read what comes back slowly. Cut anything you wouldn't say out loud to the reader.

> [!TIP]
> Fewer blocks almost always look better. If you have ten points, you have two infographics.

---

## 4. Ask Claude to build it as code

Now ask for the design, and say clearly that you want code.

You have two ways to do it.

### A quick visual in the chat

You don't need to turn anything on. Ask with words like "draw this as a diagram" or "chart this data", and Claude draws the visual inline.

These visuals are built with HTML and are temporary by default. To keep one, use one of the options Anthropic lists:

- **Copy as image**
- **Download**, as an .svg or .html file
- **Save as artifact**

([Anthropic's page on custom visuals](https://support.claude.com/en/articles/13979539-custom-visuals-in-chat-and-cowork).)

### An artifact you can edit and share

An artifact is anything Claude makes that you'd put in front of someone. It opens beside the conversation, and you can edit and share it.

Artifacts come with templates. **Design** is on every plan, and it's the natural home for an infographic.

Here's a prompt that gets a clean first version:

```prompt Build the infographic as code
Build this as a one-page infographic in HTML and SVG.

Content (use it exactly, don't add facts):
[paste the title, subtitle, blocks and sources from step 3]

Design:

- Size: [1080 x 1350 for LinkedIn / 1920 x 1080 for a slide / A4 portrait for print]
- Colours: [your two or three brand colours, as hex codes, or "one dark colour and one accent"]
- Font: [your font, or "a clean sans-serif"]
- Layout: a clear title at the top, the blocks in a grid, sources in small text at the bottom
- Leave plenty of space around each block

Rules:
- Keep every word as real, editable text. No text inside images.
- Draw charts and icons in SVG, not as pictures.
- Make sure nothing overlaps or gets cut off at the edges.
```

Hex codes are the six-character colour codes your brand guide uses, like #C62828.

**How to check it worked:** click on a word in the downloaded .html file in your browser and try to select it. If you can highlight it, it's real text.

::card[guide:brief-claude-like-a-colleague]

---

## 5. Fix it in words

The first version is a draft. That's normal.

The good news: because it's code, every fix is a sentence.

Change one thing per message. Several small asks work better than one long list.

```prompt Fixes you can copy
Make the title shorter and bigger. Keep the subtitle as it is.

Swap blocks 2 and 4.

Use [#hex code] for all headings and keep the body text dark grey.

Turn block 3 into a simple bar chart drawn in SVG, using these numbers: [numbers].

Give the whole layout more white space. Nothing should touch the edges.

Make it work in black and white too, for printing.
```

Then do the check that matters most.

Read every number on the page and compare it with your source. Claude can make mistakes, and a wrong figure on a pretty graphic travels fast.

> [!WARNING]
> Never publish an infographic with a number you haven't checked against the original source yourself. Your name goes on it, not the AI's.

---

## 6. Use an image generator for the pictures

Sometimes a layout of boxes isn't enough.

You want a friendly illustration, a scene or a hero picture at the top.

That's the job for an image generator. Two options with free tiers:

- **ChatGPT.** Its Images page says you can start for free with a ChatGPT account. You must log in or sign up to create images. ([ChatGPT Images](https://chatgpt.com/images/).)
- **Gemini.** Select **Create images** from the tools menu, then add a prompt or upload an image to edit. Nano Banana 2 is free; Nano Banana 2.1 needs a Google AI plan. ([Gemini image generation](https://gemini.google/overview/image-generation/).)

Google says Gemini can place clear text in images, in many languages.

Even so, keep the words out of the picture. Text inside an image is pixels. You can't fix a typo or change a number without making the image again. Your headings and numbers will live in Claude's layout instead.

```prompt An illustration with no words in it
Create an illustration for the top of an infographic about [topic].

Style: flat, simple shapes, [your two or three brand colours], plenty of empty space.
Subject: [what should be in it, for example "a small team around a table looking at a chart"].
Shape: wide, about 3:1.

Important: no text, no letters, no numbers and no logos anywhere in the image.
Leave the left third empty so I can place a heading there.
```

Download the result as a PNG.

> [!NOTE]
> Google says Gemini images carry an invisible SynthID watermark and a visible watermark. Plan your layout so the visible mark doesn't land somewhere awkward.

---

## 7. Combine the two

Here's where it comes together.

You have a picture with no words. You have a layout of live text. Now put the picture inside the layout.

1. Open the chat where you built your infographic.
2. Upload the PNG. Claude takes JPEG, PNG, GIF and WebP images, up to 8000 x 8000 pixels.
3. Paste the prompt below.

```prompt Place the illustration in the layout
I've uploaded an illustration. Add it to the infographic we built:

- Put it across the top, behind the title, on the right-hand side.
- Keep the title and every other word as live HTML text, on top of or beside the image.
- Don't redraw, trace or change the illustration.
- Make sure the title is easy to read against the picture. If it isn't, add a solid panel behind the text.
- Keep the rest of the layout as it is.
```

Now you get the best of both. The picture looks hand-made. The words stay sharp and editable.

When it's done, keep it as an artifact and export it.

Design artifacts export from the **Export** button to .zip, PDF, PowerPoint or standalone HTML. ([Anthropic's page on artifacts](https://support.claude.com/en/articles/17153992-what-are-artifacts-and-how-do-i-use-them).)

To share the live version, open the artifact and click **Share**. Choose who can open it under **Who has access**, then copy the link. Artifacts start private.

Two things to know about sharing:

- Anthropic's sharing table ticks "Share with anyone who has the link" for Pro, Max, Ultra, Team and Enterprise, but not Free.
- You can't change sharing settings in the iOS or Android apps. Use the web or Claude Desktop.

::button[Making slides too? Read the slides guide](guide:slides-with-ai)

---

## 8. Make it repeatable for your team

One good infographic is nice.

The same look every time, from anyone on your team, is what saves real time.

There are two ways to get there. Start with the first.

### Option A: a project

A project is a workspace with its own chats and its own knowledge. Every user can make one, and Free accounts can create up to five.

1. Go to **claude.ai/projects** and click **+ New Project**.
2. Add your brand guide and one infographic you like with **+**.
3. Click **Set project instructions** and paste the text below.

```prompt Project instructions for infographics
You make one-page infographics for [team or company].

Always:
- Build them in HTML and SVG, with every word as real, editable text.
- Use these colours: [hex codes]. Use this font: [font].
- Use this layout: title at the top, five to seven blocks, sources in small text at the bottom.
- Use only the facts the person gives you. Add a source line for every number.
- Ask about size and audience if they aren't given.

Never:
- Put text inside images.
- Invent numbers, quotes or logos.
```

Now anyone working in that project starts from the same rules.

### Option B: a skill

A skill goes one step further. It's a folder of instructions that Claude loads when a request needs it, in any chat. Skills are on every plan, and they need code execution switched on (step 2).

Every skill has a file called SKILL.md. It starts with two required fields:

- **name:** a short name, up to 64 characters
- **description:** what the skill does and when to use it, up to 200 characters

The description matters most: Claude reads it to decide when to use the skill. Anthropic's own example of a custom skill is a brand guidelines skill, so you're in good company. ([Anthropic's guide to custom skills](https://support.claude.com/en/articles/12512198-how-to-create-custom-skills).)

Here's a starting point. Put it in a folder called `infographic-maker`, in a file named `SKILL.md`:

```prompt SKILL.md for an infographic skill
---
name: infographic-maker
description: Builds one-page infographics in HTML and SVG in our brand colours, with editable text. Use when someone asks for an infographic, a one-page visual or a cheat sheet.
---

# Infographic maker

## Steps
1. Ask for the audience, where it will appear and the size, if not given.
2. Write the content first: a title of up to eight words, a subtitle, five to seven blocks, a source for every number. Show it and wait for approval.
3. Build the design in HTML and SVG. Every word stays real text.
4. Check that nothing overlaps or is cut off, then list every number with its source so the person can check it.

## Brand
- Colours: [hex codes]
- Font: [font]
- Layout: title at the top, blocks in a grid, sources at the bottom

## Never
- Put text inside images.
- Invent facts, numbers, quotes or logos.
- Redraw an illustration the person uploads. Place it as it is.
```

To add it:

1. Zip the folder (the folder itself, not just the file inside it).
2. Open **Customize → Skills** and click **+**.
3. Choose **+ Create skill**, then **Upload a skill**, and pick your ZIP file.

![A labelled drawing of Customize → Skills with the + button, + Create skill and Upload a skill marked in order](infographics-with-ai/upload-skill.webp "Customize → Skills → + → + Create skill → Upload a skill. An illustration, not a screenshot.")

**How to check it worked:** in a new chat, ask "Make me an infographic about [topic]". Claude should ask for the audience and show you the content before it designs anything.

Give the same ZIP to your teammates so they can upload it too. On paid plans, plugins can also bundle skills into one package.

> [!TIP]
> Test the skill with three requests that should use it and two that shouldn't. If it fires at the wrong time, rewrite the description.

::card[guide:claude-skills-from-zero]

---

## What to do today

Don't build the skill today. Build one infographic.

1. **Switch on file creation** in Settings → Capabilities.
2. **Write the content first**, with a source for every number.
3. **Ask for HTML and SVG**, with all text kept editable.
4. **Check every number**, then save it as an artifact.

When you've made three you like, turn your rules into a project. Then into a skill.

If you know someone who spends hours fixing typos in AI images, send them this guide.

## Before you finish

- [ ] Code execution and file creation is on in Settings → Capabilities
- [ ] My content is written: title, subtitle, five to seven blocks, sources
- [ ] My infographic is built in HTML and SVG, and I can select its text
- [ ] Every number matches its source
- [ ] Any illustration has no words in it
- [ ] I've saved it as an artifact or exported it
- [ ] My rules are in a project or a skill for next time

## Questions

### Can Claude draw the illustration itself?

No. Anthropic says Claude doesn't generate photos or illustrations the way image tools do. It builds diagrams, charts and visuals with HTML and SVG. Use an image generator for the picture, then let Claude build the layout around it.

### Do I need a paid plan?

Not to start. Artifacts and the Design template are on every plan, Free included. Link sharing for new artifacts is ticked for paid plans only in Anthropic's sharing table.

### Can I put the infographic in a PowerPoint deck?

Yes. Design artifacts export to PowerPoint, as well as PDF, .zip and standalone HTML.

### Why can't I see the visual on my phone?

Custom visuals in chat don't render in the iOS or Android apps. Open the same chat on the web or in Claude Desktop.

### Does sharing an interactive infographic cost me usage?

No. If a shared artifact uses Claude, Anthropic says the usage counts against each viewer's own plan, not yours.

::card[guide:slides-with-ai]

::card[issue:make-your-first-skill]

::subscribe
