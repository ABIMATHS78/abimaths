# Style guide for issues and guides

Learned from 135 example issues (10 read in full) (trained 2026-11-23). Use it for the shape of new issues and guides, written in the author's own voice about the author's own material.

## Principles
- One promise per issue or guide. The title names it; the subtitle says who it's for or what they'll be able to do.
- Write for one reader, as 'you'. Use 'I' only for experience the author actually gave you.
- Very short paragraphs: one or two sentences, often a single line. Half of them are 12 words or fewer.
- Be exact: menu paths (Settings → Capabilities), prompts to copy, real numbers from the brief, the order of clicks.
- Show, don't describe: a screenshot wherever the reader has to find something on screen, a prompt box whenever they should type something.
- Plain, warm, direct. Confident without hype. Practical steps over opinions.
- Every section earns its place: if a reader could skip it and lose nothing, cut it.

## Structure of an issue
- Open with an intro section of short lines (usually five to fifteen one-line paragraphs): the problem, the payoff, and what's inside as a short list (numbered when it is a sequence). No heading on the intro.
- End the intro with one button (subscribe, or the next step) and a divider (---).
- Body: five to twelve sections. Each opens with a ## heading of three to eight words and ends with a divider.
- Inside a section: short paragraphs, then the thing itself: steps (numbered list), a prompt box, a screenshot slot, a table for comparisons, a callout for the one tip or warning that matters. Use ### headings for items in a series.
- Most sections end with one call to action: subscribe, share it with one person, the next guide, or a related issue. Never two in a row.
- Close with a short 'what to do today' section, one or two buttons and a one-line sign-off in the author's voice.

## Measured shape of the examples
- Length: 2,258 typical (most between 1,950 and 4,305) words.
- Paragraphs: 87 typical (most between 60 and 195) per issue, 12 typical (most between 10 and 18) words each; 55% typical (most between 42% and 71%) of paragraphs are 12 words or fewer.
- Sections (split by dividers): 9 typical (most between 6 and 14); ## headings 6 typical (most between 1 and 13); ### headings 10 typical (most between 0 and 22).
- Per issue: screenshots 18 typical (most between 14 and 59), prompt boxes 2 typical (most between 0 and 23), lists 7 typical (most between 1 and 18), buttons 7 typical (most between 5 and 15), quotes 1 typical (most between 0 and 15).
- Opening: 14 typical (most between 4 and 37) paragraphs before the first divider; 80% include a list, 80% a button.
- Titles: 2 typical (most between 1 and 7) words; 92% end with a full stop; 16% contain a number.
- Subtitles: 8 typical (most between 5 and 13) words; 37% end with a colon that leads into the issue.
- Every issue has a cover image (100%). The site draws a title card when there is none.

## Kinds of issue (pick one)
- `beginner_walkthrough` (From-zero walkthrough): for when the reader has never used the tool or skill. Sections: what it is, in one plain paragraph; how to start (account, settings that matter); the first useful thing to do, step by step; three to five everyday uses, each with a prompt; the mistakes beginners make; what to learn next.
- `setup_steps` (Set-up steps): for when a one-time setup the reader must get right. Sections: why set it up (what it saves); Step 1 ... Step N as ## headings: where to click (screenshot), what to choose, how to check it worked; what to do if something looks different; the first thing to try.
- `use_case_playbook` (One tool, many jobs): for when the reader knows the tool and wants more from it. Sections: the situation each use case solves; one ## section per use case: when to use it, the prompt (prompt box), what you get back, one tip; how to combine them; the one to try first.
- `prompt_guide` (Prompt guide): for when the reader gets weak answers and needs better prompts. Sections: why answers come back generic; the parts of a good prompt; the template (prompt box); variations for common jobs (### per job, each with a prompt box); mistakes to avoid; a ready-to-copy pack.
- `tips_list` (Numbered tips): for when many small, independent improvements. Sections: two or three ## groups; each tip as a ### heading with one to three short paragraphs and, when useful, a prompt box or screenshot; the three tips to start with.
- `comparison` (Review or comparison): for when a new model, feature or tool the reader is deciding about. Sections: what changed, in plain words; what was tested and how (only tests from the brief); results (table); who should use what; how to switch or try it.
- `safety_warning` (Safety warning): for when a real risk readers should protect themselves from. Sections: the risk in one concrete example from the brief; how it works, plainly; how to protect yourself (checklist); what to do if it already happened.
- `essay` (Short essay): for when one idea the author wants to argue, with practical consequences. Sections: the claim; three or four ## sections of reasons, each with an example from the brief; what the reader should do differently.

## Guides (evergreen)
- A guide is evergreen: it stays useful and gets updated, unlike a dated issue.
- Start with who it's for, what they'll have at the end and what they need before starting (a short list).
- Number the main steps in the ## headings ('1. Turn on file creation'). One job per step.
- Each step: what to do, where to click (screenshot slot), what to type (prompt box), how to check it worked.
- Put the one warning that would cost the reader time or money in a [!WARNING] callout, at the step where it matters.
- End with a checklist (- [ ] items) the reader can tick, then a short 'Questions' section (### per question) if the brief answers common questions, then cards to the next guides.

## Titles and subtitles (fill with the brief's specifics)
- Title: [Outcome], from zero.
- Title: [Tool] for [audience].
- Title: How to [specific outcome].
- Title: [Number] [things] that [result].
- Title: Stop [costly habit].
- Title: [One word that names the idea].
- Subtitle: [Who it's for]: [what they'll be able to do].
- Subtitle: Everything you need to [outcome], in [N] steps:
- Subtitle: The [N] [things] I use every week, and exactly how:
- Subtitle: [Pain the reader feels]? Here's the fix, step by step.

## Calls to action
- subscribe: ::subscribe, or ::button[Subscribe](page:subscribe): after the opening and at the end.
- share: a one-line ask to send it to one specific kind of person who needs it.
- next: ::card[guide:<slug>] or ::button[Read <guide>](guide:<slug>) for the natural next step.
- related: ::card[issue:<slug>] when an earlier issue goes deeper.
- download: ::download[<file>](<path>) only for files the author provides.

## Marks you can use in the body
- ## Section heading / ### Item heading. No # (the title is the page heading).
- --- between sections.
- ```prompt <label>\n<the prompt>\n``` for anything the reader should type.
- > [!TIP] / [!NOTE] / [!WARNING] / [!KEY] followed by the text, for one callout per section at most.
- ::screenshot[<exactly what the screenshot must show>] where a screenshot is needed: the author adds the real image.
- ::button[<label>](<target>), ::card[<issue:slug|guide:slug>], ::subscribe, ::download[<label>](<path>).
- Links: [text](https://...) or [text](guide:<slug>), [text](issue:<slug>), [text](page:subscribe).
- - [ ] item for a checklist the reader can tick.
- Tables with | pipes | for comparisons.

## Content rules (always)
- Facts, numbers, names, quotes, prices, dates and links come only from the brief. If a sentence needs one the brief doesn't give, write around it or put [CHECK: what's needed].
- Never invent screenshots: use ::screenshot[...] slots describing exactly what the author must capture.
- Never promise a cadence, a download or a course that the brief doesn't give.
- Never teach cheating on courses, quizzes, certifications or AI detectors, and never suggest fooling anyone.
- No engagement bait, no fake urgency, no fake numbers of readers, likes or results.

## Words that make writing read as AI-generated: never use them
- "delve", "game-changer", "game changer", "unlock the power", "unleash", "in today's fast-paced world", "in today's digital age", "it's important to note", "seamless", "seamlessly", "elevate", "supercharge", "revolutionize", "harness the power", "dive in", "let's dive", "here's the thing", "at the end of the day", "navigate the complexities", "landscape", "tapestry", "testament to", "plays a crucial role", "robust", "cutting-edge", "transformative", "embark", "journey", "realm".
- No em dashes (—): use a colon, a comma or a new sentence. No 'It's not X, it's Y' sentences.
