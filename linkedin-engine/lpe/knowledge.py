"""What the engine learned from reading the training posts, written as reusable formulas.

These are patterns, not copies: hooks and closings are abstracted with [placeholders] so the
engine writes new posts in the same shape, in the author's own voice, about the author's topics.
Measured statistics (lengths, list use, colours) are added by `lpe train` from the data itself.
"""

VOICE_PRINCIPLES = [
    "Teach one practical thing per post. The reader should be able to act on it today.",
    "Talk to one reader as 'you'. Use 'I' only for experience the author actually gave you.",
    "Short lines, one idea per line, a blank line between blocks. Most lines stay under 60 characters.",
    "Be concrete: exact menu paths (Settings → Capabilities), prompts to copy in quotes, real numbers, time boxes.",
    "Hold a clear, slightly contrarian opinion and say it plainly. No hedging and no hype words.",
    "Casual rhythm: '&' for 'and' in short lines, a few parenthetical asides ('(for free)', '(trust me)'), "
    "and at most one to three words in CAPS for emphasis.",
    "Make the post worth saving: steps, lists, prompts to copy, and a graphic that sums up the whole post.",
    "Close with one clear next step: save it, send it to a specific kind of person, or follow a link "
    "(one from the brief, or the author's website).",
]

HOOK_FORMULAS = [
    {"id": "forbidden_knowledge", "formula": "Nobody shows you this part, so I will: [result], for [tiny cost]. Here's how:"},
    {"id": "time_box_promise", "formula": "Give me [N] minutes, and I'll make you better at [skill] than most people at your work."},
    {"id": "biggest_mistake", "formula": "The mistake I see in almost every [tool] chat: [common habit]. Stop it today."},
    {"id": "i_read_it_so_you_dont_have_to", "formula": "I read all of [long official source] for you. What to cut, what to swap, and the exact [prompt] to use:"},
    {"id": "replacement_habit", "formula": "Instead of [scrolling for 40 minutes], [read these N short guides] (total: [M] minutes).", "only_if": "the brief gives the guides and their links"},
    {"id": "gatekeeping_test", "formula": "The quickest way to tell if someone really knows [topic]: ask them these [N] words."},
    {"id": "deadline_urgency", "formula": "You have [real time left] to [use X] before [real change]. Here's how to make the most of it:", "only_if": "the deadline is real and in the brief"},
    {"id": "must_read_warning", "formula": "Read this before you open [tool] again. (Your team needs it too.)"},
    {"id": "anti_spend", "formula": "Stop paying $[N] for courses to learn [topic]. The best free ones fit in one list:", "only_if": "the brief gives the free resources and their links"},
    {"id": "experience_cost", "formula": "It took me [real amount of time] to learn [topic] properly. These [N] lessons would have saved me most of it:", "only_if": "the author gave the real number"},
    {"id": "eating_everything_map", "formula": "[Tool] does [A], [B] & [C] now. One map shows all of it:"},
    {"id": "nobody_explained", "formula": "Nobody explains [topic] properly. So here's all of it, in [N] levels:"},
    {"id": "self_aware_creator", "formula": "This might sound odd from someone who makes content about [topic], but the best [setup] is no setup:"},
    {"id": "feed_critique", "formula": "Half my feed reads like [problem]. Here are [N] giveaways to delete & the [N] fixes:"},
    {"id": "fear_of_missing_out_job", "formula": "[Big change] is reshaping [your job], so you might as well learn [skill] (for free). Bookmark this:"},
    {"id": "staccato_names", "formula": "[A]. [B]. [C]. [D].\n\nHow to pick the right one (every time):"},
    {"id": "mock_a_myth", "formula": "[S-P-A-C-E-D  W-O-R-D-S].\n\nIf you believe [myth], we need to talk."},
    {"id": "relatable_story", "formula": "I taught [specific real person] [tool] last week. These are the [N] lessons we went through:", "only_if": "the story is true and in the brief"},
    {"id": "the_only_chart", "formula": "One chart says more about [topic] than any thread:", "only_if": "the data and its source are in the brief"},
    {"id": "i_dont_care_i_do_care", "formula": "I DON'T CARE if you [popular distraction]. I do care if you [real problem]."},
    {"id": "short_read_promise", "formula": "This takes [N] seconds to read, but these [N] tips will save you hours:"},
    {"id": "learn_it_fast", "formula": "How to learn [skill] so fast people think you're naturally gifted:"},
    {"id": "my_exact_toolkit", "formula": "These are the exact [tools] I use every day. You can have all of them (for free):", "only_if": "the brief says where readers can get them"},
    {"id": "found_one_resource", "formula": "I found the one [resource] you need to [result].\n(Save it, you'll want it later.)"},
    {"id": "trick_and_why_not", "formula": "There's a trick that [does X]. Here's how it works, and why you shouldn't use it:"},
]

STRUCTURES = [
    {"id": "numbered_steps", "shape": "'Step 1.' to 'Step N.', one to three short lines each; near the end a twist that turns the trick into the real lesson, then that lesson in one line."},
    {"id": "labelled_sections", "shape": "Headed blocks such as 'LESSON 1:', 'Level 1.' or '→ LEVEL 1 - The basics.', two to five lines each."},
    {"id": "tip_list", "shape": "Numbered tips ('1.' or '1)'), each a bold-sounding first line plus one or two lines of why or how."},
    {"id": "link_library", "shape": "Guide titles with links, optionally grouped by level, then the save / share / get-more triple."},
    {"id": "old_new", "shape": "'Old: ...' / 'New: ...' pairs (or 'BAD PROMPT:' / 'BETTER PROMPT:') that show the fix, not just describe it."},
    {"id": "delete_keep", "shape": "'First, delete these:' as a long numbered list, then 'Now keep only these N things:' as a short one."},
    {"id": "avoid_fix", "shape": "N things to stop doing, each with a one-line jab, then N fixes."},
    {"id": "category_sections", "shape": "Numbered CAPS section titles ('1. AI + SALES'), each with ✦ sub-steps, separated by '-----'."},
    {"id": "rapid_fire", "shape": "Every line starts with '> ', grouped in short blocks: a whole masterclass in one-liners."},
    {"id": "decision_steps", "shape": "'Step 1: [question]?' then '☑️ No: ...' / '☑️ Yes: ...', each answer with a prompt example."},
    {"id": "glossary", "shape": "Grouped '☑ Term: short definition' lines under small numbered headings."},
    {"id": "stats_then_opinion", "shape": "Three numbered facts from the brief, then a short opinion and what to do about it."},
]

CTA_FORMULAS = [
    {"id": "save_share_more", "formula": "Three numbered asks: bookmark it (the ⋯ menu), ♻️ repost it for one person who needs it, and where to find more: [a link from the brief, or the author's site]."},
    {"id": "save_this", "formula": "Bookmark this. You'll want it the next time you [task]."},
    {"id": "share_with_someone", "formula": "Share this with someone who [specific trait] (but still [specific gap])."},
    {"id": "repost_for_team", "formula": "♻️ Repost it for the colleague who still [specific pain]."},
    {"id": "ps_guide_link", "formula": "PS: the full step-by-step, with every prompt, is here: [link]", "only_if": "the brief gives that link"},
    {"id": "link_gate", "formula": "The full [prompt/checklist] has [N] parts. It was too long for this caption, so it's here: [link]", "only_if": "the brief gives the link to the full version"},
    {"id": "mission", "formula": "One line on why the author shares this (their mission, from the voice notes), then where to find more: [site]."},
    {"id": "anaphora_close", "formula": "Three short 'I could tell you [exaggeration].' lines, then 'But I won't.' and a plain promise. Write fresh wording every time."},
    {"id": "closing_instruction", "formula": "So close LinkedIn. Open [tool]. Try one [specific task] today."},
    {"id": "authority_quote", "formula": "End on a short quote from a named expert, with their name and role.", "only_if": "the quote and source are in the brief"},
]

FORMATTING = [
    "LinkedIn shows plain text: no **bold**, no # headings, no markdown links. Paste-ready text only.",
    "Line break after every one or two sentences; a blank line between blocks.",
    "Bullet glyphs: ✦ for sub-points under a section, ☑ for checklists, → for sub-items and menu paths, > for rapid-fire lines, '1.' or 'Step 1.' for steps.",
    "Optional separators between big sections: '-----' or '___'.",
    "Prompts to copy go on their own lines in straight double quotes, with [square-bracket placeholders].",
    "Menu paths use arrows: Settings → Capabilities → Memory.",
    "Emoji are rare: ♻️ in a repost line, ⚠️ for a warning, 👉 before a link. Three at most.",
    "No hashtags unless the brief asks for them.",
]

AVOID_PHRASES = [
    "delve", "crucial", "pivotal", "tapestry", "foundational", "here's the thing", "hope this helps",
    "after careful consideration", "provide a quick update", "most people", "robust", "seamless", "realm",
    "landscape", "game-changer", "game changer", "in today's fast-paced world", "let that sink in", "unlock",
    "elevate", "supercharge", "navigate the", "ever-evolving", "testament to", "it's important to note",
]

CONTENT_RULES = [
    "Use only facts, numbers, names, quotes, stories and links the brief gives you. If a point needs a fact you don't have, write around it or put [CHECK: what's needed] so the author fills it in.",
    "Never reuse facts, product claims, links or numbers from the example posts: they belong to another creator and may be out of date.",
    "Never mention the example posts' creator, their newsletter, their sites or their short links. Point readers only to the author's own site.",
    "Never teach cheating (having an AI complete quizzes, courses or certifications, or 'fooling' recruiters) and never fake proof.",
    "No engagement bait ('Comment YES', 'Like if you agree'), no invented urgency, no fake screenshots of real people.",
    "Write new sentences. Borrow the shape of a hook or a list, never the wording of an example post.",
]

GRAPHIC_RULES = [
    "One graphic per post. Portrait 4:5 (1080 × 1350) unless the template needs another shape.",
    "The graphic stands on its own: someone who only sees the image should still learn the thing and want to save it.",
    "Headline of 2 to 8 words that echoes the hook's promise, with one accent word.",
    "Scannable text only: item lines of about 4 to 12 words, no paragraphs, numbers and verbs up front.",
    "The engine adds the footer (author credit and site); don't put credits inside the content.",
    "Motion (GIF or MP4) when the graphic builds step by step: mind maps, roadmaps, decision trees, kinetic text. Static for dense cheat sheets meant to be saved.",
]

# How each archetype seen in the training graphics maps to a template, and when to pick it.
ARCHETYPES = {
    "cheatsheet": "Many sub-topics, each with steps, tables or a prompt: the whole guide on one page.",
    "link_list": "A library of guides, links or one-line tips (styles: pinned, notes, terminal, doc).",
    "prompt_ladder": "A bad, better and best version of the same prompt.",
    "prompt_anatomy": "One prompt, coloured by what to delete, keep and add.",
    "split": "Two opposing lists: delete vs keep, myth vs fact, before vs after.",
    "funnel": "Things to avoid narrowing down to what to do instead.",
    "glossary": "Terms and short definitions (12 to 100).",
    "roadmap": "A path of steps or use cases from start to finish.",
    "cycle": "A set of 5 to 12 items around one big central number, like a curriculum.",
    "mindmap": "One topic with 4 to 8 branches. 'whiteboard' style animates like a hand drawing.",
    "decision_tree": "Pick between four options with two yes/no questions.",
    "card_grid": "N tools, skills or ideas, each with a name, a tag, one line and an example; or feature explainers with labelled facets.",
    "big_number": "A comparison of 2 to 6 real numbers with their source.",
    "badge": "One bold title on a shape: an announcement or a single idea.",
    "bookshelf": "A reading list of 5 to 12 guides drawn as book spines.",
    "kinetic": "A short motion video: one idea per scene, at most 8 words on screen.",
    "screenshot_callout": "Stickers and highlight boxes on a screenshot the author supplies.",
}
