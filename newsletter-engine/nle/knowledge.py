"""What reading the example issues taught the engine, written as reusable patterns.

These describe shape, never wording: titles and openings are abstracted with [placeholders], and every example the
patterns came from belongs to another creator. Measured numbers (lengths, counts) are added by `nle train`.
"""

PRINCIPLES = [
    "One promise per issue or guide. The title names it; the subtitle says who it's for or what they'll be able to do.",
    "Write for one reader, as 'you'. Use 'I' only for experience the author actually gave you.",
    "Very short paragraphs: one or two sentences, often a single line. Half of them are 12 words or fewer.",
    "Be exact: menu paths (Settings → Capabilities), prompts to copy, real numbers from the brief, the order of clicks.",
    "Show, don't describe: a screenshot wherever the reader has to find something on screen, a prompt box whenever "
    "they should type something.",
    "Plain, warm, direct. Confident without hype. Practical steps over opinions.",
    "Every section earns its place: if a reader could skip it and lose nothing, cut it.",
]

STRUCTURE = [
    "Open with an intro section of short lines (usually five to fifteen one-line paragraphs): the problem, the payoff, "
    "and what's inside as a short list (numbered when it is a sequence). No heading on the intro.",
    "End the intro with one button (subscribe, or the next step) and a divider (---).",
    "Body: five to twelve sections. Each opens with a ## heading of three to eight words and ends with a divider.",
    "Inside a section: short paragraphs, then the thing itself: steps (numbered list), a prompt box, a screenshot slot, "
    "a table for comparisons, a callout for the one tip or warning that matters. Use ### headings for items in a series.",
    "Most sections end with one call to action: subscribe, share it with one person, the next guide, or a related issue. "
    "Never two in a row.",
    "Close with a short 'what to do today' section, one or two buttons and a one-line sign-off in the author's voice.",
]

ISSUE_ARCHETYPES = [
    {"id": "beginner_walkthrough", "name": "From-zero walkthrough",
     "use": "the reader has never used the tool or skill",
     "sections": ["what it is, in one plain paragraph", "how to start (account, settings that matter)",
                  "the first useful thing to do, step by step", "three to five everyday uses, each with a prompt",
                  "the mistakes beginners make", "what to learn next"]},
    {"id": "setup_steps", "name": "Set-up steps",
     "use": "a one-time setup the reader must get right",
     "sections": ["why set it up (what it saves)", "Step 1 ... Step N as ## headings: where to click (screenshot), "
                  "what to choose, how to check it worked", "what to do if something looks different", "the first thing to try"]},
    {"id": "use_case_playbook", "name": "One tool, many jobs",
     "use": "the reader knows the tool and wants more from it",
     "sections": ["the situation each use case solves", "one ## section per use case: when to use it, the prompt (prompt box), "
                  "what you get back, one tip", "how to combine them", "the one to try first"]},
    {"id": "prompt_guide", "name": "Prompt guide",
     "use": "the reader gets weak answers and needs better prompts",
     "sections": ["why answers come back generic", "the parts of a good prompt", "the template (prompt box)",
                  "variations for common jobs (### per job, each with a prompt box)", "mistakes to avoid", "a ready-to-copy pack"]},
    {"id": "tips_list", "name": "Numbered tips",
     "use": "many small, independent improvements",
     "sections": ["two or three ## groups", "each tip as a ### heading with one to three short paragraphs and, when useful, "
                  "a prompt box or screenshot", "the three tips to start with"]},
    {"id": "comparison", "name": "Review or comparison",
     "use": "a new model, feature or tool the reader is deciding about",
     "sections": ["what changed, in plain words", "what was tested and how (only tests from the brief)",
                  "results (table)", "who should use what", "how to switch or try it"]},
    {"id": "safety_warning", "name": "Safety warning",
     "use": "a real risk readers should protect themselves from",
     "sections": ["the risk in one concrete example from the brief", "how it works, plainly", "how to protect yourself (checklist)",
                  "what to do if it already happened"]},
    {"id": "essay", "name": "Short essay",
     "use": "one idea the author wants to argue, with practical consequences",
     "sections": ["the claim", "three or four ## sections of reasons, each with an example from the brief",
                  "what the reader should do differently"]},
]

GUIDE_BLUEPRINT = [
    "A guide is evergreen: it stays useful and gets updated, unlike a dated issue.",
    "Start with who it's for, what they'll have at the end and what they need before starting (a short list).",
    "Number the main steps in the ## headings ('1. Turn on file creation'). One job per step.",
    "Each step: what to do, where to click (screenshot slot), what to type (prompt box), how to check it worked.",
    "Put the one warning that would cost the reader time or money in a [!WARNING] callout, at the step where it matters.",
    "End with a checklist (- [ ] items) the reader can tick, then a short 'Questions' section (### per question) if the "
    "brief answers common questions, then cards to the next guides.",
]

TITLE_FORMULAS = [
    "[Outcome], from zero.",
    "[Tool] for [audience].",
    "How to [specific outcome].",
    "[Number] [things] that [result].",
    "Stop [costly habit].",
    "[One word that names the idea].",
]

SUBTITLE_FORMULAS = [
    "[Who it's for]: [what they'll be able to do].",
    "Everything you need to [outcome], in [N] steps:",
    "The [N] [things] I use every week, and exactly how:",
    "[Pain the reader feels]? Here's the fix, step by step.",
]

CALLS_TO_ACTION = {
    "subscribe": "::subscribe, or ::button[Subscribe](page:subscribe): after the opening and at the end.",
    "share": "a one-line ask to send it to one specific kind of person who needs it.",
    "next": "::card[guide:<slug>] or ::button[Read <guide>](guide:<slug>) for the natural next step.",
    "related": "::card[issue:<slug>] when an earlier issue goes deeper.",
    "download": "::download[<file>](<path>) only for files the author provides.",
}

MARKS = [
    "## Section heading / ### Item heading. No # (the title is the page heading).",
    "--- between sections.",
    "```prompt <label>\\n<the prompt>\\n``` for anything the reader should type.",
    "> [!TIP] / [!NOTE] / [!WARNING] / [!KEY] followed by the text, for one callout per section at most.",
    "::screenshot[<exactly what the screenshot must show>] where a screenshot is needed: the author adds the real image.",
    "::button[<label>](<target>), ::card[<issue:slug|guide:slug>], ::subscribe, ::download[<label>](<path>).",
    "Links: [text](https://...) or [text](guide:<slug>), [text](issue:<slug>), [text](page:subscribe).",
    "- [ ] item for a checklist the reader can tick.",
    "Tables with | pipes | for comparisons.",
]

CONTENT_RULES = [
    "Facts, numbers, names, quotes, prices, dates and links come only from the brief. If a sentence needs one the brief "
    "doesn't give, write around it or put [CHECK: what's needed].",
    "Never invent screenshots: use ::screenshot[...] slots describing exactly what the author must capture.",
    "Never promise a cadence, a download or a course that the brief doesn't give.",
    "Never teach cheating on courses, quizzes, certifications or AI detectors, and never suggest fooling anyone.",
    "No engagement bait, no fake urgency, no fake numbers of readers, likes or results.",
]

AVOID_PHRASES = [
    "delve", "game-changer", "game changer", "unlock the power", "unleash", "in today's fast-paced world",
    "in today's digital age", "it's important to note", "seamless", "seamlessly", "elevate", "supercharge",
    "revolutionize", "harness the power", "dive in", "let's dive", "here's the thing", "at the end of the day",
    "navigate the complexities", "landscape", "tapestry", "testament to", "plays a crucial role", "robust",
    "cutting-edge", "transformative", "embark", "journey", "realm",
]
