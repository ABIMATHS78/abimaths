"""The graphic templates the engine learned from the training set: their data schemas, default looks,
layout helpers and sample data. The HTML for each lives in templates/<name>.html.j2."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .schemas import array, boolean, integer, number, obj, string

SIZES = {"portrait": (1080, 1350), "square": (1080, 1080), "landscape": (1200, 675)}
MARKUP_HINT = "Markup: *word* accent colour, **word** bold, ==word== highlight, `code` chip."


@dataclass
class Template:
    name: str
    description: str
    schema: dict
    default_theme: str
    sample: dict
    style_themes: dict = field(default_factory=dict)
    prepare_fn: Callable[[dict, dict], dict] | None = None
    motion_fn: Callable[[dict], dict] | None = None
    shapes: tuple = ("portrait",)
    motion_default: str = "png"     # what the engine renders unless told otherwise

    @property
    def file(self) -> str:
        return f"{self.name}.html.j2"

    def theme_for(self, data: dict) -> str:
        return self.style_themes.get(data.get("style"), self.default_theme)

    def size_for(self, data: dict) -> tuple[int, int]:
        shape = data.get("shape") if data.get("shape") in self.shapes else self.shapes[0]
        return SIZES[shape]

    def prepare(self, data: dict, context: dict) -> dict:
        return self.prepare_fn(data, context) if self.prepare_fn else {}

    def motion_config(self, data: dict) -> dict:
        return self.motion_fn(data) if self.motion_fn else {}


# ---------- cheatsheet ----------
BLOCK = obj({
    "kind": string("text, bullets, steps, table, prompt, compare or note",
                   enum=["text", "bullets", "steps", "table", "prompt", "compare", "note"]),
    "text": string("For text, prompt and note blocks. One or two short lines. " + MARKUP_HINT),
    "items": array(string(), "For bullets and steps: 2 to 6 short lines."),
    "header": array(string(), "For tables: optional 2 or 3 column titles."),
    "rows": array(array(string()), "For tables: 2 to 5 rows of 2 or 3 short cells."),
    "old": string("For compare: the weak version."),
    "new": string("For compare: the better version."),
}, ["kind"])

CHEATSHEET = Template(
    name="cheatsheet",
    description="The whole guide on one page: 4 to 9 panels, each with steps, bullets, a small table or a prompt to copy.",
    schema=obj({
        "title": string("2 to 7 words. Wrap the key word in *asterisks* for the accent colour."),
        "subtitle": string("Optional one line under the title."),
        "headline_style": string("serif (cream look) or condensed (heavy caps)", enum=["serif", "condensed"], default="serif"),
        "number_panels": boolean("Number the panel titles (1., 2., ...).", default=True),
        "headline_in_column": boolean("Put the headline at the top of the first column, beside the panels, instead of across the page. "
                                      "Split it into lines with |.", default=False),
        "columns": integer("Number of columns: 1, 2 or 3. Usually 2.", default=2),
        "panels": array(obj({
            "title": string("2 to 6 words."),
            "tone": string("normal, dark (a black warning panel) or accent", enum=["normal", "dark", "accent"], default="normal"),
            "column": integer("Optional: the column this panel goes in (1, 2 or 3). Set it on every panel, or on none."),
            "blocks": array(BLOCK, "1 to 3 blocks."),
        }, ["title", "blocks"]), "4 to 9 panels."),
    }, ["title", "panels"]),
    default_theme="cream",
    sample={
        "title": "*Prompting,* explained in one page",
        "subtitle": "Keep it. Use it on your next chat.",
        "panels": [
            {"title": "Start every chat with this", "blocks": [
                {"kind": "prompt", "text": "Goal: [the outcome, in one line]\nAudience: [who reads it]\nDone when: [what finished looks like]\nAsk me what's missing first."}]},
            {"title": "Swap the rules", "blocks": [
                {"kind": "compare", "old": "Don't touch the budget tab.", "new": "Check with me before editing the budget tab: finance signed it off."}]},
            {"title": "Pick the right effort", "blocks": [
                {"kind": "table", "header": ["Task", "Effort"], "rows": [["Fix grammar", "Low"], ["Draft an email", "Medium"], ["Plan a project", "High"]]}]},
            {"title": "Give it a finish line", "blocks": [
                {"kind": "steps", "items": ["Say what 'done' looks like", "Say what it may change", "Ask for a 3-line recap"]}]},
            {"title": "Stop doing this", "tone": "dark", "blocks": [
                {"kind": "bullets", "items": ["Arguing with a bad answer", "Pasting 40 rules 'just in case'", "Switching models mid-chat"]}]},
            {"title": "When it goes wrong", "blocks": [
                {"kind": "text", "text": "Edit the prompt *above* the bad answer instead of replying to it."},
                {"kind": "note", "text": "New topic? New chat."}]},
        ],
    },
)


# ---------- link_list ----------
LINK_LIST = Template(
    name="link_list",
    description="A library of guides, links or one-line tips. Styles: pinned (white page, red pins), notes (phone notes app), terminal (black, monospace links), doc (plain document with a highlighted title).",
    schema=obj({
        "title": string("2 to 6 words."),
        "subtitle": string("Optional one line."),
        "aside": string("Optional small parenthetical after the title, e.g. '(you didn't know)'."),
        "style": string("pinned, notes, terminal or doc", enum=["pinned", "notes", "terminal", "doc"], default="pinned"),
        "item_prefix": string("Optional word before each number, e.g. 'Lesson' gives 'Lesson 1:'. Empty for plain numbers."),
        "numbered": boolean("Number the items.", default=True),
        "groups": array(obj({
            "label": string("Optional group heading, e.g. 'Level 1'."),
            "items": array(obj({
                "text": string("The title or tip. **bold** key words in doc style."),
                "link": string("Optional link or short URL, exactly as given in the brief."),
                "checked": boolean("Notes style: show a filled check.", default=True),
            }, ["text"]), "Items in this group."),
        }, ["items"]), "1 to 4 groups; 5 to 25 items in total."),
        "footnote": string("Optional one or two lines at the bottom. " + MARKUP_HINT),
    }, ["title", "groups"]),
    default_theme="paper",
    style_themes={"pinned": "paper", "notes": "notes", "terminal": "ink", "doc": "paper"},
    sample={
        "title": "Free AI course",
        "subtitle": "For complete beginners who want to stay in the game",
        "style": "pinned",
        "item_prefix": "Lesson",
        "groups": [{"items": [
            {"text": "Your first useful prompt", "link": "example.com/first-prompt"},
            {"text": "Make AI write like you", "link": "example.com/your-voice"},
            {"text": "Turn notes into actions", "link": "example.com/meeting-notes"},
            {"text": "Check the numbers it gives you", "link": "example.com/check-numbers"},
            {"text": "Build your first skill", "link": "example.com/first-skill"},
            {"text": "Plan a presentation in 10 minutes", "link": "example.com/presentations"},
            {"text": "Research without the rabbit holes", "link": "example.com/research"},
        ]}],
        "footnote": "Read one a day instead of scrolling.",
    },
)


# ---------- prompt_ladder ----------
PROMPT_LADDER = Template(
    name="prompt_ladder",
    description="The same request written as a bad, a better and a best prompt, each in a chat box.",
    schema=obj({
        "title": string("Optional headline above the boxes."),
        "rungs": array(obj({
            "label": string("e.g. 'Bad prompt'."),
            "tone": string("bad, better or best", enum=["bad", "better", "best"]),
            "prompt": string("The prompt exactly as typed. New lines allowed. Wrap key phrases in ==double equals== to highlight them."),
        }, ["label", "tone", "prompt"]), "2 or 3 rungs, worst first."),
        "model_label": string("Optional text in each box's footer, e.g. 'Opus 5.5 · Medium'."),
    }, ["rungs"]),
    default_theme="paper",
    sample={
        "rungs": [
            {"label": "Bad prompt", "tone": "bad", "prompt": "Write a LinkedIn post about AI. Make it good. Go."},
            {"label": "Better prompt", "tone": "better", "prompt": "Write a LinkedIn post for ==finance managers== about using AI to check spreadsheets. Keep it under 200 words."},
            {"label": "Best prompt", "tone": "best", "prompt": "==Goal:== a LinkedIn post that gets finance managers to try AI on one spreadsheet this week.\n\n==Context:== I run AI training. My readers are busy and sceptical. Attached: two of my past posts.\n\nAsk me 3 questions before you write. End with a 2-line recap of the angle you chose."},
        ],
        "model_label": "Opus 5.5 · Medium",
    },
)


# ---------- prompt_anatomy ----------
PROMPT_ANATOMY = Template(
    name="prompt_anatomy",
    description="One prompt split into coloured parts: what to delete, what to keep and what to add, with a reason beside each.",
    schema=obj({
        "title": string("Headline. Wrap the key phrase in __double underscores__ for a red underline."),
        "subtitle": string("Optional line under the title, often in parentheses."),
        "segments": array(obj({
            "tone": string("delete, keep or add", enum=["delete", "keep", "add"]),
            "label": string("One word shown beside the block, e.g. DELETE."),
            "note": string("Why, in 3 to 8 words."),
            "text": string("The prompt text of this part (2 to 12 short lines)."),
        }, ["tone", "label", "note", "text"]), "2 to 4 segments."),
        "model_label": string("Optional model name shown in the box footer."),
    }, ["title", "segments"]),
    default_theme="cream",
    sample={
        "title": "__Delete half__ of your prompt",
        "subtitle": "(the new models already do it)",
        "segments": [
            {"tone": "delete", "label": "DELETE", "note": "The model does this itself", "text": "You are a world-class marketer.\nThink step by step.\nTake a deep breath.\nDo not hallucinate.\nUse short sentences. No jargon."},
            {"tone": "keep", "label": "KEEP", "note": "The situation and the goal", "text": "Three customers asked for refunds this week. Goal: keep them, and fix the reason they left."},
            {"tone": "add", "label": "ADD", "note": "A finish line and a check-in", "text": "Ask me before offering a discount. Finish all three replies before you stop."},
        ],
        "model_label": "Opus 5.5 · High",
    },
)


# ---------- split ----------
SIDE = obj({
    "label": string("Column heading, e.g. DELETE or KEEP."),
    "tone": string("bad (red), good (green), neutral or accent", enum=["bad", "good", "neutral", "accent"]),
    "items": array(string("One line. **bold** the key words."), "3 to 14 items."),
}, ["label", "tone", "items"])

SPLIT = Template(
    name="split",
    description="Two opposing lists side by side: delete vs keep, myth vs fact, before vs after.",
    schema=obj({
        "title": string("2 to 8 words."),
        "subtitle": string("Optional."),
        "left": SIDE,
        "right": SIDE,
        "numbered": boolean("Number the items instead of bullets.", default=False),
    }, ["title", "left", "right"]),
    default_theme="ink",
    sample={
        "title": "Clean up your AI setup",
        "left": {"label": "Delete", "tone": "bad", "items": [
            "The **'about me' file** you rewrote five times", "**Prompt libraries** you never open",
            "Rules written for **last year's models**", "**Every connector** switched on 'just in case'",
            "Thirty **projects** with three files each", "**Screenshots** of other people's prompts"]},
        "right": {"label": "Keep", "tone": "good", "items": [
            "**One clear goal** per chat", "**Memory on**, kept short", "**One project** per client",
            "A **finish line** in every prompt", "A reminder to **clean up again** in six months"]},
    },
)


# ---------- funnel ----------
def _hex_to_rgb(color: str) -> tuple:
    return tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))


def ramp(stops: list[str], n: int) -> list[str]:
    """n colours evenly spread along a gradient through the given stops."""
    if n <= 1:
        return [stops[0]]
    out = []
    for i in range(n):
        pos = i / (n - 1) * (len(stops) - 1)
        lo = min(int(pos), len(stops) - 2)
        frac = pos - lo
        a, b = _hex_to_rgb(stops[lo]), _hex_to_rgb(stops[lo + 1])
        out.append("#{:02x}{:02x}{:02x}".format(*(round(x + (y - x) * frac) for x, y in zip(a, b))))
    return out


def ink_on(color: str) -> str:
    r, g, b = (c / 255 for c in _hex_to_rgb(color))
    return "#FFFFFF" if 0.299 * r + 0.587 * g + 0.114 * b < 0.6 else "#1A1816"


FUNNEL_RAMPS = {
    None: (["#7A1714", "#C42B23", "#E2582F", "#EF9F3C", "#F3CD6A", "#F5E6AE"],
           ["#E3F0D6", "#B5D99A", "#7CBC5F", "#3F8A31", "#25591D"]),
    # The /learn look has no green: what to avoid runs from deep red to peach, what to do from beige to charcoal.
    "learn": (["#6E1512", "#B82828", "#D02E2E", "#D06E2E", "#E9A36B", "#F6D9B8"],
              ["#F6EED2", "#E2D5B2", "#A8A090", "#5C5752", "#2F2D2D"]),
}


def _funnel(data: dict, context: dict) -> dict:
    top, bottom = data.get("top_items", []), data.get("bottom_items", [])
    top_stops, bottom_stops = FUNNEL_RAMPS.get(context.get("theme"), FUNNEL_RAMPS[None])
    top_colors = ramp(top_stops, len(top))
    bottom_colors = ramp(bottom_stops, len(bottom))
    spread = lambda n, a, b: [a + (b - a) * i / max(1, n - 1) for i in range(n)]  # noqa: E731
    return {
        "top_bars": [{"text": t, "bg": c, "fg": ink_on(c), "w": w} for t, c, w in zip(top, top_colors, spread(len(top), 30, 100))],
        "bottom_bars": [{"text": t, "bg": c, "fg": ink_on(c), "w": w} for t, c, w in zip(bottom, bottom_colors, spread(len(bottom), 96, 34))],
    }


FUNNEL = Template(
    name="funnel",
    description="What to stop (a widening stack of red to yellow bars) narrowing into what to do instead (green bars).",
    schema=obj({
        "title": string("2 to 6 words."),
        "subtitle": string("Optional, e.g. '(and everyone can tell)'."),
        "top_label": string("Label for the top stack.", default="Avoid this:"),
        "top_items": array(string("2 to 8 words."), "4 to 12 things to stop, the worst first."),
        "bottom_label": string("Label for the bottom stack.", default="Do this instead:"),
        "bottom_items": array(string("2 to 8 words."), "4 to 12 fixes."),
    }, ["title", "top_items", "bottom_items"]),
    default_theme="cream",
    prepare_fn=_funnel,
    sample={
        "title": "You sound like AI.",
        "subtitle": "(and your readers can tell)",
        "top_items": ['"Delve"', '"In today\'s fast-paced world"', '"Here\'s the thing"', "Three adjectives in a row",
                      '"It\'s not X, it\'s Y"', "An em dash in every sentence", '"Hope this helps"', "A grand closing line"],
        "bottom_items": ["Read it out loud once", "Cut the first sentence", "Name one real example",
                         "Use the words your clients use", "Keep one opinion", "End with a question you'd answer"],
    },
)


# ---------- glossary ----------
GLOSSARY = Template(
    name="glossary",
    description="Numbered terms with one-line definitions in two columns (12 to 100 entries).",
    schema=obj({
        "title": string("First headline line."),
        "title_line2": string("Optional second line, shown in the accent colour."),
        "entries": array(obj({
            "term": string("1 to 3 words."),
            "definition": string("At most 12 words."),
        }, ["term", "definition"]), "12 to 100 entries, in alphabetical order."),
    }, ["title", "entries"]),
    default_theme="cream",
    sample={
        "title": "30 AI words",
        "title_line2": "you should know",
        "entries": [{"term": t, "definition": d} for t, d in [
            ("Agent", "AI that takes actions for you, not just answers"), ("Artifact", "A page or app the AI builds and you can share"),
            ("Brief", "A summary that lets a new chat continue the work"), ("Cache", "A saved copy of your chat, so re-reading is cheap"),
            ("Connector", "A link between the AI and one of your apps"), ("Context", "Everything the AI can see in this chat"),
            ("Context window", "The most text the AI can hold at once"), ("Cutoff", "The last date the model's training data covers"),
            ("Effort", "How long the model thinks before answering"), ("Em dash", "The long dash that makes text look AI-written"),
            ("Goal", "The outcome you want, in one line"), ("Hallucination", "A confident answer that is simply wrong"),
            ("HTML page", "Ask for this instead of a Word file"), ("Instructions", "Standing rules the AI reads before every chat"),
            ("LLM", "Large language model: the engine behind the chat"), ("MCP", "A standard way to plug your tools into AI"),
            ("Memory", "What the AI remembers about you between chats"), ("Model", "The version of the AI doing the work"),
            ("One-shot", "One prompt, one finished result"), ("Plugin", "A pack of skills you add in one click"),
            ("Project", "A space with shared files and instructions"), ("Prompt", "What you ask, with the goal and context"),
            ("Reasoning", "The model thinking step by step before replying"), ("Skill", "Saved instructions the AI loads when needed"),
            ("Slop", "Generic AI output with no taste or point of view"), ("Subagent", "A helper AI that does one part of a task"),
            ("System prompt", "The hidden instructions behind every reply"), ("Token", "About three quarters of a word: how AI is billed"),
            ("Usage limit", "The point where your plan says stop for now"), ("Vibe coding", "Building software by describing it, not writing code"),
        ]],
    },
)


# ---------- roadmap ----------
STOP = obj({
    "heading": string("2 to 4 words."),
    "lines": array(string("Short line."), "1 to 3 lines."),
    "code": string("Optional command or prompt shown as a monospace chip."),
    "result": string("Optional bold outcome line, e.g. 'A draft in 5 minutes.'"),
}, ["heading", "lines"])

ROADMAP = Template(
    name="roadmap",
    description="A winding path through 3 to 7 numbered stops, from a start box to a black finish box.",
    schema=obj({
        "title": string("2 to 8 words. Wrap the key word in *asterisks*."),
        "subtitle": string("Optional second headline line, e.g. '(for your business)'."),
        "start": obj({"heading": string(default="Start"), "lines": array(string(), "1 to 3 lines.")}, ["lines"]),
        "stops": array(STOP, "3 to 7 stops."),
        "finish": obj({"heading": string(default="Finish"), "lines": array(string(), "1 to 3 lines.")}, ["lines"]),
    }, ["title", "stops"]),
    default_theme="cream",
    sample={
        "title": "5 ways to use *AI* at work",
        "subtitle": "(starting Monday)",
        "start": {"heading": "Start", "lines": ["Open your AI app.", "Pick one task you do every week."]},
        "stops": [
            {"heading": "Meetings", "lines": ["Paste the transcript."], "code": "turn this into actions", "result": "Owners and dates, done."},
            {"heading": "Email", "lines": ["Paste the thread and your goal."], "result": "A reply in your tone."},
            {"heading": "Spreadsheets", "lines": ["Upload the CSV.", "Ask what changed since last month."], "result": "Three numbers that matter."},
            {"heading": "Research", "lines": ["Ask for sources you can check."], "code": "cite every claim", "result": "A one-page brief."},
            {"heading": "Slides", "lines": ["Ask for the deck, not the outline."], "result": "Ten slides to edit."},
        ],
        "finish": {"heading": "Finish", "lines": ["Pick one.", "Block 20 minutes.", "Use it today."]},
    },
)


# ---------- cycle ----------
def rounded_rect_loop(cx: float, cy: float, w: float, h: float, r: float, samples: int = 720) -> list[tuple]:
    """Points evenly spaced along a rounded rectangle, clockwise from the middle of the top edge."""
    r = min(r, w / 2, h / 2)
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    segments = [  # (kind, data): straight lines and quarter arcs, clockwise from top-middle
        ("line", (cx, y0, x1 - r, y0)), ("arc", (x1 - r, y0 + r, -90, 0)),
        ("line", (x1, y0 + r, x1, y1 - r)), ("arc", (x1 - r, y1 - r, 0, 90)),
        ("line", (x1 - r, y1, x0 + r, y1)), ("arc", (x0 + r, y1 - r, 90, 180)),
        ("line", (x0, y1 - r, x0, y0 + r)), ("arc", (x0 + r, y0 + r, 180, 270)),
        ("line", (x0 + r, y0, cx, y0)),
    ]
    lengths = [math.dist(d[:2], d[2:]) if k == "line" else math.pi * r / 2 for k, d in segments]
    total = sum(lengths)
    points = []
    for i in range(samples):
        target = total * i / samples
        for (kind, d), length in zip(segments, lengths):
            if target <= length or (kind, d) == segments[-1]:
                t = target / length if length else 0
                if kind == "line":
                    points.append((d[0] + (d[2] - d[0]) * t, d[1] + (d[3] - d[1]) * t))
                else:
                    angle = math.radians(d[2] + (d[3] - d[2]) * t)
                    points.append((d[0] + r * math.cos(angle), d[1] + r * math.sin(angle)))
                break
            target -= length
    return points


def _cycle_layout(data: dict, context: dict) -> dict:
    n = max(1, len(data.get("items", [])))
    width = context["width"]
    loop = rounded_rect_loop(width / 2, 650, width - 330, 1020, 230)
    idx = [round(i * len(loop) / n) % len(loop) for i in range(n)]
    card_w = 290 if n <= 7 else (262 if n <= 9 else 236)
    return {
        "positions": [{"x": round(loop[i][0]), "y": round(loop[i][1])} for i in idx],
        "loop": [[round(x, 1), round(y, 1)] for x, y in loop],
        "loop_idx": idx,
        "card_w": card_w,
    }


CYCLE = Template(
    name="cycle",
    description="5 to 12 cards in a circle around one big central number, joined by arrows, like a curriculum.",
    schema=obj({
        "center_big": string("The big number or 1 to 2 characters, e.g. '11'."),
        "center_title": string("2 to 5 words under it. " + MARKUP_HINT),
        "center_sub": string("Optional, e.g. '(for free)'."),
        "items": array(obj({
            "title": string("2 to 5 words."),
            "tag": string("Optional small pill, e.g. '8 min read'."),
            "text": string("At most 14 words."),
        }, ["title", "text"]), "5 to 12 items, in order."),
    }, ["center_big", "center_title", "items"]),
    default_theme="cream",
    prepare_fn=_cycle_layout,
    sample={
        "center_big": "8",
        "center_title": "lessons to learn **AI**",
        "center_sub": "(for free)",
        "items": [
            {"title": "AI for beginners", "tag": "6 min read", "text": "Open the app tonight. Ask it one real question."},
            {"title": "Better prompts", "tag": "8 min read", "text": "A goal, the context, and a finish line."},
            {"title": "Your own voice", "tag": "7 min read", "text": "Give it three of your posts before it writes."},
            {"title": "Check its numbers", "tag": "5 min read", "text": "Ask where every figure came from."},
            {"title": "Meetings to actions", "tag": "6 min read", "text": "Paste the transcript, get owners and dates."},
            {"title": "Your first skill", "tag": "9 min read", "text": "Turn a task you repeat into one command."},
            {"title": "Research briefs", "tag": "8 min read", "text": "Sources you can click, in one page."},
            {"title": "Keep it private", "tag": "4 min read", "text": "What never to paste, and the settings to change."},
        ],
    },
)


# ---------- mindmap ----------
def _mindmap_layout(data: dict, context: dict) -> dict:
    n = max(1, len(data.get("branches", [])))
    slots_by_n = {
        3: [(-0.62, -0.55), (0.62, -0.55), (0.0, 0.72)],
        4: [(-0.62, -0.58), (0.62, -0.58), (-0.62, 0.58), (0.62, 0.58)],
        5: [(0.0, -0.78), (0.64, -0.34), (0.64, 0.5), (-0.64, 0.5), (-0.64, -0.34)],
        6: [(-0.6, -0.7), (0.6, -0.7), (0.68, 0.0), (0.6, 0.72), (-0.6, 0.72), (-0.68, 0.0)],
        7: [(0.0, -0.8), (0.64, -0.5), (0.66, 0.16), (0.42, 0.8), (-0.42, 0.8), (-0.66, 0.16), (-0.64, -0.5)],
        8: [(-0.38, -0.8), (0.38, -0.8), (0.67, -0.24), (0.67, 0.38), (0.38, 0.86), (-0.38, 0.86), (-0.67, 0.38), (-0.67, -0.24)],
    }
    slots = slots_by_n.get(n) or slots_by_n[min(max(n, 3), 8)]
    return {"slots": [{"x": round(50 + x * 50, 2), "y": round(50 + y * 50, 2)} for x, y in slots]}


MINDMAP = Template(
    name="mindmap",
    description="One topic in a central box with 3 to 8 branches of short notes. 'whiteboard' style looks hand-drawn and animates like a drawing; 'clean' is crisp.",
    schema=obj({
        "title": string("2 to 7 words. " + MARKUP_HINT),
        "subtitle": string("Optional, e.g. '(from scratch)'."),
        "center": string("1 to 3 words in the central box."),
        "style": string("whiteboard or clean", enum=["whiteboard", "clean"], default="whiteboard"),
        "branches": array(obj({
            "heading": string("2 to 6 words, e.g. 'MIN 1 - Pick your tool'."),
            "items": array(string("At most 9 words."), "1 to 4 lines."),
        }, ["heading", "items"]), "3 to 8 branches, in reading order."),
    }, ["title", "center", "branches"]),
    default_theme="whiteboard",
    style_themes={"whiteboard": "whiteboard", "clean": "paper"},
    prepare_fn=_mindmap_layout,
    motion_fn=lambda data: {"perItem": 0.3, "maxBuild": 11, "minBuild": 5, "dur": 0.7, "hold": 3},
    motion_default="gif",
    sample={
        "title": "Set up your AI in 10 minutes",
        "subtitle": "(from scratch)",
        "center": "YOUR AI",
        "style": "whiteboard",
        "branches": [
            {"heading": "MIN 1 - Pick one app", "items": ["Install the desktop app", "Sign in with a work account"]},
            {"heading": "MIN 2 - Clear old rules", "items": ["Empty the instructions box", "Delete skills you never use"]},
            {"heading": "MIN 3 - Memory", "items": ["Turn it on", "Keep it short"]},
            {"heading": "MIN 4 - Goals, not steps", "items": ['"Outcome: [one line]"', '"Repeat the goal back first"']},
            {"heading": "MIN 5 - Ask for a page", "items": ['"Make it an HTML page"', "Share a link, not a file"]},
            {"heading": "MIN 6-10 - One real task", "items": ["Pick a task you do weekly", "Finish it start to end"]},
        ],
    },
)


# ---------- decision_tree ----------
LEAF = obj({
    "title": string("The answer, 1 to 3 words."),
    "subtitle": string("Optional one line under it."),
    "steps": array(string("At most 6 words."), "1 to 3 short steps."),
    "example": string("Optional prompt example (at most 40 words)."),
    "example_label": string("Label above the example.", default="Prompt example"),
}, ["title", "steps"])
BRANCH = obj({"question": string("A yes/no question, at most 10 words."), "no": LEAF, "yes": LEAF}, ["question", "no", "yes"])

DECISION_TREE = Template(
    name="decision_tree",
    description="Choose between four options with a root yes/no question and two follow-up questions.",
    schema=obj({
        "title": string("2 to 5 words. Wrap one word in [[double brackets]] to box it in the accent colour."),
        "question": string("The first yes/no question, at most 10 words."),
        "no": BRANCH,
        "yes": BRANCH,
        "no_label": string(default="NO"),
        "yes_label": string(default="YES"),
    }, ["title", "question", "no", "yes"]),
    default_theme="ink",
    motion_default="mp4",
    sample={
        "title": "[[Which]] AI tool, when?",
        "question": "Does the task need more than a quick answer?",
        "no": {"question": "Is it about your own files?",
               "no": {"title": "Chat", "subtitle": "Fast answers", "steps": ["No files needed", "Turn on web search"], "example": "Explain [topic] to a new hire in 5 bullets."},
               "yes": {"title": "Projects", "subtitle": "Your documents", "steps": ["Upload once", "Ask across files"], "example": "Using the files here, list every deadline in March."}},
        "yes": {"question": "Will it take more than an hour by hand?",
                "no": {"title": "Deep chat", "subtitle": "Think it through", "steps": ["Effort on high", "Name the finish line"], "example": "Plan [project]. Ask me 3 questions first."},
                "yes": {"title": "Agent", "subtitle": "Let it work", "steps": ["Give the goal", "Check in at the end"], "example": "Outcome: [result]. Keep going until the checklist is complete. Check with me before deleting anything."}},
    },
)


# ---------- card_grid ----------
def _card_columns(data: dict, context: dict) -> dict:
    n = len(data.get("cards", []))
    if data.get("style") == "facets":
        return {"cols": 1 if n <= 2 else 2}
    return {"cols": 2 if n <= 4 else (3 if n <= 9 else 4)}


CARD_GRID = Template(
    name="card_grid",
    description="3 to 12 cards: tools, skills or ideas with a name, a tag, one line and an example. 'facets' style explains 2 to 6 features with labelled facets.",
    schema=obj({
        "title": string("Headline. Wrap the key noun in [[double brackets]] for an accent pill."),
        "subtitle": string("Optional one line."),
        "style": string("cards or facets", enum=["cards", "facets"], default="cards"),
        "numbered": boolean("Show big numbers on the cards.", default=True),
        "cards": array(obj({
            "title": string("1 to 4 words."),
            "tag": string("Optional chip, e.g. a /command."),
            "text": string("At most 16 words."),
            "quote": string("Optional example in italics, at most 10 words."),
            "facets": array(obj({"label": string("e.g. IN PLAIN ENGLISH"), "text": string("At most 20 words.")}, ["label", "text"]),
                            "facets style only: 2 to 4 facets."),
        }, ["title"]), "3 to 12 cards."),
    }, ["title", "cards"]),
    default_theme="cream",
    prepare_fn=_card_columns,
    sample={
        "title": "10 free AI [[skills]]",
        "subtitle": "Upload once. They load themselves when you need them.",
        "cards": [
            {"title": "Meeting to actions", "tag": "/meeting-to-actions", "text": "Turns a transcript into decisions, owners and dates.", "quote": "\"Action items from this call.\""},
            {"title": "Check the numbers", "tag": "/check-the-numbers", "text": "Profiles a spreadsheet and flags what looks off.", "quote": "\"Anything odd in this CSV?\""},
            {"title": "Skill maker", "tag": "/skill-maker", "text": "Writes and tests a new skill from your description.", "quote": "\"Make this a skill.\""},
            {"title": "News explainer", "tag": "/news-explainer", "text": "Explains a story plainly, with what it means for you.", "quote": "\"What does this mean for me?\""},
            {"title": "Presentation planner", "tag": "/presentation-planner", "text": "Plans a talk slide by slide before you open the app.", "quote": "\"Plan a 10-minute talk.\""},
            {"title": "Job application", "tag": "/job-application", "text": "Tailors your CV and letter to one job ad.", "quote": "\"Fit my CV to this role.\""},
        ],
    },
)


# ---------- big_number ----------
def _bars(data: dict, context: dict) -> dict:
    bars = data.get("bars", [])
    values = [abs(float(b.get("value") or 0)) for b in bars]
    top = max(values) if values and max(values) > 0 else 1
    n = max(1, len(bars))
    column = (context["width"] - 128) * 0.9 / n
    longest = max([len(str(b.get("display", ""))) for b in bars] or [1])
    base = {1: 150, 2: 132, 3: 100, 4: 80}.get(n, 64)
    return {"heights": [max(0.04, v / top) for v in values], "fig_size": round(min(base, column / (0.6 * max(longest, 1))))}


BIG_NUMBER = Template(
    name="big_number",
    description="A bar comparison of 2 to 5 real numbers, with huge figures under the bars and the source.",
    schema=obj({
        "title": string("e.g. '*Team A* vs Team B'. " + MARKUP_HINT),
        "subtitle": string("What is measured, one line. **bold** one word."),
        "bars": array(obj({
            "label": string("What this bar is."),
            "value": number("The number used for the bar's height."),
            "display": string("The figure as printed, e.g. '+$15.6B' or '42%'."),
            "caption": string("Small line under the figure."),
            "highlight": boolean("Draw in the accent colour.", default=False),
        }, ["label", "value", "display"]), "2 to 5 bars."),
        "source": string("Where the numbers come from. Required: never chart numbers without a source."),
    }, ["title", "bars", "source"]),
    default_theme="paper",
    prepare_fn=_bars,
    motion_default="mp4",
    sample={
        "title": "*With AI* vs without",
        "subtitle": "Minutes to write one client report (sample data)",
        "bars": [
            {"label": "Without AI", "value": 95, "display": "95 min", "caption": "drafting from a blank page"},
            {"label": "With AI", "value": 25, "display": "25 min", "caption": "editing a first draft", "highlight": True},
        ],
        "source": "Sample data: replace with your own numbers and source",
    },
)


# ---------- badge ----------
BADGE = Template(
    name="badge",
    description="One bold title on a cream shape over a solid colour: an announcement or a single idea.",
    schema=obj({
        "title": string("2 to 6 words."),
        "kicker": string("Optional small line above the title."),
        "subtitle": string("Optional small line below the title."),
        "shape": string("portrait, square or landscape", enum=["portrait", "square", "landscape"], default="portrait"),
    }, ["title"]),
    default_theme="terracotta",
    shapes=("portrait", "square", "landscape"),
    sample={"kicker": "A new series", "title": "AI for Absolute Beginners", "subtitle": "No jargon, one step at a time"},
)


# ---------- bookshelf ----------
SPINE_COLORS = {
    "orange": ("#C8582E", "#F3E6D8"), "cream": ("#E9E1D0", "#2B2622"), "blue": ("#3F5A78", "#F1EADD"),
    "olive": ("#5E6440", "#F1EADD"), "mustard": ("#D99A2B", "#2B2216"), "brick": ("#B2463A", "#F3E6D8"),
    "sage": ("#9DAE8C", "#26301F"), "brown": ("#3A2A24", "#EFE3D2"), "charcoal": ("#2E2D2B", "#EFE7DA"),
    "plum": ("#5B3A55", "#F1E6EE"),
}
SPINE_ORDER = ["orange", "cream", "blue", "olive", "mustard", "brick", "sage", "brown", "charcoal", "plum"]


def _shelf(data: dict, context: dict) -> dict:
    books = data.get("books", [])
    n = max(1, len(books))
    usable = context["width"] - 40
    weights = [1 + min(len(b.get("title", "")), 34) / 60 for b in books] or [1]
    total = sum(weights)
    spines = []
    for index, (book, weight) in enumerate(zip(books, weights)):
        color = book.get("color") if book.get("color") in SPINE_COLORS else SPINE_ORDER[index % len(SPINE_ORDER)]
        bg, ink = SPINE_COLORS[color]
        spines.append({
            "width": round(usable * weight / total),
            "height": 0.9 + 0.1 * ((index * 37) % 7) / 6,
            "bg": bg, "ink": ink, "tilt": 0,
        })
    return {"spines": spines, "count": n}


BOOKSHELF = Template(
    name="bookshelf",
    description="A reading list of 5 to 12 guides drawn as cloth book spines.",
    schema=obj({
        "caption": string("Optional line above the books."),
        "books": array(obj({
            "title": string("2 to 6 words."),
            "color": string("Spine colour", enum=SPINE_ORDER),
            "emblem": string("Small mark near the foot", enum=["burst", "rule", "network", "dot", "square", "chain", "warning", "none"]),
        }, ["title"]), "5 to 12 books."),
    }, ["books"]),
    default_theme="cream",
    prepare_fn=_shelf,
    sample={"books": [
        {"title": "AI for Beginners", "emblem": "burst"}, {"title": "Prompts That Work", "emblem": "rule"},
        {"title": "Save Hours on Email", "emblem": "network"}, {"title": "Clean Up AI Slop", "emblem": "square"},
        {"title": "Build Your First Skill", "emblem": "dot"}, {"title": "AI for Finance Teams", "emblem": "rule"},
        {"title": "Meetings to Actions", "emblem": "square"}, {"title": "Research Without Rabbit Holes", "emblem": "chain"},
        {"title": "Keep Your Data Private", "emblem": "warning"},
    ]},
)


# ---------- kinetic ----------
WORD_POP = 0.45      # how long one word takes to pop in
READ_AFTER = 1.5     # seconds a scene stays fully readable after its last word (or its sub-line) lands


def _kinetic(data: dict, context: dict) -> dict:
    """Each scene lasts until its words have popped in and stayed readable; a sub-line comes in after the words
    and gets its own reading time."""
    scenes = data.get("scenes", [])
    timings = []
    t = 0.25
    for scene in scenes:
        word_count = max(1, len(str(scene.get("text", "")).split()))
        word_step = round(min(0.16, 0.9 / word_count), 3)
        words_done = 0.15 + (word_count - 1) * word_step + WORD_POP
        sub_at = words_done + 0.1
        content_done = sub_at + 0.5 if scene.get("sub") else words_done
        length = max(2.2, min(5.0, content_done + READ_AFTER + 0.35))
        timings.append({"start": round(t, 2), "end": round(t + length, 2), "word_step": word_step,
                        "sub_at": round(t + sub_at, 2)})
        t += length
    return {"timings": timings, "total": round(t + 0.4, 2)}


def _kinetic_motion(data: dict) -> dict:
    total = _kinetic(data, {})["total"]
    return {"duration": total + 1.2, "hold": 0}


KINETIC = Template(
    name="kinetic",
    description="A short motion video: one idea per scene, at most 8 words on screen, words popping in. Best as MP4.",
    schema=obj({
        "scenes": array(obj({
            "kicker": string("Optional small line above, e.g. 'Step 1'."),
            "text": string("At most 8 words. Wrap one or two words in *asterisks* for the accent."),
            "sub": string("Optional small line below."),
        }, ["text"]), "3 to 7 scenes. The first is the hook, the last is the call to action."),
        "shape": string("portrait or square", enum=["portrait", "square"], default="portrait"),
    }, ["scenes"]),
    default_theme="cream",
    shapes=("portrait", "square"),
    prepare_fn=_kinetic,
    motion_fn=_kinetic_motion,
    motion_default="mp4",
    sample={"scenes": [
        {"kicker": "Before you post", "text": "Your post *sounds like AI*", "sub": "and readers notice"},
        {"kicker": "Fix 1", "text": "Delete the *first sentence*"},
        {"kicker": "Fix 2", "text": "Name *one real* example"},
        {"kicker": "Fix 3", "text": "Read it *out loud* once"},
        {"kicker": "Your turn", "text": "Try one fix *today*"},
    ]},
)


# ---------- screenshot_callout ----------
def _screenshot(data: dict, context: dict) -> dict:
    path = Path(str(data.get("image") or "")).expanduser()
    info = {"image_uri": "", "image_ratio": 0.8}
    if str(data.get("image") or "") and path.is_file():
        info["image_uri"] = path.resolve().as_uri()
        try:
            from PIL import Image

            with Image.open(path) as image:
                info["image_ratio"] = image.width / image.height
        except Exception:
            pass
    box_w, box_h = context["width"] * 0.88, (context["height"] - 76) * 0.88
    ratio = info["image_ratio"]
    if box_w / box_h > ratio:
        info["frame_h"], info["frame_w"] = round(box_h), round(box_h * ratio)
    else:
        info["frame_w"], info["frame_h"] = round(box_w), round(box_w / ratio)
    return info


SCREENSHOT_CALLOUT = Template(
    name="screenshot_callout",
    description="Big sticker captions and red highlight boxes over a screenshot the author supplies (never a made-up one).",
    schema=obj({
        "image": string("Path to the author's screenshot."),
        "stickers": array(string("At most 6 words."), "1 to 3 captions."),
        "boxes": array(obj({
            "x": number("Left edge, % of the screenshot's width."), "y": number("Top edge, %."),
            "w": number("Width, %."), "h": number("Height, %."),
        }, ["x", "y", "w", "h"]), "Optional highlight boxes."),
        "shape": string("portrait, square or landscape", enum=["portrait", "square", "landscape"], default="portrait"),
    }, ["image", "stickers"]),
    default_theme="ink",
    shapes=("portrait", "square", "landscape"),
    prepare_fn=_screenshot,
    sample={"image": "", "stickers": ["Wait for it", "This changes everything"], "boxes": []},
)


# ---------- puzzle ----------
# Five features as interlocking jigsaw pieces (four corners and a centre piece), each explained with the same
# labelled parts. The outlines are drawn here; the text in each piece wraps around the centre piece and the knobs
# with CSS shapes, and the runtime shrinks it to fit.
PUZZLE_W, PUZZLE_H = 992, 1010


def _knob(x0, y0, x1, y1, at, bulge, neck=17, head=27):
    """An axis-aligned line from (x0, y0) to (x1, y1) with one jigsaw knob at `at` (a coordinate along the line),
    bulging towards +1 (right or down) or -1 (left or up). Returns SVG path data."""
    horizontal = y0 == y1
    reach = head * 2 + 6

    def pt(along, across):            # along the line, and the distance off it in the bulge direction
        return (along, y0 + bulge * across) if horizontal else (x0 + bulge * across, along)

    start, end = (x0, x1) if horizontal else (y0, y1)
    a, b = pt(at - neck, 0), pt(at + neck, 0)
    c1, c2 = pt(at - neck, reach * .35), pt(at - head - 4, reach * .3)
    d = pt(at - head * .9, reach - head * 1.1)
    e = pt(at + head * .9, reach - head * 1.1)
    c3, c4 = pt(at + head + 4, reach * .3), pt(at + neck, reach * .35)
    s0, s1 = pt(start, 0), pt(end, 0)
    # the arc's sweep: which way round the head goes depends on the direction of travel and the bulge
    sweep = (1 if bulge < 0 else 0) if horizontal else (1 if bulge > 0 else 0)
    f = lambda q: f"{q[0]:.1f} {q[1]:.1f}"
    return (f"M {f(s0)} L {f(a)} C {f(c1)} {f(c2)} {f(d)} A {head} {head} 0 1 {sweep} {f(e)} "
            f"C {f(c3)} {f(c4)} {f(b)} L {f(s1)}")


def _puzzle(data: dict, context: dict) -> dict:
    W, H, vx, hy = PUZZLE_W, PUZZLE_H, 496, 505
    cx0, cy0, cx1, cy1 = 268, 352, 724, 660            # the centre piece
    g, reach = 16, 60                                   # text gap from the strokes; how far a knob reaches
    paths = [
        _knob(vx, 0, vx, cy0, 176, +1),                 # top divider, knob into the top-right piece
        _knob(vx, cy1, vx, H, 835, -1),                 # bottom divider, knob into the bottom-left piece
        _knob(0, hy, cx0, hy, 134, +1),                 # left divider, knob into the bottom-left piece
        _knob(cx1, hy, W, hy, 858, -1),                 # right divider, knob into the top-right piece
        _knob(cx0, cy0, cx0, cy1, 430, -1),             # centre's left edge, knob into the top-left piece
        _knob(cx1, cy0, cx1, cy1, 585, +1),             # centre's right edge, knob into the bottom-right piece
        f"M {cx0} {cy0} L {cx1} {cy0} M {cx0} {cy1} L {cx1} {cy1}",
    ]

    def poly(points, w, h):
        return "polygon(" + ", ".join(f"{x / w * 100:.2f}% {y / h * 100:.2f}%" for x, y in points) + ")"

    boxes = []
    # top-left: the centre piece and its left knob cut into the bottom-right corner
    L, T, R, B = 22, 22, vx - g, hy - g; w, h = R - L, B - T
    fw = R - (cx0 - reach - g); ox = R - fw
    pts = [(cx0 - g - ox, cy0 - g - T), (fw, cy0 - g - T), (fw, h), (cx0 - g - ox, h), (cx0 - g - ox, 460 + g - T),
           (0, 460 + g - T), (0, 400 - g - T), (cx0 - g - ox, 400 - g - T)]
    boxes.append({"l": L, "t": T, "w": w, "h": h, "floats": [{"side": "right", "w": fw, "shape": poly(pts, fw, h)}]})
    # top-right: the top knob cuts in on the left, the centre at the bottom left, the right knob at the bottom
    L, T, R, B = vx + g, 22, W - 22, hy - g; w, h = R - L, B - T
    fw = cx1 + g - L
    knob_x = vx + reach + g - L
    pts = [(0, 146 - g - T), (knob_x, 146 - g - T), (knob_x, 206 + g - T), (1, 206 + g - T), (1, cy0 - g - T),
           (fw, cy0 - g - T), (fw, h), (0, h)]
    rfw = W - 22 - (858 - 30 - g); rpts = [(0, hy - reach - g - T), (rfw - (W - 22 - (858 + 30 + g)), hy - reach - g - T),
                                          (rfw - (W - 22 - (858 + 30 + g)), h), (0, h)]
    boxes.append({"l": L, "t": T, "w": w, "h": h, "floats": [{"side": "left", "w": fw, "shape": poly(pts, fw, h)},
                                                            {"side": "right", "w": rfw, "shape": poly(rpts, rfw, h)}]})
    # bottom-left: the centre at the top right, the bottom knob on the right, the left knob at the top left
    L, T, R, B = 22, hy + g, vx - g, H - 22; w, h = R - L, B - T
    fw = R - (cx0 - g); ox = R - fw
    kx = vx - reach - g - ox
    pts = [(0, 0), (fw, 0), (fw, 865 + g - T), (kx, 865 + g - T), (kx, 805 - g - T), (fw - 1, 805 - g - T),
           (fw - 1, cy1 + g - T), (0, cy1 + g - T)]
    lfw = 134 + 30 + g - L; lfh = hy + reach + g - T
    boxes.append({"l": L, "t": T, "w": w, "h": h, "floats": [{"side": "right", "w": fw, "shape": poly(pts, fw, h)},
                                                            {"side": "left", "w": lfw, "h": lfh}]})
    # bottom-right: the centre and its right knob at the top left
    L, T, R, B = vx + g, hy + g, W - 22, H - 22; w, h = R - L, B - T
    cw = cx1 + g - L; fw = cx1 + reach + g - L
    pts = [(0, 0), (cw, 0), (cw, 555 - g - T), (fw, 555 - g - T), (fw, 615 + g - T), (cw, 615 + g - T),
           (cw, cy1 + g - T), (0, cy1 + g - T)]
    boxes.append({"l": L, "t": T, "w": w, "h": h, "floats": [{"side": "left", "w": fw, "shape": poly(pts, fw, h)}]})
    # the centre piece
    boxes.insert(2, {"l": cx0 + g, "t": cy0 + g, "w": cx1 - cx0 - 2 * g, "h": cy1 - cy0 - 2 * g, "floats": [], "centre": True})
    pieces = list(data.get("pieces", []))[:5]
    # order on the page: top-left, top-right, centre, bottom-left, bottom-right
    return {"paths": paths, "boxes": [dict(b, piece=p) for b, p in zip(boxes, pieces)], "pw": W, "ph": H}


PUZZLE = Template(
    name="puzzle",
    description="Five features explained as interlocking jigsaw pieces: each with a name tag, a plain-English line, "
                "an everyday comparison, setup steps, what it's best for and the biggest mistake people make.",
    schema=obj({
        "title": string("The features named, e.g. 'Claude Instructions, Projects, Connectors'."),
        "title_tag": string("The word in the black block at the end.", default="Explained"),
        "pieces": array(obj({
            "name": string("1 to 3 words."),
            "plain": string("In plain English, at most 14 words."),
            "like": string("Think of it like..., at most 14 words."),
            "setup": array(string("At most 12 words."), "How to set it up: 2 to 4 steps."),
            "best": string("Best for, at most 16 words."),
            "mistake": string("The biggest mistake people make, at most 28 words."),
        }, ["name", "plain", "like", "setup", "best", "mistake"]), "Exactly 5 pieces: top left, top right, centre, bottom left, bottom right."),
    }, ["title", "pieces"]),
    default_theme="cream",
    prepare_fn=_puzzle,
    sample={
        "title": "Claude Instructions, Projects, Connectors, Skills, Chats",
        "title_tag": "Explained",
        "pieces": [
            {"name": "Instructions", "plain": "How you like things, in every chat.", "like": "A note on the fridge for a new housemate.",
             "setup": ["Open your settings", "Write your preferences in plain words"], "best": "Tone, length, words to avoid.",
             "mistake": "Putting one job's steps here. They follow you into every chat."},
            {"name": "Projects", "plain": "Background files for one piece of work.", "like": "The folder you hand over for one client.",
             "setup": ["Create a project", "Add the files", "Chat inside it"], "best": "A client, a course, a big piece of work.",
             "mistake": "Expecting it to work in chats outside the project."},
            {"name": "Chats", "plain": "Just ask, for a one-off job.", "like": "Asking a colleague a quick question.",
             "setup": ["Explain it in the chat", "Save the prompt you liked"], "best": "Jobs you do once.",
             "mistake": "Making a skill for a job you'll never repeat."},
            {"name": "Connectors", "plain": "Access to your other apps and data.", "like": "A spare set of keys.",
             "setup": ["Connect the app", "Ask about what's in it"], "best": "Answers that live in another app.",
             "mistake": "Expecting access to teach Claude how you work."},
            {"name": "Skills", "plain": "One job, done your way, in any chat.", "like": "A recipe card.",
             "setup": ["Write a SKILL.md", "Zip the folder", "Upload it"], "best": "The job you've explained three times.",
             "mistake": "Three jobs in one skill."},
        ],
    },
)


TEMPLATES: dict[str, Template] = {t.name: t for t in [
    CHEATSHEET, LINK_LIST, PROMPT_LADDER, PROMPT_ANATOMY, SPLIT, FUNNEL, GLOSSARY, ROADMAP, CYCLE, MINDMAP,
    DECISION_TREE, CARD_GRID, BIG_NUMBER, BADGE, BOOKSHELF, KINETIC, SCREENSHOT_CALLOUT, PUZZLE,
]}


def get_template(name: str) -> Template:
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template {name!r}. Choose from: {', '.join(TEMPLATES)}")
    return TEMPLATES[name]
