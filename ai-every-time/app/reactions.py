"""Emoji reactions in chat. The learner can react to the companion's messages, and the companion to the learner's.

The companion decides as it starts its reply: it may open with a reaction tag, <reaction>🎉</reaction>, which the
server takes off the reply before anyone sees it (split) and saves on the learner's message. In demo mode a few
plain rules stand in for that decision (offline).
"""

from __future__ import annotations

import re
from typing import AsyncIterator, Callable

REACTIONS = ("👍", "❤️", "😂", "😮", "😢", "🙏", "🎉", "💡")
OPEN, CLOSE = "<reaction>", "</reaction>"
_BY_BASE = {r.replace("️", ""): r for r in REACTIONS}
_SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
_TAGS = re.compile(r"<reaction>[^<]{0,16}</reaction>\s*")

RULES = (
    "# Reacting to messages\n"
    f"Like a friend in a chat app, you can react to the learner's latest message with one emoji: {' '.join(REACTIONS)}. "
    "React only where a friend naturally would: thanks or kind words (❤️ or 🙏), good news or a breakthrough (🎉), "
    "a joke (😂), a sharp question or idea (💡), something surprising (😮) or sad (😢), plain agreement (👍). "
    "Most messages get no reaction. To react, begin your reply with the emoji in a reaction tag, like "
    f"{OPEN}🎉{CLOSE}, then reply as usual. Never mention the tag or explain your reaction."
)


def canonical(emoji: str) -> str | None:
    """The reaction from REACTIONS this emoji is, however it was written (with or without a variation selector or
    a skin tone), or None if it isn't one of them."""
    bare = "".join(ch for ch in emoji.strip() if ch != "️" and ch not in _SKIN_TONES)
    return _BY_BASE.get(bare)


def strip(text: str) -> str:
    """The text without any reaction tags, should the companion put one somewhere other than the start."""
    return _TAGS.sub("", text)


async def split(chunks: AsyncIterator[str], react: Callable[[str], None]) -> AsyncIterator[str]:
    """Pass the reply through, taking a reaction tag off its very start and handing the reaction to `react`.
    The first few characters are held back only until it's clear whether the reply starts with a tag."""
    head = ""
    async for chunk in chunks:
        if head is None:  # past the start: just pass it on
            yield chunk
            continue
        head += chunk
        start = head.lstrip()
        if start.startswith(OPEN):
            end = start.find(CLOSE)
            if end == -1 and len(start) < len(OPEN) + 16 + len(CLOSE):
                continue  # the tag is still arriving
            if end != -1 and (emoji := canonical(start[len(OPEN):end])):
                react(emoji)
            rest = start[end + len(CLOSE):].lstrip() if end != -1 else head
            head = None
            if rest:
                yield rest
        elif start and not OPEN.startswith(start):
            yield head
            head = None
        # else: only whitespace, or what may be the beginning of a tag ("<rea"), so far
    if head:  # the reply ended while its start was held back
        start = head.lstrip()
        end = start.find(CLOSE)
        if start.startswith(OPEN) and end != -1:
            if emoji := canonical(start[len(OPEN):end]):
                react(emoji)
            if rest := start[end + len(CLOSE):].lstrip():
                yield rest
        else:
            yield head


_OFFLINE = [
    (re.compile(r"\b(thanks|thank you|thx|cheers|appreciate)\b|🙏", re.I), "❤️"),
    (re.compile(r"\b(haha+|lol|lmao)\b|😂|🤣", re.I), "😂"),
    (re.compile(r"\b(i did it|i passed|got it|it works|makes sense now|i understand now|finally)\b|🎉", re.I), "🎉"),
    (re.compile(r"\b(wow|whoa|no way)\b", re.I), "😮"),
]


def offline(message: str) -> str | None:
    """Demo mode's stand-in for the companion deciding whether to react to a message."""
    return next((emoji for pattern, emoji in _OFFLINE if pattern.search(message)), None)
