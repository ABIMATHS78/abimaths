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
MAX_INSIDE = 16  # characters between the two: room for one emoji however it's written, not for a sentence
_BY_BASE = {r.replace("️", ""): r for r in REACTIONS}
_SKIN_TONES = {chr(c) for c in range(0x1F3FB, 0x1F400)}
_TAGS = re.compile(rf"<reaction>[^<]{{0,{MAX_INSIDE}}}</reaction>\s*")

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


def _tag(start: str) -> tuple[str | None, str] | None:
    """For text that starts with a reaction tag: the reaction it holds (None if it isn't one of REACTIONS) and the
    text after it. None if the text doesn't start with a whole tag of at most MAX_INSIDE characters."""
    end = start.find(CLOSE, len(OPEN))
    if not start.startswith(OPEN) or end == -1 or end - len(OPEN) > MAX_INSIDE:
        return None
    return canonical(start[len(OPEN):end]), start[end + len(CLOSE):].lstrip()


async def split(chunks: AsyncIterator[str], react: Callable[[str], None]) -> AsyncIterator[str]:
    """Pass the reply through, taking a reaction tag off its very start and handing the reaction to `react`.
    The first few characters are held back only until it's clear whether the reply starts with a tag. Whatever
    the chunks, the outcome is the same: a tag that's too long, or isn't closed, is left in the reply."""
    head: str | None = ""
    async for chunk in chunks:
        if head is None:  # past the start: just pass it on
            yield chunk
            continue
        head += chunk
        start = head.lstrip()
        if OPEN.startswith(start):  # only whitespace so far, or what may be the start of a tag ("<rea")
            continue
        if start.startswith(OPEN) and CLOSE not in start and len(start) < len(OPEN) + MAX_INSIDE + len(CLOSE):
            continue  # the tag may still be arriving
        held, head = head, None
        tag = _tag(start)
        if tag is None:  # not a reaction tag after all
            yield held
            continue
        emoji, rest = tag
        if emoji:
            react(emoji)
        if rest:
            yield rest
    if head:  # the reply ended while its start was held back
        tag = _tag(head.lstrip())
        if tag is None:
            yield head
        else:
            if tag[0]:
                react(tag[0])
            if tag[1]:
                yield tag[1]


_OFFLINE = [
    (re.compile(r"\b(thanks|thank you|thx|cheers|appreciate)\b|🙏", re.I), "❤️"),
    (re.compile(r"\b(haha+|lol|lmao)\b|😂|🤣", re.I), "😂"),
    (re.compile(r"\b(i did it|i passed|got it|it works|makes sense now|i understand now|finally)\b|🎉", re.I), "🎉"),
    (re.compile(r"\b(wow|whoa|no way)\b", re.I), "😮"),
]


def offline(message: str) -> str | None:
    """Demo mode's stand-in for the companion deciding whether to react to a message."""
    return next((emoji for pattern, emoji in _OFFLINE if pattern.search(message)), None)
