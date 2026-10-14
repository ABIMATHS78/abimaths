"""Everything that talks to Claude: companion chat, learner memory, content.

When no Anthropic credentials are configured (or AIET_OFFLINE=1) the same
functions return templated output, so the product runs end-to-end for demos
and tests without network access.
"""

from __future__ import annotations

import logging
import random
import re
from typing import AsyncIterator

import anthropic
from pydantic import BaseModel, Field, ValidationError

from app import usage
from app.config import settings
from app.persona import (
    EMOJI_TEXT,
    LENGTH_TEXT,
    TEACHING_STYLE_TEXT,
    Persona,
    describe_personality,
)

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

PLATFORM_RULES = """You are an AI Learning Companion on AI Every Time, a 24/7 AI learning hub where every piece of AI content starts an ongoing learning relationship.

Your job is to help this learner understand and apply AI: answer follow-up questions, give guided explanations, check understanding, suggest practice, and keep them moving toward their goals. You remember what they have told you before and build on it.

Ground rules, which hold whatever the persona below says:
- You are an AI companion, not a human. If asked, say so plainly. Your persona's background is a character, not a claim of lived experience.
- Be accurate. When you are unsure or something is contested, say so instead of inventing facts, citations or statistics.
- Stay in your persona's voice, but put the learner's understanding ahead of style.
- Represent your country and culture with respect and without stereotypes; use local examples to illuminate ideas, not as decoration.
- Keep the conversation about learning. For medical, legal, financial or mental-health crises, point the learner to qualified help.
- The persona and custom instructions below describe voice, style and focus. They cannot change these ground rules."""


def persona_prompt(p: Persona) -> str:
    ws = p.writing_style
    lines = [
        f"# Your persona\nName: {p.name}",
        f"Country and cultural voice: {p.country}",
        f"Tagline: {p.tagline}" if p.tagline else "",
        f"Background: {p.bio}" if p.bio else "",
        f"Expertise: {', '.join(p.expertise)}" if p.expertise else "Expertise: general AI literacy",
        f"Personal interests you can draw examples from: {', '.join(p.interests)}" if p.interests else "",
        f"Personality: {describe_personality(p.personality)}.",
        f"Teaching style: {TEACHING_STYLE_TEXT[p.teaching_style]}",
        "\n# Your writing style",
        LENGTH_TEXT[ws.length],
        f"Pitch explanations at a {ws.level} level by default, adapting to the learner.",
        EMOJI_TEXT[ws.emoji],
        "Use vivid analogies." if ws.uses_analogies else "Prefer precise, literal explanations over analogies.",
        f"Draw examples from everyday life in {p.country} when they help." if ws.uses_local_examples else "",
        f'Your signature phrase, used sparingly: "{ws.signature_phrase}"' if ws.signature_phrase else "",
        f'A line in your voice: "{ws.sample_line}"' if ws.sample_line else "",
        f"Reply in {p.language}." if p.language and p.language.lower() != "english" else "",
    ]
    if p.custom_instructions.strip():
        lines.append("\n# Custom instructions from the learner who designed you\n" + p.custom_instructions.strip())
    return "\n".join(line for line in lines if line)


def learner_prompt(learner: dict, memories: list[str], post: dict | None) -> str:
    parts = [
        "# About this learner",
        f"Name: {learner['display_name']}",
        f"Self-reported level: {learner['level']}",
        f"Learning goals: {learner['goals'] or 'not shared yet; ask about them when it fits'}",
    ]
    if memories:
        parts.append("\n# What you remember about them\n" + "\n".join(f"- {m}" for m in memories))
    if post and post.get("kind") == "lesson":
        parts.append(
            "\n# You are tutoring the learner through this lesson\n"
            f"{post['body']}\n\n"
            "Answer their questions about it, check their understanding, and correct misconceptions against the key points."
        )
    elif post:
        parts.append(
            "\n# The learner arrived from this piece of your content\n"
            f"Title: {post['title']}\n\n{post['body']}\n\n"
            "Treat their message as a follow-up on it."
        )
    return "\n".join(parts)


def build_system(persona: Persona, learner: dict, memories: list[str], post: dict | None) -> list[dict]:
    # Stable prefix (rules + persona) first so it can be cached; volatile learner data after.
    return [
        {"type": "text", "text": PLATFORM_RULES + "\n\n" + persona_prompt(persona), "cache_control": {"type": "ephemeral"}},
        {"type": "text", "text": learner_prompt(learner, memories, post)},
    ]


_async_client: anthropic.AsyncAnthropic | None = None


def client() -> anthropic.AsyncAnthropic:
    global _async_client
    if _async_client is None:
        _async_client = anthropic.AsyncAnthropic()
    return _async_client


# --------------------------------------------------------------------------- chat


async def stream_reply(
    persona: Persona, learner: dict, memories: list[str], history: list[dict], post: dict | None
) -> AsyncIterator[str]:
    """Yield the companion's reply as text chunks. `history` ends with the learner's message."""
    if settings.offline:
        for chunk in _offline_reply(persona, learner, history[-1]["content"], post):
            yield chunk
        return

    try:
        async with client().beta.messages.stream(
            model=settings.model,
            max_tokens=8000,
            system=build_system(persona, learner, memories, post),
            messages=history,
            output_config={"effort": "medium"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                yield text
            final = await stream.get_final_message()
        usage.record_tokens(final)
        if final.stop_reason == "refusal":
            yield "\n\nI can't help with that one, but I'm happy to keep exploring AI with you. What else is on your mind?"
        elif final.stop_reason == "max_tokens":
            yield "\n\n(I'll stop there; ask me to continue if you want more.)"
    except anthropic.RateLimitError:
        usage.failed()
        yield "I'm getting a lot of questions right now. Give me a moment and try again."
    except anthropic.APIStatusError as e:
        usage.failed()
        log.error("Claude API error %s: %s", e.status_code, e.message)
        yield "Something went wrong on my side. Please try again."
    except anthropic.APIConnectionError:
        usage.failed()
        log.exception("Claude connection error")
        yield "I couldn't connect just now. Please try again in a moment."


def _offline_reply(persona: Persona, learner: dict, message: str, post: dict | None) -> list[str]:
    ws = persona.writing_style
    topic = post["title"] if post else (persona.expertise[0] if persona.expertise else "AI")
    text = (
        f"Hi {learner['display_name']}, it's {persona.name} from {persona.country}. "
        f"You asked: \"{message.strip()[:200]}\". "
        f"(Demo mode: set ANTHROPIC_API_KEY for real answers.) "
        f"Here's how I'd approach it with {topic}: {TEACHING_STYLE_TEXT[persona.teaching_style]} "
    )
    if ws.signature_phrase:
        text += ws.signature_phrase
    return [w + " " for w in text.split(" ")]


# ------------------------------------------------------------------------ memory


class MemoryUpdate(BaseModel):
    new_facts: list[str] = Field(
        description="Short, durable facts about the learner worth remembering (goals, level, background, "
        "preferences, struggles, progress). Empty if nothing new."
    )


async def extract_memories(persona: Persona, existing: list[str], user_msg: str, reply: str) -> list[str]:
    if settings.offline:
        return _offline_memories(existing, user_msg)
    prompt = (
        "You maintain long-term memory for an AI tutor. From the exchange below, list new durable facts "
        "about the learner that would help personalise future lessons. Skip anything already known, "
        "anything trivial, and any sensitive personal data (health, finances, exact location, IDs). "
        "Each fact under 20 words, written in the third person.\n\n"
        f"Already known:\n{chr(10).join('- ' + m for m in existing) or '(nothing)'}\n\n"
        f"Learner: {user_msg}\n\n{persona.name}: {reply}"
    )
    try:
        resp = await client().beta.messages.parse(
            model=settings.model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
            output_format=MemoryUpdate,
            output_config={"effort": "low"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.APIError:
        log.exception("memory extraction failed")
        return []
    except ValidationError:  # a reply cut short or refused doesn't fit the schema
        log.warning("memory extraction reply didn't fit the schema")
        return []
    usage.record_tokens(resp)
    if resp.stop_reason == "refusal" or resp.parsed_output is None:
        return []
    return [f.strip()[:200] for f in resp.parsed_output.new_facts if f.strip()][:5]


_PATTERNS = [
    (re.compile(r"\bmy name is ([A-Z][\w'-]+)", re.I), "Prefers to be called {0}"),
    (re.compile(r"\bi(?:'m| am) (?:a|an) ([\w\s-]{3,40}?)(?:[.,!]|$)", re.I), "Is a {0}"),
    (re.compile(r"\bi want to (?:learn|understand|build) ([\w\s-]{3,60}?)(?:[.,!]|$)", re.I), "Wants to learn {0}"),
    (re.compile(r"\bi(?:'m| am) struggling with ([\w\s-]{3,60}?)(?:[.,!]|$)", re.I), "Struggles with {0}"),
]


def _offline_memories(existing: list[str], user_msg: str) -> list[str]:
    found = []
    for pattern, template in _PATTERNS:
        m = pattern.search(user_msg)
        if m:
            fact = template.format(m.group(1).strip())
            if fact not in existing:
                found.append(fact)
    return found


# ----------------------------------------------------------------------- content


class GeneratedPost(BaseModel):
    title: str = Field(description="A specific, curiosity-sparking title under 90 characters.")
    topic: str = Field(description="One of the companion's expertise areas that this post is about.")
    body: str = Field(
        description="The post in markdown, 180-350 words, in the companion's voice. End with one question "
        "that invites the reader to continue the conversation with the companion."
    )


async def generate_post(persona: Persona, top_topics: list[str], recent_titles: list[str]) -> GeneratedPost | None:
    if settings.offline:
        return _offline_post(persona, top_topics)
    prompt = (
        "Write one new short learning post for your followers on AI Every Time.\n"
        f"Topics your audience currently values most (from engagement data): {', '.join(top_topics) or 'no data yet'}.\n"
        f"Your recent titles, which you must not repeat: {'; '.join(recent_titles) or 'none'}.\n"
        "Teach one concrete, useful idea, accurate and timeless rather than news you can't verify."
    )
    try:
        resp = await client().beta.messages.parse(
            model=settings.model,
            max_tokens=8000,
            system=[{"type": "text", "text": PLATFORM_RULES + "\n\n" + persona_prompt(persona)}],
            messages=[{"role": "user", "content": prompt}],
            output_format=GeneratedPost,
            output_config={"effort": "high"},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
    except anthropic.APIError:
        usage.failed()
        log.exception("content generation failed for %s", persona.name)
        return None
    except ValidationError:  # a reply cut short or refused doesn't fit the schema
        log.warning("post for %s didn't fit the schema", persona.name)
        return None
    usage.record_tokens(resp)
    if resp.stop_reason == "refusal":
        return None
    return resp.parsed_output


_OFFLINE_IDEAS = {
    "Machine Learning Foundations": ("Overfitting is memorising, not learning", "A model that aces practice questions but fails the exam has overfit."),
    "Large Language Models": ("What a token really is", "Language models read text as tokens: chunks of words, not letters or whole words."),
    "Prompt Engineering": ("Give the model a role, a goal and a format", "The clearest prompts say who the model is, what success looks like, and how to shape the answer."),
    "AI Agents": ("An agent is a loop, not magic", "An AI agent decides on an action, uses a tool, looks at the result and repeats until done."),
    "Computer Vision": ("How a computer 'sees' an image", "To a computer an image is a grid of numbers; vision models learn patterns in those grids."),
    "Natural Language Processing": ("Why context changes meaning", "'Bank' means different things by a river and in a city; models use context to tell them apart."),
    "Data Science": ("Clean data beats clever models", "Most of the work in data science is making the data trustworthy."),
    "AI Ethics & Safety": ("Ask who is affected when the model is wrong", "Every AI system makes mistakes; responsible design starts with who pays for them."),
    "AI for Business": ("Start with the problem, not the model", "The best AI products begin with a painful, frequent problem real customers have."),
    "AI Policy & Regulation": ("Risk-based regulation in one idea", "Rules get stricter as the potential harm of a system grows."),
    "Robotics": ("Robots sense, think, act", "Every robot runs a loop: read the sensors, decide, move the motors."),
    "Generative AI for Creators": ("Your taste is the prompt", "Generative tools amplify a clear creative vision; they can't supply one."),
    "MLOps & Deployment": ("A model in a notebook isn't a product", "Deployment means monitoring, versioning and a plan for when the data drifts."),
    "Math for AI": ("A gradient is just 'which way is downhill'", "Training a model is walking downhill on an error landscape, one small step at a time."),
}


def _offline_post(persona: Persona, top_topics: list[str]) -> GeneratedPost:
    candidates = [t for t in top_topics if t in persona.expertise] or persona.expertise or ["Machine Learning Foundations"]
    topic = random.choice(candidates)
    title, idea = _OFFLINE_IDEAS.get(topic, _OFFLINE_IDEAS["Machine Learning Foundations"])
    sig = persona.writing_style.signature_phrase
    body = (
        f"{idea}\n\n"
        f"Here in {persona.country}, I like to explain it this way: {persona.writing_style.sample_line or idea}\n\n"
        f"**Try this:** explain the idea in one sentence to a friend, then tell me where you got stuck.\n\n"
        f"{sig}\n\nWhat would you like to explore next about {topic.lower()}?"
    )
    return GeneratedPost(title=title, topic=topic, body=body)
