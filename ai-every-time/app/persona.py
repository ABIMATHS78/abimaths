"""The companion persona: everything a learner can customise about a companion.

Starter companions and learner-designed companions share this single schema, so
a learner can start from a starter companion ("remix") and change any part of it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

EXPERTISE_AREAS = [
    "Machine Learning Foundations",
    "Large Language Models",
    "Prompt Engineering",
    "AI Agents",
    "Computer Vision",
    "Natural Language Processing",
    "Data Science",
    "AI Ethics & Safety",
    "AI for Business",
    "AI Policy & Regulation",
    "Robotics",
    "Generative AI for Creators",
    "MLOps & Deployment",
    "Math for AI",
]

TeachingStyle = Literal["socratic", "step_by_step", "storytelling", "examples_first", "challenge_driven"]
ResponseLength = Literal["concise", "balanced", "detailed"]
Level = Literal["beginner", "intermediate", "advanced"]
EmojiUse = Literal["none", "light", "expressive"]

HairStyle = Literal["short", "long", "curly", "bun", "braids", "bald", "afro", "waves"]
Accessory = Literal["none", "glasses", "round_glasses", "headphones", "earrings", "headwrap", "cap"]
Expression = Literal["smile", "grin", "calm", "curious"]


class Avatar(BaseModel):
    """Parameters for the generated face. Rendered to SVG by app.avatar."""

    skin: str = "#C68642"
    hair_style: HairStyle = "short"
    hair_color: str = "#2C1B10"
    background: str = "#6C5CE7"
    accessory: Accessory = "none"
    accessory_color: str = "#222222"
    expression: Expression = "smile"
    shirt: str = "#1E272E"

    @field_validator("skin", "hair_color", "background", "accessory_color", "shirt")
    @classmethod
    def _hex_colour(cls, v: str) -> str:
        v = v.strip()
        if not (len(v) == 7 and v.startswith("#") and all(c in "0123456789abcdefABCDEF" for c in v[1:])):
            raise ValueError("colours must be #RRGGBB")
        return v.upper()


class Personality(BaseModel):
    """0-100 sliders. 50 is neutral."""

    warmth: int = Field(70, ge=0, le=100)
    humor: int = Field(40, ge=0, le=100)
    formality: int = Field(40, ge=0, le=100)
    directness: int = Field(60, ge=0, le=100)
    enthusiasm: int = Field(60, ge=0, le=100)
    patience: int = Field(80, ge=0, le=100)


class WritingStyle(BaseModel):
    length: ResponseLength = "balanced"
    level: Level = "beginner"
    emoji: EmojiUse = "light"
    uses_analogies: bool = True
    uses_local_examples: bool = True
    signature_phrase: str = Field("", max_length=120)
    sample_line: str = Field("", max_length=300, description="An example sentence in the companion's voice.")


class Voice(BaseModel):
    """Read-aloud settings, used by the browser's speech synthesis."""

    enabled: bool = True
    lang: str = Field("en-US", pattern=r"^[a-z]{2,3}(-[A-Z]{2})?$", description="BCP 47 accent hint, e.g. en-NG")
    rate: float = Field(1.0, ge=0.5, le=2.0)
    pitch: float = Field(1.0, ge=0.0, le=2.0)


class Persona(BaseModel):
    name: str = Field(..., min_length=1, max_length=40)
    tagline: str = Field("", max_length=120)
    country: str = Field(..., min_length=2, max_length=60)
    flag: str = Field("", max_length=8)
    language: str = Field("English", max_length=40)
    bio: str = Field("", max_length=800)
    expertise: list[str] = Field(default_factory=list, max_length=5)
    teaching_style: TeachingStyle = "step_by_step"
    personality: Personality = Field(default_factory=Personality)
    writing_style: WritingStyle = Field(default_factory=WritingStyle)
    avatar: Avatar = Field(default_factory=Avatar)
    voice: Voice = Field(default_factory=Voice)
    interests: list[str] = Field(default_factory=list, max_length=8)
    custom_instructions: str = Field("", max_length=1500)

    @field_validator("expertise")
    @classmethod
    def _known_expertise(cls, v: list[str]) -> list[str]:
        unknown = [e for e in v if e not in EXPERTISE_AREAS]
        if unknown:
            raise ValueError(f"unknown expertise: {', '.join(unknown)}")
        return list(dict.fromkeys(v))

    @field_validator("interests")
    @classmethod
    def _short_interests(cls, v: list[str]) -> list[str]:
        return [i.strip()[:40] for i in v if i.strip()]


TEACHING_STYLE_TEXT = {
    "socratic": "Teach mainly by asking guiding questions so the learner reasons their way to the answer; give the answer when they are stuck.",
    "step_by_step": "Break explanations into small numbered steps and check understanding before moving on.",
    "storytelling": "Explain through short stories, characters and real-world narratives before formalising.",
    "examples_first": "Start with a concrete example or worked case, then generalise to the idea behind it.",
    "challenge_driven": "Give the learner small challenges and exercises, then give feedback on their attempts.",
}

LENGTH_TEXT = {
    "concise": "Keep replies short: usually under 120 words unless the learner asks for more.",
    "balanced": "Keep replies focused: usually 120-250 words.",
    "detailed": "Give thorough replies with depth and nuance when the topic warrants it.",
}

EMOJI_TEXT = {
    "none": "Never use emoji.",
    "light": "Use at most one emoji occasionally.",
    "expressive": "Use emoji freely to add warmth and energy.",
}


def _scale(value: int, low: str, high: str) -> str:
    if value >= 75:
        return f"very {high}"
    if value >= 55:
        return high
    if value > 45:
        return f"balanced between {low} and {high}"
    if value > 25:
        return low
    return f"very {low}"


def describe_personality(p: Personality) -> str:
    return "; ".join(
        [
            _scale(p.warmth, "reserved", "warm"),
            _scale(p.humor, "serious", "playful and funny"),
            _scale(p.formality, "casual", "formal"),
            _scale(p.directness, "gentle and indirect", "direct"),
            _scale(p.enthusiasm, "calm", "enthusiastic"),
            _scale(p.patience, "brisk", "patient"),
        ]
    )
