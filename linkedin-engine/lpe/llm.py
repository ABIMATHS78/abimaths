"""Calls to the Claude API. Every call asks for JSON that matches a schema (structured outputs),
streams the response, caches the long stable system prompt, and opts into server-side fallbacks
so a declined request is retried on another model instead of failing."""

from __future__ import annotations

import base64
import io
import json
import os
from pathlib import Path

from . import prompts
from .guardrails import Issue
from .schemas import api_schema

DEFAULT_MODEL = os.environ.get("LPE_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("LPE_EFFORT", "high")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens: input, output, cache read, cache write (5-minute)
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
          "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}


class LLMError(RuntimeError):
    pass


class Claude:
    """A thin wrapper so the rest of the engine (and the tests) see one method: json_call()."""

    def __init__(self, model: str | None = None, effort: str | None = None, client=None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic()
        self.client = client
        self.model = model or DEFAULT_MODEL
        self.effort = effort or DEFAULT_EFFORT
        self.calls: list[dict] = []

    def json_call(self, system: str, messages: list, schema: dict, effort: str | None = None,
                  max_tokens: int = 32000, label: str = "") -> tuple[dict, list]:
        request = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or self.effort, "format": {"type": "json_schema", "schema": api_schema(schema)}},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )
        with self.client.beta.messages.stream(**request) as stream:
            message = stream.get_final_message()
        self._record(message, label)
        if message.stop_reason == "refusal":
            details = getattr(message, "stop_details", None)
            raise LLMError(f"Claude declined the {label or 'request'}: {getattr(details, 'explanation', '') or 'no reason given'}")
        if message.stop_reason == "max_tokens":
            raise LLMError(f"The {label or 'response'} ran out of room (max_tokens); try a shorter brief.")
        text = next((block.text for block in message.content if getattr(block, "type", "") == "text"), None)
        if text is None:
            raise LLMError(f"No text came back for the {label or 'request'}.")
        try:
            return json.loads(text), list(message.content)
        except json.JSONDecodeError as error:
            raise LLMError(f"The {label or 'response'} wasn't valid JSON: {error}") from error

    def _record(self, message, label: str) -> None:
        usage = getattr(message, "usage", None)
        entry = {"label": label, "model": getattr(message, "model", self.model)}
        for field in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            entry[field] = getattr(usage, field, 0) or 0
        self.calls.append(entry)

    def cost(self) -> dict:
        totals = {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}
        dollars = 0.0
        for call in self.calls:
            price = PRICES.get(call["model"], PRICES.get(self.model, PRICES["claude-opus-5-5"]))
            for key in totals:
                totals[key] += call[key]
            dollars += (call["input_tokens"] * price[0] + call["output_tokens"] * price[1]
                        + call["cache_read_input_tokens"] * price[2] + call["cache_creation_input_tokens"] * price[3]) / 1e6
        return {**totals, "usd_estimate": round(dollars, 4), "calls": len(self.calls)}


def write_post(claude: Claude, brief: dict, brand: dict, profile: dict, examples: list[dict],
               template: str | None = None, motion: str | None = None) -> tuple[dict, list]:
    """Returns the post plan and the conversation so far (for a repair turn)."""
    messages = [{"role": "user", "content": prompts.write_user(brief, examples, template, motion)}]
    data, content = claude.json_call(prompts.write_system(profile, brand), messages, prompts.write_schema(template), label="post")
    messages.append({"role": "assistant", "content": content})
    return data, messages


def repair_post(claude: Claude, messages: list, issues: list[Issue], brand: dict, profile: dict,
                template: str | None = None) -> tuple[dict, list]:
    messages = messages + [{"role": "user", "content": prompts.repair_user(issues)}]
    data, content = claude.json_call(prompts.write_system(profile, brand), messages, prompts.write_schema(template), label="repair")
    return data, messages + [{"role": "assistant", "content": content}]


def design_graphic(claude: Claude, post: str, graphic_brief: str, template_name: str, brand: dict, profile: dict,
                   effort: str | None = None) -> dict:
    from .templates import get_template

    template = get_template(template_name)
    messages = [{"role": "user", "content": prompts.design_user(post, graphic_brief, template_name)}]
    data, _ = claude.json_call(prompts.design_system(profile, brand), messages, template.schema, effort=effort, label="graphic")
    return data


def _image_block(path: Path) -> dict | None:
    """The graphic as an image block; a GIF's last frame (the finished design), a video is skipped."""
    from PIL import Image

    path = Path(path)
    if path.suffix.lower() in {".mp4", ".mov", ".webm", ".m4v"}:
        return None
    with Image.open(path) as image:
        if getattr(image, "n_frames", 1) > 1:
            image.seek(image.n_frames - 1)
        frame = image.convert("RGB")
        frame.thumbnail((1568, 1568))
        buffer = io.BytesIO()
        frame.save(buffer, format="JPEG", quality=88)
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(buffer.getvalue()).decode("ascii")}}


def annotate_example(example, claude: Claude | None = None, profile_voice: dict | None = None) -> dict:
    """Ask Claude what a training post's graphic is, for posts added after the hand-written annotations."""
    claude = claude or Claude(effort="medium")
    content = []
    if example.media_file:
        block = _image_block(example.media_file)
        if block:
            content.append(block)
    content.append({"type": "text", "text": prompts.annotate_user_text(example.text)})
    data, _ = claude.json_call(prompts.annotate_system(profile_voice), [{"role": "user", "content": content}],
                               prompts.ANNOTATE_SCHEMA, label=f"annotation of {example.key}")
    if not data.get("content_warning"):
        data.pop("content_warning", None)
    return data
