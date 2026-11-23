"""Calls to the Claude API. Each call asks for JSON matching a schema (structured outputs), streams the response,
caches the long stable system prompt, and opts into server-side fallbacks so a declined request is retried on
another model instead of failing."""

from __future__ import annotations

import json
import os

DEFAULT_MODEL = os.environ.get("NLE_MODEL", "claude-opus-5-5")
DEFAULT_EFFORT = os.environ.get("NLE_EFFORT", "high")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens: input, output, cache read, cache write (5-minute)
PRICES = {"claude-opus-5-5": (4.0, 20.0, 0.20, 5.0), "claude-fable-5-2": (8.0, 40.0, 0.20, 10.0),
          "claude-sonnet-5-5": (2.0, 10.0, 0.10, 2.5), "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125)}


class LLMError(RuntimeError):
    pass


def obj(properties: dict, description: str = "") -> dict:
    """An object schema with every property required (structured outputs limit optional properties)."""
    schema = {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}
    if description:
        schema["description"] = description
    return schema


def string(description: str = "", enum: list[str] | None = None) -> dict:
    schema: dict = {"type": "string", "description": description}
    if enum:
        schema["enum"] = list(enum)
    return schema


def array(items: dict, description: str = "") -> dict:
    return {"type": "array", "items": items, "description": description}


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
                  max_tokens: int = 64000, label: str = "") -> tuple[dict, list]:
        request = dict(
            model=self.model,
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=messages,
            thinking={"type": "adaptive"},
            output_config={"effort": effort or self.effort, "format": {"type": "json_schema", "schema": schema}},
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
            raise LLMError(f"The {label or 'response'} ran out of room (max_tokens); ask for a shorter piece.")
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
