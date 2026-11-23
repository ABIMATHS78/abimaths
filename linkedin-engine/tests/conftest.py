"""Shared fixtures. The Claude API is never called: a mock HTTP transport sits under the real Anthropic SDK,
so requests are built and responses parsed exactly as they would be, without a key or network."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ENGINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ENGINE))


def sse_stream(text: str, model: str = "claude-opus-5-5", stop_reason: str = "end_turn") -> bytes:
    events = [
        ("message_start", {"type": "message_start", "message": {
            "id": "msg_test", "type": "message", "role": "assistant", "model": model, "content": [],
            "stop_reason": None, "stop_sequence": None,
            "usage": {"input_tokens": 1200, "output_tokens": 1, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0}}}),
        ("content_block_start", {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        ("content_block_delta", {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        ("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                           "usage": {"output_tokens": 800}}),
        ("message_stop", {"type": "message_stop"}),
    ]
    return "".join(f"event: {name}\ndata: {json.dumps(data)}\n\n" for name, data in events).encode()


class FakeAPI:
    """Answers /v1/messages with queued JSON payloads (or a function of the request) and records requests."""

    def __init__(self):
        self.requests: list[dict] = []
        self.headers: list[dict] = []
        self.responder = None

    def handler(self, request):
        import httpx2

        body = json.loads(request.content)
        self.requests.append(body)
        self.headers.append(dict(request.headers))
        payload = self.responder(body, len(self.requests))
        stop_reason = "end_turn"
        if isinstance(payload, tuple):  # (payload, stop_reason) to simulate refusals or truncation
            payload, stop_reason = payload
        text = payload if isinstance(payload, str) else json.dumps(payload)
        return httpx2.Response(200, headers={"content-type": "text/event-stream"}, content=sse_stream(text, stop_reason=stop_reason))

    def client(self):
        import anthropic
        import httpx2

        return anthropic.Anthropic(api_key="test-key", max_retries=0,
                                   http_client=anthropic.DefaultHttpxClient(transport=httpx2.MockTransport(self.handler)))


@pytest.fixture
def fake_api():
    pytest.importorskip("anthropic")
    pytest.importorskip("httpx2")
    return FakeAPI()


def browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            p.chromium.launch().close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def renderer():
    if not browser_available():
        pytest.skip("Chromium for Playwright isn't installed")
    from lpe.render import Renderer

    r = Renderer()
    yield r
    r.close()
