"""
Thin wrapper around the Anthropic Claude SDK.

Centralises model selection, system prompts and error handling. Every call
degrades gracefully: if no API key is configured (or the SDK isn't installed),
``available`` is False and callers fall back to deterministic logic so the app
remains fully functional offline.
"""
from __future__ import annotations

import base64
from functools import lru_cache

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    CLAUDE_MODEL_FAST,
    CLAUDE_VISION_MODEL,
)


@lru_cache(maxsize=1)
def _client():
    if not ANTHROPIC_API_KEY:
        return None
    try:
        import anthropic

        return anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    except Exception:
        return None


def available() -> bool:
    """True when live Claude calls can be made."""
    return _client() is not None


def chat(
    prompt: str,
    system: str | None = None,
    history: list[dict] | None = None,
    fast: bool = False,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> str:
    """Single-turn (or multi-turn with ``history``) text completion."""
    client = _client()
    if client is None:
        raise RuntimeError("Claude API not configured")

    messages = list(history or [])
    messages.append({"role": "user", "content": prompt})

    resp = client.messages.create(
        model=CLAUDE_MODEL_FAST if fast else CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system or "You are a helpful inventory-management assistant for a small business.",
        messages=messages,
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


def vision(prompt: str, image_bytes: bytes, media_type: str = "image/png", max_tokens: int = 1500) -> str:
    """Multimodal call — used for OCR / invoice understanding."""
    client = _client()
    if client is None:
        raise RuntimeError("Claude API not configured")

    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    resp = client.messages.create(
        model=CLAUDE_VISION_MODEL,
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
    )
    return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")
