"""Provider interface. Requests and responses use the OpenAI chat-completions format throughout."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

RETRY_SAME_TARGET = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
TRY_NEXT_TARGET = RETRY_SAME_TARGET | {401, 403, 404}  # broken key / unknown model on one provider


class ProviderError(Exception):
    """A failed upstream call, classified so the router knows whether to retry, fall back or give up."""

    def __init__(self, message: str, status: int = 502, retry_after: float | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status in RETRY_SAME_TARGET

    @property
    def fallbackable(self) -> bool:
        return self.status in TRY_NEXT_TARGET


class Provider(ABC):
    name: str

    @abstractmethod
    async def chat(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        """Non-streaming chat completion. Returns an OpenAI-format response dict."""

    @abstractmethod
    def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """Streaming chat completion: yields OpenAI-format chunk dicts.

        Must raise ProviderError *before* yielding the first chunk if the call fails to start,
        so the router can still fall back to another target.
        """

    async def aclose(self) -> None:  # pragma: no cover - optional
        return None


def estimate_tokens(text: str) -> int:
    """Rough token count (~4 characters per token) for when a provider does not report usage."""
    return max(1, round(len(text) / 4))


def prompt_text(body: dict[str, Any]) -> str:
    parts = []
    for m in body.get("messages", []):
        c = m.get("content")
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            parts += [p.get("text", "") for p in c if isinstance(p, dict)]
    return "\n".join(parts)
