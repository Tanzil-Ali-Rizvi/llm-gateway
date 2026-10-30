"""Deterministic mock provider with fault injection: for tests, chaos benchmarks and outage demos."""

from __future__ import annotations

import asyncio
import random
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

from ..config import ProviderConfig
from .base import Provider, ProviderError, estimate_tokens, prompt_text


class MockProvider(Provider):
    def __init__(self, name: str, cfg: ProviderConfig):
        self.name = name
        self.cfg = cfg
        self.fault = cfg.fault  # mutable at runtime, e.g. to simulate an outage
        self.rng = random.Random(cfg.fault.seed)
        self.calls = 0

    async def _maybe_fail(self) -> None:
        self.calls += 1
        f = self.fault
        delay = max(0.0, f.latency_ms + self.rng.uniform(-f.latency_jitter_ms, f.latency_jitter_ms)) / 1000
        await asyncio.sleep(delay)
        if self.rng.random() < f.error_rate:
            retry_after = 1.0 if f.error_status == 429 else None
            raise ProviderError(
                f"HTTP {f.error_status}: injected failure", status=f.error_status, retry_after=retry_after
            )

    def _answer(self, model: str, body: dict[str, Any]) -> str:
        last = next((m.get("content", "") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
        if not isinstance(last, str):
            last = "multimodal input"
        return f"[{self.name}/{model}] Mock answer to: {last[:80]}"

    async def chat(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        await self._maybe_fail()
        text = self._answer(model, body)
        p, c = estimate_tokens(prompt_text(body)), estimate_tokens(text)
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
        }

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        await self._maybe_fail()
        text = self._answer(model, body)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:12]}", int(time.time())
        words = text.split(" ")
        for i, w in enumerate(words):
            delta = {"content": w if i == 0 else " " + w}
            if i == 0:
                delta["role"] = "assistant"
            yield {
                "id": cid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            await asyncio.sleep(0)
        p, c = estimate_tokens(prompt_text(body)), estimate_tokens(text)
        yield {
            "id": cid,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c},
        }
