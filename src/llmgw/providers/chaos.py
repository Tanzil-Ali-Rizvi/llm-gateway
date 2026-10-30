"""Fault injection around any provider: chaos-test fallback paths against real APIs (staging, demos)."""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from typing import Any

from .base import Provider, ProviderError


class FaultInjector(Provider):
    def __init__(self, inner: Provider, error_rate: float = 0.0, error_status: int = 503, seed: int | None = None):
        self.inner = inner
        self.name = inner.name
        self.error_rate = error_rate  # change at runtime to start/stop a simulated outage
        self.error_status = error_status
        self.rng = random.Random(seed)
        self.calls = 0

    def _maybe_fail(self) -> None:
        self.calls += 1
        if self.rng.random() < self.error_rate:
            raise ProviderError(
                f"HTTP {self.error_status}: simulated outage (fault injection)", status=self.error_status
            )

    async def chat(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        self._maybe_fail()
        return await self.inner.chat(model, body)

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        self._maybe_fail()
        async for chunk in self.inner.stream(model, body):
            yield chunk

    async def aclose(self) -> None:
        await self.inner.aclose()
