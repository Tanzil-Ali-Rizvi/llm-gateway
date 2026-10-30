"""Routing with retries, fallback across targets and per-target circuit breakers.

For each request the router walks the route's targets in order:

* a target whose circuit is open is skipped (until its cooldown passes, then one trial call is let through);
* retryable errors (429, 5xx, timeouts) are retried on the same target with backoff, honouring Retry-After
  when it is short, then the router moves to the next target;
* broken-target errors (401/403/404) move straight to the next target;
* client errors (400, 422 ...) are returned immediately, since every target would reject the request.

Streaming requests fall back only until the first chunk arrives; after that the client owns the stream.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from typing import Any

from .config import GatewayConfig
from .providers import Provider, ProviderError, build_provider
from .providers.base import estimate_tokens, prompt_text


# --------------------------------------------------------------------------- circuit breaker
@dataclass
class _BreakerState:
    failures: int = 0
    opened_at: float | None = None
    trial_in_flight: bool = False


class CircuitBreaker:
    def __init__(self, failure_threshold: int, cooldown_s: float, clock: Callable[[], float] = time.monotonic):
        self.threshold = failure_threshold
        self.cooldown = cooldown_s
        self.clock = clock
        self.states: dict[str, _BreakerState] = {}

    def _s(self, key: str) -> _BreakerState:
        return self.states.setdefault(key, _BreakerState())

    def state(self, key: str) -> str:
        s = self._s(key)
        if s.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - s.opened_at >= self.cooldown else "open"

    def allow(self, key: str) -> bool:
        st = self.state(key)
        if st == "closed":
            return True
        s = self._s(key)
        if st == "half_open" and not s.trial_in_flight:
            s.trial_in_flight = True  # let exactly one trial call through
            return True
        return False

    def success(self, key: str) -> None:
        self.states[key] = _BreakerState()

    def failure(self, key: str) -> None:
        s = self._s(key)
        s.failures += 1
        s.trial_in_flight = False
        if s.opened_at is not None or s.failures >= self.threshold:
            s.opened_at = self.clock()  # (re)open: a failed trial restarts the cooldown

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {k: {"state": self.state(k), "consecutive_failures": v.failures} for k, v in self.states.items()}


# --------------------------------------------------------------------------- results
@dataclass
class Attempt:
    target: str
    outcome: str  # ok | error | skipped_open_circuit
    status: int | None = None
    error: str | None = None
    latency_ms: float = 0.0


@dataclass
class RouteResult:
    provider: str
    model: str
    response: dict[str, Any] | None = None
    stream: AsyncIterator[dict[str, Any]] | None = None
    attempts: list[Attempt] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)  # filled when the stream finishes, for streams

    @property
    def target(self) -> str:
        return f"{self.provider}/{self.model}"


class AllTargetsFailed(Exception):
    def __init__(self, attempts: list[Attempt], last: ProviderError | None):
        self.attempts = attempts
        self.last = last
        super().__init__("all targets failed")

    @property
    def status(self) -> int:
        if self.last and self.last.status == 429 and all(a.status == 429 for a in self.attempts if a.status):
            return 429
        return 503


class ClientRequestError(Exception):
    def __init__(self, err: ProviderError, attempts: list[Attempt]):
        self.err = err
        self.attempts = attempts
        super().__init__(str(err))


# --------------------------------------------------------------------------- router
class Router:
    def __init__(
        self,
        cfg: GatewayConfig,
        providers: dict[str, Provider] | None = None,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.providers = providers or {n: build_provider(n, p) for n, p in cfg.providers.items()}
        self.breaker = CircuitBreaker(cfg.breaker.failure_threshold, cfg.breaker.cooldown_s, clock)
        self.sleep = sleep

    def _plan(self, requested: str) -> list[tuple[str, str]]:
        return self.cfg.resolve(requested)

    def _ordered(self, targets: list[tuple[str, str]], attempts: list[Attempt]) -> list[tuple[str, str]]:
        allowed = []
        for p, m in targets:
            key = f"{p}/{m}"
            if self.breaker.allow(key):
                allowed.append((p, m))
            else:
                attempts.append(Attempt(key, "skipped_open_circuit"))
        # If every circuit is open, still try the first target rather than failing without a call.
        return allowed or targets[:1]

    async def _backoff(self, err: ProviderError, i: int) -> bool:
        """Sleep before retrying the same target. Returns False if the wait would be too long."""
        wait = err.retry_after if err.retry_after is not None else self.cfg.retry.backoff_s * (2**i)
        if wait > self.cfg.retry.max_retry_after_s:
            return False
        await self.sleep(wait)
        return True

    async def complete(self, requested: str, body: dict[str, Any]) -> RouteResult:
        attempts: list[Attempt] = []
        last: ProviderError | None = None
        for p, m in self._ordered(self._plan(requested), attempts):
            key, provider = f"{p}/{m}", self.providers[p]
            for i in range(self.cfg.retry.attempts_per_target):
                t0 = time.perf_counter()
                try:
                    resp = await provider.chat(m, body)
                except ProviderError as e:
                    attempts.append(Attempt(key, "error", e.status, str(e), (time.perf_counter() - t0) * 1000))
                    last = e
                    if not e.fallbackable:
                        raise ClientRequestError(e, attempts) from None
                    self.breaker.failure(key)
                    if (
                        e.retryable
                        and i + 1 < self.cfg.retry.attempts_per_target
                        and self.breaker.state(key) == "closed"
                    ):
                        if await self._backoff(e, i):
                            continue
                    break
                attempts.append(Attempt(key, "ok", 200, None, (time.perf_counter() - t0) * 1000))
                self.breaker.success(key)
                usage = resp.get("usage") or {}
                if not usage:
                    text = "".join(c.get("message", {}).get("content") or "" for c in resp.get("choices", []))
                    usage = {
                        "prompt_tokens": estimate_tokens(prompt_text(body)),
                        "completion_tokens": estimate_tokens(text),
                    }
                return RouteResult(p, m, response=resp, attempts=attempts, usage=usage)
        raise AllTargetsFailed(attempts, last)

    async def open_stream(self, requested: str, body: dict[str, Any]) -> RouteResult:
        attempts: list[Attempt] = []
        last: ProviderError | None = None
        for p, m in self._ordered(self._plan(requested), attempts):
            key, provider = f"{p}/{m}", self.providers[p]
            for i in range(self.cfg.retry.attempts_per_target):
                t0 = time.perf_counter()
                agen = provider.stream(m, body)
                try:
                    first = await agen.__anext__()
                except StopAsyncIteration:
                    first = None
                except ProviderError as e:
                    attempts.append(Attempt(key, "error", e.status, str(e), (time.perf_counter() - t0) * 1000))
                    last = e
                    if not e.fallbackable:
                        raise ClientRequestError(e, attempts) from None
                    self.breaker.failure(key)
                    if (
                        e.retryable
                        and i + 1 < self.cfg.retry.attempts_per_target
                        and self.breaker.state(key) == "closed"
                    ):
                        if await self._backoff(e, i):
                            continue
                    break
                attempts.append(Attempt(key, "ok", 200, None, (time.perf_counter() - t0) * 1000))
                self.breaker.success(key)
                result = RouteResult(p, m, attempts=attempts)
                result.stream = self._relay(first, agen, body, result)
                return result
        raise AllTargetsFailed(attempts, last)

    @staticmethod
    async def _relay(first, agen, body, result: RouteResult) -> AsyncIterator[dict[str, Any]]:
        text = []

        def note(chunk: dict[str, Any]) -> None:
            if chunk.get("usage"):
                result.usage = chunk["usage"]
            for c in chunk.get("choices", []):
                text.append((c.get("delta") or {}).get("content") or "")

        if first is not None:
            note(first)
            yield first
        async for chunk in agen:
            note(chunk)
            yield chunk
        if not result.usage:
            result.usage = {
                "prompt_tokens": estimate_tokens(prompt_text(body)),
                "completion_tokens": estimate_tokens("".join(text)),
            }

    async def aclose(self) -> None:
        for p in self.providers.values():
            await p.aclose()
