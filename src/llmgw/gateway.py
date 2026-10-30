"""Request pipeline: auth -> route -> rate limit -> budget -> cache -> router (retry/fallback) -> cost -> log."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from .cache import Cache, Embedder
from .config import GatewayConfig, ModelPrice
from .providers import Provider
from .router import AllTargetsFailed, ClientRequestError, Router, RouteResult
from .store import ApiKey, Store, start_of_utc_day


def cost_usd(usage: dict[str, Any], price: ModelPrice) -> float:
    p = usage.get("prompt_tokens") or 0
    c = usage.get("completion_tokens") or 0
    return (p * price.input_per_1m + c * price.output_per_1m) / 1_000_000


class GatewayError(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
        extra: dict[str, Any] | None = None,
    ):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message
        self.headers = headers or {}
        self.extra = extra or {}

    def body(self) -> dict[str, Any]:
        return {"error": {"message": self.message, "type": self.code, "code": self.code, **self.extra}}


class TokenBucket:
    """Per-key requests-per-minute limiter (in memory, per gateway instance)."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.state: dict[int, tuple[float, float]] = {}

    def take(self, key_id: int, rpm: int) -> float:
        """Consume one token. Returns 0 if allowed, else seconds until a token is available."""
        now = self.clock()
        tokens, last = self.state.get(key_id, (float(rpm), now))
        tokens = min(float(rpm), tokens + (now - last) * rpm / 60.0)
        if tokens >= 1:
            self.state[key_id] = (tokens - 1, now)
            return 0.0
        self.state[key_id] = (tokens, now)
        return (1 - tokens) * 60.0 / rpm


@dataclass
class Outcome:
    """What the HTTP layer needs to send back."""

    status: int = 200
    body: dict[str, Any] | None = None
    stream: AsyncIterator[bytes] | None = None
    headers: dict[str, str] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)  # routing attempts, for debugging and the dashboard


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter(
            "llmgw_requests", "Requests", ["route", "target", "status", "cache"], registry=self.registry
        )
        self.tokens = Counter("llmgw_tokens", "Tokens", ["route", "target", "kind"], registry=self.registry)
        self.cost = Counter("llmgw_cost_usd", "Spend in USD", ["route", "target"], registry=self.registry)
        self.saved = Counter("llmgw_cache_saved_usd", "Spend avoided by the cache", ["route"], registry=self.registry)
        self.fallbacks = Counter(
            "llmgw_fallbacks", "Requests served by a non-primary target", ["route"], registry=self.registry
        )
        self.latency = Histogram(
            "llmgw_latency_seconds",
            "End-to-end latency",
            ["route"],
            registry=self.registry,
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
        )
        self.breaker = Gauge(
            "llmgw_circuit_open", "1 if the target's circuit is open", ["target"], registry=self.registry
        )


class Gateway:
    def __init__(
        self,
        cfg: GatewayConfig,
        store: Store | None = None,
        providers: dict[str, Provider] | None = None,
        embedder: Embedder | None = None,
        router: Router | None = None,
    ):
        self.cfg = cfg
        self.store = store or Store(cfg.database)
        self.router = router or Router(cfg, providers)
        self.cache = Cache(cfg.cache, self.store, embedder)
        self.limiter = TokenBucket()
        self.metrics = Metrics()

    # ------------------------------------------------------------------ auth
    def authenticate(self, authorization: str | None) -> ApiKey | None:
        """Returns the virtual key, or None for the master key / open gateways. Raises on bad credentials."""
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token and self.cfg.master_key and token == self.cfg.master_key:
            return None
        if token:
            key = self.store.verify_key(token)
            if key:
                return key
            raise GatewayError(401, "invalid_api_key", "Invalid API key.")
        if self.cfg.require_auth:
            raise GatewayError(401, "missing_api_key", "Send 'Authorization: Bearer <virtual key>'.")
        return None

    def require_admin(self, authorization: str | None) -> None:
        token = (authorization or "").removeprefix("Bearer ").strip()
        if not self.cfg.master_key or token != self.cfg.master_key:
            raise GatewayError(403, "forbidden", "Admin endpoints need the master key.")

    # ------------------------------------------------------------------ main entry
    async def chat(self, body: dict[str, Any], key: ApiKey | None, cache_mode: str = "auto") -> Outcome:
        t0 = time.perf_counter()
        route = body.get("model") or ""
        try:
            targets = self.cfg.resolve(route)
        except (KeyError, ValueError):
            raise GatewayError(
                404, "model_not_found", f"Unknown model '{route}'.", extra={"available": sorted(self.cfg.routes)}
            ) from None
        if key and not key.allows(route):
            raise GatewayError(403, "route_not_allowed", f"This key may not use '{route}'.")
        if key and key.rpm_limit:
            wait = self.limiter.take(key.id, key.rpm_limit)
            if wait:
                raise GatewayError(
                    429,
                    "rate_limited",
                    f"Rate limit of {key.rpm_limit} requests/minute exceeded.",
                    headers={"retry-after": f"{wait:.1f}"},
                )
        if key and key.daily_budget_usd is not None:
            spent = self.store.spent_since(key.id, start_of_utc_day())
            if spent >= key.daily_budget_usd:
                raise GatewayError(
                    429, "budget_exceeded", f"Daily budget of ${key.daily_budget_usd:.4f} reached (spent ${spent:.4f})."
                )

        stream = bool(body.get("stream"))
        primary = f"{targets[0][0]}/{targets[0][1]}"
        base = {"key_id": key.id if key else None, "route": route, "stream": int(stream)}

        # ---- cache
        cacheable = self.cache.eligible(body, cache_mode)
        if cacheable:
            hit = self.cache.lookup(route, body)
            if hit:
                kind, resp, orig_cost = hit
                ms = (time.perf_counter() - t0) * 1000
                usage = resp.get("usage") or {}
                self.store.log(
                    **base,
                    provider="cache",
                    model=resp.get("model"),
                    status=200,
                    cache=kind,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    cost_usd=0.0,
                    saved_usd=orig_cost,
                    latency_ms=ms,
                    attempts=0,
                    fallback=0,
                )
                self.metrics.requests.labels(route, "cache", "200", kind).inc()
                self.metrics.saved.labels(route).inc(orig_cost)
                self.metrics.latency.labels(route).observe(ms / 1000)
                headers = {
                    "x-gateway-cache": kind,
                    "x-gateway-cost-usd": "0",
                    "x-gateway-saved-usd": f"{orig_cost:.8f}",
                    "x-gateway-route": route,
                    "x-gateway-latency-ms": f"{ms:.1f}",
                }
                trace = [{"target": "cache", "outcome": f"{kind} hit", "status": 200, "error": None, "latency_ms": ms}]
                if stream:
                    return Outcome(stream=_replay_as_stream(resp), headers=headers, trace=trace)
                return Outcome(body=resp, headers=headers, trace=trace)

        cache_label = "miss" if cacheable else "bypass"

        # ---- route
        try:
            if stream:
                result = await self.router.open_stream(route, body)
            else:
                result = await self.router.complete(route, body)
        except AllTargetsFailed as e:
            ms = (time.perf_counter() - t0) * 1000
            self.store.log(
                **base,
                status=e.status,
                cache=cache_label,
                latency_ms=ms,
                attempts=len(e.attempts),
                error=str(e.last)[:500] if e.last else "all targets unavailable",
            )
            self.metrics.requests.labels(route, "none", str(e.status), cache_label).inc()
            raise GatewayError(
                e.status,
                "upstream_unavailable",
                "All upstream targets failed.",
                extra={"attempts": [a.__dict__ for a in e.attempts]},
            ) from None
        except ClientRequestError as e:
            ms = (time.perf_counter() - t0) * 1000
            self.store.log(
                **base,
                status=e.err.status,
                cache=cache_label,
                latency_ms=ms,
                attempts=len(e.attempts),
                error=str(e.err)[:500],
            )
            raise GatewayError(e.err.status, "upstream_rejected_request", str(e.err)) from None
        self._update_breaker_gauges()

        if stream:
            return Outcome(
                stream=self._finish_stream(result, body, route, base, primary, cacheable, cache_label, t0),
                headers=self._headers(result, route, primary, cache_label, None, t0),
                trace=[a.__dict__ for a in result.attempts],
            )

        cost = cost_usd(result.usage, self.cfg.price(result.provider, result.model))
        if cacheable:
            self.cache.save(route, body, result.response, cost)
        self._record(result, route, base, primary, cache_label, cost, t0)
        return Outcome(
            body=result.response,
            headers=self._headers(result, route, primary, cache_label, cost, t0),
            trace=[a.__dict__ for a in result.attempts],
        )

    # ------------------------------------------------------------------ helpers
    def _headers(
        self, r: RouteResult, route: str, primary: str, cache_label: str, cost: float | None, t0: float
    ) -> dict[str, str]:
        h = {
            "x-gateway-route": route,
            "x-gateway-target": r.target,
            "x-gateway-cache": cache_label,
            "x-gateway-attempts": str(sum(a.outcome != "skipped_open_circuit" for a in r.attempts)),
            "x-gateway-fallback": "true" if r.target != primary else "false",
            "x-gateway-latency-ms": f"{(time.perf_counter() - t0) * 1000:.1f}",
        }
        if cost is not None:
            h["x-gateway-cost-usd"] = f"{cost:.8f}"
        return h

    def _record(
        self, r: RouteResult, route: str, base: dict[str, Any], primary: str, cache_label: str, cost: float, t0: float
    ) -> None:
        ms = (time.perf_counter() - t0) * 1000
        fallback = int(r.target != primary)
        self.store.log(
            **base,
            provider=r.provider,
            model=r.model,
            status=200,
            cache=cache_label,
            prompt_tokens=r.usage.get("prompt_tokens", 0),
            completion_tokens=r.usage.get("completion_tokens", 0),
            cost_usd=cost,
            saved_usd=0.0,
            latency_ms=ms,
            attempts=sum(a.outcome != "skipped_open_circuit" for a in r.attempts),
            fallback=fallback,
        )
        self.metrics.requests.labels(route, r.target, "200", cache_label).inc()
        self.metrics.tokens.labels(route, r.target, "prompt").inc(r.usage.get("prompt_tokens", 0))
        self.metrics.tokens.labels(route, r.target, "completion").inc(r.usage.get("completion_tokens", 0))
        self.metrics.cost.labels(route, r.target).inc(cost)
        self.metrics.latency.labels(route).observe(ms / 1000)
        if fallback:
            self.metrics.fallbacks.labels(route).inc()

    async def _finish_stream(
        self,
        r: RouteResult,
        body: dict[str, Any],
        route: str,
        base: dict[str, Any],
        primary: str,
        cacheable: bool,
        cache_label: str,
        t0: float,
    ) -> AsyncIterator[bytes]:
        content, finish, first = [], None, None
        try:
            async for chunk in r.stream:
                first = first or chunk
                for c in chunk.get("choices", []):
                    content.append((c.get("delta") or {}).get("content") or "")
                    finish = c.get("finish_reason") or finish
                yield f"data: {json.dumps(chunk)}\n\n".encode()
            yield b"data: [DONE]\n\n"
        finally:
            cost = cost_usd(r.usage, self.cfg.price(r.provider, r.model))
            self._record(r, route, base, primary, cache_label, cost, t0)
            if cacheable and first is not None and finish:
                full = {
                    "id": first.get("id"),
                    "object": "chat.completion",
                    "created": first.get("created"),
                    "model": first.get("model"),
                    "usage": r.usage,
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "".join(content)},
                            "finish_reason": finish,
                        }
                    ],
                }
                self.cache.save(route, body, full, cost)

    def _update_breaker_gauges(self) -> None:
        for target, s in self.router.breaker.snapshot().items():
            self.metrics.breaker.labels(target).set(1 if s["state"] == "open" else 0)


async def _replay_as_stream(resp: dict[str, Any]) -> AsyncIterator[bytes]:
    """Serve a cached (non-streamed) response to a streaming client as SSE chunks."""
    cid = resp.get("id") or f"chatcmpl-{uuid.uuid4().hex[:12]}"
    msg = (resp.get("choices") or [{}])[0].get("message", {})
    base = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": resp.get("created", int(time.time())),
        "model": resp.get("model"),
    }
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': msg.get('content', '')}, 'finish_reason': None}]})}\n\n".encode()
    yield f"data: {json.dumps({**base, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': (resp.get('choices') or [{}])[0].get('finish_reason', 'stop')}], 'usage': resp.get('usage')})}\n\n".encode()
    yield b"data: [DONE]\n\n"
