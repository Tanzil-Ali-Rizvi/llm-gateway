"""Chaos and cache benchmarks with simulated providers (no API keys, reproducible).

Scenario 1, flaky primary: the primary fails 20% of calls (HTTP 503). Compare calling it directly with going
through the gateway with retries only, and with retries + fallback to a second provider.

Scenario 2, primary outage: the primary is completely down. Compare fallback without and with the circuit
breaker, measuring latency and wasted upstream calls.

Scenario 3, cache: 2,000 requests from a support-bot style workload (Zipf-distributed repeats over 300 distinct
questions, temperature 0). Compare spend with and without the exact-match cache.

Usage:  python benchmarks/chaos.py --out benchmarks/results
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmgw.config import GatewayConfig  # noqa: E402
from llmgw.gateway import Gateway, GatewayError  # noqa: E402
from llmgw.providers.base import ProviderError  # noqa: E402
from llmgw.store import Store  # noqa: E402

PRIMARY_MS, BACKUP_MS = 300, 450


def config(
    primary_error_rate: float, fallback: bool, retries: int, breaker: bool, seed: int = 1, latency_scale: float = 1.0
) -> GatewayConfig:
    cfg = GatewayConfig.model_validate(
        {
            "providers": {
                "primary": {
                    "type": "mock",
                    "fault": {
                        "error_rate": primary_error_rate,
                        "error_status": 503,
                        "latency_ms": PRIMARY_MS,
                        "latency_jitter_ms": 100,
                        "seed": seed,
                    },
                    "models": {"big": {"input_per_1m": 0.15, "output_per_1m": 0.60}},
                },
                "backup": {
                    "type": "mock",
                    "fault": {"latency_ms": BACKUP_MS, "latency_jitter_ms": 150, "seed": seed + 1},
                    "models": {"alt": {"input_per_1m": 0.30, "output_per_1m": 2.50}},
                },
            },
            "routes": {"smart": ["primary/big", "backup/alt"] if fallback else ["primary/big"]},
            "breaker": {"failure_threshold": 5 if breaker else 10**9, "cooldown_s": 30},
            "retry": {"attempts_per_target": retries, "backoff_s": 0.05, "max_retry_after_s": 5},
            "cache": {"enabled": True},
            "database": ":memory:",
            "require_auth": False,
        }
    )
    for p in cfg.providers.values():
        p.fault.latency_ms *= latency_scale
        p.fault.latency_jitter_ms *= latency_scale
    return cfg


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(q * len(xs)))], 1) if xs else 0.0


async def run(gw: Gateway, bodies: list[dict], concurrency: int = 25) -> dict:
    sem = asyncio.Semaphore(concurrency)
    lat, ok, fallbacks = [], 0, 0

    async def one(body):
        nonlocal ok, fallbacks
        async with sem:
            t0 = time.perf_counter()
            try:
                out = await gw.chat(body, None)
                ok += 1
                fallbacks += out.headers.get("x-gateway-fallback") == "true"
            except GatewayError:
                pass
            lat.append((time.perf_counter() - t0) * 1000)

    t0 = time.perf_counter()
    await asyncio.gather(*(one(b) for b in bodies))
    calls = sum(p.calls for p in gw.router.providers.values())
    rows = gw.store.usage("route")
    return {
        "requests": len(bodies),
        "success_rate": round(100 * ok / len(bodies), 2),
        "fallback_rate": round(100 * fallbacks / len(bodies), 2),
        "upstream_calls": calls,
        "p50_ms": pct(lat, 0.5),
        "p95_ms": pct(lat, 0.95),
        "p99_ms": pct(lat, 0.99),
        "cost_usd": round(sum(r["cost_usd"] or 0 for r in rows), 6),
        "wall_s": round(time.perf_counter() - t0, 1),
    }


async def direct(primary_error_rate: float, n: int, seed: int = 1) -> dict:
    """Baseline: the application calls the primary provider directly, no retries, no fallback."""
    gw = Gateway(config(primary_error_rate, False, 1, False, seed), store=Store(":memory:"))
    p = gw.router.providers["primary"]
    lat, ok = [], 0
    sem = asyncio.Semaphore(25)

    async def one(i):
        nonlocal ok
        async with sem:
            t0 = time.perf_counter()
            try:
                await p.chat("big", {"messages": [{"role": "user", "content": f"q{i}"}]})
                ok += 1
            except ProviderError:
                pass
            lat.append((time.perf_counter() - t0) * 1000)

    await asyncio.gather(*(one(i) for i in range(n)))
    return {
        "requests": n,
        "success_rate": round(100 * ok / n, 2),
        "fallback_rate": 0.0,
        "upstream_calls": p.calls,
        "p50_ms": pct(lat, 0.5),
        "p95_ms": pct(lat, 0.95),
        "p99_ms": pct(lat, 0.99),
    }


def unique_bodies(n: int) -> list[dict]:
    return [
        {
            "model": "smart",
            "messages": [{"role": "user", "content": f"Question {i}: explain item {i}"}],
            "temperature": 0.7,
        }
        for i in range(n)
    ]


def faq_workload(n: int, distinct: int, seed: int = 7) -> list[dict]:
    rng = random.Random(seed)
    weights = [1 / (k + 1) ** 1.1 for k in range(distinct)]  # Zipf-like popularity
    picks = rng.choices(range(distinct), weights=weights, k=n)
    system = {"role": "system", "content": "You are the support assistant for a UPI payments app. Answer briefly."}
    return [
        {
            "model": "smart",
            "temperature": 0,
            "max_tokens": 200,
            "messages": [system, {"role": "user", "content": f"FAQ #{k}: how do I handle situation {k} in the app?"}],
        }
        for k in picks
    ]


async def main_async(out: Path, n: int) -> dict:
    results: dict = {
        "setup": {
            "primary_latency_ms": f"{PRIMARY_MS}±100",
            "backup_latency_ms": f"{BACKUP_MS}±150",
            "concurrency": 25,
            "requests_per_run": n,
        }
    }

    flaky = {"direct to primary (no gateway)": await direct(0.2, n)}
    for name, fb, retries in (("gateway: retries only", False, 2), ("gateway: retries + fallback", True, 2)):
        gw = Gateway(config(0.2, fb, retries, True), store=Store(":memory:"))
        flaky[name] = await run(gw, unique_bodies(n))
    results["flaky_primary_20pct_errors"] = flaky

    outage = {}
    for name, breaker in (("fallback, no circuit breaker", False), ("fallback + circuit breaker", True)):
        gw = Gateway(config(1.0, True, 2, breaker), store=Store(":memory:"))
        outage[name] = await run(gw, unique_bodies(n))
    results["primary_down_100pct"] = outage

    cache = {}
    bodies = faq_workload(2 * n, 300)
    for name, mode in (("no cache", "no"), ("exact-match cache", "auto")):
        gw = Gateway(config(0.0, True, 2, True, latency_scale=0.02), store=Store(":memory:"))
        gw_chat = gw.chat

        async def chat(body, key, _m=mode, _c=gw_chat):
            return await _c(body, key, cache_mode=_m)

        gw.chat = chat  # type: ignore[method-assign]
        r = await run(gw, bodies, concurrency=1)  # sequential: a real cache sees earlier requests first
        saved = sum(x["saved_usd"] or 0 for x in gw.store.usage("cache"))
        hits = sum(x["requests"] for x in gw.store.usage("cache") if x["grp"] == "exact")
        r.update(cache_hit_rate=round(100 * hits / len(bodies), 1), saved_usd=round(saved, 6))
        cache[name] = r
    results["faq_cache_2000_requests"] = cache

    out.mkdir(parents=True, exist_ok=True)
    (out / "chaos.json").write_text(json.dumps(results, indent=2) + "\n")
    (out / "chaos.md").write_text(to_md(results))
    return results


def to_md(r: dict) -> str:
    lines = [
        "# Reliability and cache benchmark",
        "",
        f"Simulated providers (primary {r['setup']['primary_latency_ms']} ms, backup {r['setup']['backup_latency_ms']} ms), "
        f"{r['setup']['concurrency']} concurrent clients. Reproduce: `python benchmarks/chaos.py`.",
        "",
    ]
    for title, key in (
        ("Flaky primary: 20% of calls fail with HTTP 503", "flaky_primary_20pct_errors"),
        ("Primary completely down", "primary_down_100pct"),
    ):
        lines += [
            f"## {title}",
            "",
            "| Setup | Success | Served by fallback | Upstream calls | p50 ms | p95 ms |",
            "|---|---|---|---|---|---|",
        ]
        for name, v in r[key].items():
            lines.append(
                f"| {name} | **{v['success_rate']}%** | {v['fallback_rate']}% | {v['upstream_calls']} | {v['p50_ms']} | {v['p95_ms']} |"
            )
        lines.append("")
    lines += [
        "## Support-bot FAQ workload: 2,000 requests over 300 distinct questions",
        "",
        "| Setup | Spend (USD) | Cache hit rate | Upstream calls |",
        "|---|---|---|---|",
    ]
    for name, v in r["faq_cache_2000_requests"].items():
        lines.append(f"| {name} | ${v['cost_usd']:.5f} | {v['cache_hit_rate']}% | {v['upstream_calls']} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    ap.add_argument("-n", type=int, default=1000)
    a = ap.parse_args()
    res = asyncio.run(main_async(a.out, a.n))
    print(to_md(res))


if __name__ == "__main__":
    main()
