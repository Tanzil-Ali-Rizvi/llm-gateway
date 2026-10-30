import pytest
from conftest import make_cfg

from llmgw.config import ProviderConfig
from llmgw.providers.base import ProviderError
from llmgw.providers.mock import MockProvider
from llmgw.router import AllTargetsFailed, CircuitBreaker, ClientRequestError, Router

BODY = {"messages": [{"role": "user", "content": "hi"}]}


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def router(cfg, clock=None):
    sleeps = []

    async def sleep(s):
        sleeps.append(s)

    r = Router(cfg, sleep=sleep, clock=clock or Clock())
    return r, sleeps


async def test_healthy_primary_serves_without_fallback():
    r, _ = router(make_cfg())
    res = await r.complete("smart", BODY)
    assert res.target == "primary/big" and [a.outcome for a in res.attempts] == ["ok"]
    assert res.usage["prompt_tokens"] > 0


async def test_falls_back_after_retrying_primary():
    r, sleeps = router(make_cfg(primary_error_rate=1.0))
    res = await r.complete("smart", BODY)
    assert res.target == "backup/small"
    assert [(a.target, a.outcome) for a in res.attempts] == [
        ("primary/big", "error"),
        ("primary/big", "error"),
        ("backup/small", "ok"),
    ]
    assert len(sleeps) == 1  # one backoff between the two primary attempts


async def test_retry_after_is_honoured_for_429():
    r, sleeps = router(make_cfg(primary_error_rate=1.0, primary_status=429))
    await r.complete("smart", BODY)
    assert sleeps == [1.0]  # mock sends Retry-After: 1


async def test_circuit_opens_and_skips_primary_then_half_opens():
    clock = Clock()
    cfg = make_cfg(primary_error_rate=1.0)
    r, _ = router(cfg, clock)
    await r.complete("smart", BODY)  # 2 failures
    await r.complete("smart", BODY)  # 3rd failure opens the circuit (threshold 3)
    assert r.breaker.state("primary/big") == "open"
    primary = r.providers["primary"]
    calls = primary.calls
    res = await r.complete("smart", BODY)
    assert res.attempts[0].outcome == "skipped_open_circuit" and primary.calls == calls  # no call wasted
    clock.t += 31  # cooldown passes: one trial call goes through
    primary.fault.error_rate = 0.0
    res = await r.complete("smart", BODY)
    assert res.target == "primary/big" and r.breaker.state("primary/big") == "closed"


async def test_all_circuits_open_still_tries_first_target():
    r, _ = router(make_cfg(primary_error_rate=1.0))
    for _ in range(3):
        with pytest.raises(AllTargetsFailed):
            await r.complete("solo", BODY)
    assert r.breaker.state("primary/big") == "open"
    with pytest.raises(AllTargetsFailed) as e:
        await r.complete("solo", BODY)
    assert any(a.outcome == "error" for a in e.value.attempts)


async def test_client_errors_are_not_retried_or_routed_elsewhere():
    r, _ = router(make_cfg(primary_error_rate=1.0, primary_status=400))
    with pytest.raises(ClientRequestError) as e:
        await r.complete("smart", BODY)
    assert e.value.err.status == 400 and len(e.value.attempts) == 1
    assert r.providers["backup"].calls == 0


async def test_all_429_reports_429():
    cfg = make_cfg(primary_error_rate=1.0, primary_status=429)
    cfg.providers["backup"].fault.error_rate = 1.0
    cfg.providers["backup"].fault.error_status = 429
    r, _ = router(cfg)
    with pytest.raises(AllTargetsFailed) as e:
        await r.complete("smart", BODY)
    assert e.value.status == 429


async def test_stream_falls_back_before_first_chunk_and_reports_usage():
    r, _ = router(make_cfg(primary_error_rate=1.0))
    res = await r.open_stream("smart", BODY)
    chunks = [c async for c in res.stream]
    text = "".join((c["choices"][0]["delta"] or {}).get("content") or "" for c in chunks)
    assert res.target == "backup/small" and "Mock answer to: hi" in text
    assert res.usage["completion_tokens"] > 0


def test_breaker_reopens_after_failed_trial():
    clock = Clock()
    b = CircuitBreaker(2, 10, clock)
    b.failure("t")
    b.failure("t")
    assert b.state("t") == "open" and not b.allow("t")
    clock.t = 11
    assert b.allow("t") and not b.allow("t")  # exactly one trial
    b.failure("t")
    assert b.state("t") == "open"


def test_provider_error_classification():
    assert ProviderError("x", 503).retryable and ProviderError("x", 404).fallbackable
    assert not ProviderError("x", 400).fallbackable and not ProviderError("x", 401).retryable


async def test_mock_provider_is_deterministic_with_seed():
    cfg = ProviderConfig(type="mock", fault={"error_rate": 0.5, "seed": 7, "latency_ms": 0})
    outcomes = []
    for _ in range(2):
        p = MockProvider("m", cfg.model_copy(deep=True))
        o = []
        for _ in range(10):
            try:
                await p.chat("x", BODY)
                o.append(1)
            except ProviderError:
                o.append(0)
        outcomes.append(o)
    assert outcomes[0] == outcomes[1] and 0 < sum(outcomes[0]) < 10


async def test_fault_injector_wraps_real_providers_and_triggers_fallback():
    from llmgw.providers.chaos import FaultInjector

    cfg = make_cfg()
    r, _ = router(cfg)
    r.providers["primary"] = FaultInjector(r.providers["primary"], error_rate=1.0)
    res = await r.complete("smart", BODY)
    assert res.target == "backup/small" and r.providers["primary"].calls == 2
    r.providers["primary"].error_rate = 0.0
    r.breaker.success("primary/big")
    assert (await r.complete("smart", BODY)).target == "primary/big"
