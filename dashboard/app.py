"""Live demo: the gateway running in-process, with a playground, an outage simulator, a load test and a dashboard.

Real traffic goes to Groq (gpt-oss-120b / gpt-oss-20b) when GROQ_API_KEY is set; a simulated backup provider sits
at the end of every route so fallback is always visible. Without a key the whole demo runs on simulated providers.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llmgw.config import GatewayConfig  # noqa: E402
from llmgw.gateway import Gateway, GatewayError, TokenBucket  # noqa: E402
from llmgw.providers.chaos import FaultInjector  # noqa: E402
from llmgw.store import Store  # noqa: E402

GITHUB = "https://github.com/Tanzil-Ali-Rizvi/llm-gateway"
DB = os.environ.get("LLMGW_DEMO_DB", "/tmp/llmgw-demo.db")
DEMO_RPM, DEMO_BUDGET = 20, 0.50  # shared limits that protect the demo's API quota

st.set_page_config(page_title="LLM gateway", page_icon="🔀", layout="wide")


def groq_key() -> str | None:
    try:
        return st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    except Exception:
        return os.environ.get("GROQ_API_KEY")


def demo_config(live: bool) -> GatewayConfig:
    backup = {
        "type": "mock",
        "fault": {"latency_ms": 400, "latency_jitter_ms": 100},
        "models": {"sim-small": {"input_per_1m": 0.05, "output_per_1m": 0.08}},
    }
    if live:
        providers = {
            "groq": {
                "type": "openai",
                "base_url": "https://api.groq.com/openai/v1",
                "api_key_env": "GROQ_API_KEY",
                "timeout_s": 30,
                "models": {
                    "openai/gpt-oss-120b": {"input_per_1m": 0.15, "output_per_1m": 0.60},
                    "openai/gpt-oss-20b": {"input_per_1m": 0.075, "output_per_1m": 0.30},
                },
            },
            "backup": backup,
        }
        routes = {
            "smart": ["groq/openai/gpt-oss-120b", "groq/openai/gpt-oss-20b", "backup/sim-small"],
            "fast": ["groq/openai/gpt-oss-20b", "backup/sim-small"],
        }
    else:
        providers = {
            "sim": {
                "type": "mock",
                "fault": {"latency_ms": 250, "latency_jitter_ms": 80},
                "models": {
                    "big": {"input_per_1m": 0.15, "output_per_1m": 0.60},
                    "medium": {"input_per_1m": 0.075, "output_per_1m": 0.30},
                },
            },
            "backup": backup,
        }
        routes = {"smart": ["sim/big", "sim/medium", "backup/sim-small"], "fast": ["sim/medium", "backup/sim-small"]}
    return GatewayConfig.model_validate(
        {
            "providers": providers,
            "routes": routes,
            "default_route": "smart",
            "database": DB,
            "require_auth": False,
            "breaker": {"failure_threshold": 3, "cooldown_s": 20},
            "retry": {"attempts_per_target": 2, "backoff_s": 0.2, "max_retry_after_s": 3},
            "cache": {"enabled": True, "ttl_s": 6 * 3600},
        }
    )


# ---------------------------------------------------------------- shared resources
@st.cache_resource
def event_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    return loop


@st.cache_resource
def shared_store() -> Store:
    return Store(DB)


@st.cache_resource
def shared_limiter() -> TokenBucket:
    return TokenBucket()


@st.cache_resource
def demo_key_id() -> int:
    store = shared_store()
    existing = [k for k in store.list_keys() if k["name"] == "public-demo"]
    if existing:
        return existing[0]["id"]
    _, key = store.create_key("public-demo", daily_budget_usd=DEMO_BUDGET, rpm_limit=DEMO_RPM)
    return key.id


def run(coro):
    return asyncio.run_coroutine_threadsafe(coro, event_loop()).result(timeout=120)


def session_gateway() -> Gateway:
    """One gateway per visitor (own circuit breakers and outage switch), sharing the store and the rate limit."""
    if "gw" not in st.session_state:
        live = bool(groq_key())
        if live:
            os.environ["GROQ_API_KEY"] = groq_key()
        gw = Gateway(demo_config(live), store=shared_store())
        gw.limiter = shared_limiter()
        primary = next(iter(gw.router.providers))
        gw.router.providers[primary] = FaultInjector(gw.router.providers[primary])
        st.session_state.gw, st.session_state.live = gw, live
    return st.session_state.gw


gw = session_gateway()
store = shared_store()
key = store.get_key(demo_key_id())
primary_name = next(iter(gw.router.providers))

# ---------------------------------------------------------------- header
st.title("🔀 LLM gateway")
st.markdown(
    "One **OpenAI-compatible API** in front of many model providers: routing with **retries, fallback and circuit "
    "breakers**, a **response cache**, **per-key budgets and rate limits**, and **cost tracking**. "
    f"This page runs the real gateway code in-process. · [Code, tests and benchmarks]({GITHUB})"
)
if st.session_state.live:
    st.caption(
        "Live mode: routes call **Groq** (gpt-oss-120b, gpt-oss-20b), with a simulated backup provider last in "
        f"each route. Shared demo limits: {DEMO_RPM} requests/minute, ${DEMO_BUDGET:.2f}/day."
    )
else:
    st.caption("Offline mode: no API key configured, so every provider is simulated.")

play, load, dash = st.tabs(["Playground", "Load test (simulated)", "Dashboard"])

# ---------------------------------------------------------------- playground
with play:
    left, right = st.columns([3, 2])
    with left:
        route = st.radio(
            "Route (the `model` your app asks for)",
            list(gw.cfg.routes),
            horizontal=True,
            format_func=lambda r: f"{r}: {' → '.join(gw.cfg.routes[r])}",
        )
        prompt = st.text_area(
            "Prompt", "In two sentences, why do production apps put a gateway in front of LLM APIs?", height=90
        )
        temperature = st.select_slider(
            "Temperature",
            [0.0, 0.3, 0.7, 1.0],
            value=0.0,
            help="Temperature 0 responses are cached; ask the same thing twice to see a cache hit.",
        )
    with right:
        outage = st.slider(
            f"Simulated failure rate of `{primary_name}` (this session only)",
            0,
            100,
            0,
            step=10,
            help="Injects HTTP 503s in front of the primary provider to exercise retries, fallback "
            "and the circuit breaker.",
        )
        gw.router.providers[primary_name].error_rate = outage / 100
        if outage:
            st.caption(
                "Tip: a repeated temperature-0 prompt is answered from the cache without touching any "
                "provider. Change the prompt to watch retries and fallback."
            )
        circuits = gw.router.breaker.snapshot()
        if circuits:
            st.dataframe(
                pd.DataFrame([{"target": t, **s} for t, s in circuits.items()]), hide_index=True, width="stretch"
            )
    if st.button("Send", type="primary"):
        body = {
            "model": route,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": 300,
        }
        try:
            out = run(gw.chat(body, key))
            h = out.headers
            st.success(out.body["choices"][0]["message"]["content"] or "(empty response)")
            c = st.columns(5)
            c[0].metric("Served by", h.get("x-gateway-target", "cache"))
            c[1].metric("Cache", h.get("x-gateway-cache"))
            c[2].metric("Cost (USD)", f"${float(h.get('x-gateway-cost-usd', 0)):.6f}")
            c[3].metric("Saved by cache", f"${float(h.get('x-gateway-saved-usd', 0)):.6f}")
            c[4].metric("Latency", f"{float(h.get('x-gateway-latency-ms', 0)):.0f} ms")
            st.markdown("**Routing trace**")
            st.dataframe(pd.DataFrame(out.trace), hide_index=True, width="stretch")
        except GatewayError as e:
            st.error(f"{e.status} {e.code}: {e.message}")
            if e.extra.get("attempts"):
                st.dataframe(pd.DataFrame(e.extra["attempts"]), hide_index=True, width="stretch")

# ---------------------------------------------------------------- load test
with load:
    st.markdown(
        "Fire concurrent requests at **simulated** providers (no API quota used) and compare gateway "
        "settings. Primary ~300 ms, backup ~450 ms."
    )
    c = st.columns(4)
    n = c[0].slider("Requests", 50, 500, 200, step=50)
    err = c[1].slider("Primary failure rate", 0, 100, 30, step=10)
    fallback = c[2].toggle("Fallback to backup", True)
    breaker = c[3].toggle("Circuit breaker", True)
    if st.button("Run load test"):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
        from chaos import config as bench_config  # type: ignore
        from chaos import run as bench_run  # type: ignore
        from chaos import unique_bodies  # type: ignore

        bgw = Gateway(bench_config(err / 100, fallback, 2, breaker, seed=int(time.time())), store=Store(":memory:"))
        with st.spinner("Running..."):
            res = run(bench_run(bgw, unique_bodies(n)))
        c = st.columns(5)
        c[0].metric("Success", f"{res['success_rate']}%")
        c[1].metric("Served by fallback", f"{res['fallback_rate']}%")
        c[2].metric(
            "Upstream calls",
            res["upstream_calls"],
            delta=f"{res['upstream_calls'] - n:+d} vs requests",
            delta_color="inverse",
        )
        c[3].metric("p50 latency", f"{res['p50_ms']:.0f} ms")
        c[4].metric("p95 latency", f"{res['p95_ms']:.0f} ms")
        st.caption(
            "Try: 100% failure with the circuit breaker off, then on. The breaker stops wasting calls on a dead "
            "provider and cuts latency."
        )

# ---------------------------------------------------------------- dashboard
with dash:
    since = time.time() - 24 * 3600
    by_target = pd.DataFrame(store.usage("provider", since))
    recent = pd.DataFrame(store.recent(200))
    recent = recent[recent["ts"] >= since] if not recent.empty else recent
    if recent.empty:
        st.info("No traffic yet in the last 24 hours. Send a few prompts in the Playground.")
    else:
        total = len(recent)
        ok = int((recent["status"] == 200).sum())
        hits = int(recent["cache"].isin(["exact", "semantic"]).sum())
        lat = recent.loc[recent["status"] == 200, "latency_ms"]
        c = st.columns(6)
        c[0].metric("Requests (24 h)", total)
        c[1].metric("Success rate", f"{100 * ok / total:.1f}%")
        c[2].metric("Spend", f"${recent['cost_usd'].fillna(0).sum():.5f}")
        c[3].metric("Saved by cache", f"${recent['saved_usd'].fillna(0).sum():.5f}")
        c[4].metric("Fallback rate", f"{100 * recent['fallback'].fillna(0).sum() / total:.1f}%")
        c[5].metric("p50 / p95", f"{lat.quantile(0.5):.0f} / {lat.quantile(0.95):.0f} ms" if len(lat) else "–")
        a, b = st.columns(2)
        with a:
            st.markdown("**Requests by target**")
            st.bar_chart(by_target.set_index("grp")["requests"])
        with b:
            st.markdown("**Spend by target (USD)**")
            st.bar_chart(by_target.set_index("grp")["cost_usd"])
        st.markdown("**Recent requests**")
        show = recent[
            [
                "ts",
                "route",
                "provider",
                "model",
                "status",
                "cache",
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "latency_ms",
                "attempts",
                "fallback",
            ]
        ].copy()
        show["ts"] = pd.to_datetime(show["ts"], unit="s").dt.strftime("%H:%M:%S")
        st.dataframe(show, hide_index=True, width="stretch")
    budget_left = max(0.0, DEMO_BUDGET - store.spent_since(key.id, time.time() - (time.time() % 86400)))
    st.caption(f"Demo key budget left today: ${budget_left:.4f} of ${DEMO_BUDGET:.2f}.")
