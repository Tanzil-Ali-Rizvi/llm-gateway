import httpx
import pytest
import pytest_asyncio

from llmgw.app import create_app
from llmgw.cache import HashingEmbedder
from llmgw.config import GatewayConfig
from llmgw.gateway import Gateway
from llmgw.store import Store

MASTER = "master-test-key"


def make_cfg(primary_error_rate=0.0, primary_status=503, semantic=False, **over) -> GatewayConfig:
    data = {
        "providers": {
            "primary": {
                "type": "mock",
                "fault": {"error_rate": primary_error_rate, "error_status": primary_status, "latency_ms": 1, "seed": 1},
                "models": {"big": {"input_per_1m": 1.0, "output_per_1m": 2.0}},
            },
            "backup": {
                "type": "mock",
                "fault": {"latency_ms": 1, "seed": 2},
                "models": {"small": {"input_per_1m": 0.5, "output_per_1m": 1.0}},
            },
        },
        "routes": {"smart": ["primary/big", "backup/small"], "solo": ["primary/big"]},
        "default_route": "smart",
        "cache": {"enabled": True, "semantic": semantic, "semantic_threshold": 0.8},
        "breaker": {"failure_threshold": 3, "cooldown_s": 30},
        "retry": {"attempts_per_target": 2, "backoff_s": 0.0, "max_retry_after_s": 5},
        "database": ":memory:",
        "master_key_env": "TEST_MASTER_KEY",
    }
    data.update(over)
    return GatewayConfig.model_validate(data)


@pytest.fixture(autouse=True)
def _master(monkeypatch):
    monkeypatch.setenv("TEST_MASTER_KEY", MASTER)


async def _noop_sleep(_):
    return None


def make_gateway(cfg: GatewayConfig) -> Gateway:
    gw = Gateway(cfg, store=Store(":memory:"), embedder=HashingEmbedder() if cfg.cache.semantic else None)
    gw.router.sleep = _noop_sleep
    return gw


@pytest_asyncio.fixture
async def client_factory():
    opened = []

    async def make(cfg: GatewayConfig):
        gw = make_gateway(cfg)
        app = create_app(gateway=gw)
        ctx = app.router.lifespan_context(app)
        await ctx.__aenter__()
        client = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://gw")
        opened.append((client, ctx))
        return client, gw

    yield make
    for client, ctx in opened:
        await client.aclose()
        await ctx.__aexit__(None, None, None)


def chat(content="Hello", model="smart", **kw):
    return {"model": model, "messages": [{"role": "user", "content": content}], **kw}


def admin():
    return {"Authorization": f"Bearer {MASTER}"}
