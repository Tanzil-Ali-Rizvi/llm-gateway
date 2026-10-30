"""The OpenAI-compatible adapter against a fake upstream (httpx.MockTransport): no network needed."""

import json

import httpx
import pytest

from llmgw.config import ProviderConfig
from llmgw.providers.base import ProviderError
from llmgw.providers.openai_compat import OpenAICompatProvider

BODY = {"model": "smart", "messages": [{"role": "user", "content": "hi"}], "temperature": 0, "stream": False}


def provider(handler, monkeypatch=None):
    cfg = ProviderConfig(type="openai", base_url="https://upstream.test/v1")
    client = httpx.AsyncClient(
        base_url="https://upstream.test/v1",
        transport=httpx.MockTransport(handler),
        headers={"Authorization": "Bearer k"},
    )
    return OpenAICompatProvider("up", cfg, client=client)


async def test_rewrites_model_and_returns_response():
    seen = {}

    def handler(req: httpx.Request):
        seen.update(json.loads(req.content))
        seen["auth"] = req.headers["authorization"]
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3}})

    out = await provider(handler).chat("llama-3.1-8b-instant", BODY)
    assert out["choices"][0]["message"]["content"] == "ok"
    assert seen["model"] == "llama-3.1-8b-instant" and "stream" not in seen and seen["auth"] == "Bearer k"


async def test_rate_limit_error_carries_retry_after():
    def handler(req):
        return httpx.Response(429, headers={"retry-after": "2"}, json={"error": {"message": "slow down"}})

    with pytest.raises(ProviderError) as e:
        await provider(handler).chat("m", BODY)
    assert e.value.status == 429 and e.value.retry_after == 2.0 and "slow down" in str(e.value)


async def test_timeouts_and_connection_errors_are_retryable():
    def timeout(req):
        raise httpx.ReadTimeout("slow", request=req)

    def refused(req):
        raise httpx.ConnectError("refused", request=req)

    for handler, status in ((timeout, 504), (refused, 502)):
        with pytest.raises(ProviderError) as e:
            await provider(handler).chat("m", BODY)
        assert e.value.status == status and e.value.retryable


async def test_streaming_parses_sse_and_requests_usage():
    seen = {}
    sse = (
        'data: {"choices":[{"delta":{"role":"assistant","content":"Hel"}}]}\n\n'
        'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n'
        ": keep-alive\n\n"
        'data: {"choices":[],"usage":{"prompt_tokens":5,"completion_tokens":2}}\n\n'
        "data: [DONE]\n\n"
    )

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, content=sse.encode(), headers={"content-type": "text/event-stream"})

    chunks = [c async for c in provider(handler).stream("m", BODY)]
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert len(chunks) == 3 and chunks[-1]["usage"]["completion_tokens"] == 2


async def test_streaming_error_raises_before_first_chunk():
    def handler(req):
        return httpx.Response(503, json={"error": {"message": "overloaded"}})

    agen = provider(handler).stream("m", BODY)
    with pytest.raises(ProviderError) as e:
        await agen.__anext__()
    assert e.value.status == 503
