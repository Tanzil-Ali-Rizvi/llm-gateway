import json

from conftest import admin, chat, make_cfg


async def new_key(client, **kw):
    r = await client.post("/admin/keys", json={"name": "team-a", **kw}, headers=admin())
    assert r.status_code == 200
    return {"Authorization": f"Bearer {r.json()['api_key']}"}


async def test_auth_is_required_and_bad_keys_rejected(client_factory):
    client, _ = await client_factory(make_cfg())
    assert (await client.post("/v1/chat/completions", json=chat())).status_code == 401
    r = await client.post("/v1/chat/completions", json=chat(), headers={"Authorization": "Bearer sk-gw-nope"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    assert (await client.post("/admin/keys", json={"name": "x"})).status_code == 403


async def test_openai_compatible_response_with_gateway_headers(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    r = await client.post("/v1/chat/completions", json=chat("What is UPI?"), headers=h)
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion" and "What is UPI?" in body["choices"][0]["message"]["content"]
    assert r.headers["x-gateway-target"] == "primary/big" and r.headers["x-gateway-fallback"] == "false"
    assert float(r.headers["x-gateway-cost-usd"]) > 0


async def test_fallback_is_visible_in_headers_and_usage(client_factory):
    client, _ = await client_factory(make_cfg(primary_error_rate=1.0))
    h = await new_key(client)
    r = await client.post("/v1/chat/completions", json=chat(), headers=h)
    assert r.headers["x-gateway-target"] == "backup/small" and r.headers["x-gateway-fallback"] == "true"
    assert r.headers["x-gateway-attempts"] == "3"
    usage = (await client.get("/admin/usage?group_by=route", headers=admin())).json()["rows"][0]
    assert usage["requests"] == 1 and usage["fallbacks"] == 1


async def test_unknown_model_lists_routes(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    r = await client.post("/v1/chat/completions", json=chat(model="gpt-9"), headers=h)
    assert r.status_code == 404 and r.json()["error"]["available"] == ["smart", "solo"]
    models = (await client.get("/v1/models", headers=h)).json()["data"]
    assert {m["id"] for m in models} == {"smart", "solo"}


async def test_direct_provider_model_addressing(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    r = await client.post("/v1/chat/completions", json=chat(model="backup/small"), headers=h)
    assert r.headers["x-gateway-target"] == "backup/small"


async def test_exact_cache_hit_costs_nothing_and_records_savings(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    first = await client.post("/v1/chat/completions", json=chat("cache me", temperature=0), headers=h)
    second = await client.post("/v1/chat/completions", json=chat("cache me", temperature=0), headers=h)
    assert first.headers["x-gateway-cache"] == "miss" and second.headers["x-gateway-cache"] == "exact"
    assert second.json() == first.json() and second.headers["x-gateway-cost-usd"] == "0"
    assert float(second.headers["x-gateway-saved-usd"]) == float(first.headers["x-gateway-cost-usd"])


async def test_non_deterministic_requests_bypass_cache_unless_forced(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    for _ in range(2):
        r = await client.post("/v1/chat/completions", json=chat("hot", temperature=0.7), headers=h)
        assert r.headers["x-gateway-cache"] == "bypass"
    h2 = {**h, "x-gateway-cache": "force"}
    await client.post("/v1/chat/completions", json=chat("hot", temperature=0.7), headers=h2)
    r = await client.post("/v1/chat/completions", json=chat("hot", temperature=0.7), headers=h2)
    assert r.headers["x-gateway-cache"] == "exact"


async def test_cache_key_includes_parameters(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    await client.post("/v1/chat/completions", json=chat("q", temperature=0, max_tokens=10), headers=h)
    r = await client.post("/v1/chat/completions", json=chat("q", temperature=0, max_tokens=20), headers=h)
    assert r.headers["x-gateway-cache"] == "miss"


async def test_semantic_cache_matches_paraphrase_within_same_context(client_factory):
    client, _ = await client_factory(make_cfg(semantic=True))
    h = await new_key(client)
    await client.post("/v1/chat/completions", json=chat("how do I reset my UPI PIN", temperature=0), headers=h)
    r = await client.post(
        "/v1/chat/completions", json=chat("how do I reset my UPI PIN please", temperature=0), headers=h
    )
    assert r.headers["x-gateway-cache"] == "semantic"
    other = await client.post(
        "/v1/chat/completions", json=chat("what is the weather in Delhi", temperature=0), headers=h
    )
    assert other.headers["x-gateway-cache"] == "miss"


async def test_streaming_relays_sse_and_is_logged(client_factory):
    client, gw = await client_factory(make_cfg())
    h = await new_key(client)
    async with client.stream(
        "POST", "/v1/chat/completions", json=chat("stream it", stream=True, temperature=0), headers=h
    ) as r:
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
        lines = [ln async for ln in r.aiter_lines() if ln.startswith("data:")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[5:]) for ln in lines[:-1]]
    text = "".join((c["choices"][0]["delta"] or {}).get("content") or "" for c in chunks)
    assert "stream it" in text and chunks[-1]["usage"]["completion_tokens"] > 0
    assert gw.store.recent(1)[0]["stream"] == 1
    # the finished stream was cached: the next identical streaming request is replayed from cache
    async with client.stream(
        "POST", "/v1/chat/completions", json=chat("stream it", stream=True, temperature=0), headers=h
    ) as r:
        assert r.headers["x-gateway-cache"] == "exact"
        replay = [ln async for ln in r.aiter_lines() if ln.startswith("data:")]
    assert "stream it" in replay[0] and replay[-1] == "data: [DONE]"


async def test_daily_budget_blocks_once_spent(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client, daily_budget_usd=0.0001)
    statuses = []
    for i in range(20):
        r = await client.post("/v1/chat/completions", json=chat(f"question number {i} " * 20), headers=h)
        statuses.append(r.status_code)
        if r.status_code == 429:
            assert r.json()["error"]["code"] == "budget_exceeded"
            break
    assert statuses[0] == 200 and statuses[-1] == 429
    keys = (await client.get("/admin/keys", headers=admin())).json()["keys"]
    assert keys[0]["spent_today_usd"] >= 0.0001


async def test_rate_limit_returns_retry_after(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client, rpm_limit=2)
    codes = [(await client.post("/v1/chat/completions", json=chat(), headers=h)).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = await client.post("/v1/chat/completions", json=chat(), headers=h)
    assert r.json()["error"]["code"] == "rate_limited" and float(r.headers["retry-after"]) > 0


async def test_allowed_routes_are_enforced(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client, allowed_routes=["solo"])
    assert (await client.post("/v1/chat/completions", json=chat(model="smart"), headers=h)).status_code == 403
    assert (await client.post("/v1/chat/completions", json=chat(model="solo"), headers=h)).status_code == 200


async def test_all_targets_down_returns_503_with_attempts(client_factory):
    cfg = make_cfg(primary_error_rate=1.0)
    cfg.providers["backup"].fault.error_rate = 1.0
    client, _ = await client_factory(cfg)
    h = await new_key(client)
    r = await client.post("/v1/chat/completions", json=chat(), headers=h)
    assert r.status_code == 503 and len(r.json()["error"]["attempts"]) == 4


async def test_revoked_key_stops_working(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    await client.delete("/admin/keys/1", headers=admin())
    assert (await client.post("/v1/chat/completions", json=chat(), headers=h)).status_code == 401


async def test_metrics_and_health(client_factory):
    client, _ = await client_factory(make_cfg(primary_error_rate=1.0))
    h = await new_key(client)
    await client.post("/v1/chat/completions", json=chat(), headers=h)
    text = (await client.get("/metrics")).text
    assert 'llmgw_requests_total{cache="bypass",route="smart",status="200",target="backup/small"} 1.0' in text
    assert 'llmgw_fallbacks_total{route="smart"} 1.0' in text
    assert (await client.get("/health")).json()["status"] == "ok"
    circuits = (await client.get("/admin/circuits", headers=admin())).json()["circuits"]
    assert circuits["primary/big"]["consecutive_failures"] == 2


async def test_bad_requests_are_rejected_before_routing(client_factory):
    client, _ = await client_factory(make_cfg())
    h = await new_key(client)
    r = await client.post(
        "/v1/chat/completions", content=b"not json", headers={**h, "content-type": "application/json"}
    )
    assert r.status_code == 400
    r = await client.post("/v1/chat/completions", json={"model": "smart", "messages": []}, headers=h)
    assert r.status_code == 400
