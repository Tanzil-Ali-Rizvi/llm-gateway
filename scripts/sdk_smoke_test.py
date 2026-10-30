"""End-to-end check with the official OpenAI Python SDK against a real gateway process (mock providers)."""

import os
import subprocess
import sys
import time
import urllib.request

import httpx
from openai import OpenAI

PORT = 8099
env = {
    **os.environ,
    "LLMGW_CONFIG": "config/demo.yaml",
    "LLMGW_MASTER_KEY": "smoke-master",
    "LLMGW_DATABASE": "/tmp/llmgw-smoke.db",
}
if os.path.exists("/tmp/llmgw-smoke.db"):
    os.remove("/tmp/llmgw-smoke.db")
proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "llmgw.app:create_app", "--factory", "--port", str(PORT)], env=env
)
try:
    for _ in range(50):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health")
            break
        except OSError:
            time.sleep(0.2)
    key = httpx.post(
        f"http://127.0.0.1:{PORT}/admin/keys",
        json={"name": "sdk", "daily_budget_usd": 1},
        headers={"Authorization": "Bearer smoke-master"},
    ).json()["api_key"]
    client = OpenAI(base_url=f"http://127.0.0.1:{PORT}/v1", api_key=key)

    raw = client.chat.completions.with_raw_response.create(
        model="smart", messages=[{"role": "user", "content": "Hello from the OpenAI SDK"}], temperature=0
    )
    resp = raw.parse()
    assert "Hello from the OpenAI SDK" in resp.choices[0].message.content, resp
    assert resp.usage.total_tokens > 0
    assert raw.headers["x-gateway-target"] == "primary/big"

    again = client.chat.completions.with_raw_response.create(
        model="smart", messages=[{"role": "user", "content": "Hello from the OpenAI SDK"}], temperature=0
    )
    assert again.headers["x-gateway-cache"] == "exact"

    text = "".join(
        c.choices[0].delta.content or ""
        for c in client.chat.completions.create(
            model="fast", messages=[{"role": "user", "content": "stream please"}], stream=True
        )
        if c.choices
    )
    assert "stream please" in text, text

    assert {m.id for m in client.models.list()} == {"smart", "fast"}
    print("OpenAI SDK smoke test passed: chat, cache hit, streaming, models")
finally:
    proc.terminate()
    proc.wait(timeout=10)
