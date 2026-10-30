"""Adapter for any OpenAI-compatible API: OpenAI, Groq, Gemini (OpenAI endpoint), OpenRouter, Ollama, vLLM ..."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from ..config import ProviderConfig
from .base import Provider, ProviderError


def _retry_after(resp: httpx.Response) -> float | None:
    for h in ("retry-after", "x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = resp.headers.get(h)
        if not v:
            continue
        try:
            return float(v.rstrip("s"))
        except ValueError:
            continue
    return None


def _error(resp: httpx.Response, body: bytes) -> ProviderError:
    try:
        detail = json.loads(body).get("error", {})
        msg = detail.get("message") if isinstance(detail, dict) else str(detail)
    except (ValueError, AttributeError):
        msg = body[:300].decode(errors="replace")
    return ProviderError(f"HTTP {resp.status_code}: {msg}", status=resp.status_code, retry_after=_retry_after(resp))


class OpenAICompatProvider(Provider):
    def __init__(self, name: str, cfg: ProviderConfig, client: httpx.AsyncClient | None = None):
        self.name = name
        self.cfg = cfg
        headers = {"Content-Type": "application/json", **cfg.extra_headers}
        if cfg.api_key:
            headers["Authorization"] = f"Bearer {cfg.api_key}"
        self.client = client or httpx.AsyncClient(
            base_url=cfg.base_url.rstrip("/"), headers=headers, timeout=cfg.timeout_s
        )

    def _payload(self, model: str, body: dict[str, Any], stream: bool) -> dict[str, Any]:
        payload = {k: v for k, v in body.items() if k not in ("model", "stream", "stream_options")}
        payload["model"] = model
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        return payload

    async def chat(self, model: str, body: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self.client.post("/chat/completions", json=self._payload(model, body, stream=False))
        except httpx.TimeoutException as e:
            raise ProviderError(f"timeout: {e}", status=504) from None
        except httpx.TransportError as e:
            raise ProviderError(f"connection error: {e}", status=502) from None
        if resp.status_code >= 400:
            raise _error(resp, resp.content)
        return resp.json()

    async def stream(self, model: str, body: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        req = self.client.build_request("POST", "/chat/completions", json=self._payload(model, body, stream=True))
        try:
            resp = await self.client.send(req, stream=True)
        except httpx.TimeoutException as e:
            raise ProviderError(f"timeout: {e}", status=504) from None
        except httpx.TransportError as e:
            raise ProviderError(f"connection error: {e}", status=502) from None
        try:
            if resp.status_code >= 400:
                raise _error(resp, await resp.aread())
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    yield json.loads(data)
                except ValueError:
                    continue
        finally:
            await resp.aclose()

    async def aclose(self) -> None:
        await self.client.aclose()
