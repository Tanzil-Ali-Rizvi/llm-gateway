"""FastAPI app: an OpenAI-compatible API plus admin and observability endpoints."""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import generate_latest
from pydantic import BaseModel

from . import __version__
from .config import GatewayConfig
from .gateway import Gateway, GatewayError


class KeyRequest(BaseModel):
    name: str
    daily_budget_usd: float | None = None
    rpm_limit: int | None = None
    allowed_routes: list[str] | None = None


def create_app(gateway: Gateway | None = None, config_path: str | None = None) -> FastAPI:
    holder: dict[str, Gateway] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        gw = gateway or Gateway(
            GatewayConfig.load(config_path or os.environ.get("LLMGW_CONFIG", "config/gateway.yaml"))
        )
        holder["gw"] = gw
        yield
        await gw.router.aclose()

    app = FastAPI(
        title="llm-gateway",
        version=__version__,
        lifespan=lifespan,
        description="OpenAI-compatible LLM gateway with routing, fallback, caching, budgets and cost tracking.",
    )

    def gw() -> Gateway:
        return holder["gw"]

    @app.exception_handler(GatewayError)
    async def _gateway_error(_: Request, e: GatewayError) -> JSONResponse:
        return JSONResponse(e.body(), status_code=e.status, headers=e.headers)

    # ------------------------------------------------------------------ OpenAI-compatible API
    @app.post("/v1/chat/completions")
    async def chat_completions(
        request: Request, authorization: str | None = Header(None), x_gateway_cache: str = Header("auto")
    ):
        try:
            body: dict[str, Any] = await request.json()
        except ValueError:
            raise GatewayError(400, "invalid_json", "Request body must be JSON.") from None
        if not isinstance(body.get("messages"), list) or not body["messages"]:
            raise GatewayError(400, "invalid_request", "'messages' must be a non-empty list.")
        key = gw().authenticate(authorization)
        out = await gw().chat(body, key, cache_mode=x_gateway_cache.lower())
        if out.stream is not None:
            return StreamingResponse(out.stream, media_type="text/event-stream", headers=out.headers)
        return JSONResponse(out.body, status_code=out.status, headers=out.headers)

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(None)):
        gw().authenticate(authorization)
        now = int(time.time())
        data = [
            {"id": r, "object": "model", "created": now, "owned_by": "llm-gateway", "targets": t}
            for r, t in gw().cfg.routes.items()
        ]
        return {"object": "list", "data": data}

    # ------------------------------------------------------------------ admin
    @app.post("/admin/keys")
    async def create_key(req: KeyRequest, authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        raw, key = gw().store.create_key(req.name, req.daily_budget_usd, req.rpm_limit, req.allowed_routes)
        return {"api_key": raw, "id": key.id, "name": key.name, "note": "Store this key now; it is not shown again."}

    @app.get("/admin/keys")
    async def list_keys(authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        return {"keys": gw().store.list_keys()}

    @app.delete("/admin/keys/{key_id}")
    async def revoke_key(key_id: int, authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        gw().store.revoke_key(key_id)
        return {"revoked": key_id}

    @app.get("/admin/usage")
    async def usage(group_by: str = "route", hours: float = 24, authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        if group_by not in ("route", "provider", "key", "day", "cache"):
            raise GatewayError(400, "invalid_request", "group_by must be route|provider|key|day|cache")
        return {"group_by": group_by, "rows": gw().store.usage(group_by, time.time() - hours * 3600)}

    @app.get("/admin/requests")
    async def recent(limit: int = 50, authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        return {"requests": gw().store.recent(min(limit, 500))}

    @app.get("/admin/circuits")
    async def circuits(authorization: str | None = Header(None)):
        gw().require_admin(authorization)
        return {"circuits": gw().router.breaker.snapshot()}

    # ------------------------------------------------------------------ ops
    @app.get("/health")
    async def health():
        return {"status": "ok", "version": __version__, "routes": sorted(gw().cfg.routes)}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(
            generate_latest(gw().metrics.registry).decode(), media_type="text/plain; version=0.0.4"
        )

    return app
