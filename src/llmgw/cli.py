"""Command line: run the gateway and manage virtual keys.

llmgw serve --config config/gateway.yaml --port 8080
llmgw keys create --name team-a --budget 5 --rpm 60 --routes fast,smart
llmgw keys list
llmgw usage --by key
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import GatewayConfig
from .store import Store


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="llmgw", description="OpenAI-compatible LLM gateway")
    p.add_argument("--config", default="config/gateway.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="run the HTTP server")
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8080)

    k = sub.add_parser("keys", help="manage virtual API keys")
    ks = k.add_subparsers(dest="action", required=True)
    c = ks.add_parser("create")
    c.add_argument("--name", required=True)
    c.add_argument("--budget", type=float, help="daily budget in USD")
    c.add_argument("--rpm", type=int, help="requests per minute")
    c.add_argument("--routes", help="comma-separated allowed routes")
    ks.add_parser("list")
    r = ks.add_parser("revoke")
    r.add_argument("id", type=int)

    u = sub.add_parser("usage", help="spend and traffic summary")
    u.add_argument("--by", default="route", choices=["route", "provider", "key", "day", "cache"])

    a = p.parse_args(argv)
    cfg = GatewayConfig.load(a.config)

    if a.cmd == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(config_path=a.config), host=a.host, port=a.port)
        return

    store = Store(cfg.database)
    if a.cmd == "keys" and a.action == "create":
        raw, key = store.create_key(a.name, a.budget, a.rpm, a.routes.split(",") if a.routes else None)
        print(f"Created key #{key.id} '{key.name}'. Store it now; it is not shown again:\n{raw}")
    elif a.cmd == "keys" and a.action == "list":
        json.dump(store.list_keys(), sys.stdout, indent=2)
        print()
    elif a.cmd == "keys" and a.action == "revoke":
        store.revoke_key(a.id)
        print(f"Revoked key #{a.id}")
    elif a.cmd == "usage":
        json.dump(store.usage(a.by), sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
