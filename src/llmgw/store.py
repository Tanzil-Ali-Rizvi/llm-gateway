"""SQLite persistence: virtual API keys, the request/usage log and the response cache.

One file, WAL mode, a process-wide lock. Each statement takes well under a millisecond, which is fine for a
single gateway instance; a multi-instance deployment would swap this module for Postgres/Redis.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    key_hash TEXT UNIQUE NOT NULL,
    prefix TEXT NOT NULL,
    created_at REAL NOT NULL,
    daily_budget_usd REAL,
    rpm_limit INTEGER,
    allowed_routes TEXT,
    active INTEGER NOT NULL DEFAULT 1
);
CREATE TABLE IF NOT EXISTS requests (
    id INTEGER PRIMARY KEY,
    ts REAL NOT NULL,
    key_id INTEGER,
    route TEXT,
    provider TEXT,
    model TEXT,
    status INTEGER,
    cache TEXT,
    stream INTEGER,
    prompt_tokens INTEGER DEFAULT 0,
    completion_tokens INTEGER DEFAULT 0,
    cost_usd REAL DEFAULT 0,
    saved_usd REAL DEFAULT 0,
    latency_ms REAL,
    attempts INTEGER,
    fallback INTEGER DEFAULT 0,
    error TEXT
);
CREATE INDEX IF NOT EXISTS requests_ts ON requests(ts);
CREATE INDEX IF NOT EXISTS requests_key_ts ON requests(key_id, ts);
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    route TEXT,
    response TEXT NOT NULL,
    cost_usd REAL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hits INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS semantic_cache (
    id INTEGER PRIMARY KEY,
    scope TEXT NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL,
    response TEXT NOT NULL,
    cost_usd REAL DEFAULT 0,
    created_at REAL NOT NULL,
    expires_at REAL NOT NULL,
    hits INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS semantic_scope ON semantic_cache(scope);
"""


@dataclass
class ApiKey:
    id: int
    name: str
    prefix: str
    daily_budget_usd: float | None
    rpm_limit: int | None
    allowed_routes: list[str] | None
    active: bool

    def allows(self, route: str) -> bool:
        return not self.allowed_routes or route in self.allowed_routes


def hash_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()


def start_of_utc_day(ts: float | None = None) -> float:
    d = datetime.fromtimestamp(ts or time.time(), tz=UTC)
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


class Store:
    def __init__(self, path: str = "gateway.db"):
        self.path = path
        self.db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        if path != ":memory:":
            self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self.lock:
            return self.db.execute(sql, args)

    def _all(self, sql: str, args: tuple = ()) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    # ------------------------------------------------------------------ keys
    def create_key(
        self,
        name: str,
        daily_budget_usd: float | None = None,
        rpm_limit: int | None = None,
        allowed_routes: list[str] | None = None,
    ) -> tuple[str, ApiKey]:
        """Create a virtual key. The raw key is returned once and only its hash is stored."""
        raw = "sk-gw-" + secrets.token_urlsafe(24)
        cur = self._exec(
            "INSERT INTO api_keys (name, key_hash, prefix, created_at, daily_budget_usd, rpm_limit, allowed_routes)"
            " VALUES (?,?,?,?,?,?,?)",
            (
                name,
                hash_key(raw),
                raw[:12],
                time.time(),
                daily_budget_usd,
                rpm_limit,
                json.dumps(allowed_routes) if allowed_routes else None,
            ),
        )
        return raw, self.get_key(cur.lastrowid)

    def _to_key(self, r: dict[str, Any]) -> ApiKey:
        return ApiKey(
            r["id"],
            r["name"],
            r["prefix"],
            r["daily_budget_usd"],
            r["rpm_limit"],
            json.loads(r["allowed_routes"]) if r["allowed_routes"] else None,
            bool(r["active"]),
        )

    def get_key(self, key_id: int) -> ApiKey:
        return self._to_key(self._all("SELECT * FROM api_keys WHERE id=?", (key_id,))[0])

    def verify_key(self, raw: str) -> ApiKey | None:
        rows = self._all("SELECT * FROM api_keys WHERE key_hash=? AND active=1", (hash_key(raw),))
        return self._to_key(rows[0]) if rows else None

    def list_keys(self) -> list[dict[str, Any]]:
        rows = self._all(
            "SELECT id, name, prefix, created_at, daily_budget_usd, rpm_limit, allowed_routes, active "
            "FROM api_keys ORDER BY id"
        )
        today = start_of_utc_day()
        for r in rows:
            r["spent_today_usd"] = round(self.spent_since(r["id"], today), 6)
            r["allowed_routes"] = json.loads(r["allowed_routes"]) if r["allowed_routes"] else None
        return rows

    def revoke_key(self, key_id: int) -> None:
        self._exec("UPDATE api_keys SET active=0 WHERE id=?", (key_id,))

    def spent_since(self, key_id: int, since: float) -> float:
        rows = self._all("SELECT COALESCE(SUM(cost_usd),0) s FROM requests WHERE key_id=? AND ts>=?", (key_id, since))
        return float(rows[0]["s"])

    # ------------------------------------------------------------------ request log
    def log(self, **f: Any) -> None:
        cols = [
            "ts",
            "key_id",
            "route",
            "provider",
            "model",
            "status",
            "cache",
            "stream",
            "prompt_tokens",
            "completion_tokens",
            "cost_usd",
            "saved_usd",
            "latency_ms",
            "attempts",
            "fallback",
            "error",
        ]
        f.setdefault("ts", time.time())
        self._exec(
            f"INSERT INTO requests ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
            tuple(f.get(c) for c in cols),
        )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._all(
            "SELECT r.*, k.name AS key_name FROM requests r LEFT JOIN api_keys k ON k.id=r.key_id "
            "ORDER BY r.id DESC LIMIT ?",
            (limit,),
        )

    def usage(self, group_by: str = "route", since: float = 0.0) -> list[dict[str, Any]]:
        col = {
            "route": "r.route",
            "provider": "r.provider || '/' || r.model",
            "key": "COALESCE(k.name, '(none)')",
            "day": "date(r.ts, 'unixepoch')",
            "cache": "r.cache",
        }[group_by]
        return self._all(
            f"SELECT {col} AS grp, COUNT(*) AS requests, SUM(r.status=200) AS ok, SUM(r.cache!='miss' AND r.cache IS NOT NULL"
            f" AND r.cache!='bypass') AS cache_hits, SUM(r.fallback) AS fallbacks, SUM(r.prompt_tokens) AS prompt_tokens,"
            f" SUM(r.completion_tokens) AS completion_tokens, ROUND(SUM(r.cost_usd), 6) AS cost_usd,"
            f" ROUND(SUM(r.saved_usd), 6) AS saved_usd, ROUND(AVG(r.latency_ms), 1) AS avg_latency_ms"
            f" FROM requests r LEFT JOIN api_keys k ON k.id=r.key_id WHERE r.ts>=? GROUP BY grp ORDER BY requests DESC",
            (since,),
        )

    def latencies(self, since: float = 0.0) -> list[float]:
        return [
            r["latency_ms"] for r in self._all("SELECT latency_ms FROM requests WHERE ts>=? AND status=200", (since,))
        ]

    # ------------------------------------------------------------------ exact cache
    def cache_get(self, key: str) -> tuple[dict[str, Any], float] | None:
        rows = self._all("SELECT response, cost_usd FROM cache WHERE key=? AND expires_at>?", (key, time.time()))
        if not rows:
            return None
        self._exec("UPDATE cache SET hits=hits+1 WHERE key=?", (key,))
        return json.loads(rows[0]["response"]), rows[0]["cost_usd"]

    def cache_put(self, key: str, route: str, response: dict[str, Any], cost_usd: float, ttl_s: int) -> None:
        now = time.time()
        self._exec(
            "INSERT OR REPLACE INTO cache (key, route, response, cost_usd, created_at, expires_at, hits)"
            " VALUES (?,?,?,?,?,?,0)",
            (key, route, json.dumps(response), cost_usd, now, now + ttl_s),
        )

    # ------------------------------------------------------------------ semantic cache
    def semantic_candidates(self, scope: str) -> list[dict[str, Any]]:
        return self._all(
            "SELECT id, embedding, response, cost_usd FROM semantic_cache WHERE scope=? AND expires_at>?",
            (scope, time.time()),
        )

    def semantic_hit(self, row_id: int) -> None:
        self._exec("UPDATE semantic_cache SET hits=hits+1 WHERE id=?", (row_id,))

    def semantic_put(
        self, scope: str, text: str, embedding: bytes, response: dict[str, Any], cost_usd: float, ttl_s: int
    ) -> None:
        now = time.time()
        self._exec(
            "INSERT INTO semantic_cache (scope, text, embedding, response, cost_usd, created_at, expires_at)"
            " VALUES (?,?,?,?,?,?,?)",
            (scope, text, embedding, json.dumps(response), cost_usd, now, now + ttl_s),
        )

    def purge_expired(self) -> None:
        now = time.time()
        self._exec("DELETE FROM cache WHERE expires_at<=?", (now,))
        self._exec("DELETE FROM semantic_cache WHERE expires_at<=?", (now,))
