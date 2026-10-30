"""Response cache: exact match, plus an optional semantic layer for paraphrased questions.

* **Exact**: key = hash of the route and every generation-relevant request field. Safe by construction.
* **Semantic**: embeds the last user message; a hit needs the same route and the same earlier conversation
  (system prompt, history, parameters) *and* cosine similarity >= threshold. Off by default.

By default only deterministic requests (temperature 0) are cached; clients can override per request with the
header `x-gateway-cache: no | auto | force`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from .config import CacheConfig
from .store import Store

KEY_FIELDS = (
    "messages",
    "temperature",
    "top_p",
    "max_tokens",
    "max_completion_tokens",
    "tools",
    "tool_choice",
    "response_format",
    "seed",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "reasoning_effort",
)

Embedder = Callable[[Sequence[str]], np.ndarray]


def exact_key(route: str, body: dict[str, Any]) -> str:
    payload = {"route": route, **{k: body[k] for k in KEY_FIELDS if k in body}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def semantic_scope(route: str, body: dict[str, Any]) -> tuple[str, str] | None:
    """(scope hash, last user text) or None if the request is not eligible for semantic caching."""
    msgs = body.get("messages") or []
    if not msgs or msgs[-1].get("role") != "user" or not isinstance(msgs[-1].get("content"), str) or body.get("tools"):
        return None
    context = {
        "route": route,
        "history": msgs[:-1],
        **{k: body[k] for k in KEY_FIELDS if k in body and k != "messages"},
    }
    scope = hashlib.sha256(json.dumps(context, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    return scope, msgs[-1]["content"]


class HashingEmbedder:
    """Dependency-free bag-of-words embedder (for tests and offline use). Real deployments use fastembed."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


class FastEmbedder:
    def __init__(self, model: str):
        from fastembed import TextEmbedding  # optional dependency: pip install llm-gateway[semantic]

        self.model = TextEmbedding(model)

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        v = np.asarray(list(self.model.embed(list(texts))), dtype=np.float32)
        return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


class Cache:
    def __init__(self, cfg: CacheConfig, store: Store, embedder: Embedder | None = None):
        self.cfg = cfg
        self.store = store
        self.embedder = embedder
        if cfg.semantic and embedder is None:
            self.embedder = FastEmbedder(cfg.embedding_model)

    def eligible(self, body: dict[str, Any], mode: str = "auto") -> bool:
        if not self.cfg.enabled or mode == "no" or (body.get("n") or 1) > 1:
            return False
        if mode == "force" or not self.cfg.only_deterministic:
            return True
        return body.get("temperature") == 0

    def lookup(self, route: str, body: dict[str, Any]) -> tuple[str, dict[str, Any], float] | None:
        """Returns (kind, cached response, cost it originally took) or None."""
        hit = self.store.cache_get(exact_key(route, body))
        if hit:
            return ("exact", *hit)
        if self.cfg.semantic and self.embedder is not None:
            sc = semantic_scope(route, body)
            if sc:
                rows = self.store.semantic_candidates(sc[0])
                if rows:
                    q = self.embedder([sc[1]])[0]
                    mat = np.stack([np.frombuffer(r["embedding"], dtype=np.float32) for r in rows])
                    sims = mat @ q
                    best = int(np.argmax(sims))
                    if sims[best] >= self.cfg.semantic_threshold:
                        self.store.semantic_hit(rows[best]["id"])
                        return "semantic", json.loads(rows[best]["response"]), rows[best]["cost_usd"]
        return None

    def save(self, route: str, body: dict[str, Any], response: dict[str, Any], cost_usd: float) -> None:
        self.store.cache_put(exact_key(route, body), route, response, cost_usd, self.cfg.ttl_s)
        if self.cfg.semantic and self.embedder is not None:
            sc = semantic_scope(route, body)
            if sc:
                emb = self.embedder([sc[1]])[0].astype(np.float32).tobytes()
                self.store.semantic_put(sc[0], sc[1], emb, response, cost_usd, self.cfg.ttl_s)
