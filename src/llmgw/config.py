"""Gateway configuration: providers, the models they serve (with prices), and routes.

A *route* is the model name clients ask for (e.g. "smart"). It maps to an ordered list of
`provider/model` targets; the gateway tries them in order and falls back on failure.

Example (YAML):

    providers:
      groq:
        type: openai            # any OpenAI-compatible API: Groq, Gemini, OpenRouter, Ollama, OpenAI ...
        base_url: https://api.groq.com/openai/v1
        api_key_env: GROQ_API_KEY
        models:
          openai/gpt-oss-120b: {input_per_1m: 0.15, output_per_1m: 0.60}
    routes:
      smart: [groq/openai/gpt-oss-120b, openrouter/meta-llama/llama-3.3-70b-instruct]
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelPrice(BaseModel):
    input_per_1m: float = 0.0
    output_per_1m: float = 0.0


class FaultConfig(BaseModel):
    """Fault injection for the mock provider (tests, chaos benchmarks, demo outages)."""

    error_rate: float = 0.0  # probability a call fails
    error_status: int = 503
    latency_ms: float = 50.0
    latency_jitter_ms: float = 0.0
    seed: int | None = None


class ProviderConfig(BaseModel):
    type: Literal["openai", "mock"] = "openai"
    base_url: str = ""
    api_key_env: str | None = None
    timeout_s: float = 30.0
    extra_headers: dict[str, str] = Field(default_factory=dict)
    models: dict[str, ModelPrice] = Field(default_factory=dict)
    fault: FaultConfig = Field(default_factory=FaultConfig)

    @property
    def api_key(self) -> str | None:
        return os.environ.get(self.api_key_env) if self.api_key_env else None


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_s: int = 24 * 3600
    only_deterministic: bool = True  # cache only temperature==0 requests unless the client forces it
    semantic: bool = False
    semantic_threshold: float = 0.92
    embedding_model: str = "BAAI/bge-small-en-v1.5"


class BreakerConfig(BaseModel):
    failure_threshold: int = 5  # consecutive failures that open the circuit
    cooldown_s: float = 30.0  # how long an open circuit skips the target before a trial call


class RetryConfig(BaseModel):
    attempts_per_target: int = 2
    backoff_s: float = 0.25
    max_retry_after_s: float = 5.0  # honour Retry-After up to this; longer waits move on to the next target


class GatewayConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    routes: dict[str, list[str]] = Field(default_factory=dict)
    default_route: str | None = None
    cache: CacheConfig = Field(default_factory=CacheConfig)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    database: str = "gateway.db"
    master_key_env: str = "LLMGW_MASTER_KEY"
    require_auth: bool = True

    @model_validator(mode="after")
    def _check_targets(self) -> GatewayConfig:
        for route, targets in self.routes.items():
            if not targets:
                raise ValueError(f"route '{route}' has no targets")
            for t in targets:
                self.split_target(t)
        if self.default_route and self.default_route not in self.routes:
            raise ValueError(f"default_route '{self.default_route}' is not a route")
        return self

    def split_target(self, target: str) -> tuple[str, str]:
        provider, _, model = target.partition("/")
        if provider not in self.providers or not model:
            raise ValueError(f"target '{target}' must be '<provider>/<model>' with a configured provider")
        if self.providers[provider].models and model not in self.providers[provider].models:
            raise ValueError(f"model '{model}' is not listed under provider '{provider}'")
        return provider, model

    def price(self, provider: str, model: str) -> ModelPrice:
        return self.providers[provider].models.get(model, ModelPrice())

    def resolve(self, requested: str) -> list[tuple[str, str]]:
        """Client model name -> ordered (provider, model) targets."""
        if requested in self.routes:
            return [self.split_target(t) for t in self.routes[requested]]
        if "/" in requested:  # direct "provider/model" request
            return [self.split_target(requested)]
        if self.default_route and requested in ("", "default"):
            return [self.split_target(t) for t in self.routes[self.default_route]]
        raise KeyError(requested)

    @property
    def master_key(self) -> str | None:
        return os.environ.get(self.master_key_env)

    @classmethod
    def load(cls, path: str | Path) -> GatewayConfig:
        data = yaml.safe_load(Path(path).read_text())
        if os.environ.get("LLMGW_DATABASE"):
            data["database"] = os.environ["LLMGW_DATABASE"]
        return cls.model_validate(data)
