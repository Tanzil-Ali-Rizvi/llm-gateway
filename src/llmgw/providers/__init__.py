from ..config import ProviderConfig
from .base import Provider, ProviderError
from .chaos import FaultInjector
from .mock import MockProvider
from .openai_compat import OpenAICompatProvider


def build_provider(name: str, cfg: ProviderConfig) -> Provider:
    if cfg.type == "mock":
        return MockProvider(name, cfg)
    provider: Provider = OpenAICompatProvider(name, cfg)
    if cfg.fault.error_rate > 0:  # chaos testing against a real API
        provider = FaultInjector(provider, cfg.fault.error_rate, cfg.fault.error_status, cfg.fault.seed)
    return provider


__all__ = ["Provider", "ProviderError", "FaultInjector", "MockProvider", "OpenAICompatProvider", "build_provider"]
