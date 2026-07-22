"""API-backed model runner: Anthropic/OpenAI/Gemini clients plus the factory
that wires one into an :class:`APIModel`."""

from pathlib import Path
from typing import Any

from balkanbench.models.api.base import APIModel, APIResponse, ProviderClient

__all__ = ["APIModel", "APIResponse", "ProviderClient", "make_api_model"]


def make_api_model(model_cfg: dict[str, Any], cache_dir: Path) -> APIModel:
    """Construct an :class:`APIModel` wired to the provider named in
    ``model_cfg["provider"]``.

    Dispatches on ``provider`` (``"anthropic"`` / ``"openai"`` / ``"gemini"``)
    to build the matching :class:`~balkanbench.models.api.providers.ProviderClient`,
    passing ``model_cfg["api_model_id"]`` through to it. Raises ``ValueError``
    for an unrecognized provider.
    """
    from balkanbench.models.api.providers import AnthropicClient, GeminiClient, OpenAIClient

    provider = model_cfg["provider"]
    api_model_id = model_cfg["api_model_id"]

    if provider == "anthropic":
        client: ProviderClient = AnthropicClient(api_model_id)
    elif provider == "openai":
        client = OpenAIClient(api_model_id)
    elif provider == "gemini":
        client = GeminiClient(api_model_id)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")

    return APIModel(model_cfg, cache_dir=cache_dir, client=client)
