"""Provider clients: thin ``ProviderClient`` transports over the anthropic,
openai, and google-genai SDKs.

Each client lazily imports its SDK inside ``__init__`` so importing this
module (or the rest of the package) never requires the optional API extras
to be installed; only constructing a client for a given provider does. Each
client reads its API key from an environment variable at construction time
and raises a clear ``RuntimeError`` naming the variable if it is unset.

``complete()`` sends exactly one user message containing the prompt (no
system prompt), ``temperature=0`` for greedy/deterministic decoding, and maps
the provider's usage fields onto :class:`~balkanbench.models.api.base.APIResponse`.
Cost is computed from the module-level ``PRICING`` table below
(``{api_model_id: (usd_per_1m_input, usd_per_1m_output)}``); it is
intentionally empty for now (filled in as part of Task 15's model-roster
step) - unknown ids resolve to ``cost_usd = 0.0`` with a ``logger.warning``.

SDK call shapes verified against (see task report for detail):
  - anthropic: the ``claude-api`` skill's Python reference
    (``client.messages.create`` / ``response.usage.input_tokens`` etc.)
  - openai: ``openai-python`` source on GitHub (``max_tokens`` is deprecated
    in favor of ``max_completion_tokens``; ``response.usage.prompt_tokens`` /
    ``completion_tokens``)
  - google-genai: ``python-genai`` source on GitHub
    (``client.models.generate_content(model=..., contents=...,
    config=types.GenerateContentConfig(...))``;
    ``response.usage_metadata.prompt_token_count`` /
    ``candidates_token_count``)
"""

from __future__ import annotations

import logging
import os

from balkanbench.models.api.base import APIResponse

logger = logging.getLogger(__name__)

# {api_model_id: (usd_per_1m_input_tokens, usd_per_1m_output_tokens)}.
# Left empty deliberately - Task 15's model-roster step fills this in.
# Unknown/missing ids fall back to cost_usd = 0.0 with a logged warning.
PRICING: dict[str, tuple[float, float]] = {}

_API_EXTRA_HINT = 'pip install "balkanbench[api]"'


def _cost_usd(api_model_id: str, *, input_tokens: int, output_tokens: int) -> float:
    prices = PRICING.get(api_model_id)
    if prices is None:
        logger.warning("No pricing entry for model %r; recording cost_usd=0.0", api_model_id)
        return 0.0
    usd_per_1m_input, usd_per_1m_output = prices
    return (input_tokens * usd_per_1m_input + output_tokens * usd_per_1m_output) / 1_000_000


class AnthropicClient:
    """``ProviderClient`` over the ``anthropic`` SDK's Messages API."""

    def __init__(self, api_model_id: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError(f"anthropic SDK not installed; run {_API_EXTRA_HINT}") from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY environment variable is not set")

        self.api_model_id = api_model_id
        self._client = anthropic.Anthropic(api_key=api_key)

    def complete(self, prompt: str, *, max_tokens: int, stop_sequences: list[str]) -> APIResponse:
        response = self._client.messages.create(
            model=self.api_model_id,
            max_tokens=max_tokens,
            temperature=0,
            stop_sequences=stop_sequences,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_usd(
                self.api_model_id, input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )


class OpenAIClient:
    """``ProviderClient`` over the ``openai`` SDK's Chat Completions API."""

    def __init__(self, api_model_id: str) -> None:
        try:
            import openai
        except ImportError as exc:
            raise ImportError(f"openai SDK not installed; run {_API_EXTRA_HINT}") from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")

        self.api_model_id = api_model_id
        self._client = openai.OpenAI(api_key=api_key)

    def complete(self, prompt: str, *, max_tokens: int, stop_sequences: list[str]) -> APIResponse:
        response = self._client.chat.completions.create(
            model=self.api_model_id,
            max_completion_tokens=max_tokens,
            temperature=0,
            stop=stop_sequences or None,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_usd(
                self.api_model_id, input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )


class GeminiClient:
    """``ProviderClient`` over the ``google-genai`` SDK."""

    def __init__(self, api_model_id: str) -> None:
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise ImportError(f"google-genai SDK not installed; run {_API_EXTRA_HINT}") from exc

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY environment variable is not set")

        self.api_model_id = api_model_id
        self._types = types
        self._client = genai.Client(api_key=api_key)

    def complete(self, prompt: str, *, max_tokens: int, stop_sequences: list[str]) -> APIResponse:
        response = self._client.models.generate_content(
            model=self.api_model_id,
            contents=prompt,
            config=self._types.GenerateContentConfig(
                temperature=0,
                max_output_tokens=max_tokens,
                stop_sequences=stop_sequences,
            ),
        )
        text = response.text or ""
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        return APIResponse(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=_cost_usd(
                self.api_model_id, input_tokens=input_tokens, output_tokens=output_tokens
            ),
        )
