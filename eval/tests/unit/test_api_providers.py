"""Tests for the anthropic/openai/gemini provider clients and the
``make_api_model`` factory.

No network calls: each provider SDK is mocked via ``sys.modules`` patching
(the lazy ``import anthropic`` / ``import openai`` / ``from google import
genai`` statements inside each client's ``__init__`` pick up the patched
module). The SDKs are not installed in this environment (by design - they
are an optional extra), which conveniently also gives us the missing-SDK
``ImportError`` path for free, with no mocking required.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from balkanbench.models.api import make_api_model
from balkanbench.models.api.providers import PRICING, AnthropicClient, GeminiClient, OpenAIClient

REPO_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = REPO_ROOT / "eval" / "configs" / "models" / "official"

# -- Anthropic ---------------------------------------------------------------


def _fake_anthropic_module(*, text: str = "hello", input_tokens=10, output_tokens=5):
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.usage.input_tokens = input_tokens
    response.usage.output_tokens = output_tokens

    client_instance = MagicMock()
    client_instance.messages.create.return_value = response

    module = MagicMock()
    module.Anthropic.return_value = client_instance
    return module, client_instance


def test_anthropic_client_payload_shape_and_response_mapping(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    module, client_instance = _fake_anthropic_module(
        text="hi there", input_tokens=7, output_tokens=3
    )

    with patch.dict(sys.modules, {"anthropic": module}):
        client = AnthropicClient("claude-opus-4-8")
        result = client.complete("the prompt", max_tokens=100, stop_sequences=["STOP"])

    module.Anthropic.assert_called_once_with(api_key="test-anthropic-key")
    kwargs = client_instance.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-4-8"
    assert kwargs["temperature"] == 0
    assert kwargs["max_tokens"] == 100
    assert kwargs["stop_sequences"] == ["STOP"]
    assert kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
    assert "system" not in kwargs

    assert result.text == "hi there"
    assert result.input_tokens == 7
    assert result.output_tokens == 3


def test_anthropic_client_empty_stop_sequences_omits_param(monkeypatch):
    """Mirrors OpenAIClient's empty->None guard: the anthropic SDK's
    Messages API is stricter than accepting an empty list, so an empty
    ``stop_sequences`` must be omitted from the payload entirely rather than
    passed through as ``[]``."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    module, client_instance = _fake_anthropic_module()

    with patch.dict(sys.modules, {"anthropic": module}):
        client = AnthropicClient("claude-opus-4-8")
        client.complete("prompt", max_tokens=10, stop_sequences=[])

    kwargs = client_instance.messages.create.call_args.kwargs
    assert "stop_sequences" not in kwargs


def test_anthropic_client_unknown_model_id_costs_zero_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    module, _ = _fake_anthropic_module()

    with patch.dict(sys.modules, {"anthropic": module}):
        client = AnthropicClient("some-unpriced-model")
        with caplog.at_level(logging.WARNING):
            result = client.complete("prompt", max_tokens=10, stop_sequences=[])

    assert result.cost_usd == 0.0
    assert any("some-unpriced-model" in record.message for record in caplog.records)


def test_anthropic_client_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    module, _ = _fake_anthropic_module()

    with (
        patch.dict(sys.modules, {"anthropic": module}),
        pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"),
    ):
        AnthropicClient("claude-opus-4-8")


def test_anthropic_client_missing_sdk_raises_import_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    # anthropic really isn't installed in this environment; no patching needed.
    with pytest.raises(ImportError, match=r'pip install "balkanbench\[api\]"'):
        AnthropicClient("claude-opus-4-8")


# -- OpenAI --------------------------------------------------------------------


def _fake_openai_module(*, text: str = "hello", prompt_tokens=10, completion_tokens=5):
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens

    client_instance = MagicMock()
    client_instance.chat.completions.create.return_value = response

    module = MagicMock()
    module.OpenAI.return_value = client_instance
    return module, client_instance


def test_openai_client_payload_shape_and_response_mapping(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    module, client_instance = _fake_openai_module(text="hi", prompt_tokens=8, completion_tokens=4)

    with patch.dict(sys.modules, {"openai": module}):
        client = OpenAIClient("gpt-5.5")
        result = client.complete("the prompt", max_tokens=200, stop_sequences=["STOP", "END"])

    module.OpenAI.assert_called_once_with(api_key="test-openai-key")
    kwargs = client_instance.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-5.5"
    assert kwargs["temperature"] == 0
    # max_tokens is deprecated in the current openai SDK in favor of
    # max_completion_tokens - verified against openai-python's
    # completion_create_params.py.
    assert kwargs["max_completion_tokens"] == 200
    assert "max_tokens" not in kwargs
    assert kwargs["stop"] == ["STOP", "END"]
    assert kwargs["messages"] == [{"role": "user", "content": "the prompt"}]
    assert "system" not in kwargs
    assert not any(m.get("role") == "system" for m in kwargs["messages"])

    assert result.text == "hi"
    assert result.input_tokens == 8
    assert result.output_tokens == 4


def test_openai_client_empty_stop_sequences_passes_none(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    module, client_instance = _fake_openai_module()

    with patch.dict(sys.modules, {"openai": module}):
        client = OpenAIClient("gpt-5.5")
        client.complete("prompt", max_tokens=10, stop_sequences=[])

    kwargs = client_instance.chat.completions.create.call_args.kwargs
    assert kwargs["stop"] is None


def test_openai_client_unknown_model_id_costs_zero_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    module, _ = _fake_openai_module()

    with patch.dict(sys.modules, {"openai": module}):
        client = OpenAIClient("some-unpriced-model")
        with caplog.at_level(logging.WARNING):
            result = client.complete("prompt", max_tokens=10, stop_sequences=[])

    assert result.cost_usd == 0.0
    assert any("some-unpriced-model" in record.message for record in caplog.records)


def test_openai_client_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module, _ = _fake_openai_module()

    with (
        patch.dict(sys.modules, {"openai": module}),
        pytest.raises(RuntimeError, match="OPENAI_API_KEY"),
    ):
        OpenAIClient("gpt-5.5")


def test_openai_client_missing_sdk_raises_import_error(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    with pytest.raises(ImportError, match=r'pip install "balkanbench\[api\]"'):
        OpenAIClient("gpt-5.5")


# -- Gemini ----------------------------------------------------------------------


def _fake_genai_modules(*, text: str = "hello", prompt_tokens=10, candidates_tokens=5):
    response = MagicMock()
    response.text = text
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = candidates_tokens

    client_instance = MagicMock()
    client_instance.models.generate_content.return_value = response

    genai_module = MagicMock()
    genai_module.Client.return_value = client_instance

    types_module = MagicMock()
    config_sentinel = MagicMock()
    types_module.GenerateContentConfig.return_value = config_sentinel

    google_module = MagicMock()
    google_module.genai = genai_module
    genai_module.types = types_module

    return google_module, genai_module, types_module, client_instance, config_sentinel


def test_gemini_client_payload_shape_and_response_mapping(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    google_module, genai_module, types_module, client_instance, config_sentinel = (
        _fake_genai_modules(text="hi", prompt_tokens=9, candidates_tokens=6)
    )

    with patch.dict(
        sys.modules,
        {
            "google": google_module,
            "google.genai": genai_module,
            "google.genai.types": types_module,
        },
    ):
        client = GeminiClient("gemini-3.5-flash")
        result = client.complete("the prompt", max_tokens=64, stop_sequences=["STOP"])

    genai_module.Client.assert_called_once_with(api_key="test-gemini-key")
    types_module.GenerateContentConfig.assert_called_once_with(
        temperature=0, max_output_tokens=64, stop_sequences=["STOP"]
    )
    call_kwargs = client_instance.models.generate_content.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert call_kwargs["contents"] == "the prompt"
    assert call_kwargs["config"] is config_sentinel

    assert result.text == "hi"
    assert result.input_tokens == 9
    assert result.output_tokens == 6


def test_gemini_client_unknown_model_id_costs_zero_and_warns(monkeypatch, caplog):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    google_module, genai_module, types_module, _, _ = _fake_genai_modules()

    with patch.dict(
        sys.modules,
        {
            "google": google_module,
            "google.genai": genai_module,
            "google.genai.types": types_module,
        },
    ):
        client = GeminiClient("some-unpriced-model")
        with caplog.at_level(logging.WARNING):
            result = client.complete("prompt", max_tokens=10, stop_sequences=[])

    assert result.cost_usd == 0.0
    assert any("some-unpriced-model" in record.message for record in caplog.records)


def test_gemini_client_missing_api_key_raises_runtime_error(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    google_module, genai_module, types_module, _, _ = _fake_genai_modules()

    with (
        patch.dict(
            sys.modules,
            {
                "google": google_module,
                "google.genai": genai_module,
                "google.genai.types": types_module,
            },
        ),
        pytest.raises(RuntimeError, match="GEMINI_API_KEY"),
    ):
        GeminiClient("gemini-3.5-flash")


def test_gemini_client_missing_sdk_raises_import_error(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    with pytest.raises(ImportError, match=r'pip install "balkanbench\[api\]"'):
        GeminiClient("gemini-3.5-flash")


# -- Factory dispatch ----------------------------------------------------------


def test_make_api_model_dispatches_anthropic(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    module, _ = _fake_anthropic_module()
    with patch.dict(sys.modules, {"anthropic": module}):
        model = make_api_model(
            {"provider": "anthropic", "api_model_id": "claude-opus-4-8"}, tmp_path
        )
    assert isinstance(model.client, AnthropicClient)


def test_make_api_model_dispatches_openai(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    module, _ = _fake_openai_module()
    with patch.dict(sys.modules, {"openai": module}):
        model = make_api_model({"provider": "openai", "api_model_id": "gpt-5.5"}, tmp_path)
    assert isinstance(model.client, OpenAIClient)


def test_make_api_model_dispatches_gemini(monkeypatch, tmp_path):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    google_module, genai_module, types_module, _, _ = _fake_genai_modules()
    with patch.dict(
        sys.modules,
        {
            "google": google_module,
            "google.genai": genai_module,
            "google.genai.types": types_module,
        },
    ):
        model = make_api_model({"provider": "gemini", "api_model_id": "gemini-3.5-flash"}, tmp_path)
    assert isinstance(model.client, GeminiClient)


def test_make_api_model_unknown_provider_raises_value_error(tmp_path):
    with pytest.raises(ValueError, match="unicorn"):
        make_api_model({"provider": "unicorn", "api_model_id": "x"}, tmp_path)


# -- PRICING / roster drift guard ------------------------------------------------


def _api_model_ids_from_configs() -> list[str]:
    ids = []
    for path in sorted(MODELS_DIR.glob("*.yaml")):
        spec = yaml.safe_load(path.read_text())
        if spec.get("access") == "api":
            ids.append(spec["api_model_id"])
    return ids


def test_every_api_access_model_yaml_has_a_pricing_entry():
    api_model_ids = _api_model_ids_from_configs()
    assert api_model_ids, "expected at least one access: api model YAML in configs/models/official"
    missing = [model_id for model_id in api_model_ids if model_id not in PRICING]
    assert not missing, f"PRICING is missing entries for: {missing}"
