"""Tests for the API model runner: concurrency, retry, cache, cost.

Exercised against ``FakeClient`` (no network); the real anthropic/openai/gemini
clients are wired up in a later task.
"""

from __future__ import annotations

import json

import pytest

from balkanbench.models.api.base import APIModel, APIResponse
from balkanbench.models.generative_base import UnsupportedProtocolError


class FakeClient:
    def __init__(self, fail_first: int = 0):
        self.calls = 0
        self.fail_first = fail_first

    def complete(self, prompt, *, max_tokens, stop_sequences):
        self.calls += 1
        if self.calls <= self.fail_first:
            raise RuntimeError("transient")
        return APIResponse(text=f"echo:{prompt}", input_tokens=3, output_tokens=2, cost_usd=0.001)


def _model_cfg(**generation_overrides):
    return {
        "api_model_id": "fake-model-1",
        "generation": {"concurrency": 4, "max_tokens": 32, **generation_overrides},
    }


def test_generate_preserves_order_under_concurrency(tmp_path):
    client = FakeClient()
    model = APIModel(_model_cfg(concurrency=8), cache_dir=tmp_path, client=client)
    prompts = [f"prompt-{i}" for i in range(20)]

    results = model.generate(prompts, stop_sequences=[], max_gen_tokens=32)

    assert results == [f"echo:{p}" for p in prompts]


def test_retry_succeeds_after_two_transient_failures(monkeypatch, tmp_path):
    sleeps: list[float] = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))
    client = FakeClient(fail_first=2)
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=client)

    results = model.generate(["hello"], stop_sequences=[], max_gen_tokens=32)

    assert results == ["echo:hello"]
    assert client.calls == 3
    assert len(sleeps) == 2


def test_retry_reraises_after_five_failures(monkeypatch, tmp_path):
    monkeypatch.setattr("time.sleep", lambda s: None)
    client = FakeClient(fail_first=5)
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=client)

    with pytest.raises(RuntimeError, match="transient"):
        model.generate(["hello"], stop_sequences=[], max_gen_tokens=32)

    assert client.calls == 5


def test_cost_and_request_count_accumulate(tmp_path):
    client = FakeClient()
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=client)
    prompts = [f"prompt-{i}" for i in range(6)]

    model.generate(prompts, stop_sequences=[], max_gen_tokens=32)

    assert model.total_cost_usd == pytest.approx(6 * 0.001)
    assert model.request_count == 6


def test_second_identical_generate_hits_cache(tmp_path):
    client = FakeClient()
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=client)
    prompts = [f"prompt-{i}" for i in range(5)]

    first = model.generate(prompts, stop_sequences=[], max_gen_tokens=32)
    calls_after_first = client.calls
    cost_after_first = model.total_cost_usd

    second = model.generate(prompts, stop_sequences=[], max_gen_tokens=32)

    assert second == first
    assert client.calls == calls_after_first
    assert model.total_cost_usd == cost_after_first
    assert model.request_count == calls_after_first


def test_loglikelihood_raises_unsupported_protocol_error(tmp_path):
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=FakeClient())

    with pytest.raises(UnsupportedProtocolError):
        model.loglikelihood([("ctx", "cont")])


def test_generate_truncates_at_stop_sequence_client_side(tmp_path):
    """Belt-and-braces: even if a provider client ignores stop_sequences and
    returns text running past the stop marker, APIModel.generate truncates it
    client-side before returning."""

    class OverrunClient:
        def complete(self, prompt, *, max_tokens, stop_sequences):
            return APIResponse(
                text="answer: 42\nEND\nunwanted trailing text",
                input_tokens=3,
                output_tokens=8,
                cost_usd=0.0,
            )

    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=OverrunClient())

    results = model.generate(["prompt"], stop_sequences=["\nEND"], max_gen_tokens=32)

    assert results == ["answer: 42"]


def test_corrupted_cache_file_is_treated_as_miss(tmp_path):
    client = FakeClient()
    model = APIModel(_model_cfg(), cache_dir=tmp_path, client=client)

    key = model._cache_key("hello", max_tokens=32, stop=[])
    cache_path = tmp_path / key[:2] / f"{key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text("{not valid json", encoding="utf-8")

    results = model.generate(["hello"], stop_sequences=[], max_gen_tokens=32)

    assert results == ["echo:hello"]
    assert client.calls == 1
    # cache file is overwritten with valid JSON on the fresh call
    assert json.loads(cache_path.read_text(encoding="utf-8"))["text"] == "echo:hello"
