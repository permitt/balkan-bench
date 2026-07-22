"""Tests for the CausalLM open-weights model runner.

Downloads ``sshleifer/tiny-gpt2`` (~5MB) once per test session; runs on CPU.
"""

from __future__ import annotations

import math

import pytest

from balkanbench.models.causal_lm import CausalLM


@pytest.fixture(scope="module")
def model():
    return CausalLM({"name": "tiny", "hf_repo": "sshleifer/tiny-gpt2"}, device="cpu")


def test_loglikelihood_returns_finite_floats(model):
    lls = model.loglikelihood([("The sky is", " blue"), ("The sky is", " green")])
    assert len(lls) == 2 and all(math.isfinite(x) for x in lls)


def test_loglikelihood_is_deterministic(model):
    a = model.loglikelihood([("Hello", " world")])
    b = model.loglikelihood([("Hello", " world")])
    assert a == b


def test_loglikelihood_longer_continuation_lower(model):
    short = model.loglikelihood([("Hi", " a")])[0]
    long = model.loglikelihood([("Hi", " a a a a a a a a")])[0]
    assert long < short  # more tokens -> more negative sum


def test_generate_greedy_and_stop(model):
    outs = model.generate(["Once upon a"], stop_sequences=["\n"], max_gen_tokens=8)
    assert len(outs) == 1 and isinstance(outs[0], str) and "\n" not in outs[0]


def test_batching_matches_single(model):
    reqs = [("A b c", " d"), ("Completely different much longer context here", " tail")]
    batched = model.loglikelihood(reqs)
    singles = [model.loglikelihood([r])[0] for r in reqs]
    assert batched == pytest.approx(singles, abs=1e-3)
