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


def _tok_encode(model, text: str) -> list[int]:
    return model.tokenizer(text, add_special_tokens=False)["input_ids"]


def _fork_encode_pair(model, context: str, continuation: str) -> tuple[list[int], list[int]]:
    """lm-evaluation-harness v0.3.0 ``BaseLM._encode_pair`` verbatim (vendored in
    gordicaleksa/serbian-llm-eval, serb_eval_run branch, lm_eval/base.py lines 200-209).
    """
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole_enc = _tok_encode(model, context + continuation)
    context_enc = _tok_encode(model, context)
    continuation_enc = whole_enc[len(context_enc) :]
    return context_enc, continuation_enc


def test_encode_pair_whitespace_heuristic_matches_fork(model):
    context, continuation = "The sky is ", "blue"  # trailing space on context

    unshifted_full = _tok_encode(model, context + continuation)
    unshifted_ctx = _tok_encode(model, context)
    unshifted_cont = unshifted_full[len(unshifted_ctx) :]

    context_enc, continuation_enc = _fork_encode_pair(model, context, continuation)

    # The heuristic must actually change the tokenization for this pair, otherwise
    # it isn't exercising anything.
    assert continuation_enc != unshifted_cont
    assert len(continuation_enc) > 0

    # Reference: manually run the model over the fork's shifted encoding and sum
    # continuation-token logprobs.
    import torch

    full_ids = context_enc + continuation_enc
    input_ids = torch.tensor([full_ids], dtype=torch.long)
    with torch.no_grad():
        logits = model.model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = 0.0
    for j, token_id in enumerate(continuation_enc):
        pos = len(context_enc) + j
        expected += log_probs[0, pos - 1, token_id].item()

    actual = model.loglikelihood([(context, continuation)])[0]
    assert actual == pytest.approx(expected, abs=1e-4)


def test_empty_continuation_raises(model):
    # "Hello wor" + "ld" merges into a single joint-encoded token ("world") that
    # is already fully covered by the context's own encoding ("Hello", " wor"),
    # so the continuation slice is empty even after the whitespace heuristic
    # (no trailing space on this context, so it's a no-op here).
    with pytest.raises(ValueError):
        model.loglikelihood([("Hello wor", "ld")])
