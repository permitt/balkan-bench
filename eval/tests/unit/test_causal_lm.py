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


def test_batching_matches_single_mixed_continuation_lengths(model):
    # Continuations of very different token lengths in the same batch exercise the
    # per-row offset arithmetic (max_len - n_cont + j) when logits are sliced to only
    # the last (max_ncont_in_batch + 1) positions instead of the full sequence.
    reqs = [
        ("A b c", " d"),
        (
            "The quick brown fox jumps over the lazy dog near the old stone bridge",
            " and then runs away quickly into the deep dark forest",
        ),
        ("Hi", " a"),
    ]
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


def test_merged_continuation_falls_back_to_common_prefix(model):
    # "Hello wor" + "ld" merges under joint encoding into "Hello" + " world" (2 tokens),
    # while the context alone tokenizes as "Hello" + " wor" (also 2 tokens) - the naive
    # suffix slice enc_full[len(enc_ctx):] is empty AND the joint encoding's tokens don't
    # even match the context's own tokens past position 0 ("wor" 476 vs "world" 995).
    # This previously raised ValueError (a case the reference fork silently mis-scores
    # as ll=0.0, which we must not replicate either). The fix falls back to the longest
    # common prefix of enc_ctx/enc_full: p=1 (only "Hello" matches), so the continuation
    # is scored as the single joint token " world" (995) at the context/continuation
    # boundary, and a finite ll must be returned rather than raising.
    context, continuation = "Hello wor", "ld"

    import torch

    enc_ctx = _tok_encode(model, context)
    enc_full = _tok_encode(model, context + continuation)
    assert enc_full[: len(enc_ctx)] != enc_ctx  # merge corrupts the naive prefix check
    assert enc_full[len(enc_ctx) :] == []  # naive slice is empty

    p = 0
    for a, b in zip(enc_ctx, enc_full):
        if a != b:
            break
        p += 1
    continuation_enc = enc_full[p:]
    assert continuation_enc == [995]  # the merged " world" token

    input_ids = torch.tensor([enc_full], dtype=torch.long)
    with torch.no_grad():
        logits = model.model(input_ids=input_ids).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected = 0.0
    for j, token_id in enumerate(continuation_enc):
        pos = p + j
        expected += log_probs[0, pos - 1, token_id].item()

    actual = model.loglikelihood([(context, continuation)])[0]
    assert math.isfinite(actual)
    assert actual == pytest.approx(expected, abs=1e-4)


def test_truly_empty_continuation_still_raises(model):
    # "Hello world" + "" (empty continuation string): enc_ctx == enc_full identically,
    # so even the common-prefix fallback yields an empty continuation slice - a genuine
    # degenerate case that must still raise rather than silently score ll=0.0.
    with pytest.raises(ValueError):
        model.loglikelihood([("Hello world", "")])
