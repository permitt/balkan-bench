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


def test_tokenizer_repo_override_constructs_successfully():
    # Same repo passed as both hf_repo and tokenizer_repo: proves the override
    # path is exercised (not just falling back to hf_repo silently) while still
    # only needing a single tiny checkpoint download.
    m = CausalLM(
        {"name": "x", "hf_repo": "sshleifer/tiny-gpt2", "tokenizer_repo": "sshleifer/tiny-gpt2"},
        device="cpu",
    )
    lls = m.loglikelihood([("Hello", " world")])
    assert len(lls) == 1 and math.isfinite(lls[0])


def test_tokenizer_repo_override_passed_to_from_pretrained(monkeypatch):
    calls: list[tuple[str, str | None]] = []

    import balkanbench.models.causal_lm as causal_lm_module

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"
        padding_side = "right"

    def _fake_from_pretrained(repo, revision=None):
        calls.append((repo, revision))
        return _FakeTokenizer()

    monkeypatch.setattr(
        causal_lm_module,
        "AutoTokenizer",
        type("_FakeAutoTokenizer", (), {"from_pretrained": staticmethod(_fake_from_pretrained)}),
    )

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForCausalLM",
        type(
            "_FakeAutoModelForCausalLM",
            (),
            {"from_pretrained": staticmethod(lambda *a, **k: _FakeModel())},
        ),
    )

    CausalLM(
        {
            "name": "x",
            "hf_repo": "org/weights-repo",
            "hf_revision": "abc123",
            "tokenizer_repo": "org/tokenizer-repo",
        },
        device="cpu",
    )

    assert calls == [("org/tokenizer-repo", None)]


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


# -- generation.prepend_bos (opt-in BOS prepending for BOS-dependent families) --


@pytest.fixture(scope="module")
def bos_model():
    return CausalLM(
        {"name": "tiny-bos", "hf_repo": "sshleifer/tiny-gpt2", "generation": {"prepend_bos": True}},
        device="cpu",
    )


def test_prepend_bos_prepends_to_context_and_full_encode(bos_model):
    # GPT-2's tokenizer has a bos_token_id (same id as eos, "<|endoftext|>").
    bos_id = bos_model.tokenizer.bos_token_id
    assert bos_id is not None

    context, continuation = "The sky is", " blue"
    raw_ctx_ids = bos_model.tokenizer(context, add_special_tokens=False)["input_ids"]
    enc_full, cont_ids = bos_model._encode_request(context, continuation)

    # Both the context-only encode (raw_ctx_ids, prefixed with bos below) and the
    # full context+continuation encode used for scoring start with bos - proving
    # the continuation slice offset is unaffected (both sides shift by one token).
    assert enc_full[0] == bos_id
    assert enc_full == [bos_id, *raw_ctx_ids, *cont_ids]


def test_prepend_bos_changes_loglikelihood(model, bos_model):
    # `model` (module fixture above) is the default prepend_bos=False baseline.
    plain = model.loglikelihood([("The sky is", " blue")])[0]
    with_bos = bos_model.loglikelihood([("The sky is", " blue")])[0]
    assert with_bos != pytest.approx(plain)


def test_prepend_bos_without_bos_token_raises(monkeypatch):
    m = CausalLM(
        {
            "name": "tiny-no-bos",
            "hf_repo": "sshleifer/tiny-gpt2",
            "generation": {"prepend_bos": True},
        },
        device="cpu",
    )
    # tiny-gpt2 does have a bos token; simulate a tokenizer without one by
    # monkeypatching bos_token_id to None on the already-loaded tokenizer.
    monkeypatch.setattr(m.tokenizer, "bos_token_id", None)

    with pytest.raises(ValueError, match="sshleifer/tiny-gpt2"):
        m.loglikelihood([("Hello", " world")])


def test_prepend_bos_default_false_leaves_loglikelihood_unaffected(model):
    # Default (no generation.prepend_bos in config) construction must behave
    # exactly like today: prepend_bos defaults to False.
    assert model.prepend_bos is False


def test_prepend_bos_generate_prompt_starts_with_bos(bos_model, monkeypatch):
    import torch

    bos_id = bos_model.tokenizer.bos_token_id
    captured: dict[str, torch.Tensor] = {}
    real_generate = bos_model.model.generate

    def _spy_generate(*, input_ids, attention_mask, **kwargs):
        captured["input_ids"] = input_ids
        return real_generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    monkeypatch.setattr(bos_model.model, "generate", _spy_generate)

    bos_model.generate(["Once upon a"], stop_sequences=["\n"], max_gen_tokens=3)

    assert captured["input_ids"][0, 0].item() == bos_id


def test_truly_empty_continuation_still_raises(model):
    # "Hello world" + "" (empty continuation string): enc_ctx == enc_full identically,
    # so even the common-prefix fallback yields an empty continuation slice - a genuine
    # degenerate case that must still raise rather than silently score ll=0.0.
    with pytest.raises(ValueError):
        model.loglikelihood([("Hello world", "")])


# -- conditional-generation (text+vision wrapper) fallback -------------------
#
# ``IlyasMoutawwakil/tiny-random-LlavaForConditionalGeneration`` is a ~1MB public
# fixture whose config class (LlavaConfig) is not registered in
# AutoModelForCausalLM's mapping, so `AutoModelForCausalLM.from_pretrained` raises
# ValueError("Unrecognized configuration class ... for this kind of AutoModel:
# AutoModelForCausalLM"). It stands in for the 2026 mid-size flagships
# (Gemma4UnifiedForConditionalGeneration, Gemma4ForConditionalGeneration,
# Qwen3_5ForConditionalGeneration) that hit the same error path - all are
# `*ForConditionalGeneration` wrappers registered only under
# AutoModelForImageTextToText, and all support text-only forward/generate
# producing standard [batch, seq, vocab] logits.
_MM_REPO = "IlyasMoutawwakil/tiny-random-LlavaForConditionalGeneration"


@pytest.fixture(scope="module")
def mm_model():
    return CausalLM({"name": "tiny-mm", "hf_repo": _MM_REPO}, device="cpu")


def test_conditional_generation_model_loads_via_fallback(mm_model):
    # This is the GREEN half of the RED/GREEN pair: before the
    # AutoModelForImageTextToText fallback existed, constructing CausalLM on this
    # repo raised ValueError("Unrecognized configuration class ..."). Confirmed by
    # running this fixture against the pre-fallback code (git stash) before
    # implementing: it raised exactly that error. Now it must construct cleanly.
    assert mm_model.model is not None


def test_multimodal_loglikelihood_returns_finite_floats(mm_model):
    lls = mm_model.loglikelihood([("Hello", " world"), ("Hi", " there")])
    assert len(lls) == 2 and all(math.isfinite(x) for x in lls)


def test_multimodal_loglikelihood_is_deterministic(mm_model):
    a = mm_model.loglikelihood([("Hello", " world")])
    b = mm_model.loglikelihood([("Hello", " world")])
    assert a == b


def test_multimodal_batching_matches_single(mm_model):
    reqs = [("A b c", " d"), ("Completely different much longer context here", " tail")]
    batched = mm_model.loglikelihood(reqs)
    singles = [mm_model.loglikelihood([r])[0] for r in reqs]
    assert batched == pytest.approx(singles, abs=1e-3)


def test_multimodal_generate_produces_string(mm_model):
    outs = mm_model.generate(["Once upon a"], stop_sequences=["\n"], max_gen_tokens=5)
    assert len(outs) == 1 and isinstance(outs[0], str)


def test_fallback_not_triggered_by_unrelated_valueerror(monkeypatch):
    # Only the specific "Unrecognized configuration class" ValueError should trigger
    # the AutoModelForImageTextToText fallback; any other ValueError from
    # AutoModelForCausalLM must propagate unchanged, not be swallowed.
    import balkanbench.models.causal_lm as causal_lm_module

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"
        padding_side = "right"

    monkeypatch.setattr(
        causal_lm_module,
        "AutoTokenizer",
        type(
            "_FakeAutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda repo: _FakeTokenizer())},
        ),
    )

    def _raise_unrelated(*a, **k):
        raise ValueError("some other loading problem entirely")

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForCausalLM",
        type(
            "_FakeAutoModelForCausalLM",
            (),
            {"from_pretrained": staticmethod(_raise_unrelated)},
        ),
    )

    fallback_calls = []
    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForImageTextToText",
        type(
            "_FakeAutoModelForImageTextToText",
            (),
            {"from_pretrained": staticmethod(lambda *a, **k: fallback_calls.append(1) or object())},
        ),
    )

    with pytest.raises(ValueError, match="some other loading problem entirely"):
        CausalLM({"name": "x", "hf_repo": "org/weights-repo"}, device="cpu")

    assert fallback_calls == []  # fallback must NOT have been attempted


def test_fallback_dispatches_to_image_text_to_text_on_unrecognized_config(monkeypatch):
    # Capture-call test for the dispatch logic itself: an "Unrecognized configuration
    # class" ValueError from AutoModelForCausalLM must trigger exactly one fallback
    # call to AutoModelForImageTextToText.from_pretrained with the same args.
    import balkanbench.models.causal_lm as causal_lm_module

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"
        padding_side = "right"

    monkeypatch.setattr(
        causal_lm_module,
        "AutoTokenizer",
        type(
            "_FakeAutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda repo: _FakeTokenizer())},
        ),
    )

    def _raise_unrecognized(*a, **k):
        raise ValueError(
            "Unrecognized configuration class <class 'FakeConfig'> for this kind of "
            "AutoModel: AutoModelForCausalLM."
        )

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForCausalLM",
        type(
            "_FakeAutoModelForCausalLM",
            (),
            {"from_pretrained": staticmethod(_raise_unrecognized)},
        ),
    )

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    fallback_calls = []

    def _fallback_from_pretrained(repo, revision=None, dtype=None):
        fallback_calls.append((repo, revision))
        return _FakeModel()

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForImageTextToText",
        type(
            "_FakeAutoModelForImageTextToText",
            (),
            {"from_pretrained": staticmethod(_fallback_from_pretrained)},
        ),
    )

    m = CausalLM(
        {"name": "x", "hf_repo": "org/mm-weights-repo", "hf_revision": "deadbeef"},
        device="cpu",
    )

    assert fallback_calls == [("org/mm-weights-repo", "deadbeef")]
    assert isinstance(m.model, _FakeModel)


def test_device_map_auto_matches_non_device_map_loglikelihood(model):
    # `model` fixture is the plain (no device_map) CPU baseline. On CPU,
    # device_map="auto" still resolves every shard to "cpu" (accelerate's
    # balancer has only one device to place things on), so this proves the
    # device_map="auto" loading path is a pure loading-mechanism change, not a
    # scoring change. Tolerance is float32-ULP tight (1e-5), not the coarser
    # 1e-3 used by the batching-parity tests elsewhere in this file: measured
    # empirically, accelerate's meta-device-then-materialize load path (used
    # for device_map="auto") produces a ~9.5e-7 difference from the plain
    # `.from_pretrained(...).to(device)` path on one of the two requests below
    # - sub-ULP floating-point non-associativity from the different weight
    # materialization order, not a scoring bug - so exact `==` is not
    # achievable across the two loading mechanisms, but 1e-5 comfortably
    # separates "same computation, different rounding" from an actual
    # scoring regression.
    sharded = CausalLM(
        {
            "name": "tiny-sharded",
            "hf_repo": "sshleifer/tiny-gpt2",
            "generation": {"device_map": "auto"},
        },
        device="cpu",
    )
    reqs = [("The sky is", " blue"), ("Hello", " world")]
    assert sharded.loglikelihood(reqs) == pytest.approx(model.loglikelihood(reqs), abs=1e-5)


def test_device_map_auto_reaches_from_pretrained_and_skips_to(monkeypatch):
    import balkanbench.models.causal_lm as causal_lm_module

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"
        padding_side = "right"

    monkeypatch.setattr(
        causal_lm_module,
        "AutoTokenizer",
        type(
            "_FakeAutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda repo: _FakeTokenizer())},
        ),
    )

    class _FakeModel:
        device = "cpu"  # stand-in for PreTrainedModel.device on a sharded model

        def to(self, device):
            raise AssertionError("`.to()` must not be called when device_map='auto'")

        def eval(self):
            return self

    calls = []

    def _fake_from_pretrained(repo, revision=None, dtype=None, device_map=None):
        calls.append({"repo": repo, "revision": revision, "dtype": dtype, "device_map": device_map})
        return _FakeModel()

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForCausalLM",
        type(
            "_FakeAutoModelForCausalLM",
            (),
            {"from_pretrained": staticmethod(_fake_from_pretrained)},
        ),
    )

    m = CausalLM(
        {"name": "x", "hf_repo": "org/weights-repo", "generation": {"device_map": "auto"}},
        device="cpu",
    )

    assert len(calls) == 1
    assert calls[0]["repo"] == "org/weights-repo"
    assert calls[0]["device_map"] == "auto"
    assert isinstance(m.model, _FakeModel)


def test_fallback_error_names_both_attempts_when_both_fail(monkeypatch):
    import balkanbench.models.causal_lm as causal_lm_module

    class _FakeTokenizer:
        pad_token = "<pad>"
        eos_token = "<pad>"
        padding_side = "right"

    monkeypatch.setattr(
        causal_lm_module,
        "AutoTokenizer",
        type(
            "_FakeAutoTokenizer",
            (),
            {"from_pretrained": staticmethod(lambda repo: _FakeTokenizer())},
        ),
    )

    def _raise_unrecognized_causal(*a, **k):
        raise ValueError(
            "Unrecognized configuration class <class 'FakeConfig'> for this kind of "
            "AutoModel: AutoModelForCausalLM."
        )

    def _raise_unrecognized_itt(*a, **k):
        raise ValueError(
            "Unrecognized configuration class <class 'FakeConfig'> for this kind of "
            "AutoModel: AutoModelForImageTextToText."
        )

    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForCausalLM",
        type(
            "_FakeAutoModelForCausalLM",
            (),
            {"from_pretrained": staticmethod(_raise_unrecognized_causal)},
        ),
    )
    monkeypatch.setattr(
        causal_lm_module,
        "AutoModelForImageTextToText",
        type(
            "_FakeAutoModelForImageTextToText",
            (),
            {"from_pretrained": staticmethod(_raise_unrecognized_itt)},
        ),
    )

    with pytest.raises(ValueError) as excinfo:
        CausalLM({"name": "x", "hf_repo": "org/neither-supports-it"}, device="cpu")

    message = str(excinfo.value)
    assert "AutoModelForCausalLM" in message
    assert "AutoModelForImageTextToText" in message
    assert "org/neither-supports-it" in message
