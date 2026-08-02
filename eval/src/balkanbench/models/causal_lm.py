"""Local open-weights causal-LM runner.

``CausalLM`` implements :class:`~balkanbench.models.generative_base.GenerativeModel`
against any Hugging Face ``AutoModelForCausalLM`` checkpoint: batched
loglikelihood scoring for multiple-choice / ranking tasks, and greedy
generation with stop-sequence truncation for free-form QA tasks.

Some 2026-era mid-size flagships (e.g. Gemma 4, Qwen 3.6) ship as text+vision
conditional-generation wrappers (``*ForConditionalGeneration``) that are not
registered in ``AutoModelForCausalLM``'s model mapping, even though we only
ever score/generate text with them. Model loading falls back to
``AutoModelForImageTextToText`` for exactly that case; see ``CausalLM.__init__``.

No custom ``Trainer`` subclass; this is inference-only.
"""

from __future__ import annotations

from typing import Any

# transformers is lazy-loaded via __getattr__ so `balkanbench --version`,
# --help, and the non-ML subcommands avoid its multi-second import cost.
# torch is imported inside methods for the same reason.
_LAZY = {
    "AutoModelForCausalLM": "transformers",
    "AutoModelForImageTextToText": "transformers",
    "AutoTokenizer": "transformers",
}

# Substring of the ValueError transformers raises from AutoXxx.from_pretrained
# when a config class isn't registered in that auto-class's model mapping, e.g.
# "Unrecognized configuration class <class '...LlavaConfig'> for this kind of
# AutoModel: AutoModelForCausalLM." Only this specific failure triggers the
# AutoModelForImageTextToText fallback below - any other ValueError (bad repo,
# network error re-raised as ValueError, etc.) propagates unchanged.
_UNRECOGNIZED_CONFIG_MARKER = "Unrecognized configuration class"


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


class CausalLM:
    """Batched loglikelihood scoring + greedy generation over an HF causal LM.

    Constructor reads from ``model_cfg``:
      - ``hf_repo`` (required): HF hub repo id for the model weights.
      - ``hf_revision`` (optional): pinned revision/commit, applied only to
        the model weights (``hf_repo``).
      - ``tokenizer_repo`` (optional): HF hub repo id to load the tokenizer
        from, when it differs from ``hf_repo``. Needed for repos that ship
        only a slow sentencepiece ``tokenizer.model`` file, which transformers
        5.x's converter cannot load (it tries to parse the file as tiktoken
        and fails). Model weights are always loaded from ``hf_repo``
        regardless of this override; only the tokenizer load is affected.
        ``hf_revision`` is intentionally *not* applied to ``tokenizer_repo`` -
        the tokenizer repo is pinned via its own default branch, keeping this
        simple since it's a distinct repo from the weights.
      - ``generation.batch_size`` (default 8).
      - ``generation.dtype`` (default "bfloat16"; forced to "float32" when
        running on CPU, since bfloat16 kernels are slow/unsupported there).
      - ``generation.prepend_bos`` (default False): opt-in only, so the default
        behavior is byte-identical to before this flag existed. When True, the
        tokenizer's ``bos_token_id`` is prepended to both the context and full
        (context+continuation) encodings used for loglikelihood scoring, and to
        generation prompts. lm-evaluation-harness v0.3.0 (the reference harness
        this scorer's ``add_special_tokens=False`` encoding is faithful to)
        predates Gemma; Gemma-family models are trained with ``<bos>`` always
        present and are known to score near-chance without it. Raises
        ``ValueError`` at encode time if the tokenizer has no ``bos_token_id``
        (fail loud rather than silently doing nothing).
      - ``generation.device_map`` (optional, only ``"auto"`` is accepted):
        passed straight through to ``from_pretrained``, letting accelerate
        shard the model's layers across every visible device (e.g. 8x L4 on
        one Vertex machine, for 24-72GB models that don't fit on a single
        24GB GPU). When set, the single-device ``.to(device)`` call is
        skipped - accelerate has already placed each shard on its target
        device, and moving the whole module with ``.to()`` would try to copy
        every parameter onto one device, destroying the sharding. Input
        tensors are instead sent to ``self.model.device`` -
        ``PreTrainedModel.device`` returns ``next(p.device for p in
        self.parameters())``, i.e. the device of the first parameter
        (typically the input embedding layer), which is exactly where a
        sharded forward pass expects its inputs to land (verified by reading
        ``transformers==5.14.1``'s ``modeling_utils.py``: this property is
        unconditional on ``hf_device_map`` and needs no accelerate-specific
        API). ``device_map="auto"`` requires the ``accelerate`` package
        (transformers raises ``ImportError`` from
        ``transformers.integrations.accelerate`` otherwise); this repo
        already depends on ``accelerate>=1.13.0`` unconditionally (see
        ``pyproject.toml``), so no extra install is needed.
    """

    def __init__(self, model_cfg: dict[str, Any], *, device: str | None = None) -> None:
        import torch

        from balkanbench.models import causal_lm as _self

        self.model_cfg = model_cfg
        repo = model_cfg["hf_repo"]
        revision = model_cfg.get("hf_revision")
        tokenizer_repo = model_cfg.get("tokenizer_repo") or repo
        gen_cfg = model_cfg.get("generation", {})
        self.batch_size = int(gen_cfg.get("batch_size", 8))
        self.prepend_bos = bool(gen_cfg.get("prepend_bos", False))
        device_map = gen_cfg.get("device_map")

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dtype_name = gen_cfg.get("dtype", "bfloat16")
        if self.device == "cpu":
            dtype_name = "float32"
        dtype = getattr(torch, dtype_name)

        self.tokenizer = _self.AutoTokenizer.from_pretrained(tokenizer_repo)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding keeps every sequence's continuation/new tokens aligned
        # at the same trailing positions across the batch, regardless of
        # each sample's own length - required for the position-id fix-up
        # below and for decoding only the newly generated tokens.
        self.tokenizer.padding_side = "left"

        from_pretrained_kwargs: dict[str, Any] = {"revision": revision, "dtype": dtype}
        if device_map is not None:
            from_pretrained_kwargs["device_map"] = device_map

        try:
            self.model = _self.AutoModelForCausalLM.from_pretrained(repo, **from_pretrained_kwargs)
        except ValueError as causal_lm_exc:
            if _UNRECOGNIZED_CONFIG_MARKER not in str(causal_lm_exc):
                raise
            # Text+vision conditional-generation wrapper (e.g. Gemma4ForConditional
            # Generation, Qwen3_5ForConditionalGeneration): not in
            # AutoModelForCausalLM's mapping, but registered under
            # AutoModelForImageTextToText. We only ever feed these text input
            # (input_ids/attention_mask/position_ids, no pixel_values), and every
            # wrapper of this shape exposes .forward()/.generate() directly at the
            # top level returning standard [batch, seq, vocab] logits - same as a
            # plain causal LM - so no .language_model unwrapping is needed and
            # everything downstream (device placement, eval mode, dtype, forward,
            # generate) goes through the exact same code paths as the primary path.
            try:
                self.model = _self.AutoModelForImageTextToText.from_pretrained(
                    repo, **from_pretrained_kwargs
                )
            except ValueError as image_text_to_text_exc:
                raise ValueError(
                    f"could not load {repo!r}: AutoModelForCausalLM raised "
                    f"{causal_lm_exc!r}; fallback AutoModelForImageTextToText also "
                    f"raised {image_text_to_text_exc!r}"
                ) from image_text_to_text_exc
        if device_map is None:
            self.model.to(self.device)
        else:
            # accelerate has already placed each shard on its target device;
            # .to() would try to move every parameter onto one device and
            # destroy the sharding. Route input tensors to the device of the
            # model's first parameter instead - see the device_map docstring
            # above for why this is the correct target for sharded models.
            self.device = self.model.device
        self.model.eval()

    def _required_bos_token_id(self) -> int:
        """Return the tokenizer's ``bos_token_id``, for use when ``prepend_bos`` is set.

        Raises ``ValueError`` naming the model repo if the tokenizer has no BOS
        token - fail loud rather than silently skipping the prepend the caller
        asked for. Checked at each call site (not cached at construction) so a
        tokenizer whose ``bos_token_id`` changes after load is still caught.
        """
        bos_id = self.tokenizer.bos_token_id
        if bos_id is None:
            raise ValueError(
                f"generation.prepend_bos is true for {self.model_cfg.get('hf_repo')!r} "
                "but its tokenizer has no bos_token_id"
            )
        return int(bos_id)

    # -- loglikelihood ----------------------------------------------------

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        """Sum of logprobs of continuation tokens given context, one per request."""
        results: list[float] = []
        for start in range(0, len(requests), self.batch_size):
            batch = requests[start : start + self.batch_size]
            results.extend(self._loglikelihood_batch(batch))
        return results

    def _encode_request(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Return (full token ids, continuation token ids) for one request.

        Context and context+continuation are tokenized separately with
        identical settings (``add_special_tokens=False``); the continuation
        ids are the suffix of the full encoding past the context's length.
        Guards against an empty-context tokenization (never happens for our
        tasks, but matches harness behavior) by prepending BOS to both.

        Before encoding, a trailing whitespace run on ``context`` is shifted
        onto the front of ``continuation`` - ports lm-evaluation-harness
        v0.3.0's ``BaseLM._encode_pair`` (vendored in
        gordicaleksa/serbian-llm-eval, serb_eval_run branch,
        lm_eval/base.py:200-209) verbatim. This keeps the continuation's
        leading-space merge consistent whether the source text split as
        "context " + "continuation" or "context" + " continuation" -
        without it, the joint encoding can swallow the boundary into a
        single token that is already fully covered by the context's own
        encoding, silently emptying the continuation.

        When ``self.prepend_bos`` is set (opt-in, default off - see
        ``generation.prepend_bos`` in the class docstring), the tokenizer's
        ``bos_token_id`` is unconditionally prepended to both ``enc_ctx`` and
        ``enc_full`` before the empty-context guard runs, so the continuation
        slice offset (``enc_full[len(enc_ctx):]``) is unaffected - both sides
        shift by exactly one token.

        Deliberate robustness deviation from harness v0.3.0: with some
        tokenizers (e.g. Ministral/tekken), the joint encoding of
        ``context + continuation`` doesn't just append tokens past the
        context's own encoding - it can re-merge across the boundary
        entirely, so ``enc_full[:len(enc_ctx)] != enc_ctx`` and the naive
        suffix slice ``enc_full[len(enc_ctx):]`` is empty even though the
        continuation clearly contributes to ``enc_full``. The reference fork
        does not handle this: its ``_encode_pair`` silently returns an empty
        continuation, and its ``loglikelihood`` scores that as ``0.0`` - a
        bug we intentionally do not replicate. Instead, when the naive slice
        is unusable, we fall back to slicing at the longest common prefix of
        ``enc_ctx`` and ``enc_full``: everything past that prefix is treated
        as the continuation. This is only a deviation from the fork's exact
        (mis-)behavior in this unrepresentable-boundary case; normal pairs,
        where the naive slice matches, are scored identically to the fork.
        """
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        enc_ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        enc_full = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]

        if self.prepend_bos:
            bos_id = self._required_bos_token_id()
            enc_ctx = [bos_id, *enc_ctx]
            enc_full = [bos_id, *enc_full]

        if len(enc_ctx) == 0:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.tokenizer.eos_token_id
            enc_ctx = [bos_id, *enc_ctx]
            enc_full = [bos_id, *enc_full]

        continuation_ids = enc_full[len(enc_ctx) :] if enc_full[: len(enc_ctx)] == enc_ctx else []

        if len(continuation_ids) == 0:
            # Naive suffix slice is unusable (boundary re-merge, or genuinely
            # empty): fall back to the longest common prefix of enc_ctx/enc_full.
            common_prefix_len = 0
            for ctx_tok, full_tok in zip(enc_ctx, enc_full, strict=False):
                if ctx_tok != full_tok:
                    break
                common_prefix_len += 1
            continuation_ids = enc_full[common_prefix_len:]

        if len(continuation_ids) == 0:
            raise ValueError(
                f"empty continuation token ids for context={context!r} "
                f"continuation={continuation!r}: the continuation contributes no "
                "tokens beyond the context under joint encoding, even after the "
                "common-prefix fallback"
            )
        return enc_full, continuation_ids

    def _loglikelihood_batch(self, batch: list[tuple[str, str]]) -> list[float]:
        import torch

        encoded = [self._encode_request(ctx, cont) for ctx, cont in batch]
        max_len = max(len(full) for full, _cont in encoded)
        pad_id = self.tokenizer.pad_token_id

        input_ids: list[list[int]] = []
        attention_mask: list[list[int]] = []
        for full, _cont_ids in encoded:
            pad_len = max_len - len(full)
            input_ids.append([pad_id] * pad_len + full)
            attention_mask.append([0] * pad_len + [1] * len(full))

        input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        attention_mask_t = torch.tensor(attention_mask, dtype=torch.long, device=self.device)
        # Left-padding shifts real tokens to the right without this: absolute
        # position embeddings would then differ from the same request scored
        # alone (unpadded), breaking batching parity. Recompute position ids
        # from the mask so each sample's real tokens start at position 0
        # exactly as they would unbatched; padded positions are irrelevant
        # (attention-masked) so clamp them to 0 to avoid negative indices.
        position_ids_t = attention_mask_t.cumsum(-1) - 1
        position_ids_t = position_ids_t.masked_fill(attention_mask_t == 0, 0)

        # Only the last (max_ncont_in_batch + 1) positions are ever read below
        # (continuations sit at the right edge of the left-padded rows; the
        # gather uses pos - 1 with pos >= max_len - n_cont). Casting the FULL
        # [batch, max_len, vocab] logits to fp32 and log_softmax-ing every
        # position is wasteful and, with a large vocab and long contexts, can
        # OOM (e.g. 150k-vocab model + ~1500-token passages on a 24GB GPU
        # already holding a ~20GB checkpoint). Slice to the needed window
        # before the fp32 cast/log_softmax; this is mathematically identical
        # since log_softmax is per-position over vocab and the discarded
        # positions are never read.
        max_ncont = max(len(cont_ids) for _full, cont_ids in encoded)
        window = max_ncont + 1
        slice_start = max(0, max_len - window)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                position_ids=position_ids_t,
            )
            logits = outputs.logits
            # Wrapper architectures loaded via the conditional-generation
            # fallback must still emit standard [batch, seq, vocab] logits
            # for text-only input; a silently different shape would corrupt
            # every score, so fail loudly instead.
            expected = (input_ids_t.shape[0], input_ids_t.shape[1])
            if logits.dim() != 3 or tuple(logits.shape[:2]) != expected:
                raise ValueError(
                    f"model returned logits of shape {tuple(logits.shape)} for input "
                    f"{tuple(input_ids_t.shape)}; expected [batch, seq, vocab]"
                )
            windowed_logits = logits[:, slice_start:, :]
            log_probs = torch.log_softmax(windowed_logits.float(), dim=-1)

        scores: list[float] = []
        for i, (_full, cont_ids) in enumerate(encoded):
            n_cont = len(cont_ids)
            total = 0.0
            for j, token_id in enumerate(cont_ids):
                pos = max_len - n_cont + j
                windowed_pos = pos - slice_start
                total += log_probs[i, windowed_pos - 1, token_id].item()
            scores.append(total)
        return scores

    # -- generate -----------------------------------------------------------

    def generate(
        self,
        prompts: list[str],
        *,
        stop_sequences: list[str],
        max_gen_tokens: int,
    ) -> list[str]:
        """Greedy completion per prompt, truncated at the first stop sequence."""
        results: list[str] = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            results.extend(
                self._generate_batch(
                    batch, stop_sequences=stop_sequences, max_gen_tokens=max_gen_tokens
                )
            )
        return results

    def _generate_batch(
        self,
        prompts: list[str],
        *,
        stop_sequences: list[str],
        max_gen_tokens: int,
    ) -> list[str]:
        import torch

        # Tokenized per-prompt (rather than one batched tokenizer(prompts, ...)
        # call) and padded manually below so prepend_bos can prepend the BOS id
        # to each prompt's own token ids before padding - a batched call with
        # padding=True has no hook for per-row prepending before pad. Per-prompt
        # tokenization is equivalent to the batched call for the default
        # (prepend_bos=False) path: each string tokenizes independently of its
        # batch-mates, and the same left-padding (self.tokenizer.padding_side)
        # is reproduced explicitly below with the tokenizer's own pad_token_id.
        encoded_ids = [
            self.tokenizer(prompt, add_special_tokens=False)["input_ids"] for prompt in prompts
        ]
        if self.prepend_bos:
            bos_id = self._required_bos_token_id()
            encoded_ids = [[bos_id, *ids] for ids in encoded_ids]

        max_len = max(len(ids) for ids in encoded_ids)
        pad_id = self.tokenizer.pad_token_id
        input_ids_list: list[list[int]] = []
        attention_mask_list: list[list[int]] = []
        for ids in encoded_ids:
            pad_len = max_len - len(ids)
            input_ids_list.append([pad_id] * pad_len + ids)
            attention_mask_list.append([0] * pad_len + [1] * len(ids))

        input_ids = torch.tensor(input_ids_list, dtype=torch.long, device=self.device)
        attention_mask = torch.tensor(attention_mask_list, dtype=torch.long, device=self.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = output_ids[:, prompt_len:]
        texts = self.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [_truncate_at_stop(text, stop_sequences) for text in texts]


def _truncate_at_stop(text: str, stop_sequences: list[str]) -> str:
    """Cut ``text`` at the earliest occurrence of any stop sequence."""
    cut = len(text)
    for stop in stop_sequences:
        if not stop:
            continue
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]
