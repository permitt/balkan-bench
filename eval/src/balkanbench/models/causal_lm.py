"""Local open-weights causal-LM runner.

``CausalLM`` implements :class:`~balkanbench.models.generative_base.GenerativeModel`
against any Hugging Face ``AutoModelForCausalLM`` checkpoint: batched
loglikelihood scoring for multiple-choice / ranking tasks, and greedy
generation with stop-sequence truncation for free-form QA tasks.

No custom ``Trainer`` subclass; this is inference-only.
"""

from __future__ import annotations

from typing import Any

# transformers is lazy-loaded via __getattr__ so `balkanbench --version`,
# --help, and the non-ML subcommands avoid its multi-second import cost.
# torch is imported inside methods for the same reason.
_LAZY = {
    "AutoModelForCausalLM": "transformers",
    "AutoTokenizer": "transformers",
}


def __getattr__(name: str) -> Any:
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    return getattr(importlib.import_module(module_name), name)


class CausalLM:
    """Batched loglikelihood scoring + greedy generation over an HF causal LM.

    Constructor reads from ``model_cfg``:
      - ``hf_repo`` (required): HF hub repo id.
      - ``hf_revision`` (optional): pinned revision/commit.
      - ``generation.batch_size`` (default 8).
      - ``generation.dtype`` (default "bfloat16"; forced to "float32" when
        running on CPU, since bfloat16 kernels are slow/unsupported there).
    """

    def __init__(self, model_cfg: dict[str, Any], *, device: str | None = None) -> None:
        import torch

        from balkanbench.models import causal_lm as _self

        self.model_cfg = model_cfg
        repo = model_cfg["hf_repo"]
        revision = model_cfg.get("hf_revision")
        gen_cfg = model_cfg.get("generation", {})
        self.batch_size = int(gen_cfg.get("batch_size", 8))

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        dtype_name = gen_cfg.get("dtype", "bfloat16")
        if self.device == "cpu":
            dtype_name = "float32"
        dtype = getattr(torch, dtype_name)

        self.tokenizer = _self.AutoTokenizer.from_pretrained(repo, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-padding keeps every sequence's continuation/new tokens aligned
        # at the same trailing positions across the batch, regardless of
        # each sample's own length - required for the position-id fix-up
        # below and for decoding only the newly generated tokens.
        self.tokenizer.padding_side = "left"

        self.model = _self.AutoModelForCausalLM.from_pretrained(
            repo, revision=revision, dtype=dtype
        )
        self.model.to(self.device)
        self.model.eval()

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
        """
        enc_ctx = self.tokenizer(context, add_special_tokens=False)["input_ids"]
        enc_full = self.tokenizer(context + continuation, add_special_tokens=False)["input_ids"]
        if len(enc_ctx) == 0:
            bos_id = self.tokenizer.bos_token_id
            if bos_id is None:
                bos_id = self.tokenizer.eos_token_id
            enc_ctx = [bos_id, *enc_ctx]
            enc_full = [bos_id, *enc_full]
        continuation_ids = enc_full[len(enc_ctx) :]
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

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids_t,
                attention_mask=attention_mask_t,
                position_ids=position_ids_t,
            )
            log_probs = torch.log_softmax(outputs.logits.float(), dim=-1)

        scores: list[float] = []
        for i, (_full, cont_ids) in enumerate(encoded):
            n_cont = len(cont_ids)
            total = 0.0
            for j, token_id in enumerate(cont_ids):
                pos = max_len - n_cont + j
                total += log_probs[i, pos - 1, token_id].item()
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

        encoded = self.tokenizer(
            prompts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
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
