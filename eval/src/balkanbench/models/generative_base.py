"""Shared protocol for generative model runners.

``GenerativeModel`` is implemented by both the local open-weights runner
(``CausalLM``, see ``causal_lm.py``) and the API model layer (a later task),
so the generative evaluator can call either through the same interface.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class GenerativeModel(Protocol):
    """Interface the generative evaluator drives.

    Implementations must be deterministic (no sampling): ``loglikelihood``
    is a pure scoring pass and ``generate`` performs greedy decoding.
    """

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        """Sum of logprobs of continuation tokens given context, one per request."""
        ...

    def generate(
        self,
        prompts: list[str],
        *,
        stop_sequences: list[str],
        max_gen_tokens: int,
    ) -> list[str]:
        """Greedy completion per prompt, truncated at the first stop sequence."""
        ...


class UnsupportedProtocolError(RuntimeError):
    """Raised when a caller invokes a ``GenerativeModel`` method a given
    implementation cannot support (e.g. an API model that has no notion of
    token-level logprobs)."""
