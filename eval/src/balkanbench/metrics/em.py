"""Exact-match (EM) metric.

Normalization is ported verbatim from the ``serb_eval_run`` branch of
gordicaleksa/serbian-llm-eval - the vendored harness the original Serbian
LLM Eval used - so EM scores here are comparable to the original results.

Ported from ``NQOpen._normalize_answer`` in
https://raw.githubusercontent.com/gordicaleksa/serbian-llm-eval/serb_eval_run/lm_eval/tasks/nqopen.py::

    def _normalize_answer(self, text):
        # Lowercase and remove punctuation, strip whitespace
        text = text.strip().lower().translate(str.maketrans("", "", string.punctuation))

        # Remove articles, resulting in duplicate whitespace
        text = regex.sub(r"\b(a|an|the)\b", " ", text)

        # Remove duplicate whitespace
        text = " ".join(text.split())

        return text

The simpler variant in ``TriviaQA.process_results`` (same fork,
https://raw.githubusercontent.com/gordicaleksa/serbian-llm-eval/serb_eval_run/lm_eval/tasks/triviaqa.py)
only lowercases and strips ASCII punctuation - it skips article removal and
whitespace collapsing. The nq_open version is the strictly more thorough
normalization and is what is ported below.

Note: the fork uses the third-party ``regex`` package for the article-removal
substitution; the pattern (``\\b(a|an|the)\\b``) is ASCII-only, so Python's
stdlib ``re`` module produces identical results and is used here to avoid an
extra dependency. The article regex intentionally stays English-only even
when normalizing Serbian text - that is faithful to the original fork.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence
from typing import Any

from balkanbench.metrics._common import validate_pair

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)
_ARTICLE_RE = re.compile(r"\b(a|an|the)\b")


def normalize_answer(s: str) -> str:
    """Normalize an answer string for exact-match comparison.

    Lowercases, strips ASCII punctuation, removes English articles
    (``a``, ``an``, ``the``) as whole words, and collapses whitespace.
    """
    text = s.strip().lower().translate(_PUNCTUATION_TABLE)
    text = _ARTICLE_RE.sub(" ", text)
    return " ".join(text.split())


def em(
    predictions: Sequence[str] | None = None,
    references: Sequence[Sequence[str]] | None = None,
    **_: Any,
) -> float:
    """Fraction of predictions whose normalization matches any reference's.

    Each element of ``references`` is itself a list of acceptable answer
    strings for that example; a prediction counts as correct if its
    normalized form equals the normalized form of any one of them.
    """
    preds, refs = validate_pair(predictions, references)
    matches = 0
    for pred, alternatives in zip(preds, refs, strict=True):
        normalized_pred = normalize_answer(pred)
        normalized_alternatives = {normalize_answer(alt) for alt in alternatives}
        if normalized_pred in normalized_alternatives:
            matches += 1
    return matches / len(preds)


__all__ = ["em", "normalize_answer"]
