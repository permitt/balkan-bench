"""Unit tests for the ``em`` metric and its ``normalize_answer`` helper.

Normalization is ported from the ``serb_eval_run`` branch of
gordicaleksa/serbian-llm-eval (the fork the original Serbian LLM Eval used);
see ``balkanbench.metrics.em`` for the exact source references. Expected
values below are hand-derived by applying that ported normalization
(lowercase -> strip ASCII punctuation -> drop English articles as whole
words -> collapse whitespace) step by step.
"""

from __future__ import annotations

import pytest

from balkanbench.metrics import get_metric
from balkanbench.metrics.em import em, normalize_answer

# ---------- normalize_answer ----------


def test_normalize_lowercases_and_strips_punct() -> None:
    assert normalize_answer("Nikola Tesla!") == "nikola tesla"


def test_normalize_removes_english_articles() -> None:
    # "The Republic of Serbia" -> lower -> strip punct (none) -> drop "the" -> collapse ws
    assert normalize_answer("The Republic of Serbia") == "republic of serbia"


def test_normalize_removes_a_and_an() -> None:
    assert normalize_answer("A cat and an apple") == "cat and apple"


def test_normalize_collapses_whitespace() -> None:
    assert normalize_answer("  Novak   Djokovic  ") == "novak djokovic"


# ---------- Serbian diacritics cases (hand-derived) ----------


def test_normalize_keeps_serbian_diacritics_only_casefolds() -> None:
    # "Đoković!" -> strip/lower -> "đoković!" (Unicode-aware .lower() maps Đ -> đ)
    # -> strip ASCII punctuation "!" -> "đoković" (no articles, no whitespace to collapse)
    assert normalize_answer("Đoković!") == "đoković"


def test_normalize_serbian_sentence_with_commas_and_dashes() -> None:
    # "Novak Đoković, šampion." -> lower -> "novak đoković, šampion."
    # -> strip ASCII punctuation "," "." -> "novak đoković šampion"
    assert normalize_answer("Novak Đoković, šampion.") == "novak đoković šampion"


def test_normalize_serbian_with_hyphen_and_extra_spaces() -> None:
    # "Beograd - grad na dve reke!" -> lower -> strip punctuation "-" "!"
    # leaves double spaces where the hyphen was -> collapse whitespace
    assert normalize_answer("Beograd - grad na dve reke!") == "beograd grad na dve reke"


# ---------- em ----------


def test_em_matches_any_alias() -> None:
    assert em(predictions=["Tesla"], references=[["Nikola Tesla", "tesla"]]) == 1.0


def test_em_no_match() -> None:
    assert em(predictions=["Edison"], references=[["Tesla"]]) == 0.0


def test_em_averages() -> None:
    assert em(predictions=["a", "b"], references=[["a"], ["c"]]) == 0.5


def test_em_matches_serbian_diacritic_reference() -> None:
    assert em(predictions=["Đoković!"], references=[["đoković"]]) == 1.0


def test_em_empty_predictions_raises() -> None:
    with pytest.raises(ValueError):
        em(predictions=[], references=[])


# ---------- registry ----------


def test_registry_resolves_em_acc_acc_norm() -> None:
    for name in ("em", "acc", "acc_norm"):
        fn = get_metric(name)
        assert callable(fn)


def test_registry_em_call() -> None:
    fn = get_metric("em")
    assert fn(predictions=["Tesla"], references=[["tesla"]]) == 1.0


def test_registry_acc_and_acc_norm_are_accuracy() -> None:
    from balkanbench.metrics.accuracy import accuracy

    assert get_metric("acc") is accuracy
    assert get_metric("acc_norm") is accuracy
