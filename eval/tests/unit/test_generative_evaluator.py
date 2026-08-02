"""Tests for the SLE generative evaluator (loglikelihood MC protocol + few-shot).

Hand-computed expectations for the acc/acc_norm test (character lengths, per
the char-vs-byte parity decision pinned in ``test_generative_tasks.py``):

- ex1: choices ["dobar", "loš"] (lengths 5, 3), gold index 0.
  scores: " dobar" -> -1.0, " loš" -> -3.0.
  raw:  argmax(-1.0, -3.0)             = 0 (correct)
  norm: argmax(-1.0/5, -3.0/3) = argmax(-0.2, -1.0) = 0 (correct)
- ex2: choices ["ide", "trči"] (lengths 3, 4), gold index 1.
  scores: " ide" -> -2.0, " trči" -> -2.5.
  raw:  argmax(-2.0, -2.5)                    = 0 (wrong, gold is 1)
  norm: argmax(-2.0/3, -2.5/4) = argmax(-0.667, -0.625) = 1 (correct)

So predictions_raw = [0, 0] vs golds [0, 1] -> acc = 1/2 = 0.5.
predictions_norm = [0, 1] vs golds [0, 1] -> acc_norm = 2/2 = 1.0.
"""

from __future__ import annotations

import pytest

from balkanbench.evaluation.generative import (
    GenerativeRunResult,
    build_fewshot_prefix,
    run_generative_eval,
)
from balkanbench.tasks import get_task_class
from balkanbench.tasks.generative import (  # noqa: F401 (ensures registration)
    GenerativeQATask,
    MultipleChoiceLoglikelihoodTask,
)


class FakeLLModel:
    """Returns rigged loglikelihoods keyed by continuation text."""

    def __init__(self, scores: dict[str, float]):
        self.scores = scores
        self.seen: list[tuple[str, str]] = []

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        self.seen.extend(requests)
        return [self.scores[cont] for _, cont in requests]

    def generate(self, prompts: list[str], **kw: object) -> list[str]:
        raise AssertionError("must not generate in LL protocol")


def _mc_task_spec(
    report: list[str], *, task: str = "arc_easy_fake", task_score: str = "acc"
) -> dict:
    return {
        "benchmark": "sle",
        "task": task,
        "task_type": "multiple_choice_loglikelihood",
        "languages": {"available": ["sr"]},
        "inputs": {"fields": ["query"], "id_field": "example_id"},
        "metrics": {"report": report, "task_score": task_score},
        "evaluation": {"num_fewshot": 0},
    }


def _winogrande_task_spec() -> dict:
    return {
        "benchmark": "sle",
        "task": "winogrande_fake",
        "task_type": "multiple_choice_loglikelihood",
        "variant": "winogrande_partial",
        "languages": {"available": ["sr"]},
        "inputs": {"fields": ["sentence", "option1", "option2"], "id_field": "example_id"},
        "metrics": {"report": ["acc"], "task_score": "acc"},
        "evaluation": {"num_fewshot": 0},
    }


_EX1 = {"example_id": "e1", "query": "Q1", "choices": ["dobar", "loš"], "gold": 0}
_EX2 = {"example_id": "e2", "query": "Q2", "choices": ["ide", "trči"], "gold": 1}
_SCORES = {" dobar": -1.0, " loš": -3.0, " ide": -2.0, " trči": -2.5}


def test_mc_acc_and_acc_norm() -> None:
    model = FakeLLModel(_SCORES)
    result = run_generative_eval(
        task_spec=_mc_task_spec(["acc", "acc_norm"]),
        model=model,
        model_cfg={},
        dataset=[_EX1, _EX2],
    )
    assert result.metrics == {"acc": 0.5, "acc_norm": 1.0}


def test_winogrande_variant_reports_acc_only() -> None:
    ex = {
        "example_id": "w1",
        "sentence": "Marko je dao Petru knjigu jer je _ bio tu.",
        "option1": "Marko",
        "option2": "Petar",
        "answer": "1",
    }
    model = FakeLLModel({" bio tu.": -1.0})
    result = run_generative_eval(
        task_spec=_winogrande_task_spec(),
        model=model,
        model_cfg={},
        dataset=[ex],
    )
    assert set(result.metrics) == {"acc"}
    assert "acc_norm" not in result.metrics


def test_limit_evaluates_one_example() -> None:
    model = FakeLLModel(_SCORES)
    result = run_generative_eval(
        task_spec=_mc_task_spec(["acc", "acc_norm"]),
        model=model,
        model_cfg={},
        dataset=[_EX1, _EX2],
        limit=1,
    )
    assert len(result.per_example) == 1
    assert result.metrics == {"acc": 1.0, "acc_norm": 1.0}
    assert len(model.seen) == 2  # only ex1's two requests


def test_per_example_entries_carry_id_and_prediction() -> None:
    model = FakeLLModel(_SCORES)
    result = run_generative_eval(
        task_spec=_mc_task_spec(["acc", "acc_norm"], task_score="acc"),
        model=model,
        model_cfg={},
        dataset=[_EX1, _EX2],
    )
    assert result.per_example == [
        {
            "example_id": "e1",
            "prediction_raw": 0,
            "prediction_norm": 0,
            "prediction": 0,
            "correct": True,
        },
        {
            "example_id": "e2",
            "prediction_raw": 0,
            "prediction_norm": 1,
            "prediction": 0,
            "correct": False,
        },
    ]


def test_per_example_prediction_follows_acc_norm_task_score() -> None:
    """arc_easy (and hellaswag/openbookqa/piqa) report task_score: acc_norm, so
    per_example "prediction"/"correct" must follow the length-normalized
    argmax, not the raw one, when the two disagree.

    ex2 (see module docstring hand-computation): raw argmax = 0, norm
    argmax = 1, gold = 1 - raw and norm genuinely disagree here.
    """
    spec = _mc_task_spec(["acc", "acc_norm"], task="arc_easy", task_score="acc_norm")
    model = FakeLLModel(_SCORES)
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[_EX2],
    )
    entry = result.per_example[0]
    assert entry["prediction_raw"] == 0
    assert entry["prediction_norm"] == 1
    assert entry["prediction"] == entry["prediction_norm"] != entry["prediction_raw"]
    assert entry["correct"] is True  # gold is 1, prediction_norm is 1


def test_per_example_prediction_follows_acc_task_score_for_winogrande() -> None:
    """winogrande has task_score: acc (never acc_norm), so per_example
    "prediction" must be the raw argmax."""
    spec = _winogrande_task_spec()
    ex = {
        "example_id": "w1",
        "sentence": "Marko je dao Petru knjigu jer je _ bio tu.",
        "option1": "Marko",
        "option2": "Petar",
        "answer": "1",
    }
    model = FakeLLModel({" bio tu.": -1.0})
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[ex],
    )
    entry = result.per_example[0]
    assert entry["prediction"] == entry["prediction_raw"]


def test_run_result_metadata_fields() -> None:
    model = FakeLLModel(_SCORES)
    result = run_generative_eval(
        task_spec=_mc_task_spec(["acc"]),
        model=model,
        model_cfg={},
        dataset=[_EX1],
    )
    assert isinstance(result, GenerativeRunResult)
    assert result.protocol == "loglikelihood"
    assert result.num_fewshot == 0
    assert result.unparsed_responses == 0
    assert result.api_cost_usd == 0.0


def test_unsupported_task_type_not_implemented() -> None:
    spec = _mc_task_spec(["acc"])
    spec["task_type"] = "some_future_task_type"
    with pytest.raises(NotImplementedError, match="some_future_task_type"):
        run_generative_eval(
            task_spec=spec,
            model=FakeLLModel({}),
            model_cfg={},
            dataset=[],
        )


def test_unsupported_api_reformulation_not_implemented() -> None:
    spec = _mc_task_spec(["acc"])
    spec["api_protocol"] = {"reformulation": "some_future_reformulation", "metrics": {}}
    with pytest.raises(NotImplementedError, match="some_future_reformulation"):
        run_generative_eval(
            task_spec=spec,
            model=FakeLLModel({}),
            model_cfg={"access": "api"},
            dataset=[],
        )


# ----------------------------------------------------------------------
# Few-shot machinery
# ----------------------------------------------------------------------

_QA_SPEC = {"benchmark": "sle", "task": "nq_open_fake", "languages": {"available": ["sr"]}}
_QA_FEWSHOT_DATASET = [
    {"question": "Q0", "answer": ["A0"]},
    {"question": "Q1", "answer": ["A1"]},
    {"question": "Q2", "answer": ["A2"]},
    {"question": "Q3", "answer": ["A3"]},
]


def _qa_task() -> GenerativeQATask:
    task_cls = get_task_class("generative_qa")
    return task_cls(_QA_SPEC, "sr")


def test_fewshot_prefix_empty_when_zero_shot() -> None:
    task = _qa_task()
    assert (
        build_fewshot_prefix(
            task=task, fewshot_dataset=_QA_FEWSHOT_DATASET, num_fewshot=0, example_id="x"
        )
        == ""
    )


def test_fewshot_prefix_exact_string_for_two_shot() -> None:
    task = _qa_task()
    # random.Random("sle-nq_open_fake-ex-42").sample(range(4), 2) == [1, 2]
    # (pinned by running it once; see module docstring policy in generative.py).
    prefix = build_fewshot_prefix(
        task=task, fewshot_dataset=_QA_FEWSHOT_DATASET, num_fewshot=2, example_id="ex-42"
    )
    assert prefix == ("Pitanje: Q1\nOdgovor: A1\n\nPitanje: Q2\nOdgovor: A2\n\n")


def test_fewshot_prefix_deterministic_same_id_same_shots() -> None:
    task = _qa_task()
    first = build_fewshot_prefix(
        task=task, fewshot_dataset=_QA_FEWSHOT_DATASET, num_fewshot=2, example_id="stable-id"
    )
    second = build_fewshot_prefix(
        task=task, fewshot_dataset=_QA_FEWSHOT_DATASET, num_fewshot=2, example_id="stable-id"
    )
    assert first == second
    assert first != ""


def test_fewshot_prefix_draws_distinct_shots() -> None:
    task = _qa_task()
    prefix = build_fewshot_prefix(
        task=task, fewshot_dataset=_QA_FEWSHOT_DATASET, num_fewshot=4, example_id="all-shots"
    )
    # All 4 pool questions must appear exactly once each - distinct indices.
    for i in range(4):
        assert prefix.count(f"Q{i}") == 1


def test_fewshot_prefix_prepended_to_every_ll_context(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_fewshot_text(self: MultipleChoiceLoglikelihoodTask, ex: dict) -> str:
        return f"SHOT:{ex['example_id']}"

    monkeypatch.setattr(
        MultipleChoiceLoglikelihoodTask, "fewshot_example_text", fake_fewshot_text, raising=False
    )

    spec = _mc_task_spec(["acc"])
    spec["evaluation"]["num_fewshot"] = 2
    fewshot_dataset = [{"example_id": f"f{i}"} for i in range(5)]

    # Recompute the per-example expected prefixes the same way the evaluator
    # will - two DIFFERENT examples with different example_ids, so a
    # regression that computes the prefix once (e.g. from the first example)
    # and reuses it for every subsequent example is caught: each example's
    # RNG is seeded from its own example_id, so their shots (and therefore
    # prefixes) generally differ.
    task_cls = get_task_class("multiple_choice_loglikelihood")
    task = task_cls(spec, "sr")
    expected_prefix_e1 = build_fewshot_prefix(
        task=task, fewshot_dataset=fewshot_dataset, num_fewshot=2, example_id="e1"
    )
    expected_prefix_e2 = build_fewshot_prefix(
        task=task, fewshot_dataset=fewshot_dataset, num_fewshot=2, example_id="e2"
    )
    assert expected_prefix_e1 != ""
    assert expected_prefix_e2 != ""
    # If this ever fails because the two example_ids collide on the same
    # shots, pick different ids above - the point is to exercise per-example
    # prefixes that are NOT identical.
    assert expected_prefix_e1 != expected_prefix_e2

    model = FakeLLModel({" dobar": -1.0, " loš": -3.0, " ide": -2.0, " trči": -2.5})
    run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[_EX1, _EX2],
        fewshot_dataset=fewshot_dataset,
    )

    assert model.seen == [
        (expected_prefix_e1 + "Q1", " dobar"),
        (expected_prefix_e1 + "Q1", " loš"),
        (expected_prefix_e2 + "Q2", " ide"),
        (expected_prefix_e2 + "Q2", " trči"),
    ]


def test_fewshot_prefix_requires_dataset_when_shots_requested() -> None:
    task = _qa_task()
    with pytest.raises(ValueError):
        build_fewshot_prefix(task=task, fewshot_dataset=None, num_fewshot=1, example_id="x")


# ----------------------------------------------------------------------
# generative_qa (native EM generation, open + API models share this path)
# ----------------------------------------------------------------------


class FakeGenModel:
    """Returns canned ``generate()`` outputs; records every call it receives."""

    def __init__(self, texts: list[str], *, total_cost_usd: float | None = None):
        self.texts = texts
        self.calls: list[dict[str, object]] = []
        if total_cost_usd is not None:
            self.total_cost_usd = total_cost_usd

    def loglikelihood(self, requests: list[tuple[str, str]]) -> list[float]:
        raise AssertionError("must not call loglikelihood in a generative-only protocol")

    def generate(
        self, prompts: list[str], *, stop_sequences: list[str], max_gen_tokens: int
    ) -> list[str]:
        self.calls.append(
            {
                "prompts": list(prompts),
                "stop_sequences": stop_sequences,
                "max_gen_tokens": max_gen_tokens,
            }
        )
        return list(self.texts)


def _qa_task_spec(
    *,
    num_fewshot: int = 0,
    stop_sequences: list[str] | None = None,
    max_gen_tokens: int = 64,
    report: list[str] | None = None,
    task_score: str = "em",
    task: str = "nq_open_fake",
) -> dict:
    return {
        "benchmark": "sle",
        "task": task,
        "task_type": "generative_qa",
        "languages": {"available": ["sr"]},
        "inputs": {"fields": ["question"], "id_field": "example_id"},
        "metrics": {
            "primary": report or ["em"],
            "report": report or ["em"],
            "task_score": task_score,
        },
        "evaluation": {
            "num_fewshot": num_fewshot,
            "fewshot_split": "train",
            "stop_sequences": stop_sequences if stop_sequences is not None else ["\n", "."],
            "max_gen_tokens": max_gen_tokens,
        },
    }


_QA_EX1 = {"example_id": "q1", "question": "Koji je glavni grad Srbije?", "answer": ["Beograd"]}
_QA_EX2 = {"example_id": "q2", "question": "Koja je najveca reka?", "answer": ["Dunav", "Sava"]}


def test_generative_qa_computes_em_via_registry() -> None:
    model = FakeGenModel([" Beograd", " neka pogresna reka"])
    result = run_generative_eval(
        task_spec=_qa_task_spec(),
        model=model,
        model_cfg={},
        dataset=[_QA_EX1, _QA_EX2],
    )
    assert result.metrics == {"em": 0.5}
    assert result.protocol == "generative"


def test_generative_qa_prompts_are_prefix_plus_qa_prompt() -> None:
    fewshot_dataset = [
        {"question": "Q0", "answer": ["A0"]},
        {"question": "Q1", "answer": ["A1"]},
        {"question": "Q2", "answer": ["A2"]},
        {"question": "Q3", "answer": ["A3"]},
    ]
    spec = _qa_task_spec(num_fewshot=2)
    model = FakeGenModel([" Beograd", " Dunav"])
    task = get_task_class("generative_qa")(spec, "sr")

    expected_prefix_1 = build_fewshot_prefix(
        task=task, fewshot_dataset=fewshot_dataset, num_fewshot=2, example_id="q1"
    )
    expected_prefix_2 = build_fewshot_prefix(
        task=task, fewshot_dataset=fewshot_dataset, num_fewshot=2, example_id="q2"
    )

    run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[_QA_EX1, _QA_EX2],
        fewshot_dataset=fewshot_dataset,
    )

    assert model.calls[0]["prompts"] == [
        expected_prefix_1 + task.qa_prompt(_QA_EX1),
        expected_prefix_2 + task.qa_prompt(_QA_EX2),
    ]


def test_generative_qa_forwards_stop_sequences_and_max_gen_tokens() -> None:
    spec = _qa_task_spec(stop_sequences=["\n", ",", "."], max_gen_tokens=64)
    model = FakeGenModel([" Beograd"])
    run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[_QA_EX1],
    )
    assert model.calls[0]["stop_sequences"] == ["\n", ",", "."]
    assert model.calls[0]["max_gen_tokens"] == 64


def test_generative_qa_per_example_prediction_and_correctness() -> None:
    model = FakeGenModel([" Beograd", " neka pogresna reka"])
    result = run_generative_eval(
        task_spec=_qa_task_spec(),
        model=model,
        model_cfg={},
        dataset=[_QA_EX1, _QA_EX2],
    )
    assert result.per_example == [
        {"example_id": "q1", "prediction": " Beograd", "correct": True},
        {"example_id": "q2", "prediction": " neka pogresna reka", "correct": False},
    ]


def test_triviaqa_spec_uses_em_triviaqa_metric() -> None:
    spec = _qa_task_spec(report=["em_triviaqa"], task_score="em_triviaqa", task="triviaqa_fake")
    ex = {"example_id": "t1", "question": "Q", "answer_value": "Answer", "answer_aliases": ["Alt"]}
    model = FakeGenModel([" Answer"])
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={},
        dataset=[ex],
    )
    assert "em_triviaqa" in result.metrics
    assert result.metrics["em_triviaqa"] == 1.0


def test_generative_qa_run_result_metadata() -> None:
    model = FakeGenModel([" Beograd"])
    result = run_generative_eval(
        task_spec=_qa_task_spec(num_fewshot=0),
        model=model,
        model_cfg={},
        dataset=[_QA_EX1],
    )
    assert result.num_fewshot == 0
    assert result.unparsed_responses == 0
    assert result.api_cost_usd == 0.0


# ----------------------------------------------------------------------
# API multiple_choice_generative reformulation
# ----------------------------------------------------------------------


def _mc_api_task_spec() -> dict:
    spec = _mc_task_spec(["acc"], task_score="acc")
    spec["evaluation"] = {"num_fewshot": 0, "stop_sequences": [], "max_gen_tokens": 16}
    spec["api_protocol"] = {
        "reformulation": "multiple_choice_generative",
        "metrics": {"primary": ["acc"], "report": ["acc"], "task_score": "acc"},
    }
    return spec


def test_api_multiple_choice_scores_and_counts_unparsed() -> None:
    spec = _mc_api_task_spec()
    model = FakeGenModel(["A", "B", "xyz nema slovo"])
    ex3 = {"example_id": "e3", "query": "Q3", "choices": ["da", "ne"], "gold": 0}
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_EX1, _EX2, ex3],
    )
    assert result.unparsed_responses == 1
    assert result.protocol == "generative"
    assert result.per_example[2]["prediction"] == -1
    assert result.per_example[2]["correct"] is False
    # ex1: gold 0, parsed "A" -> 0 (correct); ex2: gold 1, parsed "B" -> 1
    # (correct); ex3: unparsed -> -1 vs gold 0 (incorrect).
    assert result.metrics["acc"] == pytest.approx(2 / 3)


def test_api_multiple_choice_prompts_are_api_prompt_with_no_fewshot() -> None:
    spec = _mc_api_task_spec()
    task = get_task_class("multiple_choice_loglikelihood")(spec, "sr")
    model = FakeGenModel(["A", "B"])
    run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_EX1, _EX2],
    )
    assert model.calls[0]["prompts"] == [task.api_prompt(_EX1), task.api_prompt(_EX2)]


def test_api_multiple_choice_forwards_stop_sequences_and_max_gen_tokens() -> None:
    spec = _mc_api_task_spec()
    model = FakeGenModel(["A"])
    run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_EX1],
    )
    assert model.calls[0]["stop_sequences"] == []
    assert model.calls[0]["max_gen_tokens"] == 16


# ----------------------------------------------------------------------
# API access must use api_protocol.metrics, not the native metrics block
# ----------------------------------------------------------------------


def test_api_generative_qa_uses_api_protocol_metrics_not_native_metrics() -> None:
    """api_protocol.metrics is deliberately DIFFERENT from the native metrics
    block here (real task YAMLs keep them identical) so a regression that
    reads task_spec["metrics"] for API access is caught rather than passing
    by coincidence."""
    spec = _qa_task_spec(report=["em"], task_score="em")
    spec["api_protocol"] = {
        "reformulation": "generative_qa",
        "metrics": {
            "primary": ["em_triviaqa"],
            "report": ["em_triviaqa"],
            "task_score": "em_triviaqa",
        },
    }
    model = FakeGenModel([" Beograd"])
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_QA_EX1],
    )
    assert set(result.metrics) == {"em_triviaqa"}


def test_api_multiple_choice_uses_api_protocol_metrics() -> None:
    """Same guard as above for the MC reformulation: api_protocol.metrics
    reports "acc" while the (unused, absent-on-purpose) native metrics block
    is not consulted at all for API access."""
    spec = _mc_api_task_spec()
    del spec["metrics"]  # prove the native block is never read for API access
    model = FakeGenModel(["A", "B"])
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_EX1, _EX2],
    )
    assert set(result.metrics) == {"acc"}


# ----------------------------------------------------------------------
# API cost propagation
# ----------------------------------------------------------------------


def test_api_cost_propagates_from_model_total_cost_usd() -> None:
    spec = _mc_api_task_spec()
    model = FakeGenModel(["A", "B"], total_cost_usd=0.1234)
    result = run_generative_eval(
        task_spec=spec,
        model=model,
        model_cfg={"access": "api"},
        dataset=[_EX1, _EX2],
    )
    assert result.api_cost_usd == 0.1234


def test_open_model_without_total_cost_usd_defaults_to_zero() -> None:
    model = FakeGenModel([" Beograd"])
    result = run_generative_eval(
        task_spec=_qa_task_spec(),
        model=model,
        model_cfg={},
        dataset=[_QA_EX1],
    )
    assert result.api_cost_usd == 0.0
