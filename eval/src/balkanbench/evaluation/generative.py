"""Generative evaluator: loglikelihood multiple-choice protocol + few-shot machinery.

Covers the SLE track's ``multiple_choice_loglikelihood`` task_type driven
through a :class:`~balkanbench.models.generative_base.GenerativeModel`
(local open-weights or, later, API-backed). ``generative_qa`` and every
API-protocol reformulation are intentionally out of scope here - Task 11
fills them in; :func:`run_generative_eval`'s protocol dispatch already has
the branches wired up, they just raise ``NotImplementedError`` for now.

Few-shot policy (DELIBERATE deviation from the reference harness,
lm-evaluation-harness v0.3.0): the fork draws a single, global set of shots
per task from a fixed master RNG seeded once, so every example in a run
shares the same shots and their order depends on how many prior
``rng.sample`` calls happened before it. That makes results depend on
sample order / concurrency and is awkward to reproduce for a single
example in isolation. We instead give every example its **own**
independent RNG, seeded deterministically from the task and example id
(``random.Random(f"sle-{task.task_name}-{example_id}")``), and draw
``num_fewshot`` *distinct* indices from the few-shot pool with it. This
means: (a) any single example's shots can be reproduced without replaying
the whole run, (b) shot selection is embarrassingly parallel, at the cost
of (c) no longer matching the fork's shot sequence bit-for-bit. The MC
loglikelihood tasks are all 0-shot in v1 (see the SLE task YAMLs), so this
machinery is exercised here by tests and used for real by Task 11's
generative_qa path (5-shot NQ-Open / TriviaQA).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

from balkanbench.metrics import get_metric
from balkanbench.tasks import get_task_class


@dataclass
class GenerativeRunResult:
    """Outcome of :func:`run_generative_eval`."""

    metrics: dict[str, float]
    per_example: list[dict[str, Any]]
    protocol: str  # "loglikelihood" | "generative"
    num_fewshot: int
    unparsed_responses: int  # API multiple-choice only, else 0
    api_cost_usd: float  # 0.0 for open (non-API) models


def build_fewshot_prefix(
    *,
    task: Any,
    fewshot_dataset: Any,
    num_fewshot: int,
    example_id: str,
) -> str:
    """Build the few-shot prefix for one example.

    ``task`` is typed loosely (``Any``) rather than the ``Task`` ABC: the
    only members used here (``task_name``, ``fewshot_example_text``) are
    task-family-specific (currently only :class:`GenerativeQATask` defines
    ``fewshot_example_text``), so no single base-class type covers every
    caller.

    Empty string when ``num_fewshot == 0``. Otherwise draws ``num_fewshot``
    distinct indices from ``fewshot_dataset`` with a per-example RNG seeded
    from ``f"sle-{task.task_name}-{example_id}"`` (see the module docstring
    for why this deviates from the reference harness), and joins
    ``task.fewshot_example_text(shot)`` for each drawn shot with a blank
    line, plus a trailing blank line separating the prefix from the actual
    example.
    """
    if num_fewshot == 0:
        return ""
    if fewshot_dataset is None:
        raise ValueError("fewshot_dataset is required when num_fewshot > 0")

    rng = random.Random(f"sle-{task.task_name}-{example_id}")
    pool_size = len(fewshot_dataset)
    indices = rng.sample(range(pool_size), num_fewshot)
    shots = [fewshot_dataset[i] for i in indices]
    return "\n\n".join(task.fewshot_example_text(shot) for shot in shots) + "\n\n"


def run_generative_eval(
    *,
    task_spec: dict[str, Any],
    model: Any,
    model_cfg: dict[str, Any],
    dataset: Any,
    fewshot_dataset: Any = None,
    limit: int | None = None,
) -> GenerativeRunResult:
    """Evaluate ``model`` on ``dataset`` per ``task_spec``.

    Protocol selection: ``model_cfg.get("access") == "api"`` routes through
    the task's ``api_protocol.reformulation``; otherwise the task's native
    ``task_type`` is used directly. Only the native
    ``multiple_choice_loglikelihood`` path is implemented here - every other
    branch (``generative_qa``, any API reformulation) raises
    ``NotImplementedError`` with a message naming the missing protocol, so
    Task 11 can slot its implementation into the same dispatch.
    """
    if model_cfg.get("access") == "api":
        reformulation = task_spec["api_protocol"]["reformulation"]
        raise NotImplementedError(
            f"API protocol reformulation {reformulation!r} is not implemented "
            "yet (Task 11 adds API-backed generative evaluation)."
        )

    task_type = task_spec["task_type"]
    if task_type == "generative_qa":
        raise NotImplementedError(
            "task_type 'generative_qa' is not implemented yet (Task 11 adds "
            "the greedy-generation + exact-match evaluation path)."
        )
    if task_type != "multiple_choice_loglikelihood":
        raise NotImplementedError(
            f"task_type {task_type!r} is not supported by the generative evaluator."
        )

    return _run_loglikelihood_protocol(
        task_spec=task_spec,
        model=model,
        dataset=dataset,
        fewshot_dataset=fewshot_dataset,
        limit=limit,
    )


def _run_loglikelihood_protocol(
    *,
    task_spec: dict[str, Any],
    model: Any,
    dataset: Any,
    fewshot_dataset: Any,
    limit: int | None,
) -> GenerativeRunResult:
    language = task_spec["languages"]["available"][0]
    # Typed Any: get_task_class returns a plain Task statically, but the
    # loglikelihood-protocol members used below (loglikelihood_requests,
    # gold_index, continuation_lengths) live on MultipleChoiceLoglikelihoodTask,
    # not the Task ABC - see build_fewshot_prefix's docstring for the same
    # rationale applied to fewshot_example_text.
    task: Any = get_task_class(task_spec["task_type"])(task_spec, language)

    id_field = task_spec["inputs"]["id_field"]
    num_fewshot = int(task_spec["evaluation"]["num_fewshot"])
    report_names = list(task_spec["metrics"]["report"])

    examples = list(dataset)
    if limit is not None:
        examples = examples[:limit]

    predictions_raw: list[int] = []
    predictions_norm: list[int] = []
    golds: list[int] = []
    per_example: list[dict[str, Any]] = []

    for ex in examples:
        example_id = str(ex[id_field])
        prefix = build_fewshot_prefix(
            task=task,
            fewshot_dataset=fewshot_dataset,
            num_fewshot=num_fewshot,
            example_id=example_id,
        )

        requests = task.loglikelihood_requests(ex)
        if prefix:
            requests = [(prefix + context, continuation) for context, continuation in requests]
        lls = model.loglikelihood(requests)
        lengths = task.continuation_lengths(ex)

        pred_raw = lls.index(max(lls))
        norm_scores = [ll / length for ll, length in zip(lls, lengths, strict=True)]
        pred_norm = norm_scores.index(max(norm_scores))
        gold = task.gold_index(ex)

        predictions_raw.append(pred_raw)
        predictions_norm.append(pred_norm)
        golds.append(gold)
        per_example.append(
            {
                "example_id": example_id,
                "prediction": pred_raw,
                "correct": pred_raw == gold,
            }
        )

    metrics: dict[str, float] = {}
    for name in report_names:
        fn = get_metric(name)
        predictions = predictions_norm if name == "acc_norm" else predictions_raw
        metrics[name] = fn(predictions=predictions, references=golds)

    return GenerativeRunResult(
        metrics=metrics,
        per_example=per_example,
        protocol="loglikelihood",
        num_fewshot=num_fewshot,
        unparsed_responses=0,
        api_cost_usd=0.0,
    )


__all__ = ["GenerativeRunResult", "build_fewshot_prefix", "run_generative_eval"]
