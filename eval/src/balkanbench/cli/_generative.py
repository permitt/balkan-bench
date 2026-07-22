"""Shared SLE (generative) dispatch helper for the ``eval``/``run`` CLIs.

Both ``balkanbench eval`` and ``balkanbench run`` route ``task_type`` values
in :data:`GENERATIVE_TASK_TYPES` through :func:`run_generative_dispatch`
instead of the encoder (Trainer-based) path: build a
:class:`~balkanbench.models.generative_base.GenerativeModel` (local
open-weights or API-backed), load the public test split (+ few-shot split
when the task calls for it) via the caller's own ``load_dataset`` seam,
run :func:`~balkanbench.evaluation.generative.run_generative_eval`, and
write the result with
:func:`~balkanbench.scoring.artifact.write_generative_result_artifact`. Each
split is fetched with its own ``load_dataset(..., split=...)`` call (rather
than loading the whole ``DatasetDict`` up front, as the encoder path does)
since a generative run only ever needs the ``test`` split, plus the
few-shot split when the task calls for one.

``load_dataset`` is passed in by the caller (``balkanbench.cli.eval`` /
``balkanbench.cli.run``) rather than imported here so that tests can keep
monkeypatching the same per-module seam (``balkanbench.cli.eval.load_dataset``
/ ``balkanbench.cli.run.load_dataset``) they already use for the encoder path.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from balkanbench.data.repo import resolve_dataset_repo
from balkanbench.evaluation.generative import run_generative_eval
from balkanbench.models.api import make_api_model
from balkanbench.models.causal_lm import CausalLM
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import write_generative_result_artifact

GENERATIVE_TASK_TYPES = frozenset({"multiple_choice_loglikelihood", "generative_qa"})


def run_generative_dispatch(
    *,
    task_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    language: str,
    out_dir: Path,
    api_cache_dir: Path,
    dataset_revision: str,
    benchmark_version: str,
    run_type: str,
    limit: int | None,
    load_dataset: Callable[..., Any],
    token: str | None,
) -> Path:
    """Build the model, load data, evaluate, write the result artifact.

    ``limit`` (when not ``None``) slices the dataset via
    ``run_generative_eval``'s own ``limit`` param and forces the artifact's
    ``run_type`` to ``"experimental"`` regardless of ``run_type`` - a limited
    run must never be recorded as ``"official"``.

    Returns the path of the written ``result.json``.
    """
    access = model_cfg.get("access", "open_weights")
    model: Any
    if access == "api":
        model = make_api_model(model_cfg, cache_dir=api_cache_dir)
    else:
        model = CausalLM(model_cfg)

    repo_id = resolve_dataset_repo(task_cfg, language, prefer="public")
    config = task_cfg["dataset"]["config"]
    dataset = load_dataset(
        repo_id,
        config,
        split="test",
        revision=dataset_revision,
        token=token,
    )

    num_fewshot = int(task_cfg["evaluation"]["num_fewshot"])
    fewshot_dataset = None
    if num_fewshot > 0:
        fewshot_split = task_cfg["evaluation"]["fewshot_split"]
        fewshot_dataset = load_dataset(
            repo_id,
            config,
            split=fewshot_split,
            revision=dataset_revision,
            token=token,
        )

    run_result = run_generative_eval(
        task_spec=task_cfg,
        model=model,
        model_cfg=model_cfg,
        dataset=dataset,
        fewshot_dataset=fewshot_dataset,
        limit=limit,
    )

    task_score_metric = (
        task_cfg["api_protocol"]["metrics"]["task_score"]
        if access == "api"
        else task_cfg["metrics"]["task_score"]
    )

    # A limited run samples the dataset, so its score can never stand in for
    # the official (full-data) result - forced to experimental regardless of
    # what --run-type asked for.
    effective_run_type = "experimental" if limit is not None else run_type

    provenance = collect_provenance()
    return write_generative_result_artifact(
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        language=language,
        run_result=run_result,
        task_score_metric=task_score_metric,
        provenance=provenance,
        dataset_revision=dataset_revision,
        benchmark_version=benchmark_version,
        out_dir=out_dir,
        run_type=effective_run_type,
    )


__all__ = ["GENERATIVE_TASK_TYPES", "run_generative_dispatch"]
