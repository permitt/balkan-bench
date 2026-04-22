"""Single-seed + multi-seed training/eval orchestrator.

The evaluator is the only module that touches the HF ``Trainer``. It never
subclasses it: training behaviour is driven by ``TrainingArguments`` and
callbacks only.

Design contract:
- ``run_single_seed`` trains one model on ``datasets["train"]`` and
  evaluates on ``datasets["validation"]`` (or another eval split), returning
  a ``SeedResult`` with predictions + decoded class indices + scored metric
  bundle.
- ``run_multiseed`` loops ``run_single_seed`` across seeds and returns a
  list of ``SeedResult``.
- ``aggregate_seed_results`` collapses a list of ``SeedResult`` into mean +
  sample stdev (ddof=1) per metric.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from datasets import DatasetDict
from transformers import Trainer, TrainingArguments

from balkanbench.models.hf_encoder import HFEncoder
from balkanbench.seed import set_seed
from balkanbench.tasks import get_task_class


@dataclass
class SeedResult:
    """Outcome of a single-seed training + evaluation run."""

    seed: int
    primary: dict[str, float]
    secondary: dict[str, float]
    task_score: float
    predictions: list[Any]
    references: list[Any]
    group_ids: list[Any] | None = None


@dataclass
class Aggregate:
    """Per-metric mean + sample stdev across seeds."""

    mean: dict[str, float] = field(default_factory=dict)
    stdev: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------


def aggregate_seed_results(seed_results: list[SeedResult]) -> Aggregate:
    """Mean + sample stdev (ddof=1) of primary metrics across seeds."""
    if not seed_results:
        raise ValueError("aggregate_seed_results requires at least one seed result")

    per_metric: dict[str, list[float]] = {}
    for r in seed_results:
        for name, value in r.primary.items():
            per_metric.setdefault(name, []).append(float(value))

    mean: dict[str, float] = {}
    stdev: dict[str, float] = {}
    for name, values in per_metric.items():
        mean[name] = statistics.fmean(values)
        if len(values) == 1:
            stdev[name] = 0.0
        else:
            stdev[name] = statistics.stdev(values)
        if math.isnan(stdev[name]):
            stdev[name] = 0.0

    return Aggregate(mean=mean, stdev=stdev)


# ----------------------------------------------------------------------
# Single-seed orchestration
# ----------------------------------------------------------------------


def run_single_seed(
    *,
    model_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    language: str,
    datasets: DatasetDict,
    seed: int,
    output_dir: str | Path,
    eval_split: str = "validation",
    train: bool = True,
    compute_metrics: bool = True,
) -> SeedResult:
    """Train (optional) then evaluate on ``eval_split``.

    Flags:
    - ``train=False`` skips ``Trainer.train()`` - used by diagnostic-only tasks
      (AXb/AXg) which have no train split, and by any CLI that already loaded
      a trained checkpoint.
    - ``compute_metrics=False`` skips reference loading and scoring - used by
      ``balkanbench predict`` against the public test split, which has no
      label column.
    """
    set_seed(seed)

    encoder = HFEncoder.build(model_cfg=model_cfg, task_cfg=task_cfg)
    task_cls = get_task_class(task_cfg["task_type"])
    task = task_cls(task_cfg, language=language)

    tokenized = _tokenize_datasets(task=task, datasets=datasets, tokenizer=encoder.tokenizer)

    training_args = _build_training_arguments(
        merged=encoder.training_args, output_dir=output_dir, seed=seed
    )

    # Note: in transformers >=5 the `tokenizer` kwarg became `processing_class`.
    # We pre-tokenise the datasets ourselves, so the Trainer does not need it.
    trainer = Trainer(
        model=encoder.model,
        args=training_args,
        train_dataset=tokenized.get("train"),
        eval_dataset=tokenized.get(eval_split),
    )
    if train:
        trainer.train()

    eval_ds = tokenized[eval_split]
    prediction_output = trainer.predict(eval_ds)
    logits = np.asarray(prediction_output.predictions)
    decoded = task.decode(logits)
    predictions = decoded.tolist() if hasattr(decoded, "tolist") else list(decoded)

    if not compute_metrics:
        return SeedResult(
            seed=seed,
            primary={},
            secondary={},
            task_score=0.0,
            predictions=predictions,
            references=[],
            group_ids=None,
        )

    # References come from the original (pre-tokenised) split so our fake
    # tokeniser-free tests still work.
    references = list(datasets[eval_split][task_cfg.get("label_field", "label")])

    score_kwargs: dict[str, Any] = {}
    group_ids: list[Any] | None = None
    group_fields = task_cfg["inputs"].get("group_fields") or []
    if group_fields:
        group_ids = list(zip(*[datasets[eval_split][g] for g in group_fields], strict=True))
        score_kwargs["group_ids"] = group_ids

    # Generic side-channel: let a task declare extra columns the evaluator
    # forwards to ``score()`` as kwargs. AXg's ``gender_parity`` uses this to
    # carry ``is_pro_stereotype``; future metrics that need per-example
    # metadata plug in here without the evaluator having to know about them.
    for column in task_cfg["inputs"].get("metric_columns") or []:
        if column not in datasets[eval_split].column_names:
            raise KeyError(
                f"task_cfg.inputs.metric_columns references {column!r} but the "
                f"{eval_split!r} split does not have it; columns present: "
                f"{datasets[eval_split].column_names}"
            )
        score_kwargs[column] = list(datasets[eval_split][column])

    bundle = task.score(predictions=predictions, references=references, **score_kwargs)
    primary = {name: value for name, value in bundle.items() if name in task.primary_metric_names()}
    secondary = {name: value for name, value in bundle.items() if name not in primary}
    task_score = task.task_score(bundle)

    return SeedResult(
        seed=seed,
        primary=primary,
        secondary=secondary,
        task_score=task_score,
        predictions=predictions,
        references=references,
        group_ids=group_ids,
    )


def run_multiseed(
    *,
    model_cfg: dict[str, Any],
    task_cfg: dict[str, Any],
    language: str,
    datasets: DatasetDict,
    seeds: list[int],
    output_dir: str | Path,
    eval_split: str = "validation",
    train: bool = True,
    compute_metrics: bool = True,
) -> list[SeedResult]:
    """Run ``run_single_seed`` for every seed; return the list in order."""
    results: list[SeedResult] = []
    for seed in seeds:
        results.append(
            run_single_seed(
                model_cfg=model_cfg,
                task_cfg=task_cfg,
                language=language,
                datasets=datasets,
                seed=seed,
                output_dir=Path(output_dir) / f"seed-{seed}",
                eval_split=eval_split,
                train=train,
                compute_metrics=compute_metrics,
            )
        )
    return results


# ----------------------------------------------------------------------
# Internals
# ----------------------------------------------------------------------


def _tokenize_datasets(
    *,
    task: Any,
    datasets: DatasetDict,
    tokenizer: Any,
) -> DatasetDict:
    """Apply ``task.preprocess`` over every split, keeping label column intact."""
    out: dict[str, Any] = {}
    for split_name, split in datasets.items():
        preprocessed = [task.preprocess(row, tokenizer=tokenizer) for row in split]
        from datasets import Dataset as _Dataset

        if not preprocessed:
            out[split_name] = split
            continue
        keys = preprocessed[0].keys()
        columns = {k: [ex[k] for ex in preprocessed] for k in keys}
        out[split_name] = _Dataset.from_dict(columns)
    return DatasetDict(out)


def _build_training_arguments(
    *,
    merged: dict[str, Any],
    output_dir: str | Path,
    seed: int,
) -> Any:
    """Assemble ``TrainingArguments`` from the merged task+model training dict."""
    return TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(merged.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(merged.get("batch_size", 16)),
        per_device_eval_batch_size=int(merged.get("batch_size", 16)),
        num_train_epochs=int(merged.get("num_epochs", 1)),
        warmup_ratio=float(merged.get("warmup_ratio", 0.1)),
        weight_decay=float(merged.get("weight_decay", 0.0)),
        fp16=bool(merged.get("fp16", False)),
        seed=int(seed),
        report_to=[],
        logging_strategy="epoch",
        save_strategy="no",
        eval_strategy="no",
    )
