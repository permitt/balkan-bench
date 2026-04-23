"""Score a predictions.jsonl against private test labels.

Loads predictions from disk, downloads the private test split from
``task_cfg.dataset.private_repo`` via ``HF_OFFICIAL_TOKEN``, aligns by
``example_id``, computes metrics via the task's ``score()`` method, and
writes a schema-valid result artifact. Ensures no silent misalignment:
missing or extra prediction ids abort with a loud ``ScoreError``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from balkanbench.evaluation import Aggregate, SeedResult
from balkanbench.provenance import collect_provenance
from balkanbench.scoring.artifact import write_result_artifact
from balkanbench.tasks import get_task_class


# datasets.load_dataset is lazy-loaded so the scoring module can be imported
# without paying the datasets startup cost. Tests that
# ``monkeypatch.setattr("balkanbench.scoring.score.load_dataset", ...)`` win
# because monkeypatching sets the name in the module dict.
def __getattr__(name: str) -> Any:
    if name == "load_dataset":
        import datasets

        return datasets.load_dataset
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class ScoreError(RuntimeError):
    """Raised when the private-label scoring flow cannot proceed."""


def _token_or_raise() -> str:
    token = os.environ.get("HF_OFFICIAL_TOKEN")
    if not token:
        raise ScoreError(
            "HF_OFFICIAL_TOKEN is not set. Private-label scoring requires a "
            "Hugging Face token with read access to the private labels repo. "
            "Export HF_OFFICIAL_TOKEN=<token> and retry."
        )
    return token


def _load_predictions(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ScoreError(f"predictions file not found: {path}")
    mapping: dict[str, Any] = {}
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ScoreError(f"line {lineno} in {path} is not valid JSON: {exc}") from exc
        if "example_id" not in row or "prediction" not in row:
            raise ScoreError(
                f"line {lineno} in {path} must have 'example_id' and 'prediction'; "
                f"got {sorted(row)}"
            )
        mapping[str(row["example_id"])] = row["prediction"]
    if not mapping:
        raise ScoreError(f"no predictions in {path}")
    return mapping


def score_predictions(
    *,
    predictions_path: Path,
    task_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    language: str,
    dataset_revision: str,
    benchmark_version: str,
    out_dir: Path,
    run_type: str = "official",
) -> Path:
    """Score a predictions.jsonl against private labels; write a result artifact."""
    token = _token_or_raise()
    predictions_map = _load_predictions(predictions_path)

    private_repo = task_cfg["dataset"].get("private_repo")
    if not private_repo:
        raise ScoreError(
            f"task_cfg.dataset.private_repo not set for {task_cfg['task']!r}; "
            "private-label scoring requires a private repo"
        )

    from balkanbench.scoring import score as _self

    private_labels_ds = _self.load_dataset(
        private_repo,
        task_cfg["dataset"]["config"],
        split="test",
        revision=dataset_revision,
        token=token,
    )

    id_field = task_cfg["inputs"]["id_field"]
    label_field = task_cfg.get("label_field", "label")

    label_by_id: dict[str, Any] = {}
    for row in private_labels_ds:
        label_by_id[str(row[id_field])] = row[label_field]

    # Alignment: every private id must have a prediction; every prediction id
    # must be in the private labels. Either direction of drift is a bug.
    missing = sorted(set(label_by_id) - set(predictions_map))
    if missing:
        raise ScoreError(
            f"predictions missing for {len(missing)} example_id(s); first: {missing[:5]}"
        )
    unexpected = sorted(set(predictions_map) - set(label_by_id))
    if unexpected:
        raise ScoreError(
            f"predictions contain {len(unexpected)} unexpected example_id(s) "
            f"not in the private labels repo; first: {unexpected[:5]}"
        )

    ordered_ids = sorted(label_by_id)
    predictions_list = [predictions_map[eid] for eid in ordered_ids]
    references = [label_by_id[eid] for eid in ordered_ids]

    task_cls = get_task_class(task_cfg["task_type"])
    task = task_cls(task_cfg, language=language)

    score_kwargs: dict[str, Any] = {}
    group_fields = task_cfg["inputs"].get("group_fields") or []
    if group_fields:
        group_ids: list[Any] = []
        row_by_id = {str(r[id_field]): r for r in private_labels_ds}
        for eid in ordered_ids:
            row = row_by_id[eid]
            group_ids.append(tuple(row[g] for g in group_fields))
        score_kwargs["group_ids"] = group_ids
    for column in task_cfg["inputs"].get("metric_columns") or []:
        row_by_id = {str(r[id_field]): r for r in private_labels_ds}
        score_kwargs[column] = [row_by_id[eid][column] for eid in ordered_ids]

    bundle = task.score(predictions=predictions_list, references=references, **score_kwargs)
    primary = {name: value for name, value in bundle.items() if name in task.primary_metric_names()}
    secondary = {name: value for name, value in bundle.items() if name not in primary}
    task_score = task.task_score(bundle)

    # Private-label scoring is not seeded; the underlying predictions come
    # from one or more seeded runs (tracked via their own artifacts).
    seed_result = SeedResult(
        seed=0,
        primary=primary,
        secondary=secondary,
        task_score=task_score,
        predictions=predictions_list,
        references=references,
        group_ids=score_kwargs.get("group_ids"),
    )
    aggregate = Aggregate(
        mean=dict(primary),
        stdev={name: 0.0 for name in primary},
    )
    provenance = collect_provenance()
    hp_search: dict[str, Any] = {
        "tool": "optuna",
        "sampler": "TPESampler",
        "sampler_seed": 42,
        "num_trials": 0,
        "search_space_id": "private-label-scoring",
    }

    return write_result_artifact(
        task_cfg=task_cfg,
        model_cfg=model_cfg,
        language=language,
        seed_results=[seed_result],
        aggregate=aggregate,
        provenance=provenance,
        dataset_revision=dataset_revision,
        benchmark_version=benchmark_version,
        hp_search=hp_search,
        out_dir=out_dir,
        run_type=run_type,
    )
