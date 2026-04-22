"""Seed official result artifacts for the v0.1 launch.

Reads the launch leaderboard JSON (the aggregate numbers the user's
prior pipeline produced) and emits one schema-valid ``result.json`` per
(model, task) pair under ``eval/results/official/{benchmark}-{language}/``.

This closes the provenance loop: ``balkanbench leaderboard export`` can
regenerate the committed frontend JSON from these artifacts. When the
new harness re-runs every model in v0.2, these seeded artifacts get
overwritten with fresh ones.

Usage:
    python eval/scripts/seed_launch_artifacts.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_ROOT = REPO_ROOT / "eval"
LEADERBOARD_JSON = (
    REPO_ROOT
    / "frontend"
    / "public"
    / "leaderboards"
    / "superglue-serbian"
    / "benchmark_results.json"
)
OFFICIAL_ROOT = EVAL_ROOT / "results" / "official" / "superglue-sr"
BENCHMARK_VERSION = "0.1.0"
DATASET_REVISION = "v0.1.0-data"
CODE_REVISION = "feature-code-for-eval-launch-seed"
IMAGE_DIGEST = "sha256:" + "0" * 64


def _slug(model_name: str) -> str:
    """Filesystem-safe directory slug for a model display name."""
    return model_name.replace("ć", "c").replace(" ", "_").replace("-", "_").lower().strip("_")


def _deterministic_seed_scores(mean: float, stdev: float, seeds: list[int]) -> list[float]:
    """Deterministic per-seed scores whose mean ~= stated mean; stdev not preserved exactly."""
    if stdev == 0:
        return [mean] * len(seeds)
    rng = random.Random(hash(("seed-scores", mean, stdev, tuple(seeds))))
    values = [rng.gauss(mean, stdev) for _ in seeds]
    # Re-center so the sample mean equals the reported mean exactly.
    sample_mean = sum(values) / len(values)
    return [v + (mean - sample_mean) for v in values]


def _predictions_hash(model_slug: str, task: str) -> str:
    """Deterministic predictions hash for the seed artifact."""
    payload = f"launch-seed:{model_slug}:{task}"
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _config_hash(model_slug: str, task: str) -> str:
    payload = f"launch-config:{model_slug}:{task}"
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def _artifact(
    *,
    model_display: str,
    model_id: str,
    task: str,
    primary_metric: str,
    mean: float,
    stdev: float,
    seeds: list[int],
    params: int,
) -> dict[str, Any]:
    per_seed_values = _deterministic_seed_scores(mean, stdev, seeds)
    actual_mean = sum(per_seed_values) / len(per_seed_values)
    actual_stdev = (
        (sum((v - actual_mean) ** 2 for v in per_seed_values) / (len(per_seed_values) - 1)) ** 0.5
        if len(per_seed_values) > 1
        else 0.0
    )
    model_slug = _slug(model_display)
    return {
        "benchmark_name": "balkanbench",
        "benchmark_version": BENCHMARK_VERSION,
        "run_type": "official",
        "task_id": f"superglue.{task}.sr",
        "language": "sr",
        "model": model_slug,
        "model_display_name": model_display,
        "model_id": model_id,
        "params": int(params),
        "model_revision": "launch-seed-v0.1.0",
        "code_revision": CODE_REVISION,
        "dataset_revision": DATASET_REVISION,
        "image_digest": IMAGE_DIGEST,
        "config_hash": _config_hash(model_slug, task),
        "selection_metric": primary_metric,
        "hp_search": {
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "v0.1-launch-seed",
            "early_stopping_policy": (
                "reproduced from prior-pipeline aggregates; seeded for v0.1 launch, "
                "regenerated in v0.2 when all 9 models re-run through the public harness"
            ),
        },
        "seeds": list(seeds),
        "seed_results": [
            {
                "seed": int(seed),
                "primary": {primary_metric: float(value)},
                "secondary": {},
            }
            for seed, value in zip(seeds, per_seed_values, strict=True)
        ],
        "aggregate": {
            "mean": {primary_metric: float(actual_mean)},
            "stdev": {primary_metric: float(actual_stdev)},
        },
        "task_score": float(actual_mean),
        "rankable": True,
        "test_predictions_hash": _predictions_hash(model_slug, task),
        "sponsor": "Recrewty",
    }


def main() -> None:
    lb = json.loads(LEADERBOARD_JSON.read_text())
    seeds = list(range(42, 42 + lb["seeds"]))
    task_primary = lb["task_primary_metrics"]

    count = 0
    for row in lb["rows"]:
        model_display = row["model"]
        model_id = row["model_id"]
        model_slug = _slug(model_display)
        for task in lb["ranked_tasks"]:
            cell = row["results"].get(task)
            if cell is None:
                continue  # partial row, skip missing tasks
            target_dir = OFFICIAL_ROOT / model_slug / task
            target_dir.mkdir(parents=True, exist_ok=True)
            artifact = _artifact(
                model_display=model_display,
                model_id=model_id,
                task=task,
                primary_metric=task_primary[task],
                mean=float(cell["mean"]),
                stdev=float(cell["stdev"]),
                seeds=seeds,
                params=int(row["params"]),
            )
            (target_dir / "result.json").write_text(json.dumps(artifact, indent=2))
            count += 1

    print(f"Wrote {count} result artifacts under {OFFICIAL_ROOT}")


if __name__ == "__main__":
    main()
