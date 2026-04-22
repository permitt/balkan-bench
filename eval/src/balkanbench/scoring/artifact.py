"""Write per-run result artifacts matching ``schemas/result_artifact.json``.

The artifact is the canonical record of a single (benchmark, task, language,
model) evaluation: all seeds, all primary + secondary metrics, aggregate
mean/stdev, and the provenance needed to reproduce it.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from balkanbench.evaluation import Aggregate, SeedResult


def compute_predictions_hash(
    predictions: list[Any],
    references: list[Any] | None = None,
) -> str:
    """Deterministic ``sha256:<hex>`` hash of a canonical predictions JSONL.

    Predictions are always hashed as-is. References are included when
    available so someone can confirm the label mapping is what they expected.
    """
    payload = {"predictions": list(predictions)}
    if references is not None:
        payload["references"] = list(references)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """Deterministic hash of a config dict, independent of key order."""
    canonical = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "sha256:" + hashlib.sha256(canonical).hexdigest()


def write_result_artifact(
    *,
    task_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    language: str,
    seed_results: list[SeedResult],
    aggregate: Aggregate,
    provenance: dict[str, Any],
    dataset_revision: str,
    benchmark_version: str,
    hp_search: dict[str, Any],
    out_dir: Path,
    run_type: str = "official",
    sponsor: str = "Recrewty",
) -> Path:
    """Assemble + validate + write a ``result.json`` artifact.

    Returns the absolute path of the written file. Raises ``ValueError`` on
    empty ``seed_results`` or when the assembled artifact fails schema
    validation (so upstream drift is caught before the artifact lands on disk).
    """
    if not seed_results:
        raise ValueError("write_result_artifact requires at least one SeedResult")

    benchmark = task_cfg["benchmark"]
    task = task_cfg["task"]
    model = model_cfg["name"]
    model_id = model_cfg["hf_repo"]
    task_id = f"{benchmark}.{task}.{language}"

    predictions_hash = compute_predictions_hash(
        seed_results[-1].predictions, seed_results[-1].references
    )
    config_hash = compute_config_hash(
        {"task": task_cfg, "model": model_cfg, "dataset_revision": dataset_revision}
    )

    selection_metric = task_cfg["metrics"].get("task_score", "accuracy")

    artifact: dict[str, Any] = {
        "benchmark_name": "balkanbench",
        "benchmark_version": benchmark_version,
        "run_type": run_type,
        "task_id": task_id,
        "language": language,
        "model": model,
        "model_id": model_id,
        "model_revision": model_cfg.get("hf_revision", "unknown"),
        "code_revision": provenance["code_revision"],
        "dataset_revision": dataset_revision,
        "image_digest": provenance["image_digest"],
        "config_hash": config_hash,
        "selection_metric": selection_metric,
        "hp_search": dict(hp_search),
        "seeds": [r.seed for r in seed_results],
        "seed_results": [
            {
                "seed": r.seed,
                "primary": dict(r.primary),
                "secondary": dict(r.secondary),
            }
            for r in seed_results
        ],
        "aggregate": {
            "mean": dict(aggregate.mean),
            "stdev": dict(aggregate.stdev),
        },
        "task_score": float(aggregate.mean.get(selection_metric, seed_results[-1].task_score)),
        "rankable": run_type == "official",
        "test_predictions_hash": predictions_hash,
        "sponsor": sponsor,
    }

    _validate_against_schema(artifact)

    target_dir = Path(out_dir) / f"{benchmark}-{language}" / model
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path: Path = target_dir / "result.json"
    target_path.write_text(json.dumps(artifact, indent=2))
    return target_path


def _validate_against_schema(artifact: dict[str, Any]) -> None:
    schema_path = Path(__file__).resolve().parents[3] / "schemas" / "result_artifact.json"
    schema = json.loads(schema_path.read_text())
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(artifact), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}" for err in errors
        ]
        raise ValueError("result artifact failed schema:\n" + "\n".join(messages))
