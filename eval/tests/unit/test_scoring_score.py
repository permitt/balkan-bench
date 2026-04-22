"""Tests for the private-label scorer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from jsonschema import Draft202012Validator

from balkanbench.scoring.score import ScoreError, score_predictions

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _boolq_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "status": "ranked",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "source_type": "huggingface",
            "public_repo": "permitt/superglue-serbian",
            "private_repo": "permitt/superglue-private",
            "config": "boolq",
            "splits": {
                "public": ["train", "validation", "test"],
                "labeled_public": ["train", "validation"],
                "labeled_private": ["test"],
            },
        },
        "inputs": {"fields": ["question", "passage"], "id_field": "example_id"},
        "metrics": {
            "primary": ["accuracy"],
            "report": ["accuracy"],
            "task_score": "accuracy",
        },
        "prompts": {"sr": {"template_id": "boolq_sr_v1"}},
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "metric_for_best_model": "accuracy",
        },
    }


def _bertic_cfg() -> dict:
    return {
        "name": "bertic",
        "hf_repo": "classla/bcms-bertic",
        "family": "electra",
        "params_hint": "110M",
        "tier": "official",
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 1,
            "fp16": False,
        },
    }


def _private_labels() -> Dataset:
    return Dataset.from_dict({"example_id": ["e0", "e1", "e2", "e3"], "label": [1, 0, 1, 1]})


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_score_predictions_writes_schema_valid_artifact(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        "balkanbench.scoring.score.load_dataset", lambda *a, **kw: _private_labels()
    )
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")

    preds = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"example_id": "e0", "prediction": 1},
            {"example_id": "e1", "prediction": 0},
            {"example_id": "e2", "prediction": 0},  # wrong
            {"example_id": "e3", "prediction": 1},
        ],
    )

    artifact_path = score_predictions(
        predictions_path=preds,
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg(),
        language="sr",
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path / "results",
    )
    assert artifact_path.is_file()
    data = json.loads(artifact_path.read_text())
    schema = json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())
    Draft202012Validator(schema).validate(data)

    # Accuracy = 3/4
    assert data["aggregate"]["mean"]["accuracy"] == 0.75
    assert data["rankable"] is True
    assert data["sponsor"] == "Recrewty"


def test_score_predictions_requires_hf_token(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("HF_OFFICIAL_TOKEN", raising=False)
    preds = _write_predictions(
        tmp_path / "predictions.jsonl", [{"example_id": "e0", "prediction": 1}]
    )
    with pytest.raises(ScoreError, match="HF_OFFICIAL_TOKEN"):
        score_predictions(
            predictions_path=preds,
            task_cfg=_boolq_cfg(),
            model_cfg=_bertic_cfg(),
            language="sr",
            dataset_revision="v0.1.0-data",
            benchmark_version="0.1.0",
            out_dir=tmp_path / "results",
        )


def test_score_predictions_rejects_missing_example(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "balkanbench.scoring.score.load_dataset", lambda *a, **kw: _private_labels()
    )
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")

    preds = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"example_id": "e0", "prediction": 1},
            {"example_id": "e1", "prediction": 0},
            {"example_id": "e2", "prediction": 1},
            # missing e3
        ],
    )
    with pytest.raises(ScoreError, match="missing"):
        score_predictions(
            predictions_path=preds,
            task_cfg=_boolq_cfg(),
            model_cfg=_bertic_cfg(),
            language="sr",
            dataset_revision="v0.1.0-data",
            benchmark_version="0.1.0",
            out_dir=tmp_path / "results",
        )


def test_score_predictions_rejects_unknown_example(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        "balkanbench.scoring.score.load_dataset", lambda *a, **kw: _private_labels()
    )
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")

    preds = _write_predictions(
        tmp_path / "predictions.jsonl",
        [
            {"example_id": "e0", "prediction": 1},
            {"example_id": "e1", "prediction": 0},
            {"example_id": "e2", "prediction": 1},
            {"example_id": "e3", "prediction": 1},
            {"example_id": "eX", "prediction": 0},  # not in private labels
        ],
    )
    with pytest.raises(ScoreError, match="unexpected"):
        score_predictions(
            predictions_path=preds,
            task_cfg=_boolq_cfg(),
            model_cfg=_bertic_cfg(),
            language="sr",
            dataset_revision="v0.1.0-data",
            benchmark_version="0.1.0",
            out_dir=tmp_path / "results",
        )


def test_score_predictions_fails_loudly_without_private_repo(monkeypatch, tmp_path) -> None:
    cfg = _boolq_cfg()
    del cfg["dataset"]["private_repo"]

    preds = _write_predictions(
        tmp_path / "predictions.jsonl",
        [{"example_id": "e0", "prediction": 1}],
    )
    monkeypatch.setenv("HF_OFFICIAL_TOKEN", "fake-token")
    with pytest.raises(ScoreError, match="private_repo"):
        score_predictions(
            predictions_path=preds,
            task_cfg=cfg,
            model_cfg=_bertic_cfg(),
            language="sr",
            dataset_revision="v0.1.0-data",
            benchmark_version="0.1.0",
            out_dir=tmp_path / "results",
        )
