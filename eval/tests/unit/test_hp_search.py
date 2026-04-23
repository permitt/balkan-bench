"""Tests for the Optuna HP search driver."""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from datasets import Dataset, DatasetDict

from balkanbench.evaluation import SeedResult
from balkanbench.hp_search import (
    HPSearchError,
    default_search_space_for,
    run_hp_search,
)


def _boolq_cfg() -> dict:
    return {
        "benchmark": "superglue",
        "task": "boolq",
        "task_type": "binary_classification",
        "status": "ranked",
        "languages": {"available": ["sr"], "ranked": ["sr"]},
        "dataset": {
            "public_repo": "permitt/superglue-serbian",
            "config": "boolq",
            "splits": {
                "public": ["train", "validation"],
                "labeled_public": ["train", "validation"],
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
            "num_epochs": 3,
            "metric_for_best_model": "accuracy",
        },
    }


def _bertic_cfg() -> dict:
    return {
        "name": "bertic",
        "hf_repo": "classla/bcms-bertic",
        "family": "electra",
        "params_hint": "110M",
        "tier": "experimental",
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3,
            "fp16": False,
        },
    }


def _fake_datasets() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {
                    "example_id": ["t0", "t1"],
                    "question": ["a", "b"],
                    "passage": ["x", "y"],
                    "label": [0, 1],
                }
            ),
            "validation": Dataset.from_dict(
                {
                    "example_id": ["v0", "v1"],
                    "question": ["a", "b"],
                    "passage": ["x", "y"],
                    "label": [1, 0],
                }
            ),
        }
    )


def test_default_search_space_known_families() -> None:
    space = default_search_space_for("binary_classification")
    assert "learning_rate" in space
    assert "num_epochs" in space
    space_mc = default_search_space_for("multiple_choice")
    assert "learning_rate" in space_mc


def test_default_search_space_unknown_family_raises() -> None:
    with pytest.raises(HPSearchError):
        default_search_space_for("weird_new_type")


def _effective_lr(kwargs: Any) -> float:
    """Read learning_rate as it would be after HFEncoder._merge_training_args."""
    training = dict(kwargs["model_cfg"]["training"])
    key = f"{kwargs['task_cfg']['benchmark']}.{kwargs['task_cfg']['task']}"
    training.update(kwargs["model_cfg"].get("task_overrides", {}).get(key, {}))
    return training["learning_rate"]


def test_run_hp_search_returns_best_trial(monkeypatch, tmp_path) -> None:
    # Each trial picks a deterministic accuracy from the suggested learning_rate.
    # Read the effective LR through task_overrides (that is where run_hp_search
    # writes its per-trial params now).
    def fake_run_single_seed(**kwargs: Any) -> SeedResult:
        lr = _effective_lr(kwargs)
        acc = 0.5 + min(0.49, lr * 1000.0)
        return SeedResult(
            seed=kwargs["seed"],
            primary={"accuracy": acc},
            secondary={},
            task_score=acc,
            predictions=[0, 1],
            references=[0, 1],
            group_ids=None,
        )

    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", fake_run_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=4,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
    )

    assert result.best_trial_number >= 0
    assert 0.5 <= result.best_value <= 1.0
    # The winning config stores search results under task_overrides[superglue.boolq].
    overrides = result.best_model_cfg["task_overrides"]["superglue.boolq"]
    assert overrides["learning_rate"] > 0
    # The written config file exists and round-trips through YAML.
    written = yaml.safe_load(result.best_config_path.read_text())
    assert written["task_overrides"]["superglue.boolq"]["learning_rate"] == pytest.approx(
        overrides["learning_rate"]
    )
    # Provenance header on the file.
    header = result.best_config_path.read_text().splitlines()[0]
    assert header.startswith("#")
    assert "sweep_id" in result.best_config_path.read_text()


def test_run_hp_search_uses_task_score_as_objective(monkeypatch, tmp_path) -> None:
    objectives: list[float] = []

    def fake_run_single_seed(**kwargs: Any) -> SeedResult:
        lr = _effective_lr(kwargs)
        acc = 0.5 + min(0.49, lr * 1000.0)
        objectives.append(acc)
        return SeedResult(
            seed=kwargs["seed"],
            primary={"accuracy": acc},
            secondary={},
            task_score=acc,
            predictions=[0, 1],
            references=[0, 1],
            group_ids=None,
        )

    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", fake_run_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=3,
        sampler_seed=7,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
    )
    assert result.best_value == max(objectives)
