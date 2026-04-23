"""Regression tests for the HP search override layer + provenance fields.

Two review findings:

1. HP search results must land in ``model_cfg.task_overrides["{benchmark}.{task}"]``,
   not at top-level ``model_cfg.training``. The runtime merge order in
   ``HFEncoder._merge_training_args`` applies task_overrides last, so a
   search that only mutates ``training`` becomes a no-op for tasks that
   already carry task_overrides (BERTic's real config has them for cb,
   wsc, copa).

2. Sweep outputs must record the dataset revision, search_space_id, and
   early_stopping_policy so the benchmark-contract provenance promise
   survives the promote-to-official step.
"""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, DatasetDict

from balkanbench.evaluation import SeedResult
from balkanbench.hp_search import run_hp_search


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
            "early_stopping_patience": 5,
            "metric_for_best_model": "accuracy",
        },
    }


def _bertic_cfg_with_cb_override() -> dict:
    return {
        "name": "bertic",
        "hf_repo": "classla/bcms-bertic",
        "family": "electra",
        "params_hint": "110M",
        "tier": "official",
        "training": {
            "learning_rate": 2e-5,
            "batch_size": 16,
            "num_epochs": 3,
            "fp16": False,
        },
        # Pre-existing task overrides (mirrors real bertic.yaml)
        "task_overrides": {
            "superglue.cb": {"num_epochs": 30},
            "superglue.wsc": {"learning_rate": 1.0e-5, "num_epochs": 30},
        },
    }


def _fake_datasets() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_dict(
                {"example_id": ["t0"], "question": ["a"], "passage": ["x"], "label": [0]}
            ),
            "validation": Dataset.from_dict(
                {"example_id": ["v0"], "question": ["a"], "passage": ["x"], "label": [1]}
            ),
        }
    )


def _fake_single_seed(**kwargs: Any) -> SeedResult:
    lr = kwargs["model_cfg"]["training"].get("learning_rate", 0)
    # Simulate effective training args after HFEncoder merge: task_overrides wins.
    overrides = kwargs["model_cfg"].get("task_overrides", {})
    key = f"{kwargs['task_cfg']['benchmark']}.{kwargs['task_cfg']['task']}"
    merged = dict(kwargs["model_cfg"]["training"])
    merged.update(overrides.get(key, {}))
    effective_lr = merged["learning_rate"]
    return SeedResult(
        seed=kwargs["seed"],
        primary={"accuracy": 0.5 + min(0.49, effective_lr * 1000.0)},
        secondary={},
        task_score=0.5 + min(0.49, effective_lr * 1000.0),
        predictions=[0],
        references=[0],
        group_ids=None,
    )


# ---------- Finding 1: override layer ----------


def test_hp_search_writes_result_into_task_overrides_not_training(monkeypatch, tmp_path) -> None:
    """Winning params must land in task_overrides[{benchmark}.{task}]."""
    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg_with_cb_override(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=3,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
        search_space_id="encoder-small-v1",
    )

    cfg = result.best_model_cfg
    # The searched-for task (boolq) should have a fresh override block.
    assert "superglue.boolq" in cfg["task_overrides"]
    override = cfg["task_overrides"]["superglue.boolq"]
    assert "learning_rate" in override or "num_epochs" in override

    # Global training must be unchanged from the input (not mutated).
    base = _bertic_cfg_with_cb_override()
    assert cfg["training"]["learning_rate"] == base["training"]["learning_rate"]
    assert cfg["training"]["num_epochs"] == base["training"]["num_epochs"]


def test_hp_search_preserves_unrelated_task_overrides(monkeypatch, tmp_path) -> None:
    """The CB + WSC overrides from the real config must survive a boolq sweep."""
    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg_with_cb_override(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=2,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
    )
    overrides = result.best_model_cfg["task_overrides"]
    assert overrides["superglue.cb"] == {"num_epochs": 30}
    assert overrides["superglue.wsc"] == {"learning_rate": 1.0e-5, "num_epochs": 30}


def test_hp_search_merges_into_existing_task_override_block(monkeypatch, tmp_path) -> None:
    """When an override block already exists for the searched task, merge into it."""
    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)

    # CB has existing num_epochs=30. Search tunes learning_rate; num_epochs=30 must survive.
    cb_cfg = _boolq_cfg()
    cb_cfg["task"] = "cb"
    cb_cfg["task_type"] = "multiclass_classification"
    cb_cfg["num_labels"] = 3
    cb_cfg["metrics"] = {
        "primary": ["f1_macro"],
        "report": ["f1_macro"],
        "task_score": "f1_macro",
    }

    result = run_hp_search(
        task_cfg=cb_cfg,
        model_cfg=_bertic_cfg_with_cb_override(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=2,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
        # Only tune learning_rate; num_epochs=30 must survive from existing override.
        search_space={"learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-5}},
    )
    override = result.best_model_cfg["task_overrides"]["superglue.cb"]
    assert override["num_epochs"] == 30, "existing num_epochs override must survive the search"
    assert "learning_rate" in override


# ---------- Finding 2: provenance ----------


def test_hp_search_records_dataset_revision_search_space_id_early_stopping(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg_with_cb_override(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=2,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
        search_space_id="encoder-small-v1",
    )

    config_text = result.best_config_path.read_text()
    assert "dataset_revision: v0.1.0-data" in config_text
    assert "search_space_id: encoder-small-v1" in config_text
    assert "early_stopping_policy:" in config_text
    assert "patience=5" in config_text  # from task_cfg.training.early_stopping_patience
    assert "accuracy" in config_text  # metric_for_best_model

    summary_path = result.best_config_path.parent / "sweep_summary.json"
    summary = json.loads(summary_path.read_text())
    assert summary["dataset_revision"] == "v0.1.0-data"
    assert summary["search_space_id"] == "encoder-small-v1"
    assert "patience=5" in summary["early_stopping_policy"]


def test_hp_search_cli_threads_dataset_revision_into_provenance(monkeypatch, tmp_path) -> None:
    """End-to-end regression: the CLI's --dataset-revision must reach the sweep output."""
    from typer.testing import CliRunner

    from balkanbench.cli.main import app

    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)
    monkeypatch.setattr(
        "balkanbench.cli.hp_search.load_dataset",
        lambda repo, config, **_: _fake_datasets(),
    )

    # Use the real fixtures that validate against the schemas
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "hp-search",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--task",
            "boolq",
            "--language",
            "sr",
            "--n-trials",
            "2",
            "--sampler-seed",
            "7",
            "--out",
            str(tmp_path),
            "--dataset-revision",
            "v0.1.2-data",
        ],
    )
    assert result.exit_code == 0, result.output

    # Find the sweep_summary.json that was written (one sweep dir under tmp)
    sweep_dirs = [p for p in tmp_path.iterdir() if p.is_dir() and p.name.startswith("sweep-")]
    assert len(sweep_dirs) == 1
    summary = json.loads((sweep_dirs[0] / "sweep_summary.json").read_text())
    assert summary["dataset_revision"] == "v0.1.2-data"


def test_hp_search_default_search_space_id_is_informative(monkeypatch, tmp_path) -> None:
    """If the caller does not supply --search-space-id, the default reflects the task_type."""
    monkeypatch.setattr("balkanbench.hp_search.run_single_seed", _fake_single_seed)

    result = run_hp_search(
        task_cfg=_boolq_cfg(),
        model_cfg=_bertic_cfg_with_cb_override(),
        language="sr",
        datasets=_fake_datasets(),
        n_trials=2,
        sampler_seed=42,
        out_dir=tmp_path,
        dataset_revision="v0.1.0-data",
    )
    summary = json.loads((result.best_config_path.parent / "sweep_summary.json").read_text())
    # Default format: "default-{task_type}-v1" or similar, not a bare empty string.
    assert summary["search_space_id"]
    assert (
        "binary_classification" in summary["search_space_id"]
        or "default" in summary["search_space_id"]
    )
