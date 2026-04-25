"""Smoke tests for the `balkanbench run` orchestrator."""

from __future__ import annotations

import json
from typing import Any

from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from balkanbench.cli.main import app
from balkanbench.evaluation import SeedResult
from balkanbench.hp_search import HPSearchResult

runner = CliRunner()


def _fake_datasets() -> DatasetDict:
    """A bare DatasetDict that satisfies run_multiseed's signature."""
    rows = {
        "example_id": ["t0", "t1"],
        "question": ["q", "q"],
        "passage": ["p", "p"],
        "label": [0, 1],
    }
    return DatasetDict(
        {
            "train": Dataset.from_dict(rows),
            "validation": Dataset.from_dict(rows),
            "test": Dataset.from_dict(rows),
        }
    )


def test_run_orchestrates_hp_search_then_eval_then_export(tmp_path, monkeypatch) -> None:
    """End-to-end: 5 ranked HR tasks each get HP search + eval + a leaderboard export."""
    captured_calls: list[tuple[str, str]] = []

    def fake_run_hp_search(*, task_cfg: dict[str, Any], **_: Any) -> HPSearchResult:
        captured_calls.append(("hp_search", task_cfg["task"]))
        return HPSearchResult(
            best_trial_number=3,
            best_value=0.81,
            best_model_cfg={
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
                "task_overrides": {
                    f"superglue.{task_cfg['task']}": {"learning_rate": 3e-5, "num_epochs": 5}
                },
            },
            best_config_path=tmp_path / "_unused" / "bertic_best.yaml",
            sweep_id="sweep-fake",
        )

    def fake_run_multiseed(
        *, task_cfg: dict[str, Any], seeds: list[int], **_: Any
    ) -> list[SeedResult]:
        captured_calls.append(("eval", task_cfg["task"]))
        primary_metric = task_cfg["metrics"]["task_score"]
        return [
            SeedResult(
                seed=s,
                primary={primary_metric: 0.80 + 0.001 * s},
                secondary={},
                task_score=0.80 + 0.001 * s,
                predictions=[0, 1],
                references=[0, 1],
                group_ids=None,
            )
            for s in seeds
        ]

    monkeypatch.setattr("balkanbench.cli.run.run_hp_search", fake_run_hp_search)
    monkeypatch.setattr("balkanbench.cli.run.run_multiseed", fake_run_multiseed)
    monkeypatch.setattr(
        "balkanbench.cli.run.load_dataset",
        lambda repo, cfg, **_: _fake_datasets(),
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    out = tmp_path / "runs"
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--language",
            "hr",
            "--n-trials",
            "2",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output

    # Croatian has 5 ranked tasks: boolq, cb, copa, multirc, rte (no wsc).
    expected_tasks = ["boolq", "cb", "copa", "multirc", "rte"]
    assert [t for kind, t in captured_calls if kind == "hp_search"] == expected_tasks
    assert [t for kind, t in captured_calls if kind == "eval"] == expected_tasks

    for task in expected_tasks:
        artifact = out / "results" / "superglue-hr" / "bertic" / task / "result.json"
        assert artifact.is_file(), f"missing artifact for {task}"
        data = json.loads(artifact.read_text())
        # HP-search winner promoted into model_cfg.task_overrides[{benchmark}.{task}]
        # is what the eval uses (verified indirectly via the artifact's hp_search field).
        assert data["hp_search"]["sweep_id"] == "sweep-fake"
        assert data["hp_search"]["num_trials"] == 2

    export = out / "benchmark_results.json"
    assert export.is_file()
    export_data = json.loads(export.read_text())
    assert export_data["language"] == "hr"
    assert sorted(export_data["ranked_tasks"]) == expected_tasks


def test_run_resumes_by_skipping_existing_results(tmp_path, monkeypatch) -> None:
    """If result.json already exists for a task, skip both HP search and eval."""
    hp_calls: list[str] = []
    eval_calls: list[str] = []

    def fake_run_hp_search(*, task_cfg: dict[str, Any], **_: Any) -> HPSearchResult:
        hp_calls.append(task_cfg["task"])
        return HPSearchResult(
            best_trial_number=0,
            best_value=0.5,
            best_model_cfg={
                "name": "bertic",
                "hf_repo": "classla/bcms-bertic",
                "family": "electra",
                "params_hint": "110M",
                "tier": "official",
                "training": {"learning_rate": 2e-5, "batch_size": 16, "num_epochs": 1, "fp16": False},
            },
            best_config_path=tmp_path / "_x" / "bertic_best.yaml",
            sweep_id="sweep-fake",
        )

    def fake_run_multiseed(*, task_cfg: dict[str, Any], seeds: list[int], **_: Any) -> list[SeedResult]:
        eval_calls.append(task_cfg["task"])
        return [
            SeedResult(
                seed=s,
                primary={task_cfg["metrics"]["task_score"]: 0.7},
                secondary={},
                task_score=0.7,
                predictions=[0],
                references=[0],
                group_ids=None,
            )
            for s in seeds
        ]

    monkeypatch.setattr("balkanbench.cli.run.run_hp_search", fake_run_hp_search)
    monkeypatch.setattr("balkanbench.cli.run.run_multiseed", fake_run_multiseed)
    monkeypatch.setattr(
        "balkanbench.cli.run.load_dataset",
        lambda repo, cfg, **_: _fake_datasets(),
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    out = tmp_path / "runs"
    # Pre-seed boolq so the resume path skips it.
    seeded = out / "results" / "superglue-hr" / "bertic" / "boolq" / "result.json"
    seeded.parent.mkdir(parents=True)
    seeded.write_text("{}")

    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "bertic",
            "--benchmark",
            "superglue",
            "--language",
            "hr",
            "--tasks",
            "boolq",
            "--tasks",
            "cb",
            "--n-trials",
            "1",
            "--out",
            str(out),
        ],
    )
    # cb has no validation split etc on the leaderboard side, so allow either pass
    # or a leaderboard-stage error. The thing we care about is the resume skip.
    assert "skip boolq" in result.output
    assert hp_calls == ["cb"]
    assert eval_calls == ["cb"]
