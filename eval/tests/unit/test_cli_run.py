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
                "training": {
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "num_epochs": 1,
                    "fp16": False,
                },
            },
            best_config_path=tmp_path / "_x" / "bertic_best.yaml",
            sweep_id="sweep-fake",
        )

    def fake_run_multiseed(
        *, task_cfg: dict[str, Any], seeds: list[int], **_: Any
    ) -> list[SeedResult]:
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
    assert result.exit_code == 0, result.output
    assert "skip boolq" in result.output
    assert hp_calls == ["cb"]
    assert eval_calls == ["cb"]
    # Subset run (boolq+cb out of 5 ranked HR tasks) must NOT emit a leaderboard;
    # otherwise stale or partial rows would leak into a run that wasn't asked
    # to compute the full set.
    assert not (out / "benchmark_results.json").exists()
    assert "subset run" in result.output


def test_run_aborts_on_fingerprint_mismatch(tmp_path, monkeypatch) -> None:
    """A second run with different invocation params into the same --out must abort."""
    monkeypatch.setattr("balkanbench.cli.run.run_hp_search", lambda **_: None)
    monkeypatch.setattr("balkanbench.cli.run.run_multiseed", lambda **_: [])
    monkeypatch.setattr(
        "balkanbench.cli.run.load_dataset",
        lambda repo, cfg, **_: _fake_datasets(),
    )
    monkeypatch.setenv("HF_TOKEN", "fake-token")

    out = tmp_path / "runs"
    # Manually plant a fingerprint matching a prior --seeds 42 invocation.
    out.mkdir(parents=True)
    (out / ".run_fingerprint.json").write_text(
        '{"hash": "stale", "fields": {"seeds": [42]}}'
    )

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
            "--seeds",
            "43",
            "--n-trials",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 1
    assert "different settings" in result.output
    assert "seeds:" in result.output


def test_run_hp_cache_rejects_when_settings_differ(tmp_path, monkeypatch) -> None:
    """The HP cache must not reuse a winner from a sweep with different settings."""
    hp_calls: list[dict[str, Any]] = []

    def fake_run_hp_search(**kwargs: Any) -> HPSearchResult:
        hp_calls.append(
            {k: kwargs[k] for k in ("n_trials", "sampler_seed", "seed_for_trials")}
        )
        return HPSearchResult(
            best_trial_number=0,
            best_value=0.9,
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
            },
            best_config_path=tmp_path / "_x.yaml",
            sweep_id="sweep-new",
        )

    def fake_run_multiseed(**kwargs: Any) -> list[SeedResult]:
        seeds = kwargs["seeds"]
        primary = kwargs["task_cfg"]["metrics"]["task_score"]
        return [
            SeedResult(
                seed=s,
                primary={primary: 0.7},
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
    # Pre-plant a sweep_summary.json from a sweep with different settings
    # (n_trials=99, sampler_seed=999) and a fingerprint that matches the
    # *current* invocation - so the only thing under test is HP-cache logic.
    sweep_dir = out / "sweeps" / "boolq" / "sweep-stale"
    sweep_dir.mkdir(parents=True)
    (sweep_dir / "sweep_summary.json").write_text(
        json.dumps(
            {
                "sweep_id": "sweep-stale",
                "dataset_revision": "v0.1.0-data",
                "search_space_id": "default-binary_classification-v1",
                "early_stopping_policy": "patience=5 on accuracy",
                "best_trial_number": 7,
                "best_value": 0.42,
                "sampler_seed": 999,
                "seed_for_trials": 999,
                "n_trials": 99,
                "benchmark": "superglue",
                "task": "boolq",
                "task_score_metric": "accuracy",
                "best_model_cfg": {"name": "bertic", "tag": "stale"},
            }
        )
    )

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
            "--n-trials",
            "1",
            "--sampler-seed",
            "42",
            "--seed-for-trials",
            "42",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    # HP search must have been invoked with the CURRENT settings, ignoring the cache.
    assert hp_calls == [{"n_trials": 1, "sampler_seed": 42, "seed_for_trials": 42}]
    artifact_path = (
        out / "results" / "superglue-hr" / "bertic" / "boolq" / "result.json"
    )
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text())
    assert artifact["hp_search"]["sweep_id"] == "sweep-new"
    assert artifact["hp_search"]["num_trials"] == 1


def test_run_uses_public_repo_for_validation_split(tmp_path, monkeypatch) -> None:
    captured_repos: list[str] = []

    def fake_run_hp_search(*, task_cfg: dict[str, Any], **_: Any) -> HPSearchResult:
        return HPSearchResult(
            best_trial_number=0,
            best_value=0.8,
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
            },
            best_config_path=tmp_path / "_unused.yaml",
            sweep_id="sweep-fake",
        )

    def fake_run_multiseed(
        *, seeds: list[int], task_cfg: dict[str, Any], **_: Any
    ) -> list[SeedResult]:
        primary = task_cfg["metrics"]["task_score"]
        return [
            SeedResult(
                seed=s,
                primary={primary: 0.7},
                secondary={},
                task_score=0.7,
                predictions=[0],
                references=[0],
                group_ids=None,
            )
            for s in seeds
        ]

    def fake_load_dataset(repo: str, cfg: str, **_: Any) -> DatasetDict:
        captured_repos.append(repo)
        return _fake_datasets()

    monkeypatch.setattr("balkanbench.cli.run.run_hp_search", fake_run_hp_search)
    monkeypatch.setattr("balkanbench.cli.run.run_multiseed", fake_run_multiseed)
    monkeypatch.setattr("balkanbench.cli.run.load_dataset", fake_load_dataset)
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
            "--tasks",
            "boolq",
            "--eval-split",
            "validation",
            "--n-trials",
            "1",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured_repos == ["permitt/superglue-hr"]
