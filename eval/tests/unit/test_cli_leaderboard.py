"""Smoke test for `balkanbench leaderboard export`."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


_ALL_TASKS = ["boolq", "cb", "copa", "rte", "multirc", "wsc"]
_PRIMARY = {
    "boolq": "accuracy",
    "cb": "f1_macro",
    "copa": "accuracy",
    "rte": "accuracy",
    "multirc": "f1_a",
    "wsc": "accuracy",
}


def _artifact(task: str, model: str, value: float) -> dict:
    metric = _PRIMARY[task]
    return {
        "benchmark_name": "balkanbench",
        "benchmark_version": "0.1.0",
        "run_type": "official",
        "task_id": f"superglue.{task}.sr",
        "language": "sr",
        "model": model,
        "model_id": f"hf/{model}",
        "model_revision": "a" * 40,
        "code_revision": "b" * 40,
        "dataset_revision": "v0.1.0-data",
        "image_digest": "sha256:" + "0" * 64,
        "config_hash": "sha256:" + "1" * 64,
        "selection_metric": metric,
        "hp_search": {
            "tool": "optuna",
            "sampler": "TPESampler",
            "sampler_seed": 42,
            "num_trials": 0,
            "search_space_id": "none",
        },
        "seeds": [42],
        "seed_results": [{"seed": 42, "primary": {metric: value}, "secondary": {}}],
        "aggregate": {"mean": {metric: value}, "stdev": {metric: 0.01}},
        "task_score": value,
        "rankable": True,
        "test_predictions_hash": "sha256:" + "2" * 64,
        "sponsor": "Recrewty",
    }


def _seed(tmp_path: Path) -> Path:
    """Scaffolds ``tmp/official/superglue-sr/{model}/{task}/result.json`` and
    returns the ``tmp/official/`` root (what the CLI's ``--results-dir`` expects)."""
    results_dir = tmp_path / "official"
    root = results_dir / "superglue-sr"
    for model in ("bertic", "xlmr"):
        for t in _ALL_TASKS:
            d = root / model / t
            d.mkdir(parents=True, exist_ok=True)
            val = 0.80 if model == "bertic" else 0.70
            (d / "result.json").write_text(json.dumps(_artifact(t, model, val)))
    return results_dir


def test_leaderboard_export_cli_writes_valid_json(tmp_path) -> None:
    root = _seed(tmp_path)
    out_path = tmp_path / "frontend_public" / "benchmark_results.json"
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "export",
            "--benchmark",
            "superglue",
            "--language",
            "sr",
            "--results-dir",
            str(root),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert out_path.is_file()
    data = json.loads(out_path.read_text())
    assert data["benchmark"] == "superglue"
    assert data["language"] == "sr"
    assert len(data["rows"]) == 2
    # BERTić should rank 1 (higher avg)
    ranks = {r["model"]: r["rank"] for r in data["rows"]}
    assert ranks["bertic"] == 1
    assert ranks["xlmr"] == 2


def test_leaderboard_export_cli_errors_on_empty_results(tmp_path) -> None:
    missing = tmp_path / "does-not-exist"
    out_path = tmp_path / "out.json"
    result = runner.invoke(
        app,
        [
            "leaderboard",
            "export",
            "--benchmark",
            "superglue",
            "--language",
            "sr",
            "--results-dir",
            str(missing),
            "--out",
            str(out_path),
        ],
    )
    assert result.exit_code == 1
