"""End-to-end CPU smoke test: `balkanbench eval` dispatches SLE (generative)
tasks through the loglikelihood evaluator and writes a valid result artifact.

Everything downstream of the dataset-loading seam is real: a real tiny
open-weights model (``sshleifer/tiny-gpt2``) is downloaded and run on CPU,
the real :func:`~balkanbench.evaluation.generative.run_generative_eval` and
:func:`~balkanbench.scoring.artifact.write_generative_result_artifact`
compute and validate the artifact. Only the HF Hub dataset fetch
(``balkanbench.cli.eval.load_dataset``) is monkeypatched, mirroring the seam
the existing encoder-path CLI tests already use (see
``tests/unit/test_cli_eval.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from datasets import Dataset
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"

_TASK_YAML = """\
benchmark: sle
task: arc_easy
status: ranked
task_type: multiple_choice_loglikelihood
languages:
  available: [sr]
  ranked: [sr]
dataset:
  source_type: huggingface
  config: arc_easy
  per_language:
    sr: { public_repo: permitt/serbian-llm-eval }
  splits:
    public: [test]
    labeled_public: [test]
inputs:
  fields: [query, choices]
  id_field: example_id
metrics:
  primary: [acc_norm]
  report: [acc, acc_norm]
  task_score: acc_norm
prompts:
  sr: { template_id: sle_mc_v1 }
evaluation:
  num_fewshot: 0
  stop_sequences: []
  max_gen_tokens: 16
api_protocol:
  reformulation: multiple_choice_generative
  metrics:
    primary: [acc]
    report: [acc]
    task_score: acc
"""

_MODEL_YAML = """\
name: tiny_gpt2
hf_repo: sshleifer/tiny-gpt2
family: gpt2
params_hint: 5M
tier: experimental
access: open_weights
generation:
  dtype: float32
"""


def _fake_arc_easy_dataset() -> Dataset:
    return Dataset.from_dict(
        {
            "example_id": [f"e{i}" for i in range(5)],
            "query": [f"Pitanje broj {i}?" for i in range(5)],
            "choices": [["dobar", "loš", "srednji", "nepoznat"] for _ in range(5)],
            "gold": [0, 1, 2, 3, 0],
        }
    )


def _write_configs(configs_dir: Path) -> None:
    task_path = configs_dir / "benchmarks" / "sle" / "tasks" / "arc_easy.yaml"
    task_path.parent.mkdir(parents=True, exist_ok=True)
    task_path.write_text(_TASK_YAML)

    model_path = configs_dir / "models" / "experimental" / "tiny_gpt2.yaml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(_MODEL_YAML)


@pytest.fixture
def sle_configs(tmp_path, monkeypatch) -> Path:
    configs_dir = tmp_path / "configs"
    _write_configs(configs_dir)
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(configs_dir))
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    return configs_dir


def test_eval_dispatches_sle_task_end_to_end(sle_configs, tmp_path, monkeypatch) -> None:
    captured: dict[str, Any] = {}

    def fake_load_dataset(repo: str, config: str, **kwargs: Any) -> Dataset:
        captured["repo"] = repo
        captured["config"] = config
        captured["split"] = kwargs.get("split")
        return _fake_arc_easy_dataset()

    monkeypatch.setattr("balkanbench.cli.eval.load_dataset", fake_load_dataset)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "eval",
            "--model",
            "tiny_gpt2",
            "--benchmark",
            "sle",
            "--task",
            "arc_easy",
            "--language",
            "sr",
            "--limit",
            "5",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["repo"] == "permitt/serbian-llm-eval"
    assert captured["config"] == "arc_easy"
    assert captured["split"] == "test"

    artifact_path = out_dir / "sle-sr" / "tiny_gpt2" / "arc_easy" / "result.json"
    assert artifact_path.is_file()

    artifact = json.loads(artifact_path.read_text())
    schema = json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(artifact))
    assert not errors, [e.message for e in errors]

    assert artifact["run_config"]["protocol"] == "loglikelihood"
    assert artifact["run_type"] == "experimental"
    assert artifact["rankable"] is False
    assert artifact["task_id"] == "sle.arc_easy.sr"


def test_predict_rejects_generative_task(sle_configs, tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "predict",
            "--model",
            "tiny_gpt2",
            "--benchmark",
            "sle",
            "--task",
            "arc_easy",
            "--language",
            "sr",
            "--out",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code != 0
    assert "predict is not supported for generative tasks" in result.output
    assert "use eval" in result.output
