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
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset
from jsonschema import Draft202012Validator
from typer.testing import CliRunner

from balkanbench.cli.main import app
from balkanbench.models.api.base import APIResponse
from balkanbench.models.api.providers import AnthropicClient

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

# An api-access model config (Finding 1/2b/2c): make_api_model raises a plain
# RuntimeError/ImportError from providers.py when the SDK or API key is
# missing; the CLI must catch and style that, not let a traceback through.
_API_MODEL_YAML = """\
name: claude_fake
access: api
provider: anthropic
api_model_id: claude-fake-model
family: claude
params_hint: API
generation:
  max_tokens: 16
"""

# nq_open-shaped generative_qa task (Finding 2, fewshot test): num_fewshot=2
# so build_fewshot_prefix draws from the separately-loaded train split.
_NQ_OPEN_TASK_YAML = """\
benchmark: sle
task: nq_open
status: ranked
task_type: generative_qa
languages:
  available: [sr]
  ranked: [sr]
dataset:
  source_type: huggingface
  config: nq_open
  per_language:
    sr: { public_repo: permitt/serbian-llm-eval }
  splits:
    public: [test, train]
    labeled_public: [test, train]
inputs:
  fields: [question]
  id_field: example_id
metrics:
  primary: [em]
  report: [em]
  task_score: em
prompts:
  sr: { template_id: sle_qa_v1 }
evaluation:
  num_fewshot: 2
  fewshot_split: train
  stop_sequences: ["\\n", ".", ","]
  max_gen_tokens: 64
api_protocol:
  reformulation: generative_qa
  metrics:
    primary: [em]
    report: [em]
    task_score: em
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


def _fake_nq_dataset(split_name: str, n: int = 5) -> Dataset:
    return Dataset.from_dict(
        {
            "example_id": [f"{split_name}-{i}" for i in range(n)],
            "question": [f"Pitanje {split_name} {i}?" for i in range(n)],
            "answer": [[f"Odgovor{i}"] for i in range(n)],
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


def test_run_dispatches_sle_task_end_to_end(sle_configs, tmp_path, monkeypatch) -> None:
    """`balkanbench run`'s generative dispatch mirrors the eval smoke test
    above: same seam (this time `balkanbench.cli.run.load_dataset`), same
    experimental/loglikelihood assertions, but through the `run` command's
    results directory layout (`{out}/results/...`).
    """
    captured: dict[str, Any] = {}

    def fake_load_dataset(repo: str, config: str, **kwargs: Any) -> Dataset:
        captured["repo"] = repo
        captured["split"] = kwargs.get("split")
        return _fake_arc_easy_dataset()

    monkeypatch.setattr("balkanbench.cli.run.load_dataset", fake_load_dataset)

    out_dir = tmp_path / "out"
    result = runner.invoke(
        app,
        [
            "run",
            "--model",
            "tiny_gpt2",
            "--benchmark",
            "sle",
            "--language",
            "sr",
            "--tasks",
            "arc_easy",
            "--seeds",
            "0",
            "--limit",
            "5",
            "--out",
            str(out_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["repo"] == "permitt/serbian-llm-eval"
    assert captured["split"] == "test"

    artifact_path = out_dir / "results" / "sle-sr" / "tiny_gpt2" / "arc_easy" / "result.json"
    assert artifact_path.is_file()

    artifact = json.loads(artifact_path.read_text())
    schema = json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())
    validator = Draft202012Validator(schema)
    errors = list(validator.iter_errors(artifact))
    assert not errors, [e.message for e in errors]

    assert artifact["run_config"]["protocol"] == "loglikelihood"
    assert artifact["run_type"] == "experimental"
    assert artifact["rankable"] is False


def test_eval_api_access_uses_client_and_wires_cache(sle_configs, tmp_path, monkeypatch) -> None:
    """access: api branch: make_api_model + APIModel + cache_dir wiring stay
    real; only the provider SDK transport (AnthropicClient.complete) is
    mocked, at the narrowest seam."""
    monkeypatch.setattr(
        "balkanbench.cli.eval.load_dataset", lambda *a, **k: _fake_arc_easy_dataset()
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    model_path = sle_configs / "models" / "experimental" / "claude_fake.yaml"
    model_path.write_text(_API_MODEL_YAML)

    def fake_complete(
        self: AnthropicClient, prompt: str, *, max_tokens: int, stop_sequences: list[str]
    ) -> APIResponse:
        return APIResponse(text="A", input_tokens=5, output_tokens=1, cost_usd=0.0)

    monkeypatch.setattr(AnthropicClient, "complete", fake_complete)

    out_dir = tmp_path / "out"
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        result = runner.invoke(
            app,
            [
                "eval",
                "--model",
                "claude_fake",
                "--benchmark",
                "sle",
                "--task",
                "arc_easy",
                "--language",
                "sr",
                "--out",
                str(out_dir),
            ],
        )

    assert result.exit_code == 0, result.output

    artifact_path = out_dir / "sle-sr" / "claude_fake" / "arc_easy" / "result.json"
    assert artifact_path.is_file()
    artifact = json.loads(artifact_path.read_text())
    assert artifact["run_config"]["access"] == "api"
    assert artifact["run_config"]["protocol"] == "generative"

    cache_dir = out_dir / ".api_cache" / "claude_fake"
    cache_files = list(cache_dir.rglob("*.json"))
    assert cache_files, f"expected cache files under {cache_dir}"


def test_eval_api_missing_key_shows_styled_error_not_traceback(
    sle_configs, tmp_path, monkeypatch
) -> None:
    """Finding 1: a missing ANTHROPIC_API_KEY must surface as the CLI's
    styled red-echo + exit(1), not a raw RuntimeError traceback."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(
        "balkanbench.cli.eval.load_dataset", lambda *a, **k: _fake_arc_easy_dataset()
    )

    model_path = sle_configs / "models" / "experimental" / "claude_fake.yaml"
    model_path.write_text(_API_MODEL_YAML)

    out_dir = tmp_path / "out"
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        result = runner.invoke(
            app,
            [
                "eval",
                "--model",
                "claude_fake",
                "--benchmark",
                "sle",
                "--task",
                "arc_easy",
                "--language",
                "sr",
                "--out",
                str(out_dir),
            ],
        )

    assert result.exit_code != 0
    assert "ANTHROPIC_API_KEY" in result.output
    assert "Traceback" not in result.output
    assert not (out_dir / "sle-sr").exists()


def test_eval_generative_qa_loads_fewshot_split(sle_configs, tmp_path, monkeypatch) -> None:
    """num_fewshot > 0 must load the fewshot split via the same load_dataset
    seam, distinct from the test split, and use it in the real fewshot
    prefix-building code path (build_fewshot_prefix)."""
    task_path = sle_configs / "benchmarks" / "sle" / "tasks" / "nq_open.yaml"
    task_path.write_text(_NQ_OPEN_TASK_YAML)

    model_path = sle_configs / "models" / "experimental" / "claude_fake.yaml"
    model_path.write_text(_API_MODEL_YAML)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fake-anthropic-key")

    calls: list[str | None] = []

    def fake_load_dataset(repo: str, config: str, **kwargs: Any) -> Dataset:
        split = kwargs.get("split")
        calls.append(split)
        return _fake_nq_dataset("train" if split == "train" else "test")

    monkeypatch.setattr("balkanbench.cli.eval.load_dataset", fake_load_dataset)

    def fake_complete(
        self: AnthropicClient, prompt: str, *, max_tokens: int, stop_sequences: list[str]
    ) -> APIResponse:
        return APIResponse(text="Odgovor0", input_tokens=5, output_tokens=2, cost_usd=0.0)

    monkeypatch.setattr(AnthropicClient, "complete", fake_complete)

    out_dir = tmp_path / "out"
    with patch.dict(sys.modules, {"anthropic": MagicMock()}):
        result = runner.invoke(
            app,
            [
                "eval",
                "--model",
                "claude_fake",
                "--benchmark",
                "sle",
                "--task",
                "nq_open",
                "--language",
                "sr",
                "--limit",
                "3",
                "--out",
                str(out_dir),
            ],
        )

    assert result.exit_code == 0, result.output
    assert "train" in calls, f"expected a load_dataset(split='train') call, got {calls}"
    assert "test" in calls


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
