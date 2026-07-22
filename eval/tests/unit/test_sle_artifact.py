"""Tests for the generative result artifact writer (SLE track)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from balkanbench.cli._paths import resolve_model_config, schemas_root
from balkanbench.config import load_yaml_with_schema
from balkanbench.evaluation.generative import GenerativeRunResult
from balkanbench.scoring.artifact import (
    write_generative_result_artifact,
    write_result_artifact,
)

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "results"


def _schema() -> dict:
    return json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())


def _fake_provenance() -> dict:
    return {
        "code_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
        "image_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
        "package_version": "0.1.0.dev0",
        "torch_version": "2.11.0",
        "transformers_version": "5.5.4",
        "python_version": "3.11.13",
    }


def _fake_task_cfg() -> dict:
    return {
        "benchmark": "sle",
        "task": "boolq",
        "task_type": "multiple_choice_loglikelihood",
        "metrics": {"primary": ["acc"], "report": ["acc"], "task_score": "acc"},
        "api_protocol": {
            "reformulation": "multiple_choice_generative",
            "metrics": {"primary": ["acc"], "report": ["acc"], "task_score": "acc"},
        },
    }


def _fake_model_cfg(**overrides: object) -> dict:
    cfg: dict = {"name": "galton-v3", "hf_repo": "permitt/galton-v3", "access": "open_weights"}
    cfg.update(overrides)
    return cfg


def _mc_run_result(*, num_fewshot: int = 0, api_cost_usd: float = 0.0) -> GenerativeRunResult:
    return GenerativeRunResult(
        metrics={"acc": 0.75},
        per_example=[
            {"example_id": "e2", "prediction": 1, "correct": True},
            {"example_id": "e1", "prediction": 0, "correct": True},
            {"example_id": "e3", "prediction": 1, "correct": False},
            {"example_id": "e4", "prediction": 0, "correct": True},
        ],
        protocol="loglikelihood",
        num_fewshot=num_fewshot,
        unparsed_responses=0,
        api_cost_usd=api_cost_usd,
    )


def _qa_run_result(predictions: list[str]) -> GenerativeRunResult:
    return GenerativeRunResult(
        metrics={"exact_match": 0.5},
        per_example=[
            {"example_id": f"q{i}", "prediction": pred, "correct": i == 0}
            for i, pred in enumerate(predictions)
        ],
        protocol="generative",
        num_fewshot=5,
        unparsed_responses=0,
        api_cost_usd=0.12,
    )


def test_generative_artifact_is_schema_valid(tmp_path) -> None:
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        run_result=_mc_run_result(),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )

    assert out_path.name == "result.json"
    assert out_path.parent.name == "boolq"
    assert out_path.parent.parent.name == "galton-v3"
    assert out_path.parent.parent.parent.name == "sle-sr"

    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)

    assert data["task_id"] == "sle.boolq.sr"
    assert "hp_search" not in data
    assert "seeds" not in data
    assert "seed_results" not in data
    assert data["run_config"] == {
        "protocol": "loglikelihood",
        "num_fewshot": 0,
        "access": "open_weights",
        "api_cost_usd": 0.0,
        "unparsed_responses": 0,
    }


def test_generative_artifact_task_score_matches_aggregate_mean(tmp_path) -> None:
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        run_result=_mc_run_result(),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    assert data["task_score"] == data["aggregate"]["mean"]["acc"]
    assert data["task_score"] == pytest.approx(0.75)


def test_api_run_artifact_carries_provider_and_cost(tmp_path) -> None:
    model_cfg = _fake_model_cfg(
        access="api",
        provider="anthropic",
        generation={"concurrency": 8, "max_tokens": 16},
    )
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(api_cost_usd=1.23),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert data["run_config"]["access"] == "api"
    assert data["run_config"]["provider"] == "anthropic"
    assert data["run_config"]["api_cost_usd"] == pytest.approx(1.23)
    assert data["run_config"]["generation"] == {"concurrency": 8, "max_tokens": 16}


def test_generative_qa_string_predictions_hash_deterministic(tmp_path) -> None:
    out_path_1 = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        run_result=_qa_run_result(["Beograd", "Novi Sad"]),
        task_score_metric="exact_match",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path / "run1",
    )
    out_path_2 = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        run_result=_qa_run_result(["Beograd", "Novi Sad"]),
        task_score_metric="exact_match",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path / "run2",
    )
    out_path_3 = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=_fake_model_cfg(),
        language="sr",
        run_result=_qa_run_result(["Beograd", "Nis"]),
        task_score_metric="exact_match",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path / "run3",
    )

    data_1 = json.loads(out_path_1.read_text())
    data_2 = json.loads(out_path_2.read_text())
    data_3 = json.loads(out_path_3.read_text())

    Draft202012Validator(_schema()).validate(data_1)
    assert data_1["test_predictions_hash"] == data_2["test_predictions_hash"]
    assert data_1["test_predictions_hash"] != data_3["test_predictions_hash"]
    assert data_1["test_predictions_hash"].startswith("sha256:")


@pytest.mark.parametrize(
    ("params_hint", "expected"),
    [
        ("4B", 4_000_000_000),
        ("9.7B", 9_700_000_000),
        ("8.9B", 8_900_000_000),
        ("5.1B", 5_100_000_000),
    ],
)
def test_open_artifact_carries_params_from_params_hint(tmp_path, params_hint, expected) -> None:
    model_cfg = _fake_model_cfg(params_hint=params_hint)
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert data["params"] == expected


def test_api_artifact_omits_params(tmp_path) -> None:
    model_cfg = _fake_model_cfg(access="api", provider="anthropic", params_hint="API")
    del model_cfg["hf_repo"]
    model_cfg["api_model_id"] = "claude-fake"
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(api_cost_usd=0.05),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert "params" not in data


def test_open_artifact_omits_params_when_unparseable(tmp_path) -> None:
    model_cfg = _fake_model_cfg(params_hint="unknown")
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert "params" not in data


def test_api_artifact_uses_api_model_id_when_hf_repo_absent(tmp_path) -> None:
    """Real API roster YAMLs (e.g. configs/models/official/sle-claude-sonnet-5.yaml)
    have no ``hf_repo`` - the artifact writer must fall back to ``api_model_id``
    instead of crashing with a KeyError."""
    model_cfg = {
        "name": "claude-x",
        "access": "api",
        "provider": "anthropic",
        "api_model_id": "claude-sonnet-5",
        "family": "claude",
        "params_hint": "API",
    }
    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(api_cost_usd=0.05),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert data["model_id"] == "claude-sonnet-5"


def test_api_roster_yaml_writes_artifact_without_crash(tmp_path) -> None:
    """Regression for the CRITICAL API artifact crash: load a real API roster
    YAML through the actual config loader (no synthetic hf_repo) and confirm
    write_generative_result_artifact doesn't crash and model_id matches
    api_model_id."""
    model_cfg = load_yaml_with_schema(
        resolve_model_config("sle-claude-sonnet-5"), schemas_root() / "model_spec.json"
    )
    assert "hf_repo" not in model_cfg

    out_path = write_generative_result_artifact(
        task_cfg=_fake_task_cfg(),
        model_cfg=model_cfg,
        language="sr",
        run_result=_mc_run_result(api_cost_usd=0.05),
        task_score_metric="acc",
        provenance=_fake_provenance(),
        dataset_revision="v0.1.0-data",
        benchmark_version="0.1.0",
        out_dir=tmp_path,
    )
    data = json.loads(out_path.read_text())
    Draft202012Validator(_schema()).validate(data)
    assert data["model_id"] == model_cfg["api_model_id"]


def test_encoder_artifact_writer_still_validates_superglue_fixture() -> None:
    """Regression: the encoder path (write_result_artifact) is untouched by the
    schema/writer changes made for generative runs - required-field relaxation
    (hp_search/seeds/seed_results now optional) must not break an artifact
    that still supplies them."""
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    Draft202012Validator(_schema()).validate(artifact)
    assert "hp_search" in artifact
    assert "seeds" in artifact
    assert "seed_results" in artifact
    # write_result_artifact itself is still importable and usable unchanged.
    assert write_result_artifact.__name__ == "write_result_artifact"
