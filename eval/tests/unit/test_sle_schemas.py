"""Schema tests for SLE (generative) task and benchmark specs."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from balkanbench.config import ConfigError, load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures" / "configs"

TASK_SPEC_SCHEMA = SCHEMAS_DIR / "task_spec.json"
BENCHMARK_SPEC_SCHEMA = SCHEMAS_DIR / "benchmark_spec.json"
MODEL_SPEC_SCHEMA = SCHEMAS_DIR / "model_spec.json"


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "task.yaml"
    p.write_text(text)
    return p


GENERATIVE_TASK = """\
benchmark: sle
task: arc_easy
status: ranked
task_type: multiple_choice_loglikelihood
languages: {available: [sr], ranked: [sr]}
dataset:
  source_type: huggingface
  config: arc_easy
  per_language:
    sr: {public_repo: permitt/serbian-llm-eval}
  splits: {public: [test], labeled_public: [test]}
inputs: {fields: [query, choices], id_field: example_id}
metrics: {primary: [acc_norm], report: [acc, acc_norm], task_score: acc_norm}
prompts: {sr: {template_id: sle_mc_v1}}
evaluation:
  num_fewshot: 0
  stop_sequences: []
  max_gen_tokens: 16
api_protocol: {reformulation: multiple_choice_generative, metrics: {primary: [acc], report: [acc], task_score: acc}}
"""


def test_generative_task_without_training_block_validates(tmp_path: Path) -> None:
    load_yaml_with_schema(_write(tmp_path, GENERATIVE_TASK), TASK_SPEC_SCHEMA)


def test_generative_task_requires_evaluation_block(tmp_path: Path) -> None:
    bad = GENERATIVE_TASK.replace("evaluation:\n  num_fewshot: 0\n  stop_sequences: []\n  max_gen_tokens: 16\n", "")
    with pytest.raises(ConfigError):
        load_yaml_with_schema(_write(tmp_path, bad), TASK_SPEC_SCHEMA)


def test_encoder_task_still_requires_training(tmp_path: Path) -> None:
    bad = GENERATIVE_TASK.replace("multiple_choice_loglikelihood", "binary_classification")
    with pytest.raises(ConfigError):
        load_yaml_with_schema(_write(tmp_path, bad), TASK_SPEC_SCHEMA)


def test_variant_field_accepted(tmp_path: Path) -> None:
    with_variant = GENERATIVE_TASK.replace(
        "task_type: multiple_choice_loglikelihood",
        "task_type: multiple_choice_loglikelihood\nvariant: winogrande_partial",
    )
    load_yaml_with_schema(_write(tmp_path, with_variant), TASK_SPEC_SCHEMA)


def test_benchmark_spec_seeds_now_optional(tmp_path: Path) -> None:
    superglue_benchmark = yaml.safe_load(
        (FIXTURES / "benchmarks" / "superglue_valid.yaml").read_text()
    )
    del superglue_benchmark["seeds"]

    p = tmp_path / "benchmark.yaml"
    p.write_text(yaml.safe_dump(superglue_benchmark, sort_keys=False))

    load_yaml_with_schema(p, BENCHMARK_SPEC_SCHEMA)


API_MODEL = """\
name: claude-sonnet
access: api
provider: anthropic
api_model_id: claude-sonnet-x-x
family: claude
params_hint: API
generation: {temperature: 0.0, max_tokens: 64}
"""

OPEN_MODEL_SLE = """\
name: some-slm
access: open_weights
hf_repo: org/some-slm
family: qwen
params_hint: 4B
"""


def test_api_model_validates(tmp_path: Path) -> None:
    load_yaml_with_schema(_write(tmp_path, API_MODEL), MODEL_SPEC_SCHEMA)


def test_api_model_rejects_missing_provider(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        load_yaml_with_schema(
            _write(tmp_path, API_MODEL.replace("provider: anthropic\n", "")), MODEL_SPEC_SCHEMA
        )


def test_open_weights_sle_model_without_training_validates(tmp_path: Path) -> None:
    load_yaml_with_schema(_write(tmp_path, OPEN_MODEL_SLE), MODEL_SPEC_SCHEMA)


def test_existing_encoder_model_still_validates() -> None:
    bertic_yaml = REPO_ROOT / "eval" / "configs" / "models" / "official" / "bertic.yaml"
    load_yaml_with_schema(bertic_yaml, MODEL_SPEC_SCHEMA)
