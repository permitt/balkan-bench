"""Validation tests for the SLE benchmark and its 9 task YAMLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from balkanbench.config import load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "eval" / "configs"
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"

SLE_DIR = CONFIGS / "benchmarks" / "sle"
MODELS_DIR = CONFIGS / "models" / "official"

BENCHMARK_SPEC_SCHEMA = SCHEMAS_DIR / "benchmark_spec.json"
TASK_SPEC_SCHEMA = SCHEMAS_DIR / "task_spec.json"
MODEL_SPEC_SCHEMA = SCHEMAS_DIR / "model_spec.json"

TASK_NAMES = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "nq_open",
    "openbookqa",
    "piqa",
    "triviaqa",
    "winogrande",
]

# SLE launch roster: open-weights SLMs + closed API models.
SLE_OPEN_MODEL_NAMES = [
    "sle-qwen3-5-4b",
    "sle-qwen3-5-9b",
    "sle-gemma-4-e2b-it",
    "sle-gemma-4-e4b-it",
    "sle-granite-4-1-8b",
    "sle-ministral-8b",
    "sle-olmo3-7b",
    "sle-smollm3-3b",
    "sle-phi4-mini",
    "sle-yugogpt",
]

SLE_API_MODEL_NAMES = [
    "sle-claude-sonnet-5",
    "sle-claude-haiku-4-5",
    "sle-gpt-4-1",
    "sle-gpt-4-1-mini",
    "sle-gemini-3-5-flash",
    "sle-gemini-3-1-flash-lite",
]


def test_sle_benchmark_yaml_validates() -> None:
    spec = load_yaml_with_schema(SLE_DIR / "benchmark.yaml", BENCHMARK_SPEC_SCHEMA)
    assert spec["benchmark"] == "sle"
    assert "seeds" not in spec
    assert sorted(spec["tasks"]["ranked"]) == sorted(TASK_NAMES)


@pytest.mark.parametrize("name", TASK_NAMES)
def test_sle_task_yaml_validates(name: str) -> None:
    spec = load_yaml_with_schema(SLE_DIR / "tasks" / f"{name}.yaml", TASK_SPEC_SCHEMA)
    assert spec["benchmark"] == "sle"
    assert spec["task"] == name


def test_generative_tasks_include_train_split() -> None:
    for name in ("nq_open", "triviaqa"):
        spec = load_yaml_with_schema(SLE_DIR / "tasks" / f"{name}.yaml", TASK_SPEC_SCHEMA)
        assert spec["task_type"] == "generative_qa"
        assert "train" in spec["dataset"]["splits"]["public"]
        assert "train" in spec["dataset"]["splits"]["labeled_public"]
        assert spec["evaluation"]["num_fewshot"] == 5
        assert spec["evaluation"]["fewshot_split"] == "train"
        assert spec["evaluation"]["stop_sequences"] == ["\n", ".", ","]
        assert spec["evaluation"]["max_gen_tokens"] == 64


def test_winogrande_has_partial_variant() -> None:
    spec = load_yaml_with_schema(SLE_DIR / "tasks" / "winogrande.yaml", TASK_SPEC_SCHEMA)
    assert spec["variant"] == "winogrande_partial"


def test_boolq_has_da_ne_variant() -> None:
    spec = load_yaml_with_schema(SLE_DIR / "tasks" / "boolq.yaml", TASK_SPEC_SCHEMA)
    assert spec["variant"] == "boolq_da_ne"


@pytest.mark.parametrize("name", SLE_OPEN_MODEL_NAMES)
def test_sle_open_model_yaml_validates(name: str) -> None:
    spec = load_yaml_with_schema(MODELS_DIR / f"{name}.yaml", MODEL_SPEC_SCHEMA)
    assert spec["name"] == name
    assert spec["access"] == "open_weights"
    assert spec["hf_repo"]
    assert spec["tier"] == "official"
    # batch_size is per-model GPU-memory tuning (large models need smaller
    # batches on the 24GB L4), so assert presence and sanity, not a value.
    assert spec["generation"]["batch_size"] >= 1
    assert spec["generation"]["dtype"] == "bfloat16"


@pytest.mark.parametrize("name", SLE_API_MODEL_NAMES)
def test_sle_api_model_yaml_validates(name: str) -> None:
    spec = load_yaml_with_schema(MODELS_DIR / f"{name}.yaml", MODEL_SPEC_SCHEMA)
    assert spec["name"] == name
    assert spec["access"] == "api"
    assert spec["provider"] in ("anthropic", "openai", "gemini")
    assert spec["api_model_id"]
    assert spec["tier"] == "official"
    assert spec["generation"]["temperature"] == 0.0
    assert spec["generation"]["max_tokens"] == 64
    assert spec["generation"]["concurrency"] == 8
