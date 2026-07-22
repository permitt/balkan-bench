"""Validation tests for the SLE benchmark and its 9 task YAMLs."""

from __future__ import annotations

from pathlib import Path

import pytest

from balkanbench.config import load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "eval" / "configs"
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"

SLE_DIR = CONFIGS / "benchmarks" / "sle"

BENCHMARK_SPEC_SCHEMA = SCHEMAS_DIR / "benchmark_spec.json"
TASK_SPEC_SCHEMA = SCHEMAS_DIR / "task_spec.json"

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
