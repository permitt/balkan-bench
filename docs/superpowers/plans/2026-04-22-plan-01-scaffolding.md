# BalkanBench v0.1 - Plan 1: Scaffolding

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce the monorepo layout, an installable `balkanbench` Python package with a `typer` CLI skeleton, six JSON Schemas, a config loader, `list` + `validate-env` + `validate-config` + `validate-data` subcommands, pinned Docker image, GitHub Actions CI (lint + type + tests + coverage), plus READMEs. End state: `balkanbench --help` works, validation subcommands work against fixture configs, CI green on push.

**Architecture:** Two-folder monorepo. `frontend/` keeps the existing React app intact. `eval/` hosts a new Python package `balkanbench` installable in editable mode via `uv pip install -e ".[dev]"`. The CLI lives at `balkanbench.cli.main:app` (typer). JSON Schemas in `eval/schemas/` drive the `validate-config` and `validate-data` subcommands. `pydantic` models wrap YAML configs; `jsonschema` validates raw JSON artifacts.

**Tech Stack:** Python 3.11, uv, typer, pydantic v2, jsonschema, PyYAML, pytest, pytest-cov, ruff, mypy, GitHub Actions, Docker.

**Branch:** `feature/code-for-eval` (continuation of the spec commits).

**Assumptions:** run on macOS or Linux with `git`, `python3.11`, `uv`, and `node` already installed. `cd`-prefixed commands must stay consistent with CWD at the repo root unless stated otherwise.

---

## Task 1: Move frontend into `frontend/` subfolder

Existing React app lives at repo root (`index.html`, `src/`, `public/`, `package.json`, etc). Move it under `frontend/` so the monorepo layout matches the spec. Vercel rootDirectory is reconfigured in Plan 6.

**Files:**
- Move to `frontend/`: `index.html`, `package.json`, `package-lock.json`, `vite.config.js`, `eslint.config.js`, `vercel.json`, `src/`, `public/`, `README.md`
- Modify: `.gitignore` (repo root)

- [ ] **Step 1: Verify clean tree**

Run:
```bash
git status --short
```
Expected: clean (no uncommitted changes).

- [ ] **Step 2: Create target directory and move files**

Run:
```bash
mkdir -p frontend
git mv index.html frontend/
git mv package.json frontend/
git mv package-lock.json frontend/
git mv vite.config.js frontend/
git mv eslint.config.js frontend/
git mv vercel.json frontend/
git mv src frontend/src
git mv public frontend/public
git mv README.md frontend/README.md
```

- [ ] **Step 3: Delete untracked `node_modules` and `dist`**

Run:
```bash
rm -rf node_modules dist
```

- [ ] **Step 4: Rewrite `.gitignore`**

File: `.gitignore` (replace entire contents)

```
# node
node_modules/
frontend/node_modules/
frontend/dist/

# python
eval/.venv/
eval/**/__pycache__/
eval/**/*.egg-info/
eval/.pytest_cache/
eval/.mypy_cache/
eval/.ruff_cache/
eval/htmlcov/
eval/.coverage
eval/coverage.xml

# build
dist/
build/

# mac
.DS_Store

# local results (only /official/ is committed)
eval/results/local/
eval/results/submissions/

# stray empty files from prior iterations
/validation
```

- [ ] **Step 5: Verify the frontend still builds**

Run:
```bash
cd frontend && npm install && npm run build && cd ..
```
Expected: Vite build succeeds, `frontend/dist/` exists.

- [ ] **Step 6: Commit**

Run:
```bash
git add -A
git commit -m "chore: move React app into frontend/ subfolder

Monorepo split: frontend/ will be Vercel's rootDirectory; eval/ (added
next) hosts the Python package. Updates .gitignore for the new layout."
```

---

## Task 2: Scaffold the `balkanbench` Python package

Create `eval/pyproject.toml`, package source tree, and install in editable mode. No functionality yet beyond `__version__`.

**Files:**
- Create: `eval/pyproject.toml`
- Create: `eval/src/balkanbench/__init__.py`
- Create: `eval/src/balkanbench/py.typed`
- Create: `eval/README.md`

- [ ] **Step 1: Create directories**

Run:
```bash
mkdir -p eval/src/balkanbench eval/tests/unit
```

- [ ] **Step 2: Write `eval/pyproject.toml`**

File: `eval/pyproject.toml`

```toml
[project]
name = "balkanbench"
version = "0.1.0.dev0"
description = "Reproducible evaluation benchmark for BCMS language models."
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
authors = [{ name = "BalkanBench contributors" }]
keywords = ["benchmark", "nlp", "evaluation", "serbian", "superglue", "bcms"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

dependencies = [
    "typer>=0.12",
    "pydantic>=2.7",
    "jsonschema>=4.22",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8",
    "pytest-cov>=5",
    "ruff>=0.5",
    "mypy>=1.10",
    "types-PyYAML",
    "types-jsonschema",
]

[project.scripts]
balkanbench = "balkanbench.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/balkanbench"]

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM"]
ignore = []

[tool.ruff.lint.per-file-ignores]
"tests/**" = ["B"]

[tool.ruff.format]
quote-style = "double"

[tool.mypy]
python_version = "3.11"
strict = true
packages = ["balkanbench"]
mypy_path = "src"

[[tool.mypy.overrides]]
module = ["jsonschema.*"]
ignore_missing_imports = true

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=balkanbench --cov-report=term-missing --cov-report=xml --cov-fail-under=80"

[tool.coverage.run]
source = ["src/balkanbench"]
branch = true

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "if __name__ == .__main__.:",
    "raise NotImplementedError",
]
```

- [ ] **Step 3: Write package init**

File: `eval/src/balkanbench/__init__.py`

```python
"""BalkanBench: reproducible benchmark for BCMS language models."""

__version__ = "0.1.0.dev0"

__all__ = ["__version__"]
```

File: `eval/src/balkanbench/py.typed`

(Empty marker file for PEP 561 type distribution.)

- [ ] **Step 4: Write eval README stub**

File: `eval/README.md`

````markdown
# balkanbench (Python package)

The evaluation framework for [BalkanBench](https://balkanbench.com).

## Install (development)

```bash
cd eval
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Quick check

```bash
balkanbench --version
balkanbench --help
```

## Layout

```
src/balkanbench/   # package source
configs/           # benchmark, task, model configs
schemas/           # JSON Schemas for configs + artifacts
scripts/           # publish + aggregate + GCP launchers
tests/             # unit + integration + smoke
results/           # official results (committed); local/submissions gitignored
```

Full design: [`docs/superpowers/specs/2026-04-22-balkanbench-v0.1-design.md`](../docs/superpowers/specs/2026-04-22-balkanbench-v0.1-design.md).
````

- [ ] **Step 5: Install in editable mode**

Run:
```bash
cd eval
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
cd ..
```
Expected: install succeeds; `balkanbench` executable on PATH inside `.venv`.

- [ ] **Step 6: Verify import**

Run:
```bash
eval/.venv/bin/python -c "import balkanbench; print(balkanbench.__version__)"
```
Expected: `0.1.0.dev0`.

- [ ] **Step 7: Commit**

Run:
```bash
git add eval/pyproject.toml eval/src eval/README.md
git commit -m "feat(eval): scaffold balkanbench Python package

- pyproject.toml with typer, pydantic, jsonschema deps
- ruff + mypy + pytest + coverage config
- empty src/balkanbench/__init__.py exposing __version__
- PEP 561 py.typed marker
- eval/README.md with dev install instructions

Installable in editable mode via \`uv pip install -e \".[dev]\"\`."
```

---

## Task 3: Typer CLI root with `--version`

Add `balkanbench` CLI root. No subcommands yet beyond `--version` / `--help`.

**Files:**
- Create: `eval/src/balkanbench/cli/__init__.py`
- Create: `eval/src/balkanbench/cli/main.py`
- Create: `eval/tests/__init__.py`
- Create: `eval/tests/unit/__init__.py`
- Create: `eval/tests/unit/test_cli_skeleton.py`

- [ ] **Step 1: Write failing CLI test**

File: `eval/tests/unit/test_cli_skeleton.py`

```python
"""Smoke tests for the typer CLI skeleton."""
from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_version_flag_prints_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0, result.output
    assert "0.1.0.dev0" in result.stdout


def test_help_mentions_balkanbench() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "balkanbench" in result.stdout.lower()


def test_no_args_shows_help() -> None:
    result = runner.invoke(app, [])
    assert result.exit_code != 0  # typer exits non-zero when no_args_is_help
    assert "Usage" in result.stdout or "Usage" in result.output
```

File: `eval/tests/__init__.py` and `eval/tests/unit/__init__.py` are empty files.

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
cd eval && source .venv/bin/activate && pytest tests/unit/test_cli_skeleton.py -v
```
Expected: `ModuleNotFoundError: No module named 'balkanbench.cli'`.

- [ ] **Step 3: Implement CLI**

File: `eval/src/balkanbench/cli/__init__.py`

```python
"""balkanbench CLI package."""
```

File: `eval/src/balkanbench/cli/main.py`

```python
"""Root typer app for balkanbench."""
from __future__ import annotations

import typer

from balkanbench import __version__

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show balkanbench version and exit.",
    ),
) -> None:
    """BalkanBench CLI root."""


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests, verify they pass**

Run:
```bash
pytest tests/unit/test_cli_skeleton.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Verify installed entry point**

Run:
```bash
balkanbench --version
balkanbench --help
```
Expected: prints `0.1.0.dev0`; prints usage.

- [ ] **Step 6: Commit**

Run:
```bash
cd ..
git add eval/src/balkanbench/cli eval/tests
git commit -m "feat(cli): add typer root app with --version and --help

Empty root command. Subcommands (list, validate-*, eval, predict, score,
hp-search, throughput, leaderboard, submit) land in later tasks."
```

---

## Task 4: JSON Schema - `task_spec.json`

Add the schema file plus a pydantic model, plus a `balkanbench validate-config` that loads a YAML and validates it.

**Files:**
- Create: `eval/schemas/task_spec.json`
- Create: `eval/tests/fixtures/configs/tasks/boolq_valid.yaml`
- Create: `eval/tests/fixtures/configs/tasks/boolq_invalid_missing_metrics.yaml`
- Create: `eval/tests/unit/test_schema_task_spec.py`

- [ ] **Step 1: Write task_spec JSON Schema**

File: `eval/schemas/task_spec.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/task_spec.json",
  "title": "BalkanBench Task Specification",
  "type": "object",
  "required": [
    "benchmark",
    "task",
    "status",
    "task_type",
    "languages",
    "dataset",
    "inputs",
    "metrics",
    "prompts",
    "training"
  ],
  "additionalProperties": false,
  "properties": {
    "benchmark": { "type": "string", "minLength": 1 },
    "task": { "type": "string", "minLength": 1 },
    "status": {
      "type": "string",
      "enum": ["ranked", "diagnostic", "experimental", "archived"]
    },
    "task_type": {
      "type": "string",
      "enum": [
        "binary_classification",
        "multiclass_classification",
        "multiple_choice",
        "grouped_binary_classification"
      ]
    },
    "languages": {
      "type": "object",
      "required": ["available", "ranked"],
      "additionalProperties": false,
      "properties": {
        "available": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "ranked":    { "type": "array", "items": { "type": "string" } },
        "roadmap":   { "type": "array", "items": { "type": "string" } }
      }
    },
    "dataset": {
      "type": "object",
      "required": ["public_repo", "config", "splits"],
      "additionalProperties": false,
      "properties": {
        "public_repo":  { "type": "string", "minLength": 1 },
        "private_repo": { "type": "string" },
        "config":       { "type": "string", "minLength": 1 },
        "splits": {
          "type": "object",
          "required": ["public", "labeled_public"],
          "additionalProperties": false,
          "properties": {
            "public":          { "type": "array", "items": { "type": "string" }, "minItems": 1 },
            "labeled_public":  { "type": "array", "items": { "type": "string" }, "minItems": 1 },
            "labeled_private": { "type": "array", "items": { "type": "string" } }
          }
        }
      }
    },
    "inputs": {
      "type": "object",
      "required": ["fields", "id_field"],
      "additionalProperties": false,
      "properties": {
        "fields":   { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "id_field": { "type": "string", "minLength": 1 }
      }
    },
    "metrics": {
      "type": "object",
      "required": ["primary", "report", "task_score"],
      "additionalProperties": false,
      "properties": {
        "primary":    { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "report":     { "type": "array", "items": { "type": "string" }, "minItems": 1 },
        "task_score": { "type": "string", "minLength": 1 }
      }
    },
    "prompts": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "type": "object",
        "required": ["template_id"],
        "additionalProperties": false,
        "properties": {
          "template_id": { "type": "string", "minLength": 1 }
        }
      }
    },
    "training": {
      "type": "object",
      "required": [
        "learning_rate",
        "batch_size",
        "num_epochs",
        "metric_for_best_model"
      ],
      "additionalProperties": true,
      "properties": {
        "learning_rate":           { "type": "number", "exclusiveMinimum": 0 },
        "batch_size":              { "type": "integer", "minimum": 1 },
        "num_epochs":              { "type": "integer", "minimum": 1 },
        "warmup_ratio":            { "type": "number", "minimum": 0, "maximum": 1 },
        "weight_decay":            { "type": "number", "minimum": 0 },
        "early_stopping_patience": { "type": "integer", "minimum": 0 },
        "metric_for_best_model":   { "type": "string", "minLength": 1 }
      }
    }
  }
}
```

- [ ] **Step 2: Write fixtures**

File: `eval/tests/fixtures/configs/tasks/boolq_valid.yaml`

```yaml
benchmark: superglue
task: boolq
status: ranked
task_type: binary_classification
languages:
  available: [sr]
  ranked: [sr]
  roadmap: [hr, cnr, bs]
dataset:
  public_repo: permitt/superglue-serbian
  private_repo: permitt/superglue-private
  config: boolq
  splits:
    public: [train, validation, test]
    labeled_public: [train, validation]
    labeled_private: [test]
inputs:
  fields: [question, passage]
  id_field: example_id
metrics:
  primary: [accuracy]
  report: [accuracy]
  task_score: accuracy
prompts:
  sr:
    template_id: boolq_sr_v1
training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 10
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping_patience: 5
  metric_for_best_model: accuracy
```

File: `eval/tests/fixtures/configs/tasks/boolq_invalid_missing_metrics.yaml`

```yaml
benchmark: superglue
task: boolq
status: ranked
task_type: binary_classification
languages:
  available: [sr]
  ranked: [sr]
dataset:
  public_repo: permitt/superglue-serbian
  config: boolq
  splits:
    public: [train, validation, test]
    labeled_public: [train, validation]
inputs:
  fields: [question, passage]
  id_field: example_id
prompts:
  sr:
    template_id: boolq_sr_v1
training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 10
  metric_for_best_model: accuracy
```

- [ ] **Step 3: Write failing schema test**

File: `eval/tests/unit/test_schema_task_spec.py`

```python
"""Validate task YAMLs against `schemas/task_spec.json`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "tasks"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "task_spec.json").read_text())


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_boolq_passes() -> None:
    Draft202012Validator(_load_schema()).validate(_load_yaml("boolq_valid.yaml"))


def test_invalid_boolq_missing_metrics_fails() -> None:
    validator = Draft202012Validator(_load_schema())
    errors = list(validator.iter_errors(_load_yaml("boolq_invalid_missing_metrics.yaml")))
    assert errors, "expected at least one schema error"
    assert any("metrics" in (e.message + str(e.path)) for e in errors)
```

- [ ] **Step 4: Run test, verify it fails**

Run:
```bash
pytest tests/unit/test_schema_task_spec.py -v
```
Expected: ImportError or FileNotFoundError on the schema file, or a syntax error. Before the schema exists the first test should fail with `FileNotFoundError`.

- [ ] **Step 5: Make sure the schema file is present (step 1 already wrote it) and rerun**

Run:
```bash
pytest tests/unit/test_schema_task_spec.py -v
```
Expected: 2 passed.

- [ ] **Step 6: Commit**

Run:
```bash
cd ..
git add eval/schemas/task_spec.json eval/tests
git commit -m "feat(schemas): add task_spec JSON Schema + fixtures

task_spec.json enforces benchmark/task/status/task_type/languages/dataset/
inputs/metrics/prompts/training sections with strict additionalProperties:
false and required-field rules. Includes valid + invalid fixtures for BoolQ."
```

---

## Task 5: JSON Schema - `model_spec.json`

**Files:**
- Create: `eval/schemas/model_spec.json`
- Create: `eval/tests/fixtures/configs/models/bertic_valid.yaml`
- Create: `eval/tests/fixtures/configs/models/bertic_invalid_no_hf_repo.yaml`
- Create: `eval/tests/unit/test_schema_model_spec.py`

- [ ] **Step 1: Write the schema**

File: `eval/schemas/model_spec.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/model_spec.json",
  "title": "BalkanBench Model Specification",
  "type": "object",
  "required": ["name", "hf_repo", "family", "params_hint", "training"],
  "additionalProperties": false,
  "properties": {
    "name":         { "type": "string", "minLength": 1 },
    "hf_repo":      { "type": "string", "minLength": 1 },
    "hf_revision":  { "type": "string" },
    "family":       { "type": "string", "minLength": 1 },
    "params_hint":  { "type": "string", "minLength": 1 },
    "tier":         { "type": "string", "enum": ["official", "experimental"] },
    "hf_auth": {
      "type": "object",
      "additionalProperties": false,
      "properties": { "required": { "type": "boolean" } }
    },
    "training": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "learning_rate": { "type": "number", "exclusiveMinimum": 0 },
        "batch_size":    { "type": "integer", "minimum": 1 },
        "num_epochs":    { "type": "integer", "minimum": 1 },
        "fp16":          { "type": "boolean" }
      }
    },
    "task_overrides": {
      "type": "object",
      "additionalProperties": {
        "type": "object",
        "additionalProperties": true
      }
    },
    "seeds": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 1
    }
  }
}
```

- [ ] **Step 2: Write fixtures**

File: `eval/tests/fixtures/configs/models/bertic_valid.yaml`

```yaml
name: bertic
hf_repo: classla/bcms-bertic
family: electra
params_hint: 110M
tier: official
training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 10
  fp16: true
task_overrides:
  cb:
    num_epochs: 30
  wsc:
    learning_rate: 1.0e-5
    num_epochs: 30
seeds: [42, 43, 44, 45, 46]
```

File: `eval/tests/fixtures/configs/models/bertic_invalid_no_hf_repo.yaml`

```yaml
name: bertic
family: electra
params_hint: 110M
training:
  learning_rate: 2.0e-5
  batch_size: 16
  num_epochs: 10
```

- [ ] **Step 3: Write failing test**

File: `eval/tests/unit/test_schema_model_spec.py`

```python
"""Validate model YAMLs against `schemas/model_spec.json`."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "configs" / "models"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "model_spec.json").read_text())


def _load_yaml(name: str) -> dict:
    return yaml.safe_load((FIXTURES / name).read_text())


def test_valid_bertic_passes() -> None:
    Draft202012Validator(_load_schema()).validate(_load_yaml("bertic_valid.yaml"))


def test_missing_hf_repo_fails() -> None:
    errors = list(
        Draft202012Validator(_load_schema()).iter_errors(
            _load_yaml("bertic_invalid_no_hf_repo.yaml")
        )
    )
    assert errors
    assert any("hf_repo" in (e.message + str(e.path)) for e in errors)
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_schema_model_spec.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/schemas/model_spec.json eval/tests
git commit -m "feat(schemas): add model_spec JSON Schema + fixtures"
```

---

## Task 6: JSON Schema - `dataset_manifest.json`

**Files:**
- Create: `eval/schemas/dataset_manifest.json`
- Create: `eval/tests/fixtures/manifests/superglue_sr_valid.json`
- Create: `eval/tests/unit/test_schema_dataset_manifest.py`

- [ ] **Step 1: Write the schema**

File: `eval/schemas/dataset_manifest.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/dataset_manifest.json",
  "title": "BalkanBench Dataset Manifest",
  "type": "object",
  "required": [
    "benchmark",
    "language",
    "public_repo",
    "dataset_revision",
    "configs",
    "license",
    "hidden_test_labels"
  ],
  "additionalProperties": false,
  "properties": {
    "benchmark":         { "type": "string", "minLength": 1 },
    "language":          { "type": "string", "minLength": 1 },
    "public_repo":       { "type": "string", "minLength": 1 },
    "private_repo":      { "type": "string" },
    "dataset_revision":  { "type": "string", "minLength": 1 },
    "license":           { "type": "string", "minLength": 1 },
    "hidden_test_labels": { "type": "boolean" },
    "configs": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": {
        "type": "object",
        "required": ["splits", "fields"],
        "additionalProperties": false,
        "properties": {
          "splits": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
              "type": "object",
              "required": ["num_rows"],
              "additionalProperties": false,
              "properties": {
                "num_rows":   { "type": "integer", "minimum": 0 },
                "has_labels": { "type": "boolean" }
              }
            }
          },
          "fields": { "type": "array", "items": { "type": "string" }, "minItems": 1 }
        }
      }
    }
  }
}
```

- [ ] **Step 2: Write fixture**

File: `eval/tests/fixtures/manifests/superglue_sr_valid.json`

```json
{
  "benchmark": "superglue",
  "language": "sr",
  "public_repo": "permitt/superglue-serbian",
  "private_repo": "permitt/superglue-private",
  "dataset_revision": "v0.1.0-data",
  "license": "CC-BY-4.0",
  "hidden_test_labels": true,
  "configs": {
    "boolq": {
      "splits": {
        "train":      { "num_rows": 9427, "has_labels": true },
        "validation": { "num_rows": 3270, "has_labels": true },
        "test":       { "num_rows": 3245, "has_labels": false }
      },
      "fields": ["example_id", "question", "passage", "language", "task_id"]
    }
  }
}
```

- [ ] **Step 3: Write failing test**

File: `eval/tests/unit/test_schema_dataset_manifest.py`

```python
"""Validate dataset manifests against `schemas/dataset_manifest.json`."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "manifests"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "dataset_manifest.json").read_text())


def test_valid_manifest_passes() -> None:
    manifest = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(manifest)


def test_missing_hidden_test_labels_fails() -> None:
    manifest = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    del manifest["hidden_test_labels"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(manifest))
    assert errors
    assert any("hidden_test_labels" in e.message for e in errors)
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_schema_dataset_manifest.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/schemas/dataset_manifest.json eval/tests
git commit -m "feat(schemas): add dataset_manifest JSON Schema + fixture"
```

---

## Task 7: JSON Schema - `result_artifact.json`

**Files:**
- Create: `eval/schemas/result_artifact.json`
- Create: `eval/tests/fixtures/results/bertic_boolq_sr_valid.json`
- Create: `eval/tests/unit/test_schema_result_artifact.py`

- [ ] **Step 1: Write the schema**

File: `eval/schemas/result_artifact.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/result_artifact.json",
  "title": "BalkanBench Per-Run Result Artifact",
  "type": "object",
  "required": [
    "benchmark_name",
    "benchmark_version",
    "run_type",
    "task_id",
    "language",
    "model",
    "model_id",
    "model_revision",
    "code_revision",
    "dataset_revision",
    "image_digest",
    "config_hash",
    "selection_metric",
    "hp_search",
    "seeds",
    "seed_results",
    "aggregate",
    "task_score",
    "rankable",
    "test_predictions_hash",
    "sponsor"
  ],
  "additionalProperties": false,
  "properties": {
    "benchmark_name":    { "type": "string", "minLength": 1 },
    "benchmark_version": { "type": "string", "minLength": 1 },
    "run_type":          { "type": "string", "enum": ["official", "experimental"] },
    "task_id":           { "type": "string", "minLength": 1 },
    "language":          { "type": "string", "minLength": 1 },
    "model":             { "type": "string", "minLength": 1 },
    "model_id":          { "type": "string", "minLength": 1 },
    "model_revision":    { "type": "string", "minLength": 1 },
    "code_revision":     { "type": "string", "minLength": 1 },
    "dataset_revision":  { "type": "string", "minLength": 1 },
    "image_digest":      { "type": "string", "minLength": 1 },
    "config_hash":       { "type": "string", "minLength": 1 },
    "selection_metric":  { "type": "string", "minLength": 1 },
    "hp_search": {
      "type": "object",
      "required": ["tool", "sampler", "sampler_seed", "num_trials", "search_space_id"],
      "additionalProperties": false,
      "properties": {
        "tool":                  { "type": "string" },
        "sampler":               { "type": "string" },
        "sampler_seed":          { "type": "integer" },
        "num_trials":            { "type": "integer", "minimum": 0 },
        "search_space_id":       { "type": "string" },
        "early_stopping_policy": { "type": "string" }
      }
    },
    "seeds": {
      "type": "array",
      "items": { "type": "integer" },
      "minItems": 1
    },
    "seed_results": {
      "type": "array",
      "minItems": 1,
      "items": {
        "type": "object",
        "required": ["seed", "primary", "secondary"],
        "additionalProperties": false,
        "properties": {
          "seed": { "type": "integer" },
          "primary": {
            "type": "object",
            "additionalProperties": { "type": "number" },
            "minProperties": 1
          },
          "secondary": {
            "type": "object",
            "additionalProperties": { "type": "number" }
          }
        }
      }
    },
    "aggregate": {
      "type": "object",
      "required": ["mean", "stdev"],
      "additionalProperties": false,
      "properties": {
        "mean":  { "type": "object", "additionalProperties": { "type": "number" } },
        "stdev": { "type": "object", "additionalProperties": { "type": "number" } }
      }
    },
    "task_score":            { "type": "number" },
    "rankable":              { "type": "boolean" },
    "test_predictions_hash": { "type": "string", "pattern": "^sha256:[0-9a-f]{64}$" },
    "throughput": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "ex_per_sec":     { "type": "number" },
        "tok_per_sec":    { "type": "number" },
        "peak_vram_mib":  { "type": "number" },
        "hardware":       { "type": "string" },
        "batch_size":     { "type": "integer" },
        "max_seq_len":    { "type": "integer" },
        "precision":      { "type": "string" },
        "torch_version":  { "type": "string" },
        "driver_version": { "type": "string" }
      }
    },
    "sponsor": { "type": "string", "minLength": 1 }
  }
}
```

- [ ] **Step 2: Write fixture**

File: `eval/tests/fixtures/results/bertic_boolq_sr_valid.json`

```json
{
  "benchmark_name": "balkanbench",
  "benchmark_version": "0.1.0",
  "run_type": "official",
  "task_id": "superglue.boolq.sr",
  "language": "sr",
  "model": "bertic",
  "model_id": "classla/bcms-bertic",
  "model_revision": "abc123def456abc123def456abc123def4567890",
  "code_revision": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
  "dataset_revision": "v0.1.0-data",
  "image_digest": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
  "config_hash": "sha256:1111111111111111111111111111111111111111111111111111111111111111",
  "selection_metric": "accuracy",
  "hp_search": {
    "tool": "optuna",
    "sampler": "TPESampler",
    "sampler_seed": 42,
    "num_trials": 20,
    "search_space_id": "superglue-encoder-v1",
    "early_stopping_policy": "patience=5 on accuracy"
  },
  "seeds": [42, 43, 44, 45, 46],
  "seed_results": [
    { "seed": 42, "primary": { "accuracy": 0.7766 }, "secondary": {} },
    { "seed": 43, "primary": { "accuracy": 0.7812 }, "secondary": {} },
    { "seed": 44, "primary": { "accuracy": 0.7779 }, "secondary": {} },
    { "seed": 45, "primary": { "accuracy": 0.7805 }, "secondary": {} },
    { "seed": 46, "primary": { "accuracy": 0.7733 }, "secondary": {} }
  ],
  "aggregate": {
    "mean": { "accuracy": 0.7779 },
    "stdev": { "accuracy": 0.0018 }
  },
  "task_score": 0.7779,
  "rankable": true,
  "test_predictions_hash": "sha256:2222222222222222222222222222222222222222222222222222222222222222",
  "sponsor": "Recrewty"
}
```

- [ ] **Step 3: Write failing test**

File: `eval/tests/unit/test_schema_result_artifact.py`

```python
"""Validate per-run result artifacts against `schemas/result_artifact.json`."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "results"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "result_artifact.json").read_text())


def test_valid_artifact_passes() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(artifact)


def test_bad_hash_format_fails() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    artifact["test_predictions_hash"] = "not-a-hash"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(artifact))
    assert errors
    assert any("test_predictions_hash" in str(e.path) or "pattern" in e.message for e in errors)


def test_missing_sponsor_fails() -> None:
    artifact = json.loads((FIXTURES / "bertic_boolq_sr_valid.json").read_text())
    del artifact["sponsor"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(artifact))
    assert errors
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_schema_result_artifact.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/schemas/result_artifact.json eval/tests
git commit -m "feat(schemas): add result_artifact JSON Schema + fixture

Strict per-run artifact schema with sha256: pattern on test_predictions_hash,
required run_type enum (official|experimental), seeds array, per-seed
primary/secondary metrics, hp_search block, and Recrewty sponsor field."
```

---

## Task 8: JSON Schema - `leaderboard_export.json`

**Files:**
- Create: `eval/schemas/leaderboard_export.json`
- Create: `eval/tests/fixtures/leaderboards/superglue_sr_valid.json`
- Create: `eval/tests/unit/test_schema_leaderboard_export.py`

- [ ] **Step 1: Write the schema**

File: `eval/schemas/leaderboard_export.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/leaderboard_export.json",
  "title": "BalkanBench Leaderboard Export",
  "type": "object",
  "required": [
    "benchmark",
    "language",
    "benchmark_version",
    "generated_at",
    "sponsor",
    "seeds",
    "ranked_tasks",
    "task_primary_metrics",
    "rows"
  ],
  "additionalProperties": false,
  "properties": {
    "benchmark":         { "type": "string", "minLength": 1 },
    "language":          { "type": "string", "minLength": 1 },
    "benchmark_version": { "type": "string", "minLength": 1 },
    "generated_at":      { "type": "string", "format": "date-time" },
    "sponsor":           { "type": "string", "minLength": 1 },
    "seeds":             { "type": "integer", "minimum": 1 },
    "ranked_tasks":      { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "task_primary_metrics": {
      "type": "object",
      "minProperties": 1,
      "additionalProperties": { "type": "string" }
    },
    "throughput": {
      "type": "object",
      "additionalProperties": true,
      "properties": {
        "hardware":            { "type": "string" },
        "precision":           { "type": "string" },
        "batch_size_policy":   { "type": "string" },
        "warmup_batches":      { "type": "integer" },
        "measurement_batches": { "type": "integer" }
      }
    },
    "rows": {
      "type": "array",
      "items": {
        "type": "object",
        "required": [
          "model",
          "model_id",
          "params",
          "results",
          "avg",
          "complete",
          "tasks_completed",
          "tasks_total"
        ],
        "additionalProperties": true,
        "properties": {
          "rank":            { "type": ["integer", "null"] },
          "model":           { "type": "string", "minLength": 1 },
          "model_id":        { "type": "string", "minLength": 1 },
          "model_revision":  { "type": "string" },
          "params":          { "type": "integer", "minimum": 0 },
          "params_display":  { "type": "string" },
          "results": {
            "type": "object",
            "additionalProperties": {
              "anyOf": [
                { "type": "null" },
                {
                  "type": "object",
                  "required": ["mean", "stdev"],
                  "additionalProperties": false,
                  "properties": {
                    "mean":  { "type": "number" },
                    "stdev": { "type": "number" }
                  }
                }
              ]
            }
          },
          "avg":              { "type": "number" },
          "complete":         { "type": "boolean" },
          "tasks_completed":  { "type": "integer", "minimum": 0 },
          "tasks_total":      { "type": "integer", "minimum": 1 },
          "partial_flag":     { "type": "string" },
          "throughput": {
            "type": "object",
            "additionalProperties": false,
            "properties": {
              "ex_per_sec":    { "type": "number" },
              "peak_vram_mib": { "type": "number" }
            }
          }
        }
      }
    }
  }
}
```

- [ ] **Step 2: Write fixture**

File: `eval/tests/fixtures/leaderboards/superglue_sr_valid.json`

```json
{
  "benchmark": "superglue",
  "language": "sr",
  "benchmark_version": "0.1.0",
  "generated_at": "2026-04-27T09:00:00Z",
  "sponsor": "Recrewty",
  "seeds": 5,
  "ranked_tasks": ["boolq", "cb", "copa", "rte", "multirc", "wsc"],
  "task_primary_metrics": {
    "boolq": "accuracy",
    "cb": "f1_macro",
    "copa": "accuracy",
    "rte": "accuracy",
    "multirc": "f1_a",
    "wsc": "accuracy"
  },
  "throughput": {
    "hardware": "NVIDIA L4 24GB",
    "precision": "fp16",
    "batch_size_policy": "from_task_config",
    "warmup_batches": 2,
    "measurement_batches": 50
  },
  "rows": [
    {
      "rank": 1,
      "model": "BERTić",
      "model_id": "classla/bcms-bertic",
      "params": 110000000,
      "params_display": "110M",
      "results": {
        "cb":      { "mean": 78.61, "stdev": 2.90 },
        "copa":    { "mean": 68.87, "stdev": 0.52 },
        "rte":     { "mean": 71.70, "stdev": 0.87 },
        "wsc":     { "mean": 65.07, "stdev": 0.00 },
        "boolq":   { "mean": 77.79, "stdev": 0.18 },
        "multirc": { "mean": 66.75, "stdev": 0.87 }
      },
      "avg": 71.46,
      "complete": true,
      "tasks_completed": 6,
      "tasks_total": 6,
      "throughput": { "ex_per_sec": 234.5, "peak_vram_mib": 4820 }
    }
  ]
}
```

- [ ] **Step 3: Write failing test**

File: `eval/tests/unit/test_schema_leaderboard_export.py`

```python
"""Validate benchmark_results.json payloads."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "leaderboards"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "leaderboard_export.json").read_text())


def test_valid_export_passes() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(export)


def test_partial_row_allowed() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    export["rows"].append(
        {
            "rank": None,
            "model": "ModernBERTić small",
            "model_id": "permitt/modernbertic-small",
            "params": 149000000,
            "params_display": "149M",
            "results": {
                "cb":      {"mean": 76.96, "stdev": 3.19},
                "copa":    {"mean": 65.76, "stdev": 2.42},
                "rte":     {"mean": 65.82, "stdev": 1.14},
                "wsc":     {"mean": 64.11, "stdev": 1.11},
                "boolq":   {"mean": 76.02, "stdev": 0.63},
                "multirc": None,
            },
            "avg": 69.73,
            "complete": False,
            "tasks_completed": 5,
            "tasks_total": 6,
            "partial_flag": "(5/6)",
            "throughput": {"ex_per_sec": 312.1, "peak_vram_mib": 2410},
        }
    )
    Draft202012Validator(_load_schema()).validate(export)


def test_missing_sponsor_fails() -> None:
    export = json.loads((FIXTURES / "superglue_sr_valid.json").read_text())
    del export["sponsor"]
    errors = list(Draft202012Validator(_load_schema()).iter_errors(export))
    assert errors
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_schema_leaderboard_export.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/schemas/leaderboard_export.json eval/tests
git commit -m "feat(schemas): add leaderboard_export JSON Schema + fixtures

Allows rank=null and partial rows (results.{task}: null + partial_flag +
complete=false), required throughput metadata block, task_primary_metrics
map. Fixture covers BERTić full row and a ModernBERTić small partial row."
```

---

## Task 9: JSON Schema - `submission_metadata.json`

**Files:**
- Create: `eval/schemas/submission_metadata.json`
- Create: `eval/tests/fixtures/submissions/example_valid.json`
- Create: `eval/tests/unit/test_schema_submission_metadata.py`

- [ ] **Step 1: Write the schema**

File: `eval/schemas/submission_metadata.json`

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://balkanbench.com/schemas/submission_metadata.json",
  "title": "BalkanBench Submission Metadata",
  "type": "object",
  "required": [
    "submission_id",
    "submitter",
    "model",
    "benchmark",
    "languages",
    "predictions_package"
  ],
  "additionalProperties": false,
  "properties": {
    "submission_id": { "type": "string", "minLength": 1 },
    "submitter": {
      "type": "object",
      "required": ["name", "identity"],
      "additionalProperties": false,
      "properties": {
        "name":     { "type": "string", "minLength": 1 },
        "email":    { "type": "string", "format": "email" },
        "affiliation": { "type": "string" },
        "identity": {
          "type": "object",
          "required": ["provider", "handle"],
          "additionalProperties": false,
          "properties": {
            "provider": { "type": "string", "enum": ["github", "huggingface"] },
            "handle":   { "type": "string", "minLength": 1 }
          }
        }
      }
    },
    "model": {
      "type": "object",
      "required": ["name", "hf_repo", "license"],
      "additionalProperties": false,
      "properties": {
        "name":        { "type": "string" },
        "hf_repo":     { "type": "string" },
        "hf_revision": { "type": "string" },
        "license":     { "type": "string" },
        "params":      { "type": "integer", "minimum": 0 }
      }
    },
    "benchmark": {
      "type": "object",
      "required": ["name", "version"],
      "additionalProperties": false,
      "properties": {
        "name":    { "type": "string" },
        "version": { "type": "string" }
      }
    },
    "languages": {
      "type": "array",
      "items": { "type": "string" },
      "minItems": 1
    },
    "predictions_package": {
      "type": "object",
      "required": ["path", "sha256"],
      "additionalProperties": false,
      "properties": {
        "path":   { "type": "string", "minLength": 1 },
        "sha256": { "type": "string", "pattern": "^sha256:[0-9a-f]{64}$" }
      }
    },
    "notes": { "type": "string" }
  }
}
```

- [ ] **Step 2: Write fixture**

File: `eval/tests/fixtures/submissions/example_valid.json`

```json
{
  "submission_id": "sub_01HZN0X8E6PQK8P7",
  "submitter": {
    "name": "Mitar Perović",
    "email": "perovicmitar@gmail.com",
    "affiliation": "BalkanBench / Recrewty",
    "identity": { "provider": "github", "handle": "permitt" }
  },
  "model": {
    "name": "bertic",
    "hf_repo": "classla/bcms-bertic",
    "hf_revision": "abc123",
    "license": "apache-2.0",
    "params": 110000000
  },
  "benchmark": { "name": "balkanbench-superglue", "version": "0.1.0" },
  "languages": ["sr"],
  "predictions_package": {
    "path": "submissions/bertic-sr-2026-04-27.tar.gz",
    "sha256": "sha256:3333333333333333333333333333333333333333333333333333333333333333"
  },
  "notes": "Reference reproduction of BERTić on SuperGLUE-Serbian."
}
```

- [ ] **Step 3: Write failing test**

File: `eval/tests/unit/test_schema_submission_metadata.py`

```python
"""Validate submission metadata JSON."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

SCHEMAS_DIR = Path(__file__).resolve().parents[2] / "schemas"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "submissions"


def _load_schema() -> dict:
    return json.loads((SCHEMAS_DIR / "submission_metadata.json").read_text())


def test_valid_submission_passes() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    Draft202012Validator(_load_schema()).validate(submission)


def test_bad_identity_provider_fails() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    submission["submitter"]["identity"]["provider"] = "twitter"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(submission))
    assert errors


def test_bad_package_hash_fails() -> None:
    submission = json.loads((FIXTURES / "example_valid.json").read_text())
    submission["predictions_package"]["sha256"] = "deadbeef"
    errors = list(Draft202012Validator(_load_schema()).iter_errors(submission))
    assert errors
```

- [ ] **Step 4: Run test, verify it passes**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_schema_submission_metadata.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/schemas/submission_metadata.json eval/tests
git commit -m "feat(schemas): add submission_metadata JSON Schema + fixture

Required submitter identity via github or huggingface (anti-spam),
sha256:<64hex> hash on the predictions package, required license field
on the model block."
```

---

## Task 10: Config loader (YAML + schema)

Unified loader: load a YAML file, validate it against a named JSON Schema, return a plain dict. Later tasks can wrap with pydantic models but the plan keeps this minimal for v0.1.

**Files:**
- Create: `eval/src/balkanbench/config.py`
- Create: `eval/tests/unit/test_config_loader.py`

- [ ] **Step 1: Write failing test**

File: `eval/tests/unit/test_config_loader.py`

```python
"""Tests for the unified YAML + JSON Schema loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from balkanbench.config import ConfigError, load_yaml_with_schema

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMAS_DIR = REPO_ROOT / "eval" / "schemas"
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures" / "configs"


def test_loads_valid_task_yaml() -> None:
    cfg = load_yaml_with_schema(
        FIXTURES / "tasks" / "boolq_valid.yaml",
        SCHEMAS_DIR / "task_spec.json",
    )
    assert cfg["benchmark"] == "superglue"
    assert cfg["task"] == "boolq"
    assert cfg["metrics"]["task_score"] == "accuracy"


def test_rejects_invalid_task_yaml() -> None:
    with pytest.raises(ConfigError) as exc:
        load_yaml_with_schema(
            FIXTURES / "tasks" / "boolq_invalid_missing_metrics.yaml",
            SCHEMAS_DIR / "task_spec.json",
        )
    assert "metrics" in str(exc.value)


def test_reports_missing_file() -> None:
    with pytest.raises(ConfigError):
        load_yaml_with_schema(
            FIXTURES / "tasks" / "does_not_exist.yaml",
            SCHEMAS_DIR / "task_spec.json",
        )
```

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_config_loader.py -v
```
Expected: `ModuleNotFoundError: No module named 'balkanbench.config'`.

- [ ] **Step 3: Implement loader**

File: `eval/src/balkanbench/config.py`

```python
"""Shared YAML + JSON Schema loader."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml
from jsonschema import Draft202012Validator


class ConfigError(ValueError):
    """Raised when a YAML config fails schema validation or cannot be read."""


def load_yaml_with_schema(yaml_path: Path, schema_path: Path) -> dict[str, Any]:
    """Load `yaml_path`, validate against `schema_path`, return the parsed dict.

    Raises `ConfigError` on missing files, YAML parse errors, or schema violations.
    """
    if not yaml_path.is_file():
        raise ConfigError(f"config file not found: {yaml_path}")
    if not schema_path.is_file():
        raise ConfigError(f"schema file not found: {schema_path}")

    try:
        data = yaml.safe_load(yaml_path.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"failed to parse YAML {yaml_path}: {exc}") from exc

    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        raise ConfigError(f"failed to parse schema {schema_path}: {exc}") from exc

    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.path))
    if errors:
        messages = [
            f"  - {'.'.join(str(p) for p in err.path) or '<root>'}: {err.message}"
            for err in errors
        ]
        raise ConfigError(
            f"{yaml_path} failed schema {schema_path.name}:\n" + "\n".join(messages)
        )

    if not isinstance(data, dict):
        raise ConfigError(f"top-level YAML must be a mapping, got {type(data).__name__}")

    return data
```

- [ ] **Step 4: Run tests, verify they pass**

Run:
```bash
pytest tests/unit/test_config_loader.py -v
```
Expected: 3 passed.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add eval/src/balkanbench/config.py eval/tests
git commit -m "feat(config): add YAML + JSON Schema loader

load_yaml_with_schema() loads a YAML, validates it against a JSON Schema
using Draft 2020-12, and returns the parsed dict. Raises ConfigError with
a bullet-listed summary of all validation errors. Used by validate-config
and by every downstream CLI subcommand that reads configs."
```

---

## Task 11: `balkanbench validate-env`

Checks Python version, required third-party imports, and optional `HF_TOKEN` / `HF_OFFICIAL_TOKEN` env vars. Non-zero exit on hard-fail; zero on warnings.

**Files:**
- Create: `eval/src/balkanbench/cli/validate.py`
- Modify: `eval/src/balkanbench/cli/main.py`
- Create: `eval/tests/unit/test_cli_validate_env.py`

- [ ] **Step 1: Write failing test**

File: `eval/tests/unit/test_cli_validate_env.py`

```python
"""Smoke tests for `balkanbench validate-env`."""
from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_validate_env_runs() -> None:
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 0, result.output
    assert "python" in result.stdout.lower()


def test_validate_env_reports_hf_token_absence(monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_OFFICIAL_TOKEN", raising=False)
    result = runner.invoke(app, ["validate-env"])
    assert result.exit_code == 0
    assert "hf_token" in result.stdout.lower() or "huggingface" in result.stdout.lower()
```

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_cli_validate_env.py -v
```
Expected: FAIL (command not registered).

- [ ] **Step 3: Implement command**

File: `eval/src/balkanbench/cli/validate.py`

```python
"""`validate-*` subcommands."""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import typer

from balkanbench.config import ConfigError, load_yaml_with_schema

REQUIRED_IMPORTS = ("typer", "pydantic", "jsonschema", "yaml")
OPTIONAL_ENV_VARS = ("HF_TOKEN", "HF_OFFICIAL_TOKEN")


def _green(text: str) -> str:
    return typer.style(text, fg=typer.colors.GREEN, bold=True)


def _yellow(text: str) -> str:
    return typer.style(text, fg=typer.colors.YELLOW, bold=True)


def _red(text: str) -> str:
    return typer.style(text, fg=typer.colors.RED, bold=True)


def validate_env() -> None:
    """Check the Python + dependency + secrets environment."""
    ok = True

    py = sys.version_info
    typer.echo(f"python: {py.major}.{py.minor}.{py.micro}")
    if (py.major, py.minor) < (3, 11):
        typer.echo(_red("  required: >=3.11"))
        ok = False

    for name in REQUIRED_IMPORTS:
        try:
            importlib.import_module(name)
            typer.echo(f"import {name}: {_green('OK')}")
        except ImportError:
            typer.echo(f"import {name}: {_red('MISSING')}")
            ok = False

    for var in OPTIONAL_ENV_VARS:
        if os.environ.get(var):
            typer.echo(f"env {var}: {_green('present')}")
        else:
            typer.echo(f"env {var}: {_yellow('absent (needed for private labels)')}")

    if not ok:
        raise typer.Exit(code=1)


def validate_config(
    path: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
    schema: str = typer.Option(
        "task_spec",
        "--schema",
        "-s",
        help="Schema name under eval/schemas/ (without .json).",
    ),
) -> None:
    """Validate a YAML config against a named JSON Schema."""
    schema_path = _resolve_schema_path(schema)
    try:
        load_yaml_with_schema(path, schema_path)
    except ConfigError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    typer.echo(_green(f"OK: {path} is a valid {schema}"))


def validate_data(
    manifest: Path = typer.Argument(..., exists=True, readable=True, dir_okay=False),
) -> None:
    """Validate a dataset manifest JSON against `dataset_manifest.json`."""
    schema_path = _resolve_schema_path("dataset_manifest")
    try:
        load_yaml_with_schema(manifest, schema_path)
    except ConfigError as exc:
        typer.echo(_red(str(exc)))
        raise typer.Exit(code=1) from exc
    typer.echo(_green(f"OK: {manifest} is a valid dataset manifest"))


def _resolve_schema_path(schema_name: str) -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "schemas" / f"{schema_name}.json"
```

File: `eval/src/balkanbench/cli/main.py` (replace entire contents)

```python
"""Root typer app for balkanbench."""
from __future__ import annotations

import typer

from balkanbench import __version__
from balkanbench.cli import validate as validate_cmds

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)

app.command("validate-env", help="Check Python + deps + env vars.")(validate_cmds.validate_env)
app.command("validate-config", help="Validate a YAML config against a JSON Schema.")(
    validate_cmds.validate_config
)
app.command("validate-data", help="Validate a dataset manifest JSON.")(
    validate_cmds.validate_data
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show balkanbench version and exit.",
    ),
) -> None:
    """BalkanBench CLI root."""


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Verify `validate-env` tests pass**

Run:
```bash
pytest tests/unit/test_cli_validate_env.py -v
```
Expected: 2 passed.

- [ ] **Step 5: Write `validate-config` + `validate-data` tests**

File: `eval/tests/unit/test_cli_validate_config.py`

```python
"""Smoke tests for `balkanbench validate-config` and `validate-data`."""
from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "eval" / "tests" / "fixtures"


def test_validate_config_accepts_valid_task_yaml() -> None:
    result = runner.invoke(
        app,
        [
            "validate-config",
            str(FIXTURES / "configs" / "tasks" / "boolq_valid.yaml"),
            "--schema",
            "task_spec",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output


def test_validate_config_rejects_invalid_task_yaml() -> None:
    result = runner.invoke(
        app,
        [
            "validate-config",
            str(FIXTURES / "configs" / "tasks" / "boolq_invalid_missing_metrics.yaml"),
            "--schema",
            "task_spec",
        ],
    )
    assert result.exit_code == 1


def test_validate_data_accepts_valid_manifest() -> None:
    result = runner.invoke(
        app,
        ["validate-data", str(FIXTURES / "manifests" / "superglue_sr_valid.json")],
    )
    assert result.exit_code == 0, result.output
    assert "OK" in result.output
```

- [ ] **Step 6: Run all tests**

Run:
```bash
pytest tests/unit -v
```
Expected: all tests pass (8 from earlier tasks + ~5 new = 13+).

- [ ] **Step 7: Commit**

Run:
```bash
cd ..
git add eval/src/balkanbench/cli eval/tests
git commit -m "feat(cli): add validate-env, validate-config, validate-data

- validate-env checks Python 3.11+, required imports, and optional
  HF_TOKEN / HF_OFFICIAL_TOKEN env vars
- validate-config validates a YAML against a named JSON Schema
- validate-data validates a dataset manifest JSON
- all three wired into the typer root app"
```

---

## Task 12: `balkanbench list` discovery commands

Empty-for-now: lists whatever is in `eval/configs/{benchmarks,models}/`. When those directories don't exist yet, the commands print a friendly empty message.

**Files:**
- Create: `eval/src/balkanbench/cli/listcmd.py`
- Modify: `eval/src/balkanbench/cli/main.py`
- Create: `eval/tests/unit/test_cli_list.py`

- [ ] **Step 1: Write failing test**

File: `eval/tests/unit/test_cli_list.py`

```python
"""Smoke tests for `balkanbench list`."""
from __future__ import annotations

from typer.testing import CliRunner

from balkanbench.cli.main import app

runner = CliRunner()


def test_list_benchmarks_runs_even_with_no_configs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "benchmarks"])
    assert result.exit_code == 0, result.output
    assert "no benchmarks" in result.stdout.lower() or result.stdout.strip() == ""


def test_list_models_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "models"])
    assert result.exit_code == 0


def test_list_tasks_runs(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "tasks"])
    assert result.exit_code == 0


def test_list_languages_returns_sr_for_v01(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("BALKANBENCH_CONFIGS_DIR", str(tmp_path))
    result = runner.invoke(app, ["list", "languages"])
    assert result.exit_code == 0
    assert "sr" in result.stdout
```

- [ ] **Step 2: Run test, verify it fails**

Run:
```bash
cd eval && source .venv/bin/activate
pytest tests/unit/test_cli_list.py -v
```
Expected: FAIL (command not registered).

- [ ] **Step 3: Implement list subcommand**

File: `eval/src/balkanbench/cli/listcmd.py`

```python
"""`balkanbench list ...` discovery commands."""
from __future__ import annotations

import os
from pathlib import Path

import typer

V01_LANGUAGES: tuple[str, ...] = ("sr",)
V01_ROADMAP_LANGUAGES: tuple[str, ...] = ("hr", "cnr", "bs")

list_app = typer.Typer(
    name="list",
    help="Discover configured benchmarks, tasks, models, languages.",
    no_args_is_help=True,
    add_completion=False,
)


def _configs_root() -> Path:
    override = os.environ.get("BALKANBENCH_CONFIGS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "configs"


def _yaml_files(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*.yaml"))


@list_app.command("benchmarks")
def list_benchmarks() -> None:
    """List known benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no benchmarks configured yet")
        return
    names = sorted(p.name for p in root.iterdir() if p.is_dir())
    if not names:
        typer.echo("no benchmarks configured yet")
        return
    for name in names:
        typer.echo(name)


@list_app.command("tasks")
def list_tasks() -> None:
    """List tasks across all benchmarks."""
    root = _configs_root() / "benchmarks"
    if not root.is_dir():
        typer.echo("no tasks configured yet")
        return
    task_ids: list[str] = []
    for bench_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for task_yaml in _yaml_files(bench_dir / "tasks"):
            task_ids.append(f"{bench_dir.name}.{task_yaml.stem}")
    if not task_ids:
        typer.echo("no tasks configured yet")
        return
    for tid in task_ids:
        typer.echo(tid)


@list_app.command("models")
def list_models() -> None:
    """List model configs."""
    root = _configs_root() / "models"
    if not root.is_dir():
        typer.echo("no models configured yet")
        return
    files = _yaml_files(root)
    if not files:
        typer.echo("no models configured yet")
        return
    for f in files:
        tier = f.parent.name if f.parent.name in {"official", "experimental"} else "-"
        typer.echo(f"{f.stem}\t{tier}")


@list_app.command("languages")
def list_languages() -> None:
    """List languages in scope for v0.1 plus the roadmap."""
    for lang in V01_LANGUAGES:
        typer.echo(f"{lang}\tavailable")
    for lang in V01_ROADMAP_LANGUAGES:
        typer.echo(f"{lang}\troadmap")
```

File: `eval/src/balkanbench/cli/main.py` (append the `app.add_typer` line near the other command registrations; replace the full file for clarity)

```python
"""Root typer app for balkanbench."""
from __future__ import annotations

import typer

from balkanbench import __version__
from balkanbench.cli import validate as validate_cmds
from balkanbench.cli.listcmd import list_app

app = typer.Typer(
    name="balkanbench",
    help="BalkanBench: reproducible evaluation benchmark for BCMS language models.",
    add_completion=False,
    no_args_is_help=True,
)

app.add_typer(list_app, name="list")

app.command("validate-env", help="Check Python + deps + env vars.")(validate_cmds.validate_env)
app.command("validate-config", help="Validate a YAML config against a JSON Schema.")(
    validate_cmds.validate_config
)
app.command("validate-data", help="Validate a dataset manifest JSON.")(
    validate_cmds.validate_data
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(__version__)
        raise typer.Exit()


@app.callback()
def root(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show balkanbench version and exit.",
    ),
) -> None:
    """BalkanBench CLI root."""


if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run tests, verify they pass**

Run:
```bash
pytest tests/unit/test_cli_list.py -v
```
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite**

Run:
```bash
pytest tests/unit -v
```
Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:
```bash
cd ..
git add eval/src/balkanbench/cli eval/tests
git commit -m "feat(cli): add \`list\` subcommand group

list benchmarks | tasks | models | languages. Reads configs from
eval/configs/ by default, overridable via BALKANBENCH_CONFIGS_DIR.
v0.1 language list: sr available; hr, cnr, bs on roadmap."
```

---

## Task 13: Dockerfile

Pinned Python 3.11 image with `uv`, installing the package in editable mode. Used by CI and (in Plan 5) GCP.

**Files:**
- Create: `eval/Dockerfile`
- Create: `eval/.dockerignore`

- [ ] **Step 1: Write `eval/Dockerfile`**

File: `eval/Dockerfile`

```dockerfile
# syntax=docker/dockerfile:1.7
FROM python:3.11-slim-bookworm AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv

RUN useradd --create-home --shell /bin/bash balkanbench
WORKDIR /workspace

COPY pyproject.toml /workspace/pyproject.toml
COPY src /workspace/src
COPY README.md /workspace/README.md

RUN uv pip install --system -e ".[dev]"

USER balkanbench

ENTRYPOINT ["balkanbench"]
CMD ["--help"]
```

File: `eval/.dockerignore`

```
.venv
__pycache__
*.egg-info
.pytest_cache
.mypy_cache
.ruff_cache
htmlcov
.coverage
coverage.xml
results/local
results/submissions
tests/fixtures/large
```

- [ ] **Step 2: Build image**

Run:
```bash
cd eval && docker build -t balkanbench:dev .
```
Expected: image builds successfully.

- [ ] **Step 3: Smoke-run the image**

Run:
```bash
docker run --rm balkanbench:dev --version
docker run --rm balkanbench:dev --help
```
Expected: prints `0.1.0.dev0`, then CLI help.

- [ ] **Step 4: Commit**

Run:
```bash
cd ..
git add eval/Dockerfile eval/.dockerignore
git commit -m "build(docker): pinned Python 3.11 image for balkanbench

- Debian slim base, build-essential + git system deps
- uv installed for fast installs
- package installed in editable mode
- non-root balkanbench user at runtime
- entrypoint is the CLI (\`docker run balkanbench:dev --help\`)"
```

---

## Task 14: GitHub Actions CI workflow

Single `ci.yml` workflow: ruff lint + format check, mypy, pytest with coverage.

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write the workflow**

File: `.github/workflows/ci.yml`

```yaml
name: ci

on:
  push:
    branches: [main, feature/**]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    name: lint + type + test
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: eval
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install uv
        run: pip install uv

      - name: Install package with dev extras
        run: uv pip install --system -e ".[dev]"

      - name: Ruff check
        run: ruff check .

      - name: Ruff format check
        run: ruff format --check .

      - name: Mypy
        run: mypy

      - name: Pytest with coverage
        run: pytest --cov-fail-under=80

      - name: Upload coverage XML
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: coverage-xml
          path: eval/coverage.xml
          if-no-files-found: ignore
```

- [ ] **Step 2: Run ruff locally to pre-clean anything ruff would flag**

Run:
```bash
cd eval && source .venv/bin/activate
ruff check .
ruff format --check .
```
Expected: no errors. If ruff reports issues, fix them by running `ruff format .` and re-running `ruff check --fix .` before continuing.

- [ ] **Step 3: Run mypy locally**

Run:
```bash
mypy
```
Expected: no errors. If errors surface, fix them in-place (most likely: missing return types, untyped vars).

- [ ] **Step 4: Run tests with coverage**

Run:
```bash
pytest
```
Expected: all tests pass, coverage >= 80%.

- [ ] **Step 5: Commit**

Run:
```bash
cd ..
git add .github/workflows/ci.yml
git commit -m "ci: add lint + type + test workflow

ruff check + ruff format --check + mypy + pytest with --cov-fail-under=80.
Runs on push to main and feature/** and on PRs to main."
```

---

## Task 15: Final integration + coverage check

Make one end-to-end pass: install clean, run every test, verify the CLI, verify Docker.

- [ ] **Step 1: Clean rebuild**

Run:
```bash
cd eval
rm -rf .venv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```
Expected: fresh install succeeds.

- [ ] **Step 2: Full test + coverage**

Run:
```bash
pytest
```
Expected: all tests pass; coverage >= 80%.

- [ ] **Step 3: Verify every CLI subcommand starts**

Run:
```bash
balkanbench --help
balkanbench --version
balkanbench validate-env
balkanbench list --help
balkanbench list benchmarks
balkanbench list tasks
balkanbench list models
balkanbench list languages
balkanbench validate-config tests/fixtures/configs/tasks/boolq_valid.yaml --schema task_spec
balkanbench validate-data tests/fixtures/manifests/superglue_sr_valid.json
```
Expected: all exit 0 except that no-configs-dir list commands produce the "no ... configured yet" message.

- [ ] **Step 4: Docker smoke**

Run:
```bash
docker build -t balkanbench:dev .
docker run --rm balkanbench:dev --version
```
Expected: `0.1.0.dev0`.

- [ ] **Step 5: Commit any stragglers (coverage tweaks, formatting)**

Run:
```bash
cd ..
git status --short
```
If anything changed, commit with `chore: tidy scaffolding`. Otherwise skip.

- [ ] **Step 6: Push branch**

Run:
```bash
git log --oneline feature/code-for-eval ^main | head
```
Expected: list of Plan 1 commits since branching from `main`.

Run:
```bash
git push -u origin feature/code-for-eval
```
Expected: remote updated. (Requires the user's confirmation in interactive use.)

---

## End of Plan 1

Successful completion state:

- `frontend/` contains the React app, unchanged behaviour
- `eval/` has an installable Python package `balkanbench` with:
  - `typer` CLI: `--version`, `--help`, `list {benchmarks,tasks,models,languages}`, `validate-env`, `validate-config`, `validate-data`
  - 6 JSON Schemas (`task_spec`, `model_spec`, `dataset_manifest`, `result_artifact`, `leaderboard_export`, `submission_metadata`) with passing fixture tests
  - `balkanbench.config.load_yaml_with_schema()` used by `validate-config`
  - pinned Docker image
- `.github/workflows/ci.yml` runs ruff + mypy + pytest + coverage gate on every push + PR
- Coverage at or above 80% repo-wide
- Plan 2 (data pipeline: `publish_dataset.py`, COPA rename, HF push) can start next.
