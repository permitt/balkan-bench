# BalkanBench v0.1 - Plan 3: Tasks + Metrics + Evaluator

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` syntax.

**Goal:** A runnable `balkanbench eval` + `balkanbench predict` that fine-tunes
an HF encoder on train + validation and emits a scored result artifact. Six
task implementations (BoolQ, CB, COPA, RTE, MultiRC, WSC), five metrics
(accuracy, f1_macro, f1_a, matthews_correlation, gender_parity), a task
registry, an HF encoder wrapper, and a seed-orchestrated evaluator.

**Architecture:** Thin layers.

- `metrics/` - each metric is one function `(predictions, references) -> float`.
  Registry maps string names to callables.
- `tasks/base.py` - `Task` ABC: `load_split`, `preprocess`, `decode`,
  `score`, `score_from_private`. Concrete tasks are small, declarative files
  with all task-specific behaviour; no custom `Trainer` subclasses.
- `models/hf_encoder.py` - thin wrapper over `AutoModelForSequenceClassification`
  / `AutoModelForMultipleChoice`. Feature detection for precision (`fp16`) and
  attention implementation (`flash_attention_2` > `sdpa` > default).
- `evaluation/evaluator.py` - `run_single_seed`, `run_multiseed`, `run_task`.
  Always goes through plain HF `Trainer`.
- CLI subcommands `eval` and `predict` wire the evaluator to typer.

**Tech Stack:** `torch`, `transformers`, `scikit-learn`, `numpy`. Added as
`[ml]` optional extras plus folded into `dev` so CI still exercises the tests.

**Branch:** `feature/code-for-eval`.

---

## Task 1: Add ML deps

**Files:** `eval/pyproject.toml`

- [ ] Add `torch`, `transformers`, `scikit-learn`, `numpy` to `[ml]` extra and to `dev`.
- [ ] `uv pip install -e ".[dev]"`
- [ ] Commit: `build(eval): add torch + transformers + sklearn for the eval pipeline`

## Task 2: Metrics module

**Files:**
- Create: `eval/src/balkanbench/metrics/__init__.py` (registry)
- Create: `eval/src/balkanbench/metrics/accuracy.py`
- Create: `eval/src/balkanbench/metrics/f1.py` (exports `f1_macro`, `f1_a`)
- Create: `eval/src/balkanbench/metrics/matthews.py`
- Create: `eval/src/balkanbench/metrics/gender_parity.py`
- Create: `eval/tests/unit/test_metrics.py`

- [ ] Registry `get_metric(name) -> Callable[[predictions, references], float]`
- [ ] Each metric unit-tested with hand-crafted arrays
- [ ] `f1_a` computes F1 of the positive class (MultiRC spec)
- [ ] `gender_parity` accepts a parallel array of `is_pro_stereotype` flags
- [ ] Commit per metric in one consolidated commit

## Task 3: Pydantic wrappers for task + model YAML

**Files:**
- Create: `eval/src/balkanbench/configs/task_config.py`
- Create: `eval/src/balkanbench/configs/model_config.py`
- Create: `eval/tests/unit/test_config_models.py`

- [ ] `TaskConfig.load(path)` wraps `load_yaml_with_schema(path, task_spec.json)` and returns a pydantic model with typed fields (benchmark, task, languages, dataset, metrics, training)
- [ ] `ModelConfig.load(path)` same for model spec
- [ ] Commit: `feat(configs): pydantic wrappers for task + model YAMLs`

## Task 4: Task ABC + registry

**Files:**
- Create: `eval/src/balkanbench/tasks/__init__.py` (registry)
- Create: `eval/src/balkanbench/tasks/base.py` (`Task` ABC)
- Create: `eval/tests/unit/test_task_registry.py`

- [ ] `Task` ABC: abstract `preprocess`, `decode`, `score`, `score_from_private`; concrete `primary_metric_names`, `task_id`
- [ ] `@register_task(task_type)` decorator; `get_task_class(task_type)` lookup
- [ ] Commit: `feat(tasks): Task ABC and registry`

## Task 5: ClassificationTask (BoolQ, CB, RTE)

**Files:**
- Create: `eval/src/balkanbench/tasks/classification.py`
- Create: `eval/tests/unit/test_task_classification.py`

- [ ] Handles `binary_classification` + `multiclass_classification` (CB = 3 labels)
- [ ] `preprocess(example, tokenizer)`: concatenates the two fields or passes through single field
- [ ] `decode(logits) -> class index`
- [ ] `score(predictions, references) -> MetricBundle` using configured primary + report metrics
- [ ] Commit: `feat(tasks): ClassificationTask for BoolQ / CB / RTE`

## Task 6: WSCTask

**Files:**
- Create: `eval/src/balkanbench/tasks/wsc.py`
- Create: `eval/tests/unit/test_task_wsc.py`

- [ ] Prompt-style natural-language query formulation (binary classification under the hood)
- [ ] Commit: `feat(tasks): WSCTask with natural-language query formulation`

## Task 7: MultipleChoiceTask (COPA)

**Files:**
- Create: `eval/src/balkanbench/tasks/multiple_choice.py`
- Create: `eval/tests/unit/test_task_multiple_choice.py`

- [ ] Uses `AutoModelForMultipleChoice` model family via the task's declared `num_choices`
- [ ] Prompt-driven (COPA cause/effect prompts from the per-language YAML)
- [ ] `decode(logits) -> choice index`
- [ ] Commit: `feat(tasks): MultipleChoiceTask for COPA`

## Task 8: MultiRCTask (grouped binary)

**Files:**
- Create: `eval/src/balkanbench/tasks/multirc.py`
- Create: `eval/tests/unit/test_task_multirc.py`

- [ ] `task_type: grouped_binary_classification`
- [ ] `preprocess` keeps `group_id`, `candidate_id` columns as metadata (no positional state)
- [ ] Scoring: `f1_a` on candidate-level, `exact_match` on grouped (paragraph_id, question_id) predictions
- [ ] Mandatory regression fixture test: hand-built predictions + gold, assert expected F1 and EM
- [ ] Commit: `feat(tasks): MultiRCTask with grouped metric`

## Task 9: DiagnosticTask (AXb, AXg)

**Files:**
- Create: `eval/src/balkanbench/tasks/diagnostic.py`
- Create: `eval/tests/unit/test_task_diagnostic.py`

- [ ] Inherits from `Task`, inference-only (uses an RTE checkpoint), inherits RTE preprocess/decode
- [ ] Diagnostic sanity gate: AXb `matthews_correlation` cannot be more than 3σ below 0; AXg `accuracy` cannot be more than 3σ below 0.5. Violations raise `DiagnosticBelowRandomError`.
- [ ] Commit: `feat(tasks): DiagnosticTask with below-random sanity gate`

## Task 10: HF encoder wrapper

**Files:**
- Create: `eval/src/balkanbench/models/__init__.py`
- Create: `eval/src/balkanbench/models/hf_encoder.py`
- Create: `eval/tests/unit/test_hf_encoder.py`

- [ ] Single class `HFEncoder` that takes a `ModelConfig` + `TaskConfig` and returns a HF model via `AutoModelForSequenceClassification` or `AutoModelForMultipleChoice`
- [ ] Attention impl feature detection (`flash_attention_2` > `sdpa` > default)
- [ ] `fp16` from config
- [ ] Tests monkeypatch `AutoModel*.from_pretrained` - no real download
- [ ] Commit: `feat(models): HF encoder wrapper with attn-impl detection`

## Task 11: Seed + provenance utilities

**Files:**
- Create: `eval/src/balkanbench/seed.py`
- Create: `eval/src/balkanbench/provenance.py`
- Create: `eval/tests/unit/test_seed.py`
- Create: `eval/tests/unit/test_provenance.py`

- [ ] `set_seed(seed)` sets python, numpy, torch, transformers seeds
- [ ] `collect_provenance() -> dict`: git sha, package versions, image digest (env var), torch + cuda versions, python version
- [ ] Commit: `feat(core): seed + provenance helpers`

## Task 12: Evaluator

**Files:**
- Create: `eval/src/balkanbench/evaluation/__init__.py`
- Create: `eval/src/balkanbench/evaluation/evaluator.py`
- Create: `eval/tests/unit/test_evaluator.py`

- [ ] `run_single_seed(task, model_config, task_config, seed) -> SeedResult`: train + evaluate via plain HF `Trainer`
- [ ] `run_multiseed(task, model_config, task_config, seeds) -> list[SeedResult]`
- [ ] `aggregate(seed_results) -> Aggregate` with mean + stdev per primary metric
- [ ] Tests monkeypatch `Trainer.train`, `Trainer.predict` to return canned outputs; assert aggregation shape
- [ ] Commit: `feat(evaluation): single-seed + multi-seed evaluator`

## Task 13: Result artifact writer

**Files:**
- Create: `eval/src/balkanbench/scoring/__init__.py`
- Create: `eval/src/balkanbench/scoring/artifact.py`
- Create: `eval/tests/unit/test_artifact.py`

- [ ] `write_result_artifact(...) -> Path`: assembles seed results + aggregate + provenance + config hash + dataset revision into a schema-valid artifact matching `schemas/result_artifact.json`
- [ ] `test_predictions_hash`: sha256 of a canonical JSONL serialisation of predictions
- [ ] Commit: `feat(scoring): result artifact writer with provenance + predictions hash`

## Task 14: `balkanbench eval` CLI

**Files:**
- Modify: `eval/src/balkanbench/cli/main.py`
- Create: `eval/src/balkanbench/cli/eval.py`
- Create: `eval/tests/unit/test_cli_eval.py`

- [ ] `--model`, `--benchmark`, `--language`, `--task`, `--seeds`, `--out`
- [ ] Loads configs, runs evaluator, writes artifact
- [ ] Tests monkeypatch the evaluator
- [ ] Commit: `feat(cli): add eval subcommand`

## Task 15: `balkanbench predict` CLI

**Files:**
- Modify: `eval/src/balkanbench/cli/main.py`
- Create: `eval/src/balkanbench/cli/predict.py`
- Create: `eval/tests/unit/test_cli_predict.py`

- [ ] `--model`, `--benchmark`, `--language`, `--task`, `--out`
- [ ] Runs prediction on public test split, emits `predictions.jsonl` + `run_metadata.json`
- [ ] No label access required (private token is not used)
- [ ] Commit: `feat(cli): add predict subcommand`

## Task 16: Concrete SuperGLUE task configs

**Files:**
- Create: `eval/configs/benchmarks/superglue/benchmark.yaml`
- Create: `eval/configs/benchmarks/superglue/tasks/{boolq,cb,copa,rte,multirc,wsc}.yaml`
- Create: `eval/configs/benchmarks/superglue/prompts/copa/sr.yaml`
- Create: `eval/configs/benchmarks/superglue/prompts/wsc/sr.yaml`
- Create: `eval/configs/models/official/bertic.yaml`

- [ ] Author real YAMLs. Every file validates against the corresponding schema.
- [ ] `balkanbench list benchmarks` returns `superglue`; `list tasks` returns all 6.
- [ ] Commit: `feat(configs): author SuperGLUE tasks + BERTić model config`

## End of Plan 3

Success state:
- `balkanbench eval --model bertic --benchmark superglue --language sr --task boolq --seeds 1` runs end-to-end on a tiny fixture model in CI
- `balkanbench predict ...` produces `predictions.jsonl`
- All 6 tasks + 2 diagnostics implemented
- Coverage still ≥ 80%
- Plan 4 (HP search + scoring + leaderboard export) is next
