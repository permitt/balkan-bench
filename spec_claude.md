# BalkanBench — Clean-Slate Specification

A design document for rebuilding the South-Slavic SuperGLUE benchmark as a lean,
reproducible, locally-runnable open-source project. This specification is
informed by the audit in `VERIFICATION.md` / `VERIFICATION_CLAUDE.md` — every
design choice here is traceable to a problem we actually hit in the v6/v10
codebase.

Author: Claude. Scope: greenfield v1.0. Target repo name: `balkanbench`.

---

## 1. Goals

G1. **Runs on a single 24 GB GPU** (RTX 4090 / L4 / A5000) with no cloud
    required. Cloud is an optional runner, not a prerequisite.

G2. **BCMS from day one**: Serbian, Croatian, Montenegrin. Language is a
    first-class dimension, not a suffix hack.

G3. **One canonical number per (model, task, language)**. No parallel universe
    of sweep vs final vs smoke. One config, one result.

G4. **Every result is reproducible by string**: provenance (git SHA, dataset
    rev, image digest, seed set) is part of the result file schema, not an
    afterthought.

G5. **Official SuperGLUE metric contracts** for every task. If a task's metric
    diverges from official SuperGLUE, the divergence is documented and the
    task is labeled `superglue_variant: true`.

G6. **Public leaderboard with community submissions**: one JSON schema in, one
    ranked table out. Submission is a PR with a verified JSON artifact.

G7. **Green test suite, green CI, from commit one**.

## 2. Non-goals

- Not a training library. We fine-tune off-the-shelf encoders via HF Trainer;
  we do not reimplement training.
- Not a pretraining benchmark. Encoder fine-tuning + evaluation only.
- Not a generative-model benchmark. LLMs with generation-based scoring are v2
  scope (different eval contract).
- Not a multi-node distributed evaluator. Single-GPU or single-host-multi-GPU
  is the scope; multi-node is not a design constraint.
- Not a hyperparameter search tool. We will *consume* HP-search outputs (as
  task-level YAML files) but the search itself is an adjunct script, not a
  first-class module.

## 3. Design principles (= lessons from the audit)

P1. **One source of truth**. Every piece of semantic information lives in
    exactly one place. Primary metrics in the task YAML. Language routing in
    the dataset schema. Training config in the model YAML merged with task
    YAML. No hidden dicts in Python that shadow YAML.

P2. **Fail loudly, never silently**. Missing test split → error. Eval split
    metadata unavailable → error. Incomplete task coverage in a leaderboard
    submission → reject. No more "falls back to binary accuracy when metadata
    is missing" paths.

P3. **No positional contracts**. If a metric needs metadata per example, that
    metadata travels *with the example* through the dataset, not via a
    class-level list matched by position.

P4. **No custom Trainer subclasses unless required**. If a feature can be
    achieved via HF Trainer's public API, use it. `WeightedBCETrainer`,
    `DebugTrainer`, etc., all go away. Debug logging via HF callbacks, not
    subclasses.

P5. **Prompts are data**. Every prompt, every connective, every natural-language
    template lives in a per-language YAML. Python code renders, it doesn't
    author.

P6. **One task = one file**. A task YAML declares data paths, metric names,
    prompts per language, and training overrides. Python implementation for
    non-generic tasks lives next to the YAML, not in a separate `custom/`
    subtree.

P7. **Results are immutable artifacts**. A result JSON is fully self-describing.
    Deleting the repo and only keeping the JSON should let anyone reproduce
    the provenance string.

P8. **Local CLI is the primary interface**. Cloud runner is a thin wrapper that
    shells out to the same CLI.

## 4. Scope decisions for v1.0

### 4.1 Tasks (v1.0 task roster)

| Task    | Status | Primary metrics           | Notes |
|---------|--------|---------------------------|-------|
| BoolQ   | in     | accuracy                  | sentence-pair classification |
| CB      | in     | accuracy, f1_macro        | 3-class NLI |
| RTE     | in     | accuracy                  | binary NLI |
| COPA    | in     | accuracy                  | multiple-choice, canonical prompt |
| WSC    | in     | accuracy                  | sentence-pair reformulation |
| MultiRC | in     | f1_a, exact_match         | BCE per candidate, question-level EM |
| ReCoRD  | **cut for v1.0** | —               | metric bug in legacy repo; ship in v1.1 |
| AXb     | in     | matthews_correlation      | diagnostic, test-only |
| AXg     | in     | accuracy, gender_parity   | diagnostic, test-only |

ReCoRD is cut because the legacy repo pins all models to EM ≈ 0.17 regardless
of architecture — indicates a metric/indexing bug. Shipping a task where every
model scores identically within 0.01 is worse than not shipping it at all.
Fix the bug in v1.1 and add ReCoRD as a leaderboard column with `since_v1_1: true`.

### 4.2 Languages

- `sr` (Serbian, Cyrillic + Latin)
- `hr` (Croatian, Latin)
- `cnr` (Montenegrin, Latin)

Each task is declared *per language*. A model's benchmark score is reported
per language. Cross-language macro-averages are an optional column, not the
primary rank.

If only `sr` data exists at v1.0 release, the README states that explicitly
and `hr`/`cnr` ship as empty-config stubs with a "data in v1.1" label. No
pretending.

### 4.3 Models (seed baselines for the v1.0 leaderboard)

Curated to cover architecture × size × pretraining-data diversity:

| Model config         | HF repo                            | ~params | why |
|----------------------|------------------------------------|---------|-----|
| `bertic`             | `classla/bcms-bertic`              | 110M    | Serbian-specialised ELECTRA baseline |
| `xlm_r_bertic`       | `classla/xlm-r-bertic`             | 278M    | multilingual XLM-R adapted for BCMS |
| `mmbert`             | `jhu-clsp/mmBERT-base`             | 110M    | modern multilingual baseline |
| `crosloengual_bert`  | `EMBEDDIA/crosloengual-bert`       | 110M    | hr/sl/en baseline |
| `modern_bertic`      | `permitt/galton-modernbertic-65B-bcms-v1` | 150M | project's ModernBERT |
| `modern_bertic_large`| user-owned large variant           | 350M    | project's large model |

Community PRs can add more. The v1.0 launch ships with these six plus a
`baseline/random` config that emits uniform predictions for sanity checking.

## 5. Repository layout

```
balkanbench/
├── pyproject.toml                 # uv-managed, pinned torch extras
├── uv.lock
├── Dockerfile                     # reproducible image, digest pinned in CI
├── .github/workflows/
│   ├── test.yml                   # pytest -m "not requires_gpu" on every PR
│   ├── schema.yml                 # validate submitted JSON against result schema
│   └── release.yml                # tag → PyPI + GHCR image
├── README.md                      # user-facing; install, run, interpret
├── CONTRIBUTING.md                # add model, add task, submit a run
├── LICENSE                        # Apache-2.0
├── CITATION.cff
├── CODE_OF_CONDUCT.md
│
├── balkanbench/                   # the package
│   ├── __init__.py
│   ├── cli.py                     # typer-based CLI; single entrypoint
│   ├── registry.py                # task + model auto-discovery
│   ├── config.py                  # YAML loader with _include, per-language overrides
│   ├── provenance.py              # captures git SHA, image, revs, envs
│   ├── seed.py                    # set_seed, deterministic-mode helpers
│   │
│   ├── tasks/
│   │   ├── base.py                # Task ABC
│   │   ├── builder.py             # factory: config → Task
│   │   ├── classification.py      # generic sentence-pair / single-seq task
│   │   ├── multiple_choice.py     # COPA-style multi-choice task
│   │   ├── multirc.py             # candidate-binary + question-EM
│   │   ├── wsc.py                 # natural-language-query reformulation
│   │   └── diagnostic.py          # AXb, AXg (checkpoint-based eval)
│   │
│   ├── models/
│   │   ├── base.py                # Model ABC
│   │   └── hf_encoder.py          # HF AutoModel* wrapper, no custom Trainer
│   │
│   ├── metrics/
│   │   ├── __init__.py            # load(name) → Callable
│   │   ├── accuracy.py
│   │   ├── f1.py
│   │   ├── matthews.py
│   │   ├── gender_parity.py       # AXg-specific
│   │   └── bootstrap.py           # prediction-level bootstrap CI
│   │
│   ├── evaluator.py               # run_single, run_task_multiseed, run_benchmark
│   ├── aggregator.py              # results → leaderboard score
│   └── schema.py                  # pydantic models for result JSON
│
├── configs/
│   ├── benchmark.yaml             # canonical run spec (models, tasks, seeds)
│   ├── base/
│   │   └── encoder_defaults.yaml
│   ├── tasks/
│   │   ├── boolq.yaml
│   │   ├── cb.yaml
│   │   ├── copa.yaml
│   │   ├── rte.yaml
│   │   ├── wsc.yaml
│   │   ├── multirc.yaml
│   │   ├── axb.yaml
│   │   └── axg.yaml
│   ├── prompts/                   # per-task per-language prompt bundles
│   │   ├── copa/{sr,hr,cnr}.yaml
│   │   ├── wsc/{sr,hr,cnr}.yaml
│   │   └── ...
│   └── models/
│       ├── bertic.yaml
│       ├── xlm_r_bertic.yaml
│       ├── mmbert.yaml
│       ├── crosloengual_bert.yaml
│       ├── modern_bertic.yaml
│       └── baseline_random.yaml
│
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   ├── test_config.py
│   │   ├── test_registry.py
│   │   ├── test_metrics.py        # includes gender parity, bootstrap
│   │   ├── test_tasks.py
│   │   └── test_aggregator.py
│   ├── integration/
│   │   ├── test_evaluator_smoke.py  # tiny model, tiny data, CPU
│   │   └── test_submission_schema.py
│   └── fixtures/
│       └── mini_dataset/          # 8 rows per split, shipped in repo
│
├── scripts/
│   ├── hp_search.py               # Optuna sweep, writes best configs back as YAML
│   └── download_checkpoints.py    # convenience; uses HF Hub auth
│
├── cloud/                         # optional GCP runner
│   ├── Dockerfile                 # same base as root Dockerfile
│   ├── launch_jobs.py             # Cloud Batch wrapper around `balkanbench run`
│   └── collect_results.py
│
├── leaderboard/
│   ├── schema.json                # result JSON contract
│   ├── render.py                  # results/ → static site payload
│   ├── submissions/               # accepted community submissions, PR-reviewed
│   └── site/                      # minimal HTML/JS for Balkanbench.com
│
└── docs/
    ├── index.md                   # mkdocs site root
    ├── tasks.md                   # each task, formulation, metric, citation
    ├── models.md
    ├── adding-a-model.md
    ├── adding-a-task.md
    ├── adding-a-language.md
    ├── submission-process.md
    └── faq.md
```

Notable differences from the legacy repo:

- No `main.py`, no `CLAUDE.md` in repo root, no `combined_labels.zip` artifacts.
- `tasks/custom/` is gone; task classes live at the top level of `tasks/`.
- Single `configs/tasks/copa.yaml` (not three).
- `configs/prompts/<task>/<lang>.yaml` instead of hard-coded Python strings.
- Single `cloud/` entry point that wraps the CLI; no parallel cloud logic.

## 6. Task contract

### 6.1 Task YAML schema

```yaml
# configs/tasks/copa.yaml
task: copa
superglue_variant: false
description: "Choice of Plausible Alternatives"
citation: "Gordon et al. 2012 / Wang et al. 2019"

data:
  source: "huggingface:balkanbench/bcms-superglue"
  subset: copa
  splits:
    train:   train
    dev:     dev          # used for early stopping + best-checkpoint selection
    test:    test         # used for reported scores
  languages: [sr, hr, cnr]   # which language-columns this task supports

task_type: multiple_choice
num_choices: 2
label_field: label

tokenizer:
  max_length: 256
  padding: longest
  truncation: true

prompts:
  # path to per-language prompt bundle — rendered at preprocess time
  include: prompts/copa

metrics:
  - name: accuracy
    primary: true
    # CI computed via bootstrap over test predictions, not seeds
    bootstrap:
      iters: 1000
      confidence: 0.95

training:
  # canonical training recipe for v1.0; overridable per-model in model YAML
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 10
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping_patience: 5
  metric_for_best_model: accuracy
```

### 6.2 Per-language prompt bundle

```yaml
# configs/prompts/copa/sr.yaml
language: sr
cause_prompt: "{{premise}} Šta je bio uzrok?"
effect_prompt: "{{premise}} Šta se desilo kao rezultat?"
```

```yaml
# configs/prompts/copa/hr.yaml
language: hr
cause_prompt: "{{premise}} Što je bio uzrok?"
effect_prompt: "{{premise}} Što se dogodilo kao rezultat?"
```

Adding a language = adding a prompt file. No Python changes.

### 6.3 Task Python base

```python
# balkanbench/tasks/base.py
class Task(ABC):
    config: TaskConfig        # pydantic-validated
    language: str             # the active language

    @abstractmethod
    def load_data(self) -> DatasetDict: ...

    @abstractmethod
    def preprocess(self, examples: dict, tokenizer) -> dict: ...

    @abstractmethod
    def compute_metrics(self, eval_pred: EvalPrediction) -> dict[str, float]: ...

    # concrete
    def primary_metric_names(self) -> list[str]:
        return [m.name for m in self.config.metrics if m.primary]
```

Key rule: `compute_metrics` consumes everything it needs from `eval_pred`
(which now includes `inputs` via `include_inputs_for_metrics=True` in
`TrainingArguments`) or from per-example columns padded into the tokenised
dataset as `label`-siblings. No class-level positional state. This kills the
MultiRC / ReCoRD positional group_id contract by design.

### 6.4 Task type decisions

| Task    | Python class              | Loss / problem_type                  |
|---------|---------------------------|--------------------------------------|
| BoolQ   | `ClassificationTask`      | 2-class CE (`num_labels=2`)          |
| CB      | `ClassificationTask`      | 3-class CE (`num_labels=3`)          |
| RTE     | `ClassificationTask`      | 2-class CE                            |
| COPA    | `MultipleChoiceTask`      | `AutoModelForMultipleChoice`         |
| WSC    | `WSCTask`                 | 2-class CE on natural-language query |
| MultiRC | `MultiRCTask`             | **BCE** (`num_labels=1`, `problem_type=multi_label_classification`, weights enabled) |
| AXb/AXg | `DiagnosticTask`          | inference-only; inherits RTE checkpoint |

MultiRC moves to BCE in v1.0. The legacy MSE-regression workaround is
retired; no "worked for bertic" shortcuts. This is a potential accuracy
change; the launch release notes will document the diff from legacy numbers.

## 7. Model contract

### 7.1 Model YAML schema

```yaml
# configs/models/xlm_r_bertic.yaml
name: xlm_r_bertic
hf_repo: classla/xlm-r-bertic
params_hint: 278M
family: xlm-roberta

# optional auth (private repos)
hf_auth:
  required: false

# Model-wide defaults (merge over task defaults)
training:
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 10
  fp16: true

# Per-task overrides
task_overrides:
  cb:
    num_epochs: 30
    batch_size: 16
  wsc:
    num_epochs: 30
    learning_rate: 1e-5
```

### 7.2 What the model class does

- Load via `AutoModelForSequenceClassification` / `AutoModelForMultipleChoice`
- Apply fine-tuning dropout from config (default 0.1; pretrained models with
  0.0 are a footgun we hit in the legacy code)
- Pick `attn_implementation` via feature detection
  (`flash_attention_2` → `sdpa` → default)
- Return a plain HF model. Training goes through plain `Trainer`. No custom
  subclasses.

### 7.3 Dropping the legacy model-specific hacks

- No `te-sla/teslaXLM` manual pytorch_model.bin workaround in the baseline.
  If the model ships a broken checkpoint, that's a data issue; contributors
  can write an optional loader plugin.
- No manual pooler copying for XLM-R multiple-choice. We rely on the
  library's `AutoModelForMultipleChoice`; if mean-pooling is empirically
  better, we do it via a wrapper task class (COPA) that computes pooling
  explicitly, not by subclassing MC.

## 8. Metrics contract

### 8.1 Primary metric lookup

One function, one source:

```python
# balkanbench/metrics/__init__.py
def primary_metrics_for_task(task_config: TaskConfig) -> list[str]:
    return [m.name for m in task_config.metrics if m.primary]
```

Used by: evaluator (for `metric_for_best_model`), aggregator (for benchmark
score), HP search (for Optuna objective), leaderboard renderer (for
highlighted column). **Never** read from a separate dict.

### 8.2 Confidence intervals — prediction-level bootstrap

On the test split:
1. Evaluate best checkpoint, get `predictions` and `labels` arrays.
2. For each of `N_bootstrap=1000` iterations, resample indices with
   replacement, compute the task's primary metric on the resample.
3. Report `{mean, std, ci_lower, ci_upper}` from the bootstrap distribution.

Multi-seed runs report CI over *seeds* as a secondary "seed variance" stat,
not as the primary CI. Seed variance answers a different question than test-
set variance; we report both, label them distinctly, and never confuse them.

Three-seed percentile bootstrap over seed scores — the failure mode in the
legacy code — is explicitly deprecated.

### 8.3 Benchmark score

Per-language score = macro-average of primary metrics across tasks declared
for that language.

Overall score = macro-average of language-specific scores.

**Hard rule**: a submission must include every task declared in the
benchmark config for every language it claims, or it is rejected at
validation. No partial-coverage inflation.

### 8.4 New metrics

- `gender_parity`: per AXg contract —
  `accuracy(pro-stereotype) − accuracy(anti-stereotype)`. Lower absolute
  value is better. A secondary `accuracy` is also reported.
- `exact_match` for MultiRC — computed by grouping predictions by
  `(paragraph_id, question_id)` read from the dataset columns, not by
  position.

## 9. Data & dataset contract

### 9.1 Canonical HF dataset

- Repo: `balkanbench/bcms-superglue` (new public dataset, CC-BY 4.0).
- Each task is a subset: `balkanbench/bcms-superglue@copa`, etc.
- Each example has a `language` column (`sr`, `hr`, or `cnr`) and
  language-specific text fields (`premise`, `choice1`, …) — *no* `_srb`
  suffixes.
- Splits: `train`, `dev` (small, for early stopping), `test`. All three
  have gold labels for the v1 release. If a hidden-test-set pattern is
  adopted later, the dataset schema grows a `test_private` subset loaded
  via a submission API; the repo does not need that in v1.

### 9.2 Legacy dataset migration

- One-time migration script: `scripts/migrate_legacy_dataset.py` reads the
  legacy `permitt/superglue` repo, strips `_srb` suffixes, adds
  `language: sr`, writes to the new repo. Migration script is archived
  under `tools/` after migration; it does not ship in the published repo.

### 9.3 Dataset revision discipline

- Every dataset release has a git tag (e.g. `v1.0.0-data`). Result JSONs
  include the resolved commit SHA.
- Changing a test label = new major dataset version. No silent edits to
  test splits.

## 10. Evaluation protocol

### 10.1 One seed

1. Set seed.
2. Load model, attach task head.
3. Train on `train` split, early-stop on `dev` split using
   `metric_for_best_model = primary[0]`.
4. Load best checkpoint.
5. Evaluate on `test` split.
6. Compute bootstrap CI over test predictions.
7. Return: `{metrics, provenance, test_predictions_hash}`.

### 10.2 Multiple seeds

Default `num_seeds = 5` for stable models; `num_seeds = 10` for small tasks
(CB, WSC). Report mean, std, and seed spread alongside the prediction-level
CI.

### 10.3 Diagnostic tasks

AXb, AXg are test-only. Load the best RTE checkpoint for the given model
and language, evaluate. Diagnostic scores do not contribute to the benchmark
score but are reported in a dedicated diagnostics column.

**Fix the legacy bug**: the AXb/AXg diagnostic path in v10 produced
below-random MCC/accuracy. v1 must include a sanity test:
`test_diagnostic_not_below_random` — if any model-language diagnostic comes
back more than 3σ below chance, the evaluator raises a loud warning and
refuses to write a success result.

## 11. CLI UX

Single entrypoint, `balkanbench`, backed by typer:

```bash
# Install
pip install balkanbench

# Discover available models / tasks
balkanbench list models
balkanbench list tasks
balkanbench list languages

# Run one model on one task, one language, one seed (smoke)
balkanbench run \
  --model xlm_r_bertic \
  --task copa \
  --language sr \
  --seeds 1 \
  --output results/smoke/

# Run the canonical benchmark
balkanbench run --config configs/benchmark.yaml --output results/v1/

# Evaluate a checkpoint you trained elsewhere
balkanbench evaluate \
  --model xlm_r_bertic \
  --checkpoint path/to/ckpt \
  --task copa --language sr

# Produce a leaderboard-ready submission JSON from a results directory
balkanbench submit results/v1/ --out submission.json

# Validate a submission against the schema
balkanbench validate submission.json

# Render the leaderboard static site
balkanbench leaderboard render leaderboard/submissions/ --out leaderboard/site/
```

All commands are thin wrappers around library APIs. Anything a user does via
the CLI, they can do by importing `balkanbench` in Python.

## 12. Result JSON schema (pydantic → JSON Schema)

```json
{
  "$schema": "https://balkanbench.com/schema/v1.json",
  "submission_id": "uuid-v4",
  "submitter": { "name": "...", "email": "...", "affiliation": "..." },
  "model": {
    "name": "xlm_r_bertic",
    "hf_repo": "classla/xlm-r-bertic",
    "hf_revision": "abc123...",
    "params": 278_000_000
  },
  "benchmark": {
    "name": "balkanbench",
    "version": "1.0.0",
    "task_set_hash": "sha256:..."
  },
  "provenance": {
    "package_version": "1.0.0",
    "git_sha": "deadbeef...",
    "image_digest": "sha256:... (optional)",
    "python_version": "3.11.9",
    "torch_version": "2.5.1+cu121",
    "transformers_version": "4.45.0",
    "dataset_revision": "v1.0.0-data",
    "hardware": { "gpu": "NVIDIA L4", "vram_gb": 24, "cuda": "12.1" },
    "seeds": [42, 43, 44, 45, 46]
  },
  "languages": {
    "sr": {
      "tasks": {
        "copa": {
          "primary_metric": "accuracy",
          "test": {
            "mean": 0.753,
            "ci_lower": 0.731,
            "ci_upper": 0.771,
            "ci_method": "bootstrap_1000_test",
            "seed_std": 0.027,
            "per_seed": [0.74, 0.76, 0.75, 0.77, 0.75]
          },
          "secondary_metrics": {}
        },
        "// ... every declared task ...": {}
      },
      "diagnostics": {
        "axb": { "matthews_correlation": 0.22 },
        "axg": { "accuracy": 0.64, "gender_parity": 0.03 }
      },
      "language_score": 0.704
    }
  },
  "overall_score": 0.704
}
```

Validation rules enforced by `balkanbench validate`:

- Every language in `languages` must cover every task in `benchmark.task_set_hash`.
- Primary metric names must match the task YAML contract.
- Provenance fields are all required, none nullable.
- CIs must be consistent with means (lower ≤ mean ≤ upper).
- If diagnostics are missing and the benchmark declares them, `validate` fails.

## 13. Reproducibility contract

R1. Every result JSON carries enough provenance to rerun it.

R2. Published `balkanbench:v1.0.0` docker image is pinned by digest in the
    release notes and in the result JSON.

R3. `uv.lock` tracks every Python dep. Torch CUDA wheel URL is pinned.

R4. Seeds are explicit in the result JSON; default is
    `[42, 43, 44, 45, 46]`.

R5. Disable fp16 for the canonical reference run (one run per model at
    launch), report fp16 vs fp32 delta as a separate column. Community
    submissions default to fp16 with a "runtime_mode: fp16" field.

R6. Byte-identical reproducibility of the reference run is a stretch goal,
    not a v1 requirement. We target reproducibility up to bootstrap CI
    overlap.

## 14. Testing strategy

### 14.1 Tests that must be green at merge

Unit (run on CPU, < 30 s total):
- Config loader merges `_include` correctly; later keys override earlier.
- Registry discovers every YAML in `configs/tasks/` and `configs/models/`.
- Primary-metric lookup returns only metrics with `primary: true`.
- Bootstrap CI computes reasonable intervals on synthetic data.
- Aggregator rejects incomplete task coverage.
- Gender parity metric behaves correctly on hand-crafted pro/anti pairs.
- Each task class `preprocess` produces expected shapes on `fixtures/mini_dataset`.

Integration (CPU, < 2 min total):
- Smoke run: random-baseline model + mini dataset + 1 seed → valid result JSON.
- `balkanbench validate` rejects a tampered JSON (missing task, wrong CI,
  missing provenance).
- `balkanbench submit` round-trips: run → submit → validate → render.

GPU (marked `requires_gpu`, optional in CI):
- 1-epoch fine-tune of a tiny HF model (e.g. `prajjwal1/bert-tiny`) on the
  mini dataset, asserts non-zero training and sensible metrics.

### 14.2 CI gates

- `test.yml`: `pytest -m "not requires_gpu"` on every PR. Hard requirement.
- `schema.yml`: any PR touching `configs/tasks/*.yaml` revalidates all task
  configs against the pydantic schema.
- `release.yml`: on git tag, build + push docker image, publish to PyPI,
  attach result JSON schema to the GitHub release.

## 15. HP search (adjunct, not core)

- Lives in `scripts/hp_search.py`. Not part of the library.
- Optuna study is seeded (`TPESampler(seed=42)`) and persisted
  (`optuna.storages.RDBStorage("sqlite:///hpsearch.db")`).
- Objective metric is looked up from task YAML primary list (§8.1).
- Search space is declared in a per-task HP-search YAML:
  `configs/hpsearch/copa.yaml`.
- Best params are written back as `configs/models/<model>_sweep.yaml` with
  the sweep's dataset revision and commit SHA in the header — so a future
  reader can see which sweep generated this config.
- HP-search outputs live in `results/hpsearch/<sweep_id>/`, never in the
  main `results/` tree. Namespaces are not shared.

## 16. Leaderboard

### 16.1 Submission flow

1. User runs `balkanbench run` on their machine or cloud.
2. User runs `balkanbench submit results/... --out submission.json`.
3. User opens a PR to `balkanbench` repo with the JSON at
   `leaderboard/submissions/<model>-<date>.json`.
4. CI runs `balkanbench validate` on the JSON. Pass → mergeable.
5. Maintainer reviews submitter identity (no fake affiliations).
6. On merge, `balkanbench leaderboard render` rebuilds the static site.

No auto-upload, no submission API for v1. Lower operational cost, higher
trust.

### 16.2 Published pages

- `/` — top-level leaderboard, sortable by overall score, per-language
  scores, each task column.
- `/model/<name>` — model card: all submissions, per-seed spreads,
  provenance table.
- `/task/<name>` — task card: data stats, metric formula, SuperGLUE
  reference, all model scores.
- `/language/<code>` — language-sliced leaderboard.
- `/docs` — mkdocs-built user guide.

### 16.3 What is *not* on the leaderboard

- Below-chance submissions (e.g. tesla_xlm COPA 0.494 in legacy). CI
  rejects any task score below `chance − 3σ`.
- Submissions without diagnostics (if benchmark declares them).
- Submissions where a task delta between seeds exceeds a threshold
  (configurable; default 3× median task seed std across accepted
  submissions). High-variance submissions show up in a separate "in
  review" section until investigated.

## 17. Build plan — step-by-step roadmap for the new repo

A realistic solo-developer timeline. Each step is a self-contained commit
or small PR.

### Week 1 — skeleton and data
- D1. `uv init balkanbench`; add pyproject with pinned torch + transformers.
- D1. Write the result JSON pydantic schema; add `balkanbench validate`
      round-trip test with one canned-good and two canned-bad fixtures.
- D2. Port the legacy ConfigLoader + Registry (cleaned, typed) to the new
      package; re-run their unit tests.
- D3. Write the Task ABC + ClassificationTask + MultipleChoiceTask +
      MultiRCTask (with BCE) + WSCTask + DiagnosticTask. Unit tests on
      mini_dataset for each.
- D4. Write the HFEncoderBackend. No custom Trainer subclasses. Gate on
      `prajjwal1/bert-tiny` end-to-end smoke test in CI (CPU).
- D5. Run the legacy dataset migration script; publish
      `balkanbench/bcms-superglue@v1.0.0-data` (Serbian-only for now).

### Week 2 — metrics, evaluator, CLI
- D6. Implement `metrics/bootstrap.py`. Test on synthetic predictions.
- D7. Implement `metrics/gender_parity.py`, `metrics/matthews.py`.
- D8. Implement `evaluator.py`: run_single_seed, run_multiseed, run_task.
- D9. Implement `aggregator.py` with mandatory complete-coverage rule.
      Add failing-coverage test.
- D10. Implement the typer CLI; full smoke of `list`, `run`, `submit`,
      `validate` commands.

### Week 3 — first canonical run
- D11. `configs/models/baseline_random.yaml`. Run the whole benchmark with
      it. Confirms the pipeline produces a valid JSON end-to-end for a
      free submission.
- D12. `configs/models/xlm_r_bertic.yaml`. Run the canonical v1.0 benchmark
      on a single 24 GB GPU. Fix whatever breaks.
- D13. `configs/models/bertic.yaml`, `mmbert.yaml`, `crosloengual_bert.yaml`.
      Run one at a time. Commit each result JSON.
- D14. `modern_bertic.yaml` and `modern_bertic_large.yaml` (private repos
      require `hf_auth`). Run.
- D15. Diagnostic sanity check: no below-random scores. If the legacy AXb
      below-random bug repeats, fix it in-flight (likely a prompt or
      label-mapping issue for the new dataset).

### Week 4 — leaderboard and launch
- D16. Implement `leaderboard/render.py`. Produce the first static site.
- D17. Hook up GitHub Actions (`test.yml`, `schema.yml`, `release.yml`).
- D18. Write `docs/` (tasks, models, adding-a-model, adding-a-task,
       submission-process, faq).
- D19. Soft-launch: push tag `v1.0.0-rc1`, publish docker image, post in
       Serbian/Croatian NLP channels asking for a round of external
       reproducibility tests.
- D20. Address feedback. Tag `v1.0.0`. Announce on Balkanbench.com,
       HuggingFace forums, Twitter/Bluesky.

### Month 2 — v1.1
- ReCoRD bug fix + re-enable.
- hr / cnr data splits.
- Gender parity expanded metrics (not just accuracy delta).
- More community-submitted models.

## 18. Migration from legacy repo — pragmatic answer

The legacy repo has roughly 6 months of empirical knowledge baked in:
best learning rates per model, task-specific prompt engineering, known
OOM failure modes on T4, etc. Don't throw it away.

Concrete migration rules:

M1. **Task YAML** — rewrite from scratch using the schema in §6.1. Pick
    *one* recipe per task (the one that verifiably works locally).
    Discard the 20+ `v*` config variants.

M2. **Custom task Python** — rewrite `multirc.py`, `wsc.py`, `copa.py`
    (multiple_choice) using the no-positional-state rules (§6.3). Port
    the domain-knowledge bits (natural-language WSC query, COPA
    question-prompt) into the `configs/prompts/<task>/<lang>.yaml`
    files; keep Python generic.

M3. **ReCoRD** — do not port to v1.0. Revisit in v1.1 with a clean
    metric formulation. The legacy all-models-pinned-at-0.17 bug shows
    the current implementation is not debuggable by code-reading alone;
    rewriting is cheaper than debugging.

M4. **HF encoder backend** — port `finetune_dropout`, `classifier_pooling`
    config options, flash-attn detection. Skip `te-sla/teslaXLM` manual
    loading (that model can be excluded from v1.0, or shipped as a
    community plugin).

M5. **HP search script** — keep Optuna + TPE + median pruner. Add the
    seeded-sampler + persistent-study discipline. Rewrite the search-
    space declaration to read from `configs/hpsearch/<task>.yaml`
    instead of the hard-coded Python grid.

M6. **v6-final result numbers** — do *not* port. Re-run the canonical
    benchmark on the new codebase end-to-end; use v6 numbers as sanity
    checks, not as published results.

## 19. What lands in v2, explicitly

- Hidden-test-set / submission-API pattern for the held-out test split.
- Generative-model evaluator (decoder-only LLMs with zero-shot/few-shot
  scoring).
- Winogrande, SmartCAT-QA, and other optional tasks.
- Automated data-quality dashboards.
- Compute-budget-aware leaderboard columns (efficient inference cost).
- Cross-language transfer experiments (train on sr, test on hr / cnr).

## 20. Decision log

Design choices that are not obvious, recorded here so the next maintainer
can understand why.

- **MultiRC uses BCE, not MSE-regression.** Legacy chose MSE because it
  "worked for bertic"; that's not a reproducible argument. BCE is the
  standard and we commit to it even if it moves single-model numbers.
- **Per-prediction bootstrap CI, not per-seed.** Seed variance is a
  different question than test-set variance; we report both separately.
- **ReCoRD cut from v1.0.** Shipping a task where every model scores the
  same within 0.01 hurts credibility more than omitting the task.
- **No custom Trainer subclasses.** Debug logging is a callback. Weighted
  BCE is a `class_weights` kwarg to the loss function, not a subclass.
- **Prompts in YAML, not Python.** A contributor adding Croatian data
  should never need to touch Python.
- **One dataset repo per language set.** Legacy had `_srb` suffixes
  everywhere — that scales badly. A `language` column scales to any BCMS
  variant.
- **Local-first, cloud-optional.** Researchers with one GPU are the
  primary user. Cloud is a power-user feature that wraps the same CLI.
- **Complete-coverage hard rule for leaderboard.** Partial coverage
  inflated legacy rankings; we refuse to repeat that.

