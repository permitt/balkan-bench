---
title: BalkanBench v0.1 Design
status: approved-for-implementation
release-target: 2026-04-27
supersedes: SPECIFICATION.md (for v0.1 scope only)
---

# BalkanBench v0.1 Design

This document is the frozen design for the v0.1 public release. It supersedes `SPECIFICATION.md` where the two disagree; anything in `SPECIFICATION.md` not covered here is deferred to v0.2+.

## 1. Purpose

BalkanBench v0.1 is the first public, reproducible, auditable benchmark for Serbian language-model evaluation. It ships:

- an open-source Python framework (`balkanbench`) for running evaluations
- a public Hugging Face dataset with train + validation labeled and test unlabeled
- a private Hugging Face dataset holding hidden test labels
- an official leaderboard covering 6 SuperGLUE tasks and 9 models
- a React landing page + leaderboard UI deployed via Vercel
- a GitHub-issue-based contribution flow for community benchmarks, tasks, and submissions
- methodology, governance, data provenance, versioning, and GCP usage documentation

Compute for official evaluation is sponsored by Recrewty.

## 2. In scope for v0.1

### 2.1 Language
- Serbian (`sr`) only. The landing page may advertise `sr, me, hr, bs` as roadmap.

### 2.2 Benchmark
- SuperGLUE (Serbian adaptation)
- 6 ranked tasks: BoolQ, CB, COPA, RTE, MultiRC, WSC
- Optional diagnostics: AX-b, AX-g (included only if present in `permitt/superglue`)
- No ReCoRD in v0.1

### 2.3 Models (launch lineup)

9 models, already evaluated with 5 fixed seeds prior to this spec by the user's pre-v0.1 pipeline. v0.1 codifies that pipeline publicly; these results ship as the v0.1 official leaderboard.

| Rank | Model | Params |
|------|-------|--------|
| 1 | ModernBERTić base | 395M |
| 2 | BERTić | 110M |
| 3 | ModernBERTić small | 149M (5/6 partial) |
| 4 | mmBERT | 307M |
| 5 | XLM-R BERTić | 560M |
| 6 | CroSloEngual BERT | 124M |
| 7 | TeSLa-XLM | 560M |
| 8 | Jerteh-355 | 354M |
| 9 | mmBERT-small | 140M |

BERTić (`classla/bcms-bertic`) is the **local-dev reproducibility target**: the new framework must reproduce BERTić's row within stddev tolerance.

### 2.4 Data layout (Hugging Face)

- **Source** (already exists): `permitt/superglue`
- **Private test labels** (already exists): `permitt/superglue-private`, gated by `HF_OFFICIAL_TOKEN`
- **Public v0.1 release** (produced by this repo): `permitt/superglue-serbian`
- **v0.2+**: `permitt/superglue-croatian`, `permitt/superglue-montenegrin`, `permitt/superglue-bosnian`

### 2.5 Official protocol
- HP search on train → validation (Optuna)
- Freeze config, train final on train + validation, evaluate on hidden test
- 5 fixed seeds per final run, report per-seed + mean ± stdev
- Main score = unweighted mean of 6 primary task metrics

### 2.6 Execution
- Local: `balkanbench` CLI (installable via pip) and a pinned Docker image
- GCP: bash launcher scripts for A100 + L4 GPU VMs (manual trigger, not orchestrated)

## 3. Out of scope for v0.1 (deferred to v0.2+)

- Croatian, Montenegrin, Bosnian data
- Serbian-LLM-Eval (Gordić) suite; framework namespacing is ready but the adapter is not implemented
- MTEB-BCMS, LLM Arena
- Multi-job GCP orchestrator, automated submission-scoring server
- Prediction-level bootstrap confidence intervals (v0.1 reports mean ± stdev only)
- ReCoRD task
- Re-running the 9 launch models through the new public harness (deferred; prior-pipeline results ship as v0.1 official)

## 4. Repository layout

```
balkan-bench/
├── README.md
├── CONTRIBUTING.md
├── LICENSE                         # MIT
├── SECURITY.md
├── .gitignore
├── .github/
│   ├── ISSUE_TEMPLATE/
│   │   ├── propose-benchmark.yml
│   │   ├── propose-task.yml
│   │   ├── propose-model.yml
│   │   ├── submission.yml
│   │   └── bug.yml
│   └── workflows/
│       ├── ci.yml
│       ├── validate-configs.yml
│       ├── validate-fixtures.yml
│       ├── release-check.yml
│       └── repro-bertic.yml
├── frontend/                       # Vercel rootDirectory = this
│   ├── package.json
│   ├── vite.config.js
│   ├── vercel.json
│   ├── index.html
│   ├── public/
│   │   ├── favicon.svg
│   │   ├── recrewty-logo.png
│   │   └── leaderboards/
│   │       └── superglue-serbian/
│   │           └── benchmark_results.json
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── pages/
│       │   ├── Home.jsx
│       │   ├── Leaderboard.jsx
│       │   ├── About.jsx
│       │   └── Submit.jsx
│       ├── components/
│       └── styles/
├── eval/                           # Python package
│   ├── pyproject.toml
│   ├── README.md
│   ├── Dockerfile
│   ├── src/balkanbench/
│   │   ├── __init__.py
│   │   ├── cli/
│   │   ├── benchmarks/superglue/
│   │   ├── tasks/
│   │   ├── metrics/
│   │   ├── models/
│   │   ├── evaluation/
│   │   ├── scoring/
│   │   ├── validation/
│   │   ├── leaderboard/
│   │   ├── gcp/
│   │   └── utils/
│   ├── configs/
│   │   ├── benchmarks/superglue/
│   │   │   ├── benchmark.yaml
│   │   │   ├── tasks/{boolq,cb,copa,rte,multirc,wsc}.yaml
│   │   │   └── prompts/
│   │   ├── models/
│   │   │   ├── official/{bertic,modernbertic_base,modernbertic_small,mmbert,xlmr_bertic,crosloengualbert,tesla_xlm,jerteh_355,mmbert_small}.yaml
│   │   │   └── experimental/
│   │   └── environments/
│   ├── schemas/
│   │   ├── task_spec.json
│   │   ├── model_spec.json
│   │   ├── dataset_manifest.json
│   │   ├── result_artifact.json
│   │   ├── leaderboard_export.json
│   │   └── submission_metadata.json
│   ├── scripts/
│   │   ├── publish_dataset.py
│   │   ├── aggregate_results.py
│   │   ├── export_leaderboard.py
│   │   └── gcp/
│   │       ├── launch_a100.sh
│   │       ├── launch_l4.sh
│   │       └── common.sh
│   ├── tests/
│   │   ├── unit/
│   │   ├── integration/
│   │   ├── smoke/
│   │   └── fixtures/
│   └── results/
│       ├── local/                  # gitignored
│       ├── official/               # committed
│       └── submissions/            # gitignored
└── docs/
    ├── methodology/
    │   ├── benchmark_contract.md
    │   ├── data_provenance.md
    │   ├── versioning.md
    │   └── task_lifecycle.md
    ├── governance/
    │   ├── submissions.md
    │   └── contributions.md
    ├── leaderboard/format.md
    ├── gcp/
    │   ├── running_official_eval.md
    │   ├── costs.md
    │   └── security.md
    └── superpowers/specs/
        └── 2026-04-22-balkanbench-v0.1-design.md
```

## 5. Benchmark + task abstraction

### 5.1 Namespacing

Every task has a public identifier `{benchmark}.{task}.{language}`:
- v0.1: `superglue.boolq.sr`, `superglue.cb.sr`, `superglue.copa.sr`, `superglue.rte.sr`, `superglue.multirc.sr`, `superglue.wsc.sr`
- v0.2+: `sle.arc_challenge.sr`, `mteb_bcms.sts.hr`, etc.

### 5.2 Task spec YAML

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
  # canonical recipe for v0.1; overridable per-model in model YAML task_overrides
  learning_rate: 2e-5
  batch_size: 16
  num_epochs: 10
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping_patience: 5
  metric_for_best_model: accuracy
```

A JSON Schema at `eval/schemas/task_spec.json` enforces the shape. `balkanbench validate-config` runs it in CI. Training defaults live in the task YAML; model YAMLs override per-task via a `task_overrides` block instead of duplicating the full recipe.

### 5.3 Task implementation interface

Each task under `eval/src/balkanbench/tasks/{task}.py` provides:

- `load_split(split_name) -> Dataset`
- `preprocess(example) -> ModelInput`
- `decode(prediction) -> TaskOutput`
- `score(predictions, gold) -> MetricBundle`
- `score_from_private(predictions_path, gold_source) -> ScoredArtifact`

Tasks never hard-code Serbian prompts. All prompts are config-driven and language-keyed.

Metrics never rely on positional dataset state. Any metadata a metric needs (e.g. MultiRC `group_id`, `candidate_id`) travels with the example through the dataset columns, not through a class-level list indexed by position.

### 5.4 CLI

- Framework: `typer` (single entrypoint `balkanbench`)
- Anything the CLI does is also exposed via the `balkanbench` Python API
- Commands:
  - `balkanbench list {models,tasks,languages,benchmarks}` - discovery
  - `balkanbench validate-{env,config,data}` - validation
  - `balkanbench eval` - train + validation run, labels visible
  - `balkanbench predict` - public test inputs → `predictions.jsonl` + `run_metadata.json`
  - `balkanbench score` - predictions + private labels → scored artifact (requires `HF_OFFICIAL_TOKEN`)
  - `balkanbench hp-search` - Optuna on train → validation
  - `balkanbench throughput` - measure inference throughput on reference hardware (see §23)
  - `balkanbench leaderboard export` - aggregate official artifacts → `benchmark_results.json`
  - `balkanbench submit` - package a prediction run for leaderboard submission

## 6. Data pipeline

### 6.1 Normalization + publishing

`eval/scripts/publish_dataset.py`:

1. Authenticate via `HF_OFFICIAL_TOKEN`.
2. Download `permitt/superglue`, config by config.
3. **Rename COPA split `dev` → `validation`** (idempotent).
4. Validate schema against `eval/schemas/dataset_manifest.json`.
5. Strip label columns from `test` split.
6. Push to `permitt/superglue-serbian` with configs `boolq, cb, copa, rte, multirc, wsc` (and `ax_b, ax_g` if present in source).
7. Do not touch `permitt/superglue-private`; it already holds test labels.

The public dataset card (auto-generated by the script) states:
- test labels are hidden
- public users can generate predictions but cannot score test locally
- leaderboard scoring runs in an official environment
- license, source provenance, schema, version, revision hash

### 6.2 Dataset fields

Every row contains:
- `example_id` (string, unique within task)
- `task_id` (string, e.g. `superglue.boolq.sr`)
- `language` (string, `sr` in v0.1)
- task-specific input fields
- train + validation only: task-specific label
- MultiRC test: also `group_id` + `candidate_id` (public-safe grouping keys), no labels

## 7. Hidden-label policy

### 7.1 Public users can
- pull `permitt/superglue-serbian`
- run `balkanbench eval` on train + validation (labels visible)
- run `balkanbench predict` on public test inputs, producing `predictions.jsonl` + `run_metadata.json`
- submit predictions via the submission flow (see §13)

### 7.2 Public users cannot
- access test labels
- run `balkanbench score` on test without `HF_OFFICIAL_TOKEN`

### 7.3 Official environment (CI + GCP)
- holds `HF_OFFICIAL_TOKEN` as a GitHub repository secret (for CI workflows that need private labels) and as a GCP Secret Manager secret (for official eval jobs)
- `balkanbench score --predictions predictions.jsonl --benchmark superglue --language sr` reads private labels with the token, computes metrics, emits an artifact conforming to `eval/schemas/result_artifact.json`

### 7.4 No silent fallback

`balkanbench score` fails loudly on:
- private repo unavailable
- missing private label row for any public test `example_id`
- schema mismatch
- benchmark aggregate attempted over fewer than all 6 ranked task scores

## 8. Metrics + main score

### 8.1 Primary metric per task (v0.1)

| Task    | Primary metric |
|---------|----------------|
| BoolQ   | `accuracy`     |
| CB      | `f1_macro`     |
| COPA    | `accuracy`     |
| RTE     | `accuracy`     |
| MultiRC | `f1_a`         |
| WSC    | `accuracy`      |

These match the user's existing leaderboard table. Secondary metrics (MultiRC `exact_match`, CB `accuracy`, etc.) are still computed and emitted in the result artifact but excluded from the main score.

### 8.2 Main benchmark score

- Main score = unweighted arithmetic mean of the 6 primary task scores, expressed as a percentage (0 to 100)
- Not computed if any ranked task is missing
- Partial runs display `(N/6) partial` and receive no main rank (e.g. ModernBERTić small, 5/6)
- Leaderboard export rejects partial ranked coverage at the aggregation step

### 8.3 Uncertainty

- v0.1: mean ± standard deviation across 5 seeds
- v0.2+: prediction-level bootstrap confidence intervals

## 9. Official eval protocol

1. **Search** with Optuna on `train → validation`. Single sampler seed. Primary validation metric as objective. Search space and early stopping policy documented per task family.
2. **Select** one final config per (model, task, language). Optionally rerank top-k with multiple validation seeds before freezing.
3. **Freeze** the final config into `eval/configs/models/official/{model}.yaml`.
4. **Final training** on `train + validation` with the frozen config.
5. **Seeds**: 5 fixed seeds, documented in the model config.
6. **Hidden-test evaluation**: one evaluation per seed. No further config changes after the first test evaluation.
7. **Report**: per-seed score, mean, stdev, and all provenance (image digest, git SHA, dataset revision, config hash, Optuna sampler + seed + trial count, search space id, early-stopping policy).

Forbidden:
- tuning on test
- selecting the best seed by test score
- changing hyperparameters after seeing test results
- different final hyperparameters per seed after test begins

Recommended public phrasing (for dataset card + methodology doc):

> Hyperparameters are selected on validation only. Final leaderboard scores are computed on the hidden test split using 5 independent seeds with the selected configuration frozen before test evaluation.

## 10. Reproducibility gate

`.github/workflows/repro-bertic.yml`:
- runs nightly and on `workflow_dispatch`
- pulls the BERTić official config
- runs a single seed on a small validation-sized sample per task
- asserts per-task primary metric is within committed tolerance of `eval/results/official/superglue-sr/bertic/baseline.json`
- failure red-labels the release

A full 5-seed BERTić reproduction is a separate manual-dispatch workflow and is the release qualification check.

## 11. Leaderboard

### 11.1 Per-run result artifact

Stored at `eval/results/official/{benchmark}-{language}/{model}/result.json`, where `{model}` is the short name from `configs/models/official/{model}.yaml` (filesystem-safe; the HF repo id with its slash is stored as the `model_id` field inside the JSON). Conforms to `eval/schemas/result_artifact.json`. Required fields:

- `benchmark_name`, `benchmark_version`, `run_type` (`"official"` | `"experimental"`)
- `task_id`, `language`, `model_id`, `model_revision`
- `code_revision`, `dataset_revision`, `image_digest`, `config_hash`
- `selection_metric`
- `hp_search`: tool, sampler, sampler_seed, num_trials, search_space_id, early_stopping_policy
- `seed_results` (per-seed primary + secondary metrics), `aggregate` (mean + stdev)
- `task_score`, `rankable`
- `test_predictions_hash`: SHA-256 of the predictions file, so a reader can verify they are looking at the same predictions
- `throughput`: `{ex_per_sec, tok_per_sec, peak_vram_mib, hardware, batch_size, max_seq_len, precision, torch_version, driver_version}` (see §23)
- `sponsor`: `"Recrewty"`

### 11.2 Leaderboard export

`balkanbench leaderboard export --benchmark superglue --language sr`:
- loads all eligible artifacts under `eval/results/official/superglue-sr/`
- validates completeness: 6 ranked tasks present for rankable rows
- writes the leaderboard JSON to `frontend/public/leaderboards/superglue-serbian/benchmark_results.json`
- commits alongside the artifacts it was generated from

### 11.3 `benchmark_results.json` schema

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
      "model": "ModernBERTić base",
      "model_id": "<hf-repo-id>",
      "model_revision": "<commit-sha>",
      "params": 395000000,
      "params_display": "395M",
      "results": {
        "cb":      { "mean": 78.52, "stdev": 3.82 },
        "copa":    { "mean": 76.84, "stdev": 1.29 },
        "rte":     { "mean": 73.13, "stdev": 0.84 },
        "wsc":     { "mean": 63.56, "stdev": 2.39 },
        "boolq":   { "mean": 80.70, "stdev": 0.44 },
        "multirc": { "mean": 67.90, "stdev": 0.47 }
      },
      "avg": 73.44,
      "complete": true,
      "tasks_completed": 6,
      "tasks_total": 6,
      "throughput": {
        "ex_per_sec": 234.5,
        "peak_vram_mib": 4820
      }
    },
    {
      "rank": null,
      "model": "ModernBERTić small",
      "model_id": "<hf-repo-id>",
      "params": 149000000,
      "params_display": "149M",
      "results": {
        "cb":      { "mean": 76.96, "stdev": 3.19 },
        "copa":    { "mean": 65.76, "stdev": 2.42 },
        "rte":     { "mean": 65.82, "stdev": 1.14 },
        "wsc":     { "mean": 64.11, "stdev": 1.11 },
        "boolq":   { "mean": 76.02, "stdev": 0.63 },
        "multirc": null
      },
      "avg": 69.73,
      "complete": false,
      "tasks_completed": 5,
      "tasks_total": 6,
      "partial_flag": "(5/6)",
      "throughput": {
        "ex_per_sec": 312.1,
        "peak_vram_mib": 2410
      }
    }
  ]
}
```

The launch `benchmark_results.json` contains all 9 rows from §2.3 using the values in the user's table.

## 12. Frontend

- Stack: React 19, Vite 8, `react-router-dom`, editorial brutalist style (continues from current landing page)
- Routes:
  - `/` - existing landing page (unchanged hero, ticker, status card, sponsor footer)
  - `/leaderboard` - renders from `benchmark_results.json`
  - `/about` - methodology summary, sponsor acknowledgment
  - `/submit` - submission walkthrough, links to issue templates
- Leaderboard benchmark selector: v0.1 exposes only `superglue.sr`; the component is written so adding `sle.sr` or `superglue.hr` in v0.2 is a data-only change
- Leaderboard columns: 6 task columns + Avg + `Throughput (L4, fp16, ex/s)` + `Params`; peak VRAM shown on row hover / expand
- Deployment: Vercel, `rootDirectory = frontend`, `vercel.json` sets framework + output dir
- Sponsor: "Compute sponsored by Recrewty" appears in the leaderboard footer and the about page

## 13. Contribution flow

### 13.1 Issue templates

`.github/ISSUE_TEMPLATE/`:
- `propose-benchmark.yml`: name, description, languages, task list, license, data source, hidden-label policy, contact
- `propose-task.yml`: within an existing benchmark, task description, metric, dataset source
- `propose-model.yml`: model id, source repo, revision, license, submitter identity, contact
- `submission.yml`: for official submissions - model config + predictions package reference
- `bug.yml`: standard bug template

### 13.2 Flow

1. Contributor opens an issue using the relevant template
2. Maintainer triages (labels: `proposal-benchmark`, `proposal-task`, `proposal-model`, `submission`, `bug`)
3. Governance check (identity, license, scope) per `docs/governance/submissions.md`
4. Contributor or maintainer opens a PR with YAML configs + data card + prompt templates
5. CI runs schema validation + metric regression + smoke tests
6. Maintainer reviews and merges
7. Ships in the next minor release

`CONTRIBUTING.md` walks the above. `docs/governance/contributions.md` and `docs/governance/submissions.md` spell out identity, anti-spam, and review SLAs.

## 14. GCP execution

v0.1 provides scripts only, not an orchestrator. Targets: A100 and L4 GPU VMs on Compute Engine with container-optimized images.

`eval/scripts/gcp/launch_a100.sh` and `eval/scripts/gcp/launch_l4.sh`:
- inputs via env vars: `PROJECT_ID`, `ZONE`, `MODEL`, `TASK`, `SEED`, optional `N_TRIALS`, optional `MODE` (one of `eval`, `predict`, `score`, `hp-search`, `throughput`; default `eval`)
- create a VM with the chosen GPU type
- boot the pinned `balkanbench` Docker image
- fetch `HF_OFFICIAL_TOKEN` from GCP Secret Manager
- run `balkanbench {eval,predict,score,hp-search,throughput}` with the requested config
- write artifacts to a GCS bucket
- auto-shutdown on completion

`launch_l4.sh --mode throughput` is the canonical way to generate throughput artifacts across all official models in one sweep (see §23).

`eval/scripts/gcp/common.sh` holds shared setup (gcloud auth, project checks, image pull, VM naming).

Docs:
- `docs/gcp/running_official_eval.md` walks a launch end to end
- `docs/gcp/costs.md` documents per-task GPU-hours and Recrewty's sponsorship
- `docs/gcp/security.md` covers secret management and audit logging

## 15. Docker

`eval/Dockerfile` pins:
- Python 3.11
- `torch`, `transformers`, `datasets`, `evaluate` (exact versions pinned in `pyproject.toml`)
- system deps (git, build-essential)
- non-root runtime user

Every official result artifact records `image_digest`. The image is built and pushed to Google Artifact Registry on release tags.

## 16. CI

`.github/workflows/`:

- `ci.yml`: ruff (lint + format check), mypy, `pytest tests/unit` + `pytest tests/integration`, coverage
  - coverage thresholds: **overall 80%**, **critical modules 95%** (`metrics/`, `scoring/`, `validation/`)
  - CI fails if either threshold drops
- `validate-configs.yml`: runs on changes under `eval/configs/**`, executes `balkanbench validate-config` on every spec
- `validate-fixtures.yml`: runs on changes under `eval/tests/fixtures/**` or `eval/schemas/**`
- `release-check.yml`: runs on release tags, executes full release validation (all artifacts present, schema checks, aggregate integrity)
- `repro-bertic.yml`: nightly + manual dispatch; runs the reproducibility gate (§10)

## 17. Testing

- **Unit**: config parsing, schema validation, every metric implementation, leaderboard aggregation rules, task loaders
- **Integration**: local benchmark end-to-end on a toy model with fixture data; prediction packaging; private-label scoring with fixture private labels
- **Smoke**: CLI end-to-end on tiny fixture; GCP launch script dry-run
- **Regression** (mandatory):
  - MultiRC grouped F1 against locked fixture
  - benchmark aggregation rejects partial ranked runs
  - hidden-label scoring path uses private repo
  - public test prediction path emits no labels
  - leaderboard export uses correct primary metric per task
  - diagnostic sanity: AXb `matthews_correlation` and AXg `accuracy` cannot score more than 3σ below chance; the evaluator raises a loud warning and refuses to emit a success artifact (this rule caught a legacy-repo bug where diagnostics silently came back below random)

## 18. Versioning

Semantic versioning, with bump rules in `docs/methodology/versioning.md`:
- **major**: ranked task list, metric definition, aggregation formula, or hidden/public contract changes
- **minor**: new language, new ranked task, new diagnostic, non-breaking schema extension
- **patch**: fixes that do not change leaderboard numbers

v0.1.0 is the initial public release. The benchmark version is recorded in every artifact.

## 19. Documentation (v0.1 required)

- `README.md` - what it is, who for, quickstart, contribute link, sponsor block
- `CONTRIBUTING.md` - issue-first workflow, PR rules, commit style, code of conduct link
- `LICENSE` - MIT
- `SECURITY.md` - token + secret handling, responsible disclosure
- `docs/methodology/benchmark_contract.md` - frozen v0.1 contract
- `docs/methodology/data_provenance.md` - Serbian translation, adaptation, review, adjudication
- `docs/methodology/versioning.md` - semver rules
- `docs/methodology/task_lifecycle.md` - ranked / diagnostic / experimental / archived states
- `docs/methodology/throughput.md` - throughput protocol, reference hardware, caveats
- `docs/governance/submissions.md` - identity, anti-spam, review SLAs
- `docs/governance/contributions.md` - benchmark + task + model proposal flow
- `docs/leaderboard/format.md` - schema of `benchmark_results.json`
- `docs/gcp/running_official_eval.md` - end-to-end GCP launch
- `docs/gcp/costs.md` - cost expectations, sponsorship note
- `docs/gcp/security.md` - secret handling, audit logging
- `frontend/README.md` - dev setup, Vercel deploy notes
- `eval/README.md` - Python dev setup, CLI walkthrough

## 20. Sponsorship

"Compute sponsored by Recrewty" appears in:
- `README.md` hero
- `frontend` footer (already present) and leaderboard + about pages
- public HF dataset card (`permitt/superglue-serbian`)
- every official result artifact (`sponsor: "Recrewty"`)
- `docs/gcp/costs.md`

## 21. Commit + branch conventions

- Main branch: `main`
- Feature branches: `feat/{topic}`, `fix/{topic}`, `docs/{topic}`
- Conventional Commits: `feat`, `fix`, `test`, `refactor`, `docs`, `chore`, `ci`, `build`
- TDD cadence: red (failing test) → green (minimal impl) → refactor, each as its own commit where the slice is large enough to warrant
- One logical change per commit
- **No Claude co-author line**
- PRs: descriptive title, linked issue, review required before merge once external contributors are involved

## 22. Launch gate (release 2026-04-27)

Must all pass before tagging `v0.1.0`:

1. `permitt/superglue-serbian` published with complete dataset card
2. COPA `dev` → `validation` rename verified
3. `balkanbench` package installable, CLI `--help` works for every subcommand
4. All 6 task implementations pass unit + integration tests
5. Reproducibility gate: full 5-seed BERTić run matches baseline within tolerance
6. 9 official result artifacts committed under `eval/results/official/superglue-sr/`
7. 9 throughput artifacts committed under `eval/results/official/superglue-sr/{model}/throughput.json`
8. `frontend/public/leaderboards/superglue-serbian/benchmark_results.json` generated and committed, throughput column populated
9. Frontend `/leaderboard` renders correctly on localhost and Vercel preview
10. Frontend deployed to Vercel production at the project's domain
11. CI green: lint, type, tests, coverage thresholds on `main`
12. All docs in §19 written and committed
13. All issue templates in `.github/ISSUE_TEMPLATE/` present
14. MIT LICENSE committed; repo set public on GitHub

## 23. Throughput reporting

Readers should be able to pick a model for production by latency, not just quality.

### 23.1 Reference hardware

- NVIDIA L4 24GB
- fp16 precision
- `torch.compile` off
- Driver + CUDA versions pinned in the Docker image, recorded per artifact

Rationale: L4 is a common production inference GPU, fits every v0.1 launch model, and matches the "24 GB GPU" target of `spec_claude.md`. Additional hardware (A100, CPU, TensorRT) comes post-launch via the same `throughput` subcommand with a different launcher flag.

### 23.2 Measurement protocol

- Inputs: each task's validation split (labels not needed for throughput; validation is labeled but we ignore the labels here)
- `batch_size`: from the task config
- `max_seq_len`: from the task tokenizer spec
- Warmup: 2 batches (discarded)
- Measurement: `min(50, full_pass)` batches
- Report median wall-clock latency across measurement batches, converted to `examples_per_sec` and `tokens_per_sec`
- Record peak GPU memory during measurement via `torch.cuda.max_memory_allocated()`

### 23.3 Artifact schema

Per-task throughput at `eval/results/official/superglue-sr/{model}/throughput/{task}.json`:

```json
{
  "benchmark": "superglue",
  "language": "sr",
  "task": "boolq",
  "model_id": "classla/bcms-bertic",
  "hardware": "NVIDIA L4 24GB",
  "precision": "fp16",
  "batch_size": 16,
  "max_seq_len": 256,
  "warmup_batches": 2,
  "measurement_batches": 50,
  "torch_version": "2.5.1+cu121",
  "driver_version": "535.xx",
  "image_digest": "sha256:...",
  "throughput_ex_per_sec": 242.8,
  "throughput_tok_per_sec": 62156.8,
  "peak_vram_mib": 4820,
  "measured_at": "2026-04-26T12:34:56Z"
}
```

Per-model aggregate at `eval/results/official/superglue-sr/{model}/throughput.json`:

```json
{
  "model_id": "classla/bcms-bertic",
  "hardware": "NVIDIA L4 24GB",
  "precision": "fp16",
  "tasks": {
    "boolq":   { "ex_per_sec": 242.8, "peak_vram_mib": 4820 },
    "cb":      { "ex_per_sec": 221.4, "peak_vram_mib": 4620 },
    "copa":    { "ex_per_sec": 298.2, "peak_vram_mib": 3910 },
    "rte":     { "ex_per_sec": 236.0, "peak_vram_mib": 4701 },
    "multirc": { "ex_per_sec": 154.1, "peak_vram_mib": 6210 },
    "wsc":     { "ex_per_sec": 254.7, "peak_vram_mib": 3240 }
  },
  "mean_ex_per_sec": 234.5,
  "max_peak_vram_mib": 6210
}
```

### 23.4 CLI

```
balkanbench throughput \
  --model bertic \
  --benchmark superglue \
  --language sr \
  --hardware l4 \
  --out eval/results/official/superglue-sr/bertic/throughput.json
```

Sweep-all-models mode:

```
balkanbench throughput sweep \
  --benchmark superglue \
  --language sr \
  --hardware l4 \
  --models-from configs/benchmark.yaml
```

### 23.5 GCP

`eval/scripts/gcp/launch_l4.sh --mode throughput` boots a single L4 VM, runs the sweep over every official model (9 × 6 tasks, sequential), uploads artifacts, shuts down. Wall-clock ~1.5h.

### 23.6 Leaderboard

- Primary throughput column on the leaderboard: `Throughput (L4, fp16, ex/s)`, value = per-model `mean_ex_per_sec`
- Secondary: peak VRAM shown on row hover / expand
- Sortable independently from the quality Avg, so readers can find the best quality-per-ex/s point on their own

### 23.7 Docs

`docs/methodology/throughput.md` covers:
- reference hardware rationale
- protocol (warmup, measurement, aggregation)
- fp16 caveats (some models may have numerical differences versus fp32)
- version pinning (torch, driver, CUDA, image digest)
- how to reproduce a throughput measurement locally on a single L4 VM
- known caveats (throughput is workload- and batch-size-sensitive; the reported numbers are for the declared protocol only)

### 23.8 Out of scope for v0.1

- A100 column (post-launch via same `throughput` subcommand with `--hardware a100`)
- CPU throughput
- TensorRT / ONNX export benchmarks
- Latency percentiles beyond median (p95 / p99)
- Time-to-first-token (not meaningful for encoder models in v0.1)

## 24. Open questions for v0.2+

- Re-run all 9 models through the new official Docker-based pipeline (v0.1 ships prior-pipeline artifacts as-is)
- Orchestrated multi-job GCP pipeline
- HR, CNR, BS language data
- Serbian-LLM-Eval adapter
- MTEB-BCMS, LLM Arena
- Prediction-level bootstrap CIs
- Automated submission-scoring server with quarantined execution
