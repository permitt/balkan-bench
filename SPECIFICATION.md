<CODEX>

# BalkanBench Repository Specification

## 1. Objective

Create a new public repository for a clean, auditable, reproducible benchmark for Serbian, Croatian, and Montenegrin language model evaluation.

This repository is intended to become:
- the canonical open-source benchmark implementation
- the source of public documentation and methodology
- the runner used for local and GCP evaluation
- the producer of leaderboard-ready result artifacts

The benchmark must be neat enough to publish publicly, easy enough for contributors to run locally, and strict enough that released leaderboard results are defensible.

---

## 2. Release Reality

This specification assumes:
- Serbian data exists now
- Croatian and Montenegrin will be added very soon
- official large-scale evaluation will run on GCP
- final benchmark data will be published on Hugging Face
- public Hugging Face datasets must not expose hidden test labels

That means the repository must be built around two modes:
- public development mode
- official evaluation mode

These must be explicitly separated in the repo and in the docs.

---

## 3. Public Benchmark Principles

### 3.1 Strict separation of public vs official evaluation

The repository must distinguish between:
- `public_train`
- `public_validation`
- `public_test_inputs`
- `official_test_labels` (private, not published)

The public repo and public dataset must never contain hidden test labels.

### 3.2 One source of truth

Task definitions, metrics, language support, and aggregation rules must come from declarative specs.

There must not be:
- one metric mapping in task configs
- another one in the runner
- another one in the leaderboard frontend

One task spec must drive all of them.

### 3.3 Local-first, cloud-capable

The benchmark must run locally for small/dev runs.
The benchmark must also scale cleanly to GCP for official leaderboard evaluation.

Cloud execution is an implementation detail, not the benchmark definition.

### 3.4 No silent fallback behavior

The benchmark must fail loudly if:
- ranked tasks are missing
- grouped metrics cannot be computed
- hidden-label evaluation assets are unavailable
- split schema is wrong
- leaderboard aggregation is attempted over partial ranked coverage

### 3.5 Public neatness

The public repo should look like a maintained benchmark, not a personal experiment dump.

That means:
- no stale result directories in root
- no ambiguous "v3/v4/v7/v10" sprawl
- no private notes in repo root
- no contradictory README statements
- no dead code that claims benchmark semantics but is unused

---

## 4. Benchmark Scope

## 4.1 Benchmark name

Recommended:
- organization/project: `BalkanBench`
- benchmark: `BalkanBench NLU`
- release naming:
  - `balkanbench-bcms-v1`
  - if needed during transition: `balkanbench-sr-preview`

Since you plan to add Croatian and Montenegrin imminently, the new repo should be designed as BCMS from day one, even if the first release candidate is Serbian-heavy.

## 4.2 Languages

Supported languages:
- `sr`
- `hr`
- `cnr`

Every task must explicitly declare which languages are currently available.

Example:

```yaml
languages:
  available: [sr, hr, cnr]
  ranked: [sr, hr, cnr]
```

or during rollout:

```yaml
languages:
  available: [sr]
  ranked: [sr]
  roadmap: [hr, cnr]
```

## 4.3 Task classes

Each task must be classified as:
- `ranked`
- `diagnostic`
- `experimental`

Recommended initial ranked tasks:
- BoolQ
- CB
- COPA
- RTE
- MultiRC
- WSC
- ReCoRD

Recommended diagnostics:
- AX-b
- AX-g

Recommended rule:
- diagnostics are reported publicly
- diagnostics never contribute to main benchmark score

Canonical COPA policy:
- the new benchmark exposes exactly one active public COPA task id: `copa`
- canonical `copa` uses the valid multiple-choice formulation only
- legacy `copa_fixed` and `copa_connective` are historical implementation artifacts, not benchmark task ids
- historical variants may be described in methodology notes, but they must not exist as active public configs, leaderboard columns, or official submission targets

---

## 5. Hidden Test Label Policy

This is central to the public release.

## 5.1 Public dataset contents

Publish on Hugging Face:
- train split with labels
- validation split with labels
- test split without labels

Test set should expose:
- text/input fields
- metadata needed for formatting and grouped prediction ids
- unique example ids

Test set should not expose:
- labels
- grouped gold answers
- hidden official targets

## 5.2 Private official label store

Keep official test labels outside the public dataset.

Recommended options:
- private GCS bucket
- private Hugging Face dataset repo
- private encrypted artifact bundle in GCP

Official evaluation jobs running on GCP may access this private store.
Public local users should not.

## 5.3 Public submission mode

The repository should support:
- local benchmarking on train/validation
- local prediction generation for public test inputs
- packaging predictions for official scoring

Recommended outputs:
- `predictions.jsonl`
- `run_metadata.json`
- optional `submission.tar.gz`

This allows later leaderboard scoring without exposing labels.

## 5.4 Benchmark modes

Define three explicit modes:

### Dev mode
- uses train + validation
- labels visible
- used by contributors locally

### Public test prediction mode
- uses public test inputs only
- no local scoring
- outputs prediction file

### Official scoring mode
- runs in trusted environment
- accesses hidden labels
- produces scored result artifact

The CLI and docs must make these modes obvious.

---

## 6. Repository Layout

Recommended repository structure:

```text
.
├── README.md
├── CONTRIBUTING.md
├── LICENSE
├── pyproject.toml
├── .github/
│   └── workflows/
├── src/
│   └── balkanbench/
│       ├── cli/
│       ├── config/
│       ├── data/
│       ├── tasks/
│       ├── metrics/
│       ├── models/
│       ├── evaluation/
│       ├── scoring/
│       ├── leaderboard/
│       ├── gcp/
│       ├── validation/
│       └── utils/
├── configs/
│   ├── benchmark/
│   ├── tasks/
│   ├── models/
│   │   ├── official/
│   │   └── experimental/
│   ├── prompts/
│   └── environments/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── smoke/
│   ├── fixtures/
│   └── schemas/
├── docs/
│   ├── methodology/
│   ├── data/
│   ├── leaderboard/
│   ├── releases/
│   ├── gcp/
│   └── governance/
├── schemas/
├── examples/
├── results/
│   ├── local/
│   ├── official/
│   ├── submissions/
│   └── diagnostics/
└── leaderboard_exports/
```

Do not ship:
- mixed old experiments in root
- opaque `wandb/` dumps
- cloud-specific scratch state

---

## 7. Dataset Specification

## 7.1 Public Hugging Face dataset contract

Each task/language pair should be a separate config or equivalent documented subset.

Example:
- `boolq_sr`
- `boolq_hr`
- `boolq_cnr`

Alternative:
- one config per task with `language` column

Choose one and keep it uniform across all tasks.

Recommended simpler public contract:
- one config per task-language pair

## 7.2 Required dataset metadata

Each published task-language pair must have:
- source description
- license statement
- split sizes
- schema documentation
- known caveats
- version
- revision hash

## 7.3 Required fields

Every split must contain:
- `example_id`
- task input fields
- `language`
- `task_id`

Grouped tasks must also contain public-safe grouping keys:
- `group_id`
- `candidate_id`

Private labels/golds must remain in official scoring storage only.

## 7.4 Data docs

For each task, document:
- original source
- translation or native creation method
- review method
- normalization rules
- BCMS-specific linguistic adjustments
- any language-specific prompt or formatting differences

Required release artifact:
- `docs/methodology/data_provenance.md`

This document must describe:
- source provenance for every task and split
- translation and adaptation pipeline for Serbian, Croatian, and Montenegrin
- who performed translation, post-editing, and adjudication
- review and disagreement-resolution policy
- any BCMS-specific adjustments beyond direct translation
- normalization policy and text-cleaning rules
- dataset revision policy and how provenance is tracked across releases

---

## 8. Task Specification

Each task must have a single YAML spec that defines:
- id
- status
- languages supported
- task type
- split policy
- required input columns
- public-visible columns
- official label columns
- prompt templates per language
- primary metrics
- report metrics
- task score aggregation rule

Example:

```yaml
task_id: cb
status: ranked
task_type: sequence_classification
languages: [sr, hr, cnr]
splits:
  public: [train, validation, test]
  labeled_public: [train, validation]
  labeled_private: [test]
inputs:
  text_fields:
    - premise
    - hypothesis
prompts:
  sr:
    format: pair
  hr:
    format: pair
  cnr:
    format: pair
metrics:
  primary: [accuracy, f1_macro]
  report: [accuracy, f1_macro]
task_score: mean_primary
```

---

## 9. Metric Specification

This must be exact and public.

## 9.1 Ranked task metrics

- BoolQ:
  - `accuracy`

- CB:
  - `accuracy`
  - `f1_macro`
  - task score = mean of both

- COPA:
  - `accuracy`

- RTE:
  - `accuracy`

- MultiRC:
  - `f1_a`
  - `exact_match`
  - optional: `f1_m`
  - task score = mean of primary metrics

- WSC:
  - `accuracy`

- ReCoRD:
  - `f1`
  - `exact_match`
  - task score = mean of primary metrics

## 9.2 Diagnostic metrics

- AX-b:
  - `matthews_correlation`

- AX-g:
  - `accuracy`
  - `gender_parity_score`

## 9.3 Main benchmark score

Main score =
- macro-average of task scores across all ranked tasks

Main score must not be computed if:
- any ranked task is missing
- any ranked task failed validation

## 9.4 Confidence intervals

Recommended v1 public policy:
- report mean and standard deviation across seeds
- report CI only if methodology is fully implemented and documented

Preferred official release policy:
- prediction-level bootstrap CIs from saved predictions

Do not ship decorative CIs that are not methodologically sound.

---

## 10. Task Implementation Rules

## 10.1 Task interface

Every task implementation must support:
- loading splits
- preprocessing
- prediction decoding
- metric computation with required metadata
- metric validation

Grouped tasks must not rely on hidden dataset order assumptions if there is a cleaner metadata-driven alternative.

## 10.2 Prompt handling

All prompts must be config-driven and language-specific.

No ranked task should hard-code Serbian-only prompts in Python.

## 10.3 Private-label scoring

Task scoring code must support:
- scoring from predictions plus private gold bundle
- generating prediction packages from public test inputs

This is required because you want labels hidden publicly.

---

## 11. Model Specification

## 11.1 Official model configs

Official model configs must include:
- model id
- source repo/path
- revision
- architecture family
- tokenizer settings
- default training settings
- task-specific overrides
- max context
- precision policy

Official model configs live in:
- `configs/models/official/`

Experimental models live in:
- `configs/models/experimental/`

Only official models are leaderboard-eligible.

## 11.2 Canonical config generation

After hyperparameter search, the final chosen configuration must be materialized into official YAML.

Do not leave the final benchmark in a state where:
- sweep winners exist only in result JSON
- official configs still reflect stale hand tuning

## 11.3 Hyperparameter search protocol

Official hyperparameter search must use validation only.

Required rule:
- `train` is used for fitting
- `validation` is used for model selection, early stopping, and Optuna objective computation
- `test` is never used for hyperparameter search, checkpoint selection, early stopping, sweep analysis, or manual model selection

Recommended official search process:
1. Define the search space in config.
2. Run Optuna on `train -> validation`.
3. Optimize the task's declared primary validation metric.
4. Select one final hyperparameter configuration per model family and task.
5. Freeze that configuration before any hidden-test evaluation.
6. Materialize the frozen configuration into official YAML.

Allowed practical policy:
- Optuna may run with a single seed for efficiency.
- If compute allows, the top few validation configurations may be reranked with multiple validation seeds before freezing the final configuration.

Forbidden behavior:
- tuning on `test`
- selecting the best seed by `test` performance
- changing hyperparameters after seeing `test` results
- using different final hyperparameters per seed after hidden-test evaluation begins

## 11.4 Final evaluation protocol

Official leaderboard evaluation must use a frozen configuration.

Recommended protocol:
1. Complete hyperparameter search on `train -> validation`.
2. Freeze one final configuration per model and task.
3. Train the final system on `train + validation`.
4. Evaluate once on hidden `test`.
5. Repeat the final run across 5 fixed seeds.
6. Report per-seed scores, mean, and standard deviation.

If a stricter policy is desired, the benchmark may use `train` only for the final fit, but the policy must be uniform across all official models and documented explicitly.

Official release reporting must include:
- search space definition
- number of Optuna trials
- Optuna sampler and seed
- early stopping policy
- final frozen hyperparameters
- exact final seed list
- primary validation metric used for selection
- per-seed hidden-test scores
- hidden-test mean and standard deviation

Recommended benchmark text:
- "Hyperparameters are selected on validation only. Final leaderboard scores are computed on the hidden test split using 5 independent seeds with the selected configuration frozen before test evaluation."

---

## 12. Execution Modes

## 12.1 Local dev mode

Purpose:
- let users run small evaluations locally
- let contributors test new models/tasks

Should support:
- validation scoring
- smoke tests
- debug-friendly outputs

## 12.2 Local prediction mode

Purpose:
- let public users run on unlabeled test inputs

Should produce:
- prediction package
- run metadata
- environment metadata

Should not score test locally.

## 12.3 Official GCP scoring mode

Purpose:
- run canonical leaderboard evaluation
- access hidden labels privately

Should support:
- reproducible containerized execution
- access to private label store
- final scored artifact generation
- leaderboard export generation

---

## 13. GCP Architecture

Since official evaluation will run on GCP, define that path cleanly from the start.

## 13.1 Containerization

Use a single pinned Docker image for official runs.

The image must pin:
- Python version
- torch version
- transformers version
- dataset/evaluate versions
- system dependencies

Every official result must record:
- image digest
- git SHA
- config hash
- dataset revision

## 13.2 Job orchestration

Recommended structure:
- one job per model x task x language x seed
- one aggregation step per model-language benchmark
- one release aggregation step per benchmark version

## 13.3 Private resources

Private official scoring environment must provide:
- access to hidden label artifacts
- access to private secrets if needed
- audit log of benchmark release runs

## 13.4 GCP docs

Document:
- infra assumptions
- cost expectations
- dry-run mode
- how to launch official eval
- how to collect results

This goes in `docs/gcp/`.

---

## 14. Result Artifact Specification

## 14.1 Per-task result artifact

Each scored run must emit:

```json
{
  "benchmark_name": "BalkanBench",
  "benchmark_version": "1.0.0",
  "run_type": "official",
  "task_id": "cb",
  "language": "sr",
  "model_id": "xlm_r_bertic",
  "model_revision": "...",
  "code_revision": "...",
  "dataset_revision": "...",
  "image_digest": "...",
  "config_hash": "...",
  "selection_metric": "...",
  "hp_search": {
    "tool": "optuna",
    "sampler": "...",
    "sampler_seed": 42,
    "num_trials": 50,
    "search_space_id": "...",
    "early_stopping_policy": "..."
  },
  "seed_results": [...],
  "aggregate": {...},
  "task_score": 0.85,
  "rankable": true
}
```

## 14.2 Benchmark aggregate artifact

Each official release must emit one aggregate file:

```json
{
  "benchmark_name": "BalkanBench",
  "benchmark_version": "1.0.0",
  "languages": ["sr", "hr", "cnr"],
  "ranked_tasks": [...],
  "diagnostic_tasks": [...],
  "results": {
    "xlm_r_bertic": {
      "sr": {...},
      "hr": {...},
      "cnr": {...}
    }
  }
}
```

No incomplete model-language run may receive a main score.

## 14.3 Leaderboard export

Generate a clean machine-readable leaderboard export:
- JSON
- CSV
- markdown summary

This should be generated from validated official artifacts only.

---

## 15. Testing Specification

This repository must be professionally tested.

## 15.1 Test categories

### Unit tests

Test:
- config parsing
- schema validation
- task metric implementations
- grouped metric correctness
- model config resolution
- leaderboard aggregation

### Integration tests

Test:
- local benchmark run for a toy model
- prediction packaging
- official scoring path with fixture labels
- result validation

### Smoke tests

Test:
- CLI works end to end on tiny fixture data
- GCP job payload generation
- HF dataset schema assumptions

### Regression tests

Lock:
- metric outputs for curated fixture examples
- aggregation behavior
- ranking behavior

## 15.2 Minimum test coverage requirements

Recommended public standard:
- overall line coverage: at least 85%
- critical modules:
  - metrics
  - scoring
  - validation
  - aggregation
  - task implementations
  at least 95%

If 85% is too aggressive for early v1, set:
- repo minimum: 80%
- critical-path minimum: 95%

## 15.3 Coverage enforcement

CI must fail if:
- coverage drops below threshold
- critical modules fall below threshold

Use coverage reports in:
- terminal
- XML for CI

## 15.4 Must-have regression tests

These are mandatory:
- MultiRC grouped metric test
- ReCoRD grouped metric test
- benchmark aggregation rejects partial ranked runs
- hidden-label scoring path test
- public test prediction path test
- leaderboard export uses task primary metrics

---

## 16. Validation Specification

Validation must be a first-class subsystem.

## 16.1 Config validation

Validate:
- task specs
- model specs
- benchmark configs
- prompt specs
- environment configs
- active task ids reference only non-archived benchmark tasks
- official configs may reference only canonical `copa`
- legacy task ids such as `copa_fixed` and `copa_connective` fail validation

## 16.2 Dataset validation

Validate:
- split presence
- required columns
- schema consistency
- manifest presence
- hidden-label split separation rules

## 16.3 Result validation

Validate:
- provenance fields present
- required tasks completed
- metric schema matches task spec
- no illegal fallback metrics
- no missing seeds

## 16.4 Submission governance validation

Validate:
- official submissions are tied to an identifiable submitter
- submitter identity is linked to a public GitHub or Hugging Face account
- claimed public model repository exists and is accessible when a public checkpoint is declared
- submission metadata includes contact, license, and model provenance fields
- duplicate or spam submissions can be flagged and rejected under documented governance rules

Identity verification is leaderboard governance, not task scoring logic, but it is still a release requirement for a public benchmark.

## 16.5 Benchmark versioning policy

The benchmark must use semantic versioning with explicit bump rules:
- major: any change to the ranked task list, official metric definition, benchmark score aggregation formula, or hidden/public scoring contract that can change official numbers
- minor: addition of a new language, new ranked task column, new diagnostic task, or non-breaking schema extension
- patch: bug fixes, documentation improvements, CI changes, provenance clarifications, or implementation fixes that do not change official leaderboard numbers

The release must include:
- `docs/methodology/versioning.md`

## 16.6 Task deprecation and archival policy

Deprecated tasks must be handled explicitly:
- deprecated tasks move to an archive or experimental namespace
- deprecated tasks are removed from official benchmark configs and official leaderboard scoring
- historical submissions that used deprecated tasks remain visible with an `archived-task` flag
- new official submissions may not include deprecated tasks
- for COPA specifically, only canonical `copa` is active in the new benchmark; legacy variants are archival only

Official release generation must require successful validation.

---

## 17. CI/CD Specification

Use GitHub Actions from day one.

## 17.1 Required workflows

### `ci.yml`
- lint
- type check
- unit tests
- integration tests
- coverage gate

### `validate-configs.yml`
- validate task/model/benchmark YAMLs

### `validate-fixtures.yml`
- validate sample datasets and result schemas

### `release-check.yml`
- run official release validation before tagging

## 17.2 Style and static analysis

Recommended:
- `ruff`
- `pytest`
- `mypy` or `pyright`
- `pydantic` or JSON schema validation

---

## 18. Documentation Specification

The public repo must include:

## 18.1 README

Must explain:
- what BalkanBench is
- current languages
- ranked tasks
- what is public vs hidden
- how to run locally
- how official scoring works conceptually
- how to contribute

## 18.2 Methodology docs

Must include:
- benchmark definition
- task definitions
- metrics
- aggregation rules
- seed policy
- CI policy
- release policy
- versioning policy
- task lifecycle and archival policy

## 18.3 Data docs

Must include:
- source provenance
- licenses
- translation/native generation methodology
- known caveats
- translation, review, and adjudication provenance
- language-specific adaptation policy

## 18.4 Leaderboard docs

Must include:
- rankability requirements
- official submission format
- what counts as official
- how diagnostics are displayed
- submitter identity requirements
- anti-spam and submission review policy

## 18.5 GCP docs

Must include:
- how to run official scoring
- where hidden labels live
- cost guidance
- security rules

Required public documents:
- `docs/methodology/data_provenance.md`
- `docs/methodology/versioning.md`
- `docs/methodology/task_lifecycle.md`
- `docs/governance/submissions.md`

---

## 19. Public Hugging Face Publishing Policy

## 19.1 Dataset publication

Publish:
- benchmark task inputs
- train/validation labels
- test inputs only

Do not publish:
- test labels
- private gold bundles

## 19.2 Model/result publication

Publish:
- official result artifacts
- leaderboard export
- prediction format examples

Do not publish:
- private test golds
- internal secrets

## 19.3 Dataset card must say explicitly

- test labels are hidden
- public users can generate predictions but not score test locally
- leaderboard scoring is done in official environment

---

## 20. Step-by-Step Plan For Building The New Repository

## Phase 1: Define the benchmark contract

1. Freeze benchmark scope.
2. Decide public naming.
3. Define ranked vs diagnostic tasks.
4. Define language support policy.
5. Define hidden-label policy.
6. Define benchmark versioning policy.
7. Define task archival policy.

Deliverable:
- `docs/methodology/benchmark_contract.md`

## Phase 2: Define schemas and validators

1. Task spec schema
2. Model spec schema
3. Dataset manifest schema
4. Result schema
5. Leaderboard export schema
6. Submission metadata schema

Deliverable:
- `schemas/`
- validation code in `src/balkanbench/validation/`

## Phase 3: Create clean package and CLI

1. Create `balkanbench` package.
2. Implement config loader.
3. Implement CLI skeleton.
4. Implement validators.

Deliverable:
- runnable skeleton with `validate-env`, `validate-config`, `validate-data`

## Phase 4: Implement dataset layer

1. Build public dataset processing pipeline.
2. Build private hidden-label storage format.
3. Build HF publishing pipeline for public data.
4. Write data provenance and adaptation methodology.

Deliverable:
- public train/validation/test-input datasets
- private official test label bundle
- `docs/methodology/data_provenance.md`

## Phase 5: Implement task layer

1. Add sequence classification tasks.
2. Add multiple-choice tasks.
3. Add grouped tasks with correct metrics.
4. Add diagnostics.

Deliverable:
- all task specs and implementations

## Phase 6: Implement local run path

1. Local dev benchmark run
2. Public test prediction generation
3. Result packaging

Deliverable:
- contributors can run locally without GCP

## Phase 7: Implement official GCP path

1. Containerize environment
2. Add GCP job launcher
3. Add private-label scoring integration
4. Add result collector

Deliverable:
- official scoring workflow on GCP

## Phase 8: Implement tests and CI

1. Unit tests
2. Integration tests
3. Coverage gates
4. Release validation checks
5. Submission governance validation

Deliverable:
- green CI

## Phase 9: Implement release tooling

1. Aggregate official results
2. Validate official release package
3. Generate leaderboard export
4. Generate markdown summary
5. Generate versioned benchmark manifest

Deliverable:
- one canonical release bundle

## Phase 10: Publish public v1

1. Publish public dataset to Hugging Face
2. Publish repo
3. Publish official results
4. Publish leaderboard export
5. Publish methodology and governance docs

Deliverable:
- public benchmark release

---

## 21. Recommended Minimum Release Gate

Before public release:

1. BCMS language policy is truthful and reflected everywhere.
2. Public HF dataset contains no hidden test labels.
3. Official hidden-label scoring path is implemented and tested.
4. Ranked task metrics are correct and documented.
5. Partial ranked runs cannot receive a benchmark score.
6. CI is green.
7. Coverage gate passes.
8. Official GCP run path is reproducible and documented.
9. One canonical official release artifact exists.
10. README and dataset card are precise and non-contradictory.
11. `docs/methodology/data_provenance.md` exists and is complete.
12. `docs/governance/submissions.md` exists and defines identity and anti-spam policy.
13. `docs/methodology/versioning.md` exists and defines semantic bump rules.
14. `docs/methodology/task_lifecycle.md` exists and defines deprecation and archival rules.
15. Only canonical `copa` exists as an active benchmark task id.

---

## 22. Recommended Practical Decision

Given your stated plan, the cleanest path is:
- build the new repo as BCMS-first from the start
- publish public task inputs on Hugging Face with hidden test labels withheld
- let local users train and produce predictions
- do official scoring on GCP using private labels
- publish one final validated leaderboard artifact

That is the cleanest and most credible public design for what you want to ship.

</CODEX>

