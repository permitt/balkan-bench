# Contributing to BalkanBench

Thanks for wanting to help. The goal is a benchmark the BCMS NLP community owns
together. Every new benchmark, task, model, or submission is a schema-validated
PR - no core-code edits required.

This document walks you through the four supported contribution paths, with a
full step-by-step example for the most common one: **adding a new benchmark**.

- [Ground rules](#ground-rules)
- [Four contribution paths](#four-contribution-paths)
- [Adding a new benchmark](#adding-a-new-benchmark) (full walkthrough)
- [Adding a task to an existing benchmark](#adding-a-task-to-an-existing-benchmark)
- [Adding a model](#adding-a-model)
- [Submitting a result](#submitting-a-result)
- [Local development setup](#local-development-setup)
- [Commit conventions](#commit-conventions)
- [Code of conduct](#code-of-conduct)

---

## Ground rules

1. **Open an issue before a PR.** Proposals get triaged for scope, license, and
   identity before code review. This protects reviewer time and yours.
2. **Identity is required** for any official contribution. Your submitter
   identity must be a public GitHub or Hugging Face handle. Anonymous
   contributions are fine for bug reports but not for benchmarks, tasks,
   models, or submissions. Anti-spam rationale is documented in
   [`docs/governance/submissions.md`](docs/governance/submissions.md).
3. **Schemas are the source of truth.** Everything goes through JSON Schema
   validation in CI. You cannot invent fields; you can only change fields that
   the schema allows. Schemas live under
   [`eval/schemas/`](eval/schemas/). If your proposal genuinely needs a new
   schema field, that is a separate PR titled
   `feat(schemas): extend <schema_name> for <reason>`.
4. **Tests must stay green.** `eval/` has a coverage gate at 80% overall and
   95% on critical paths (metrics, scoring, validation). Your PR should not
   drop them.
5. **Be nice.** See the [Code of conduct](#code-of-conduct).

## Four contribution paths

| Path | Trigger | Outcome |
|------|---------|---------|
| [Add a new benchmark](#adding-a-new-benchmark) | You have a new dataset (e.g. Croatian sentiment, legal QA for BCMS, a new translation pair eval) and want it to be a first-class benchmark alongside SuperGLUE | A new `configs/benchmarks/<name>/` folder, a published HF dataset, and a leaderboard column on balkanbench.com |
| [Add a task to existing benchmark](#adding-a-task-to-an-existing-benchmark) | You want to extend SuperGLUE, SLE, etc. with one more task (e.g. adding ReCoRD to SuperGLUE) | A new `tasks/<task>.yaml` under the existing benchmark folder, a new leaderboard column |
| [Add a model](#adding-a-model) | You want your model to appear on the leaderboard | A `configs/models/experimental/<model>.yaml`, optionally upgraded to `official/` after community review |
| [Submit a result](#submitting-a-result) | An existing model-benchmark combination hasn't been run yet, or you want to challenge an existing row | A PR with a predictions package and its scored artifact |

---

## Adding a new benchmark

The full path, end to end, with a concrete example: **adding a fictitious
"Croatian Sentiment" benchmark** (single task, `sentiment`, three labels).

### Step 1 - Open a `Propose Benchmark` issue

Use the `Propose Benchmark` issue template on GitHub. Required fields:

- **Benchmark name** (short identifier, matches the directory name). Example:
  `croatian_sentiment`.
- **Description**: one paragraph. What is the task, why is it useful for BCMS
  LLM evaluation?
- **Languages**: which BCMS languages does it support now, which are on the
  roadmap?
- **Task list**: names and short descriptions.
- **Data source**: where does the data live now (paper, GitHub, HF dataset),
  what license, who owns it?
- **Hidden-label policy**: does this benchmark have a held-out test set? If
  yes, where will the private labels live?
- **Maintainer identity**: GitHub or Hugging Face handle of the person who
  will maintain this benchmark going forward.

A maintainer will label the issue `proposal-benchmark` and triage within the
SLA documented in [`docs/governance/submissions.md`](docs/governance/submissions.md).
Triage can end in: accepted, needs-more-info, or declined (with reason). Only
proceed past this step after the issue is accepted.

### Step 2 - Publish (or identify) the data on Hugging Face

BalkanBench canonicalises data on Hugging Face. Two modes are supported today:

- **You own the data**: publish a public HF dataset at `<your-org>/<name>-<lang>`
  (e.g. `teacompany/croatian-sentiment-hr`). Train and validation splits have
  labels. If you have a held-out test set, publish test **without labels** to
  the public repo and create a private sibling repo with test labels; a
  maintainer gets `HF_OFFICIAL_TOKEN` access to scoring.
- **You point at an external HF dataset**: declare the existing repo in your
  benchmark config. No publishing step needed.

Publishing is handled by `scripts/publish_dataset.py` (lands in Plan 2).

### Step 3 - Add the benchmark manifest

File: `eval/configs/benchmarks/croatian_sentiment/benchmark.yaml`

```yaml
benchmark: croatian_sentiment
version: 0.1.0
description: "Croatian Sentiment: three-class sentiment classification over tweets and product reviews."
license: CC-BY-4.0
maintainers:
  - name: Your Name
    github: your-handle
citation: "Author et al. 2024"
paper: https://arxiv.org/abs/0000.00000
homepage: https://example.com/croatian-sentiment
tasks:
  ranked: [sentiment]
languages:
  available: [hr]
  roadmap: [sr, bs, mne]
aggregation:
  formula: unweighted_mean
  over: primary_task_scores
  require_complete_ranked_coverage: true
seeds: [42, 43, 44, 45, 46]
sponsor: Recrewty
```

This manifest is validated by
[`eval/schemas/benchmark_spec.json`](eval/schemas/benchmark_spec.json). Every
required field is enforced.

### Step 4 - Add each task YAML

File: `eval/configs/benchmarks/croatian_sentiment/tasks/sentiment.yaml`

```yaml
benchmark: croatian_sentiment
task: sentiment
status: ranked
task_type: multiclass_classification
languages:
  available: [hr]
  ranked: [hr]
  roadmap: [sr, bs, mne]
dataset:
  source_type: huggingface
  config: default
  per_language:
    hr:
      public_repo: teacompany/croatian-sentiment-hr
      private_repo: teacompany/croatian-sentiment-hr-private
  splits:
    public: [train, validation, test]
    labeled_public: [train, validation]
    labeled_private: [test]
inputs:
  fields: [text]
  # `idx` is the canonical row id used by the published BCMS-SuperGLUE
  # datasets. Match that here so the predict + score alignment works.
  id_field: idx
metrics:
  primary: [f1_macro]
  report: [f1_macro, accuracy]
  task_score: f1_macro
prompts:
  hr:
    template_id: croatian_sentiment_hr_v1
training:
  learning_rate: 2.0e-5
  batch_size: 32
  num_epochs: 5
  warmup_ratio: 0.1
  weight_decay: 0.01
  early_stopping_patience: 3
  metric_for_best_model: f1_macro
```

Validated by [`eval/schemas/task_spec.json`](eval/schemas/task_spec.json).

### Step 5 - Add the prompt bundle

File: `eval/configs/benchmarks/croatian_sentiment/prompts/sentiment/hr.yaml`

```yaml
language: hr
template_id: croatian_sentiment_hr_v1
prompt: |
  Odredi sentiment sljedećeg teksta. Odgovor mora biti jedno od:
  pozitivan, negativan, neutralan.
  Tekst: "{{text}}"
  Sentiment:
labels:
  pozitivan: 2
  neutralan: 1
  negativan: 0
```

Prompts live as data, not Python. Adding another language means another prompt
file, nothing else.

### Step 6 - Run local validation

```bash
cd eval
source .venv/bin/activate

balkanbench validate-config \
  configs/benchmarks/croatian_sentiment/benchmark.yaml \
  --schema benchmark_spec

balkanbench validate-config \
  configs/benchmarks/croatian_sentiment/tasks/sentiment.yaml \
  --schema task_spec

balkanbench list benchmarks
# expects: croatian_sentiment, superglue

balkanbench list tasks
# expects: croatian_sentiment.sentiment, superglue.boolq, ...

balkanbench list languages
# expects: hr, sr, ... dynamically discovered
```

If any command exits non-zero, fix the config and re-run. Reviewers will not
look at PRs that fail local validation.

### Step 7 - Open the PR

Branch: `feat/benchmark-croatian-sentiment`.

PR title: `feat(benchmarks): add croatian_sentiment benchmark`.

PR body (template):

```
Linked issue: #<number>

## Summary
- New benchmark: croatian_sentiment (hr)
- One ranked task: sentiment (multiclass_classification, f1_macro)
- Data source: teacompany/croatian-sentiment-hr on Hugging Face (CC-BY-4.0)
- Maintainer: @your-handle

## Files added
- configs/benchmarks/croatian_sentiment/benchmark.yaml
- configs/benchmarks/croatian_sentiment/tasks/sentiment.yaml
- configs/benchmarks/croatian_sentiment/prompts/sentiment/hr.yaml

## Validation
- [x] `balkanbench validate-config ... --schema benchmark_spec` passes
- [x] `balkanbench validate-config ... --schema task_spec` passes
- [x] `balkanbench list tasks` shows croatian_sentiment.sentiment
- [x] All existing tests still green
```

### Step 8 - CI + review

CI runs `validate-configs.yml` and the full test suite. A maintainer reviews:

- identity and license match what was agreed in the issue
- dataset card on HF matches the manifest
- schemas pass
- task implementation (if custom) follows the
  [task interface rules](docs/methodology/benchmark_contract.md)

### Step 9 - Merge and release

On merge, the new benchmark ships in the next minor release
(`0.2.0`, `0.3.0`, ...) per
[`docs/methodology/versioning.md`](docs/methodology/versioning.md). New
benchmarks are always at least a minor bump.

---

## Adding a task to an existing benchmark

Similar to adding a benchmark, but only:

1. Open a `Propose Task` issue against the existing benchmark.
2. Add `eval/configs/benchmarks/<existing>/tasks/<new_task>.yaml`.
3. Add prompts under `eval/configs/benchmarks/<existing>/prompts/<new_task>/<lang>.yaml`.
4. Update the benchmark manifest's `tasks.ranked` (or `diagnostic`) list.
5. Open PR, run validation, get review.

A new task is a **minor** version bump for that benchmark.

## Adding a model

1. Open a `Propose Model` issue declaring model name, HF repo, license,
   intended tier (`experimental` or `official`), and submitter identity.
2. Add `eval/configs/models/experimental/<model>.yaml` conforming to
   [`eval/schemas/model_spec.json`](eval/schemas/model_spec.json). Use the
   namespaced `task_overrides` keys: `{benchmark}.{task}`.
3. Open a PR with the config. Running the model through the official pipeline
   and committing the result artifact can follow in a separate PR (see
   [Submitting a result](#submitting-a-result)).
4. Promotion from `experimental` to `official` tier happens after community
   review and at least one successful official run.

## Submitting a result

1. Run the benchmark locally or on GCP:
   ```bash
   balkanbench predict --model <name> --benchmark <bench> --language <lang>
   balkanbench submit eval/results/local/<name>/ --out submission.json
   ```
2. Upload the predictions package (the tarball that `submit` produces) to a
   permanent URL (HF repo file, GCS object, GitHub release asset).
3. Open a `Submission` issue with the `submission.json` metadata and the
   package URL.
4. A maintainer scores the predictions against private labels on GCP and
   commits the scored artifact and updated `benchmark_results.json`.

Full submission flow: [`docs/governance/submissions.md`](docs/governance/submissions.md).

## Local development setup

```bash
# Python package
cd eval
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# Run the full test suite
pytest

# Run lint + format + type checks (same gates CI enforces)
ruff check .
ruff format --check .
mypy

# Run the frontend (separate process)
cd ../frontend
npm install
npm run dev
```

See [`eval/README.md`](eval/README.md) for package layout, and the project
specs under `docs/superpowers/` for the full design intent.

## Commit conventions

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat(benchmarks): add croatian_sentiment benchmark
fix(metrics): MultiRC f1_a should use f1 of the positive class
test(config): cover malformed YAML path in load_yaml_with_schema
docs(contrib): clarify dataset publication step
refactor(scaffolding): close four extensibility gaps before Plan 2
```

Scope hints: `benchmarks`, `tasks`, `metrics`, `models`, `scoring`,
`leaderboard`, `cli`, `schemas`, `config`, `docker`, `ci`, `docs`, `gcp`,
`frontend`, `scaffolding`.

No Claude or AI-authored co-author trailer lines. Human authors only.

## Code of conduct

We expect courteous, respectful behaviour in issues, PRs, reviews, and any
space that uses the BalkanBench name. In short: assume good faith, disagree on
content not on people, and flag anything that crosses a line to a maintainer.
A full code of conduct document lands alongside the v0.1 release. Until it is
in the repo, reports go to
[perovicmitar@gmail.com](mailto:perovicmitar@gmail.com).
