# BalkanBench v0.1 - Plan 4: Score + HP Search + Leaderboard Export + Repro Gate

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development. Steps use `- [ ]`.

**Goal:** Close the remaining evaluation pipeline gaps. Ship:

- `balkanbench score` - score a `predictions.jsonl` against the private labels HF repo (requires `HF_OFFICIAL_TOKEN`), emits a schema-valid `result.json`.
- `balkanbench leaderboard export` - walk `eval/results/official/{benchmark}-{language}/` and assemble the frontend's `benchmark_results.json`, with strict completeness enforcement (no partial rankables) and partial-row flagging.
- `balkanbench hp-search` - Optuna TPESampler over a documented search space, single-seed objective on `train -> validation`; writes the frozen config back as YAML.
- Additional CI workflows: `validate-configs.yml`, `validate-fixtures.yml`, `release-check.yml`, `repro-bertic.yml` (nightly + manual dispatch, asserts metric tolerance against a committed baseline).

**Architecture:**

- Pure functions in `balkanbench.leaderboard.export` and `balkanbench.scoring.score`; HF interactions live behind a narrow seam.
- Optuna is a soft dependency (added under the `[ml]` extra already); `hp-search` fails loudly with an install hint if missing.
- Repro gate CI job uses `--no-train` + a committed small fixture to avoid burning GPU hours on every nightly run.

**Tech Stack:** `optuna`, existing HF + sklearn + numpy.

**Branch:** `feature/code-for-eval`.

---

## Task 1: Add optuna to the `[ml]` and `dev` extras

Install and commit.

## Task 2: `balkanbench.leaderboard.export`

- Walks `eval/results/official/{benchmark}-{language}/*/result.json`.
- Validates each artifact against `schemas/result_artifact.json`.
- Builds `benchmark_results.json` conforming to `schemas/leaderboard_export.json`.
- Rank assignment: sorted by `avg` descending; partial rows (< full task coverage) keep `rank=null`, `complete=false`, `partial_flag="(N/M) partial"`.
- Commit: `feat(leaderboard): export artifact assembler`.

## Task 3: `balkanbench leaderboard` subcommand group

- `balkanbench leaderboard export --benchmark X --language Y [--results-dir dir] --out path.json`.
- Writes the JSON; prints a short summary.
- Commit: `feat(cli): leaderboard export subcommand`.

## Task 4: `balkanbench.scoring.score`

- Takes `predictions.jsonl`, task cfg, model cfg, private-labels repo, dataset revision.
- Reads the private labels via `datasets.load_dataset(private_repo, config, split='test', revision=...)` - requires `HF_OFFICIAL_TOKEN`.
- Aligns predictions to labels by `example_id`; fails loudly on any missing / extra id.
- Computes metrics with the task's `score()`, assembles a single-seed result artifact.
- Handles `group_fields` and `metric_columns` (gender_parity side channel) transparently.
- Commit: `feat(scoring): private-label scorer`.

## Task 5: `balkanbench score` CLI

- Arg flags mirror `predict` + `--predictions <path>` + `--out <dir>`.
- Commit: `feat(cli): score subcommand`.

## Task 6: `balkanbench.hp_search`

- Optuna `TPESampler(seed=sampler_seed)`, SQLite study under `results/hpsearch/<sweep_id>/study.db`.
- Objective is the task's `task_score` on the validation split (single seed).
- Search space declared per-task-family (classification / multiple_choice / diagnostic) as defaults; overridable via a YAML block.
- Writes `configs/models/<sweep_id>_<model>.yaml` with the winning hyperparameters + sweep provenance (git sha, dataset revision, sampler seed, num_trials).
- Commit: `feat(hp-search): Optuna TPE search against validation`.

## Task 7: `balkanbench hp-search` CLI

- Flags: `--model`, `--benchmark`, `--task`, `--language`, `--n-trials`, `--sampler-seed`, `--out`.
- Commit: `feat(cli): hp-search subcommand`.

## Task 8: CI workflows

- `.github/workflows/validate-configs.yml` - validates every YAML under `eval/configs/` on change.
- `.github/workflows/validate-fixtures.yml` - validates every fixture JSON/YAML under `eval/tests/fixtures/`.
- `.github/workflows/release-check.yml` - on release tags: repo-wide test + coverage + artifact completeness check.
- `.github/workflows/repro-bertic.yml` - nightly + manual dispatch; runs `balkanbench eval` with `--no-train` on a tiny committed fixture + asserts metric tolerance vs a baseline JSON.
- Commit: `ci: add validate-configs, validate-fixtures, release-check, repro-bertic`.

## End of Plan 4

Success state:
- `balkanbench {score, hp-search, leaderboard export}` all work.
- Four additional CI workflows configured.
- Plan 5 (throughput + GCP launchers) is next.
