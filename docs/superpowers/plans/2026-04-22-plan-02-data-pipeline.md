# BalkanBench v0.1 - Plan 2: Data Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use `- [ ]` syntax.

**Goal:** A `balkanbench publish-dataset` subcommand that downloads
`permitt/superglue`, renames COPA split `dev` → `validation`, strips test
labels, generates a dataset manifest + dataset card, and pushes to the public
`permitt/superglue-serbian` repo. Unit + integration tests exercise the logic
against local fixtures; the real HF push is a manual step the maintainer runs
once with `HF_OFFICIAL_TOKEN` in env.

**Architecture:** Logic lives in `src/balkanbench/data/` as composable pure
functions (`normalize_splits`, `strip_test_labels`, `build_manifest`,
`render_dataset_card`). An orchestrator (`publish_dataset`) ties them together
and is the single thing that touches HuggingFace (`datasets.load_dataset` +
`huggingface_hub.HfApi`). Tests monkeypatch those two HF entry points so no
network is required.

**Tech Stack:** `datasets>=2.20`, `huggingface_hub>=0.24` in a new `[publish]`
optional extra (maintainer-only deps, not pulled in for plain users or CI).

**Branch:** `feature/code-for-eval` (continues from Plan 1).

---

## Task 1: Add `[publish]` optional extra

**Files:** modify `eval/pyproject.toml`

- [ ] Add optional-dependency group with `datasets>=2.20`, `huggingface_hub>=0.24`, `hf-transfer` (for fast downloads, opt-in at runtime).
- [ ] Install locally: `uv pip install -e ".[dev,publish]"`
- [ ] Commit: `build(eval): add [publish] optional extra for data pipeline deps`

## Task 2: `data.normalize` module

**Files:**
- Create: `eval/src/balkanbench/data/__init__.py`
- Create: `eval/src/balkanbench/data/normalize.py`
- Create: `eval/tests/unit/test_data_normalize.py`

Functions:
- `rename_splits(mapping: dict[str, dict[str, str]], dataset_dict) -> DatasetDict`: per-config rename rules; primary use case is `{"copa": {"dev": "validation"}}`.
- `strip_label_columns(dataset_dict, split: str, label_fields: list[str]) -> DatasetDict`: drops label columns from a split.
- `attach_task_metadata(dataset_dict, *, task_id: str, language: str) -> DatasetDict`: adds `task_id`, `language`, and (if missing) `example_id` columns.

TDD: build tiny in-memory `DatasetDict` fixtures (pyarrow + datasets), assert transformations.

Commit: `feat(data): split/label normalization helpers`

## Task 3: `data.manifest` module

**Files:**
- Create: `eval/src/balkanbench/data/manifest.py`
- Create: `eval/tests/unit/test_data_manifest.py`

`build_manifest(benchmark: str, language: str, public_repo: str, private_repo: str | None, configs: dict[str, DatasetDict], *, dataset_revision: str, license: str, hidden_test_labels: bool) -> dict`

Output conforms to `schemas/dataset_manifest.json`. Validate the result inline
via `jsonschema`; fail loudly on drift.

Commit: `feat(data): dataset manifest builder`

## Task 4: `data.card` module (dataset card renderer)

**Files:**
- Create: `eval/src/balkanbench/data/card.py`
- Create: `eval/src/balkanbench/data/_card_template.md` (Jinja-less f-string template - no extra dep)
- Create: `eval/tests/unit/test_data_card.py`

`render_dataset_card(manifest: dict, *, sponsor: str = "Recrewty") -> str`

Card must include:
- benchmark name, language, license, revision
- "test labels are hidden" disclosure block
- "compute sponsored by Recrewty" footer
- per-config table: splits + row counts + has_labels
- reproduction instructions (`balkanbench predict ...`)

Commit: `feat(data): HF dataset card renderer`

## Task 5: `data.publish` orchestrator

**Files:**
- Create: `eval/src/balkanbench/data/publish.py`
- Create: `eval/tests/unit/test_data_publish.py`

`publish_dataset(source_repo, public_repo, *, private_repo, language, license, dataset_revision, configs_to_publish, dry_run=False) -> PublishReport`

Steps:
1. Require `HF_OFFICIAL_TOKEN` (raise with a clear message if absent).
2. `datasets.load_dataset(source_repo, config)` for each config.
3. Apply `rename_splits` (COPA `dev` → `validation`).
4. Apply `strip_label_columns` on `test`.
5. Build manifest via `data.manifest.build_manifest`, validate against schema.
6. Render dataset card via `data.card.render_dataset_card`.
7. `HfApi.create_repo(public_repo, repo_type="dataset", private=False, exist_ok=True)`.
8. Push each config via `DatasetDict.push_to_hub(public_repo, config_name=...)`.
9. Upload `README.md` (card) + `dataset_manifest.json` as top-level files.
10. If `dry_run=True`, skip HF side effects and return the manifest/card text for inspection.

Tests monkeypatch `datasets.load_dataset` and `HfApi` to sinks; no network.

Commit: `feat(data): publish_dataset orchestrator (dry-run verified)`

## Task 6: `balkanbench publish-dataset` CLI wrapper

**Files:**
- Modify: `eval/src/balkanbench/cli/main.py`
- Create: `eval/src/balkanbench/cli/publish.py`
- Create: `eval/tests/unit/test_cli_publish.py`

CLI signature:
```
balkanbench publish-dataset \
  --source-repo permitt/superglue \
  --public-repo permitt/superglue-serbian \
  --private-repo permitt/superglue-private \
  --language sr \
  --license CC-BY-4.0 \
  --dataset-revision v0.1.0-data \
  [--config boolq --config cb ...] \
  [--dry-run]
```

The CLI fails loudly if `datasets` or `huggingface_hub` is not importable,
pointing to `pip install 'balkanbench[publish]'`.

Commit: `feat(cli): add publish-dataset subcommand`

## Task 7: Integration test - end-to-end with a fake HF source

**Files:** `eval/tests/integration/test_publish_e2e.py`

Build a local `DatasetDict` in a tmp dir that mimics `permitt/superglue`
(tiny fixtures: 4 rows per split, one config per of boolq, cb, copa with
`dev` split). Monkeypatch `datasets.load_dataset` to return these fixtures.
Monkeypatch `HfApi` to record calls. Run the orchestrator in `dry_run=False`.
Assert:
- `HfApi.create_repo` called for `permitt/superglue-serbian`
- `push_to_hub` called once per config
- COPA's pushed DatasetDict has `validation` not `dev`
- BoolQ's pushed `test` split has no `label` column
- A manifest is uploaded with `hidden_test_labels=true`
- A dataset card is uploaded mentioning Recrewty

Commit: `test(data): integration test for publish-dataset end to end`

## Task 8: Docs - publication walkthrough

**Files:** `docs/methodology/data_provenance.md`

Minimum content for v0.1:
- source: `permitt/superglue`
- normalization rules (COPA rename, test label stripping)
- public destination: `permitt/superglue-serbian`
- private labels location: `permitt/superglue-private` (gated by `HF_OFFICIAL_TOKEN`)
- license + revision policy
- how to reproduce the publish locally (`balkanbench publish-dataset --dry-run`)

Commit: `docs(methodology): document data provenance and publication`

## End of Plan 2

Success state:
- `balkanbench publish-dataset --dry-run ...` produces a validated manifest and a dataset card with Recrewty sponsorship
- All tests green, coverage still ≥ 80% overall
- Real publish command is one env var (`HF_OFFICIAL_TOKEN`) away from live
- Plan 3 (task layer + metrics + evaluator) can start next
