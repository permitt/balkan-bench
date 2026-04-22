# Data Provenance

This document records where BalkanBench data comes from, how it is
adapted, normalised, and published, and how the hidden-label contract
is enforced. Compute for official evaluation is sponsored by
[Recrewty](https://recrewty.com).

## v0.1 scope

- **Benchmark:** SuperGLUE (Serbian adaptation).
- **Language:** Serbian (`sr`) only. Croatian, Montenegrin, and Bosnian
  are on the roadmap and will land as sibling public datasets:
  `permitt/superglue-croatian`, `permitt/superglue-montenegrin`,
  `permitt/superglue-bosnian`.

## Sources

| Role | Hugging Face repo | Access |
|------|-------------------|--------|
| Upstream source | `permitt/superglue` | public or gated (pre-normalisation data) |
| Public release | `permitt/superglue-serbian` | public, no test labels |
| Private labels | `permitt/superglue-private` | gated; access via `HF_OFFICIAL_TOKEN` |

`permitt/superglue` is the existing Serbian SuperGLUE translation owned
by the maintainers; the public release is a normalised, sanitised view
of it.

## Normalisation pipeline

The single-entry-point `balkanbench publish-dataset` CLI (implementation in
[`eval/src/balkanbench/data/publish.py`](../../eval/src/balkanbench/data/publish.py))
produces the public release from the source. Every step is pure-function
and unit-tested against in-memory fixtures in
[`eval/tests/unit/test_data_normalize.py`](../../eval/tests/unit/test_data_normalize.py).

### Steps applied per task config

1. **Download** the config from `permitt/superglue`.
2. **Rename splits** via `data.normalize.rename_splits`. COPA is the
   only v0.1 config that needs a rename: `dev` → `validation`. Other
   configs already ship `train` / `validation` / `test` and are left
   alone.
3. **Strip test labels** via `data.normalize.strip_label_columns`. Any
   column in `LABEL_FIELDS` (currently just `label`) is removed from the
   `test` split only; train and validation splits retain labels.
4. **Attach metadata** via `data.normalize.attach_task_metadata`: every
   row gets `task_id = {benchmark}.{task}.{language}`, a `language`
   column set to the BCMS code, and a stable `example_id` column if
   the upstream did not provide one (`{split}-{i}`).
5. **Build manifest** via `data.manifest.build_manifest`, validated
   against
   [`schemas/dataset_manifest.json`](../../eval/schemas/dataset_manifest.json).
   The manifest records per-split row counts and per-split
   `has_labels`. If `hidden_test_labels=True` but any test split still
   carries labels after step 3, publishing aborts with `ManifestError`.
6. **Render dataset card** via `data.card.render_dataset_card`. Card
   ships with YAML frontmatter, a hidden-label disclosure paragraph, a
   per-config splits table, a reproduction snippet, and a Recrewty
   sponsor block.
7. **Push** the normalised `DatasetDict` to the public repo via
   `DatasetDict.push_to_hub(repo, config_name=...)`; upload `README.md`
   (the card) and `dataset_manifest.json` as top-level files.

The private repo (`permitt/superglue-private`) already holds the test
labels for the Serbian release and is **not** touched by the public
publish flow.

## Hidden-label policy

The public dataset contract is:

- train and validation splits are published **with** labels
- the test split is published **without** labels
- official scoring reads private labels from
  `permitt/superglue-private` using `HF_OFFICIAL_TOKEN`
- `balkanbench score` fails loudly if the private token is absent or
  if any public test `example_id` is missing from the private labels

Rationale: preserves leaderboard integrity while still letting the
community run `balkanbench predict` locally.

Full policy: [`docs/methodology/benchmark_contract.md`](benchmark_contract.md)
(landing with v0.1 release).

## Reproducing the publish locally

```bash
cd eval
source .venv/bin/activate
export HF_OFFICIAL_TOKEN=<your token>

balkanbench publish-dataset \
  --source-repo permitt/superglue \
  --public-repo permitt/superglue-serbian \
  --private-repo permitt/superglue-private \
  --language sr \
  --license CC-BY-4.0 \
  --dataset-revision v0.1.0-data \
  --config boolq --config cb --config copa \
  --config rte  --config multirc --config wsc \
  --dry-run
```

Drop `--dry-run` to actually create the public repo and push. `--dry-run`
returns the built manifest and dataset card without touching HuggingFace.

## Adding languages and new benchmarks

Adding Croatian, Montenegrin, or Bosnian is additive:

1. Publish a sibling upstream repo with the new language's data.
2. Run `balkanbench publish-dataset` with `--language hr|cnr|bs` and a
   sibling `--public-repo` (e.g. `permitt/superglue-croatian`).
3. Add a `languages.available` entry to the task YAMLs under
   `eval/configs/benchmarks/superglue/tasks/*.yaml`.
4. CI will automatically surface the new language in
   `balkanbench list languages`.

Adding a brand-new benchmark (e.g. Serbian-LLM-Eval, MTEB-BCMS, a
community-contributed dataset) is handled via the contribution flow in
[`CONTRIBUTING.md`](../../CONTRIBUTING.md).

## Revision policy

- Dataset revisions follow semver against the published HF repo tags
  (`v0.1.0-data`, `v0.1.1-data`, ...). Major changes to test labels or
  split composition bump the data major version.
- Every published result artifact records `dataset_revision` so a
  score is always traceable to the exact snapshot it was computed
  against.
- Test labels are never silently edited. Relabelling is a major data
  version bump, and the prior revision remains accessible on HF.

## Open questions for v0.2+

- Croatian, Montenegrin, Bosnian data acquisition and adaptation
  pipelines.
- Serbian-LLM-Eval (Aleksa Gordić) adapter: generative tasks have a
  different data shape; we will extend `data.normalize` or add a
  sibling module when that benchmark lands.
- MTEB-BCMS embeddings evaluation: different scoring paradigm, may need
  its own publishing helpers.
