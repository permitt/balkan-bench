# Benchmark Contract (v0.1)

This is the frozen methodology for the BalkanBench v0.1 release. A score
that claims to be on this benchmark means every item below was honoured.

## Languages

- Serbian (`sr`) is the only available language for v0.1.
- Croatian (`hr`), Montenegrin (`cnr`), and Bosnian (`bs`) are on the
  roadmap; each will ship as a sibling public dataset (e.g.
  `permitt/superglue-croatian`) once the data is ready. Adding a
  language does not change the v0.1 contract for Serbian.

## Ranked tasks

| Task    | Primary metric | Schema source                                                               |
|---------|----------------|-----------------------------------------------------------------------------|
| BoolQ   | `accuracy`     | [`boolq.yaml`](../../eval/configs/benchmarks/superglue/tasks/boolq.yaml)    |
| CB      | `f1_macro`     | [`cb.yaml`](../../eval/configs/benchmarks/superglue/tasks/cb.yaml)          |
| COPA    | `accuracy`     | [`copa.yaml`](../../eval/configs/benchmarks/superglue/tasks/copa.yaml)      |
| RTE     | `accuracy`     | [`rte.yaml`](../../eval/configs/benchmarks/superglue/tasks/rte.yaml)        |
| MultiRC | `f1_a`         | [`multirc.yaml`](../../eval/configs/benchmarks/superglue/tasks/multirc.yaml)|
| WSC     | `accuracy`     | [`wsc.yaml`](../../eval/configs/benchmarks/superglue/tasks/wsc.yaml)        |

Diagnostics: AX-b (`matthews_correlation`) and AX-g (`accuracy`). They
are reported but do not contribute to the main score.

## Main benchmark score

Unweighted arithmetic mean of the 6 primary task scores.

- **Not computed** when any ranked task is missing. The leaderboard
  export rejects partial rankable rows.
- **Partial runs** (e.g. ModernBERTić small, 5/6) are displayed with a
  `(N/6) partial` flag and receive no rank.

## Seeds

Every official run uses **5 fixed seeds** (recorded in the model config,
default `[42, 43, 44, 45, 46]`). Reports include per-seed scores, mean,
and sample standard deviation.

## Hyperparameter search protocol

1. Search on `train -> validation` only, with Optuna `TPESampler(seed=sampler_seed)`.
2. Single seed per trial; optional re-rank of top-k across seeds before freezing.
3. Freeze one final config per (model, task, language) into `eval/configs/models/official/{model}.yaml`.
4. Train the final system on `train + validation` with the frozen config.
5. Evaluate once on the hidden `test` split, repeated across 5 seeds.
6. Report per-seed scores + mean + stdev + all provenance (image digest,
   git SHA, dataset revision, config hash, sampler seed, num_trials,
   search space id, early-stopping policy).

Forbidden:

- tuning on `test`
- selecting the best seed by `test` score
- changing hyperparameters after seeing `test` results

## Hidden test labels

- `train` and `validation` splits are published with labels on
  `permitt/superglue-serbian`.
- `test` split is published **without** labels.
- Test labels live on `permitt/superglue-private` (gated).
  `balkanbench score` reads them via `HF_OFFICIAL_TOKEN` and fails
  loudly on missing or extra `example_id`s.
- Public users can generate predictions with `balkanbench predict`;
  they cannot score test locally.

## Result artifact

Every scored official run emits a `result.json` conforming to
[`eval/schemas/result_artifact.json`](../../eval/schemas/result_artifact.json).
Required fields include `model_revision`, `code_revision`,
`dataset_revision`, `image_digest`, `config_hash`,
`test_predictions_hash`, and `sponsor`.

## Leaderboard export

Produced by `balkanbench leaderboard export` from artifacts on disk.
Schema: [`eval/schemas/leaderboard_export.json`](../../eval/schemas/leaderboard_export.json).
Format notes: [`docs/leaderboard/format.md`](../leaderboard/format.md).

## Throughput

Optional but shipped with v0.1. Reference hardware: NVIDIA L4 24GB, fp16,
`torch.compile` off. Protocol:
[`docs/methodology/throughput.md`](throughput.md).

## Versioning

Semver for the benchmark. Rules:
[`docs/methodology/versioning.md`](versioning.md).

## Task lifecycle

Ranked -> diagnostic -> experimental -> archived transitions and the
required governance for each:
[`docs/methodology/task_lifecycle.md`](task_lifecycle.md).

## Sponsor

Compute for official v0.1 evaluation is sponsored by
**[Recrewty](https://recrewty.com)**. Every artifact records
`"sponsor": "Recrewty"`.
