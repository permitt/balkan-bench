# Leaderboard Format

The frontend leaderboard is driven by a single JSON file per
(benchmark, language) pair:

```
frontend/public/leaderboards/{benchmark}-{language}/benchmark_results.json
```

Schema: [`eval/schemas/leaderboard_export.json`](../../eval/schemas/leaderboard_export.json).

It is regenerated from on-disk official result artifacts by
`balkanbench leaderboard export`; the source of truth is the per-task
`result.json` files under
`eval/results/official/{benchmark}-{language}/{model}/{task}/result.json`.

## Top-level fields

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
    "boolq":   "accuracy",
    "cb":      "f1_macro",
    "copa":    "accuracy",
    "rte":     "accuracy",
    "multirc": "f1_a",
    "wsc":     "accuracy"
  },
  "throughput": {
    "hardware": "NVIDIA L4 24GB",
    "precision": "fp16",
    "batch_size_policy": "from_task_config",
    "warmup_batches": 2,
    "measurement_batches": 50
  },
  "rows": [ /* ... */ ]
}
```

- `benchmark`, `language`, `benchmark_version`: identity of the export.
- `generated_at`: ISO-8601 UTC timestamp when the export was produced.
- `sponsor`: compute sponsor for the rows in this export. Defaults to
  `"Recrewty"` for v0.1 official runs. Community submissions may ship
  with a per-row `sponsor` override.
- `seeds`: number of seeds used for each official row.
- `ranked_tasks`: the task identifiers that count toward the main
  score, in display order.
- `task_primary_metrics`: the single metric name per task used for
  both the leaderboard column and the main-score average.
- `throughput`: optional metadata block describing how the per-row
  throughput numbers were measured.

## Row fields

```json
{
  "rank": 1,
  "model": "ModernBERTić base",
  "model_id": "permitt/galton-modernbertic-base-bcms-v1",
  "model_revision": "abc123...",
  "params": 395000000,
  "params_display": "395M",
  "results": {
    "boolq":   { "mean": 80.70, "stdev": 0.44 },
    "cb":      { "mean": 78.52, "stdev": 3.82 },
    "copa":    { "mean": 76.84, "stdev": 1.29 },
    "rte":     { "mean": 73.13, "stdev": 0.84 },
    "multirc": { "mean": 67.90, "stdev": 0.47 },
    "wsc":     { "mean": 63.56, "stdev": 2.39 }
  },
  "avg": 73.44,
  "complete": true,
  "tasks_completed": 6,
  "tasks_total": 6,
  "throughput": { "ex_per_sec": 234.5, "peak_vram_mib": 4820 }
}
```

- `rank`: integer rank among complete rankable rows, or `null` for
  partial or non-rankable rows.
- `model`: display name.
- `model_id`: canonical Hugging Face repo id (or other identifier).
- `model_revision`: commit / branch / tag of the published checkpoint.
- `params` / `params_display`: raw count + human-readable.
- `results`: per-task primary-metric dict. Values are either a
  `{mean, stdev}` object or `null` when the task has not been scored
  for this row. Values are percentages on a 0 - 100 scale so the
  display can render them verbatim.
- `avg`: unweighted arithmetic mean of present `mean` values.
- `complete`: true only when every ranked task has a rankable artifact.
- `partial_flag`: `"(N/M) partial"` string, present only when
  `complete` is false.
- `throughput`: optional; populated once L4 throughput artifacts land
  under `eval/results/official/.../{model}/throughput.json`.

## Ranking rules

- Only rows with `complete: true` receive an integer `rank`.
- Ranks are assigned by descending `avg`, ties broken by model name.
- `partial_flag` rows keep `rank: null` and sort below ranked rows
  when the frontend defaults to rank-based sort.

## Regenerating the leaderboard

```bash
cd eval
source .venv/bin/activate

balkanbench leaderboard export \
  --benchmark superglue \
  --language sr \
  --results-dir results/official \
  --out ../frontend/public/leaderboards/superglue-serbian/benchmark_results.json
```

The CLI walks `results/official/{benchmark}-{language}/{model}/{task}/result.json`,
validates each artifact against the per-run schema, assigns ranks, and
writes the file.

## Adding a benchmark or language

New `(benchmark, language)` pairs get their own directory:

```
frontend/public/leaderboards/{benchmark}-{language}/benchmark_results.json
```

The `/leaderboard` page today reads the SuperGLUE-SR file directly.
When a second pair lands (Serbian-LLM-Eval or Croatian SuperGLUE), the
Leaderboard page gains a selector; the JSON format is unchanged.
