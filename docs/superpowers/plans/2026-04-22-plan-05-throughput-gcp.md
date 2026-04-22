# BalkanBench v0.1 - Plan 5: Throughput + GCP Launchers

> **For agentic workers:** Use superpowers:executing-plans or subagent-driven-development.

**Goal:** Publish a production-picking signal alongside quality scores.
`balkanbench throughput` measures inference examples-per-second, tokens-per-second,
and peak VRAM on a reference GPU (NVIDIA L4 24GB, fp16, `torch.compile` off),
writing schema-valid throughput artifacts. GCP bash launchers run any
`balkanbench` subcommand inside the pinned Docker image on A100 or L4 VMs and
upload the artifacts to GCS.

**Architecture:**
- `balkanbench.throughput.measure` is a pure measurement loop; the HF model +
  tokenizer + dataset are injected, the inner inference step is a callable so
  tests can stub it with cheap numpy ops.
- Per-task throughput artifact at
  `eval/results/official/{benchmark}-{language}/{model}/throughput/{task}.json`.
- Per-model aggregate artifact at
  `eval/results/official/{benchmark}-{language}/{model}/throughput.json`.
- `balkanbench throughput` CLI takes one model and iterates over the benchmark's
  ranked tasks (or a subset via `--task`).
- `eval/scripts/gcp/{common,launch_a100,launch_l4}.sh` are thin wrappers:
  boot a VM from the Docker image, mount `HF_OFFICIAL_TOKEN` from Secret
  Manager, run the requested `balkanbench` subcommand, push results to a
  GCS bucket, auto-shutdown on completion.

**Branch:** `feature/code-for-eval`.

---

## Task 1: `balkanbench.throughput.measure`

- `measure_task_throughput(task, encoder, dataset, *, hardware, precision, warmup, measurement, predict_fn, now)` returns a `ThroughputSample` dataclass.
- `predict_fn(model, batch) -> (predictions, per_batch_seconds)`; tests inject a deterministic fake.
- Median across measurement batches (less volatile than mean).
- Peak VRAM via `torch.cuda.max_memory_allocated()` when `torch.cuda.is_available()`; else 0.
- Commit: `feat(throughput): measurement loop with injectable predict_fn`.

## Task 2: Per-task + per-model artifact writers

- `write_task_throughput(sample, out_dir, task, model, benchmark, language)` - validated against schema extension (new `task_throughput` schema).
- `write_model_throughput_aggregate(samples, out_dir, ...)` - aggregate mean_ex_per_sec + max_peak_vram_mib.
- Commit: `feat(throughput): per-task + per-model aggregate writers`.

## Task 3: `balkanbench throughput` CLI

- Flags: `--model`, `--benchmark`, `--language`, `--task` (repeatable, default = all ranked tasks), `--hardware` (default `l4`), `--precision` (default `fp16`), `--out` (default `eval/results/official/...`).
- Sweep mode: `balkanbench throughput sweep --models-from configs/benchmarks/{benchmark}/benchmark.yaml` iterates over a benchmark's ranked models.
- Commit: `feat(cli): throughput subcommand (single-model + sweep)`.

## Task 4: Schema for the task throughput artifact

- Add `eval/schemas/task_throughput.json`.
- Extend `result_artifact.json.throughput` (already additive).
- Commit: `feat(schemas): add task_throughput JSON Schema`.

## Task 5: GCP launchers

- `eval/scripts/gcp/common.sh`: `check_env PROJECT_ID ZONE MODEL ...`, `fetch_hf_token`, `vm_name_for`, `upload_to_gcs`.
- `eval/scripts/gcp/launch_a100.sh` + `launch_l4.sh`: `create_vm_with_gpu --gpu a100-40gb` (or `l4`), `run_container` with the right `MODE`, `teardown_vm`.
- `bash -n` syntax check in CI.
- Commit: `feat(gcp): A100 + L4 launcher scripts + shared helpers`.

## Task 6: Methodology + GCP docs

- `docs/methodology/throughput.md`: reference hardware, protocol, caveats, how to reproduce locally.
- `docs/gcp/running_official_eval.md`: end-to-end GCP launch walkthrough.
- `docs/gcp/costs.md`: per-task GPU hours + Recrewty sponsorship note.
- `docs/gcp/security.md`: HF_OFFICIAL_TOKEN handling via Secret Manager.
- Commit: `docs(methodology,gcp): throughput protocol + GCP walkthroughs`.

## End of Plan 5

Success state:
- `balkanbench throughput --model bertic --benchmark superglue --language sr --hardware l4` produces both the per-task artifacts and the per-model aggregate.
- GCP launcher scripts exist and pass `bash -n`; real GCP run is a documented manual step.
- Plan 6 (frontend leaderboard page) is next.
