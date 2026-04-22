# Throughput Reporting

BalkanBench publishes inference throughput alongside quality scores so
readers can pick a model for production by **both** latency and quality,
not just one of them. Compute for official throughput measurement is
sponsored by [Recrewty](https://recrewty.com).

## Reference hardware

| Setting            | Value                 |
|--------------------|-----------------------|
| GPU                | NVIDIA L4, 24 GB VRAM |
| Precision          | fp16                  |
| `torch.compile`    | off                   |
| Batch size         | from task config      |
| Max sequence length| from task config      |

L4 is common as an inference-side production GPU, it fits every v0.1 launch
model, and pinning it removes the cross-hardware noise that makes throughput
numbers hard to compare.

## Protocol

1. Load the model + tokenizer as the task's normal flow does (`HFEncoder.build`).
2. Load the task's **validation** split (no labels needed for timing).
3. Run **2 warmup batches** (discarded) so the CUDA caching allocator, JIT
   autotuners, and attention implementations settle.
4. Run **`min(50, full_pass)` measurement batches**. Each batch is timed with
   `time.perf_counter()`.
5. Report the **median** per-batch wall-clock (less volatile than mean on a
   shared GPU). Convert to `throughput_ex_per_sec = batch_size / median` and
   `throughput_tok_per_sec = ex_per_sec * max_seq_len`.
6. Capture peak GPU memory via `torch.cuda.max_memory_allocated()`.

Every artifact carries `torch_version`, `driver_version`, `hardware`,
`precision`, `batch_size`, `max_seq_len`, `warmup_batches`, and
`measurement_batches` so a reader can repeat the measurement exactly.

## Artifacts

**Per task** (one file per ranked task):
`eval/results/official/{benchmark}-{language}/{model}/throughput/{task}.json`
Validated against [`eval/schemas/task_throughput.json`](../../eval/schemas/task_throughput.json).

**Per model aggregate** (one file per model):
`eval/results/official/{benchmark}-{language}/{model}/throughput.json`
Reports `mean_ex_per_sec` across ranked tasks and `max_peak_vram_mib`.

The leaderboard export picks up the aggregate and adds a
`Throughput (L4, fp16, ex/s)` column alongside the quality columns.

## Caveats

- **fp16 numerical differences.** Some older models are slightly less
  accurate in fp16 than fp32; the throughput artifact reports `precision`
  so a reader can check the matching quality measurement used the same
  setting.
- **Shared GPU noise.** Running on a GPU that is also doing other work
  inflates the median; results should come from a dedicated L4 (Compute
  Engine with the VM reserved for the run) for comparability.
- **Workload sensitivity.** Numbers are specific to the task's
  `batch_size` and `max_seq_len`. Different benchmarks compute task
  throughput with their own batch sizes; cross-benchmark aggregation is
  not meaningful.

## Reproduction

Local single-task:

```bash
cd eval
source .venv/bin/activate

balkanbench throughput \
  --model bertic \
  --benchmark superglue \
  --language sr \
  --task boolq \
  --hardware "NVIDIA L4 24GB" \
  --precision fp16 \
  --out results/official
```

Sweep every ranked task for one model:

```bash
balkanbench throughput \
  --model bertic \
  --benchmark superglue \
  --language sr \
  --hardware "NVIDIA L4 24GB" \
  --precision fp16 \
  --out results/official
```

On GCP (sweep every ranked task for one model on a dedicated L4 VM):

```bash
export PROJECT_ID=<your-gcp-project>
export GCS_BUCKET=<your-bucket>
export MODEL=bertic MODE=throughput BENCHMARK=superglue LANGUAGE=sr

bash eval/scripts/gcp/launch_l4.sh
```

See [`docs/gcp/running_official_eval.md`](../gcp/running_official_eval.md)
for the full GCP walkthrough.

## Out of scope for v0.1

- A100 throughput column (post-launch via the same subcommand with `--hardware "NVIDIA A100 40GB"`).
- CPU throughput.
- TensorRT / ONNX export benchmarks.
- p95 / p99 latency percentiles.
- Time-to-first-token (encoder-only models in v0.1).
