# GCP Compute Costs

Compute for the official v0.1 evaluation is sponsored by
[Recrewty](https://recrewty.com). This document tracks expected GPU-hour
budgets so sponsors and contributors can see where the compute goes.

## Per-task-per-model GPU hours (estimated, v0.1)

Numbers are per seed, single-GPU, on the reference BERTic model. Larger
models scale roughly linearly with parameter count for training (longer)
and sub-linearly for inference (batch-size bound).

| Task    | A100 train | L4 eval | L4 throughput |
|---------|-----------:|--------:|--------------:|
| BoolQ   | ~0.8 h     | ~2 min  | ~2 min        |
| CB      | ~0.3 h     | ~1 min  | ~1 min        |
| COPA    | ~0.4 h     | ~1 min  | ~1 min        |
| RTE     | ~0.2 h     | ~1 min  | ~1 min        |
| MultiRC | ~1.5 h     | ~3 min  | ~3 min        |
| WSC    | ~0.2 h     | ~1 min  | ~1 min        |
| AXb    | -          | ~1 min  | ~1 min        |
| AXg    | -          | ~1 min  | ~1 min        |

Full v0.1 official run, 9 models × 6 ranked tasks × 5 seeds ≈ **160-200 A100 hours**.
Throughput sweep for 9 models ≈ **~15 L4 hours**.

## Budget gates

The launcher scripts do **not** set billing alerts. Set them at the GCP
project level:

```bash
gcloud alpha billing budgets create \
  --billing-account "$BILLING_ACCOUNT" \
  --display-name "balkanbench-v0.1" \
  --budget-amount "500USD" \
  --threshold-rule "percent=0.8" \
  --threshold-rule "percent=1.0"
```

## Sponsorship acknowledgement

Every official result artifact records `"sponsor": "Recrewty"`. The
leaderboard export carries a top-level `"sponsor": "Recrewty"`. The
frontend leaderboard page and the landing page footer render the same
acknowledgement. Community submissions that use different compute may
leave the sponsor field as-is or replace it with their own sponsor; the
leaderboard renders the per-row sponsor when it differs from the
benchmark default.
