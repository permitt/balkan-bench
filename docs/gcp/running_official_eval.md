# Running Official Evaluation on GCP

Official v0.1 evaluation runs on Google Cloud Compute Engine VMs. BalkanBench
compute is sponsored by [Recrewty](https://recrewty.com).

## Prerequisites

1. A GCP project with Compute Engine + Secret Manager APIs enabled.
2. A GCS bucket for run artifacts.
3. `gcloud` + `gsutil` on the launching machine, authenticated with an
   identity that can:
   - create Compute Engine instances
   - read the `balkanbench-hf-official-token` secret
   - write to the GCS bucket
4. The pinned BalkanBench Docker image pushed to your project's Artifact
   Registry (see [`../../eval/Dockerfile`](../../eval/Dockerfile)).
5. `HF_OFFICIAL_TOKEN` stored as a Secret Manager secret named
   `balkanbench-hf-official-token` (override with `HF_SECRET_NAME`).

## One-time setup

```bash
gcloud secrets create balkanbench-hf-official-token \
  --project "$PROJECT_ID" \
  --replication-policy automatic

printf '%s' "hf_yourActualTokenHere" | \
  gcloud secrets versions add balkanbench-hf-official-token \
    --project "$PROJECT_ID" --data-file=-

gsutil mb -p "$PROJECT_ID" -l us-central1 "gs://balkanbench-artifacts"
```

## Launching a run

Set common env vars once per shell session:

```bash
export PROJECT_ID=<your-gcp-project>
export GCS_BUCKET=balkanbench-artifacts
export ZONE=us-central1-a
export BENCHMARK=superglue
export LANGUAGE=sr
export IMAGE=gcr.io/$PROJECT_ID/balkanbench:v0.1.0
```

### Fine-tune + evaluate on A100 (per-task, per-seed)

```bash
export MODEL=bertic MODE=eval TASK=boolq SEED=42
bash eval/scripts/gcp/launch_a100.sh
```

The launcher creates a short-lived VM, boots the BalkanBench Docker image,
runs `balkanbench eval`, uploads `result.json` to GCS, and prints the name
of the VM so you can tear it down once the run is complete.

### Predict on the public test split (CPU-friendly on L4)

```bash
export MODEL=bertic MODE=predict TASK=boolq SEED=42
bash eval/scripts/gcp/launch_l4.sh
```

### Score predictions against private labels

```bash
export MODEL=bertic MODE=score TASK=boolq
bash eval/scripts/gcp/launch_l4.sh
```

The VM fetches the private labels via `HF_OFFICIAL_TOKEN`, scores against
`predictions.jsonl` (shipped with the VM in `/workspace/predictions.jsonl`),
and uploads the scored artifact.

### Throughput sweep on L4 (reference hardware)

```bash
export MODEL=bertic MODE=throughput
bash eval/scripts/gcp/launch_l4.sh
```

Sweeps every ranked task for the chosen model. ~1.5h wall-clock per model.

### Hyperparameter search on A100

```bash
export MODEL=bertic MODE=hp-search TASK=boolq N_TRIALS=50
bash eval/scripts/gcp/launch_a100.sh
```

## Post-run

All artifacts land under `gs://$GCS_BUCKET/runs/{benchmark}-{language}/{model}/{mode}/{timestamp}/`.

Pull the official artifacts locally and commit them:

```bash
gsutil -m cp -r "gs://$GCS_BUCKET/runs/superglue-sr/bertic/eval/" \
  eval/results/official/superglue-sr/bertic/

cd eval
source .venv/bin/activate
balkanbench leaderboard export \
  --benchmark superglue \
  --language sr \
  --results-dir results/official \
  --out ../frontend/public/leaderboards/superglue-serbian/benchmark_results.json
```

Tear down the VM:

```bash
gcloud compute instances delete "$VM_NAME" --zone "$ZONE"
```

## Troubleshooting

- **`PERMISSION_DENIED` on secret access**: the VM's default service account
  needs `roles/secretmanager.secretAccessor` on the token secret.
- **No GPU quota**: A100 / L4 GPUs need per-region quota increases from GCP.
  L4 is the shortest path for most projects.
- **Image pull fails**: verify Artifact Registry access for the VM's
  service account (`roles/artifactregistry.reader`).

See [`./costs.md`](costs.md) for expected GPU-hour budgets and
[`./security.md`](security.md) for the token + audit-log contract.
