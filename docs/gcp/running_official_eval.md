# Running Official Evaluation on GCP

Official BalkanBench runs use **Vertex AI Custom Jobs** as the v0.1 path.
Each job is a single container that runs `balkanbench run` end-to-end
(HP search -> multi-seed test eval -> per-task `result.json`) and uploads
its artifacts to GCS via `balkanbench gcs-upload` before tearing down.
Compute sponsored by [Recrewty](https://recrewty.com).

The legacy `launch_a100.sh` / `launch_l4.sh` scripts still ship for
single-mode operations (per-task eval/predict/score/throughput on raw
Compute Engine VMs) but are **not** the recommended path for v0.1
official runs - they require a startup-script orchestration that the
repo does not provide. Use `launch_vertex.sh` for the canonical flow.

## Prerequisites

1. A GCP project with these APIs enabled: `aiplatform`, `cloudbuild`,
   `artifactregistry`, `compute`, `storage`.
2. A GCS bucket for run artifacts.
3. An Artifact Registry Docker repository in your region.
4. `gcloud` (>=458.0.0) on the launching machine, authenticated with an
   identity that can:
   - submit Cloud Build builds
   - push to your Artifact Registry repo
   - create Vertex AI custom jobs
   - write to the GCS bucket
5. An HF read token in your shell env as `HF_TOKEN`. This is passed to
   the container as an env var (no Secret Manager indirection in the
   v0.1 launcher; use Secret Manager + `--service-account` if you need
   to harden this).
6. **GPU quota** in your target region. A100 / L4 quota requests can
   take 24-48h on a fresh project.

## One-time setup

```bash
export PROJECT_ID=<your-gcp-project>
export REGION=us-central1

# bucket + AR repo
gcloud storage buckets create gs://balkanbench-artifacts \
  --project "$PROJECT_ID" --location "$REGION" --uniform-bucket-level-access

gcloud artifacts repositories create balkanbench \
  --project "$PROJECT_ID" --location "$REGION" \
  --repository-format=docker

# IAM: the Compute default SA runs both Cloud Build AND Vertex AI jobs.
PROJECT_NUM=$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')
COMPUTE_SA="${PROJECT_NUM}-compute@developer.gserviceaccount.com"

gcloud artifacts repositories add-iam-policy-binding balkanbench \
  --project "$PROJECT_ID" --location "$REGION" \
  --member "serviceAccount:${COMPUTE_SA}" \
  --role roles/artifactregistry.writer

gcloud storage buckets add-iam-policy-binding gs://balkanbench-artifacts \
  --member "serviceAccount:${COMPUTE_SA}" \
  --role roles/storage.objectAdmin
```

## Build the GPU image

The image bundles `balkanbench[ml]` plus `google-cloud-storage` (for the
`balkanbench gcs-upload` post-step). Cloud Build runs the build remotely
so no local Docker daemon is needed.

```bash
gcloud builds submit eval \
  --project "$PROJECT_ID" \
  --region "$REGION" \
  --config eval/scripts/gcp/cloudbuild.yaml \
  --substitutions=_AR_HOST=${REGION}-docker.pkg.dev,_REPO=balkanbench,_TAG=v0.1.0,_IMAGE_NAME=balkanbench-gpu
```

This pushes `${REGION}-docker.pkg.dev/${PROJECT_ID}/balkanbench/balkanbench-gpu:v0.1.0`
plus a `:latest` alias.

## Launching one cell (single model x language x task set)

```bash
export PROJECT_ID GCS_BUCKET=balkanbench-artifacts
export AR_HOST=${REGION}-docker.pkg.dev AR_REPO=balkanbench
export IMAGE_NAME=balkanbench-gpu IMAGE_TAG=v0.1.0
export HF_TOKEN=hf_...

export MODE=run MODEL=bertic BENCHMARK=superglue LANGUAGE=hr
export TASKS="cb"          # space-separated; empty = every ranked task
export N_TRIALS=20         # Optuna trials per task
export EVAL_SPLIT=test     # 'test' uses gated private repo
export DATASET_REVISION=v0.1.0-data
export ACCELERATOR=l4      # or a100

bash eval/scripts/gcp/launch_vertex.sh
```

Inside the container the launcher runs:

```
balkanbench run --model $MODEL --benchmark $BENCHMARK --language $LANGUAGE \
  --tasks $TASKS --n-trials $N_TRIALS --eval-split $EVAL_SPLIT \
  --dataset-revision $DATASET_REVISION --out /workspace/results \
&& balkanbench gcs-upload /workspace/results "$BALKANBENCH_GCS_OUT"
```

The job uploads everything (sweep state, per-task `result.json`,
`benchmark_results.json` if a full ranked set was run) to
`gs://$GCS_BUCKET/runs/{benchmark}-{language}/{model}/{timestamp}/` and
then exits, releasing the GPU.

## Launching the full official sweep

`launch_batch.sh` triple-loops `MODELS` x `LANGUAGES` x `BATCH_TASKS` and
submits one Vertex AI job per cell. Defaults match the v0.1 official
sweep (9 models x 4 ranked tasks x 2 languages = 72 jobs).

```bash
bash eval/scripts/gcp/launch_batch.sh
```

Override scope via env vars:

```bash
MODELS="bertic galton_v3_large" \
BATCH_TASKS="boolq cb" \
LANGUAGES="hr" \
N_TRIALS=10 \
bash eval/scripts/gcp/launch_batch.sh
```

Cost projection (L4, ~10 min per small-model job, more for 560M XLM-R):
**~$15-20** for the full 72-job v0.1 sweep.

## Post-run: assemble the leaderboard

```bash
# pull every per-task result.json from GCS
gcloud storage rsync -r -x "sweeps/.*|work/.*|.run_fingerprint.json" \
  gs://$GCS_BUCKET/runs/ /tmp/balkanbench-runs/

# flatten the latest-timestamp result per (model, task) into a clean tree
python3 - <<'PY'
from pathlib import Path
import shutil
src_root = Path("/tmp/balkanbench-runs")
dst_root = Path("eval/results/v1")
dst_root.mkdir(parents=True, exist_ok=True)
for src in sorted(src_root.rglob("result.json")):
    rel = src.relative_to(src_root)
    # rel = superglue-{lang}/{model}/{ts}/results/superglue-{lang}/{model}/{task}/result.json
    task_path = str(rel).split("/results/", 1)[1]
    dst = dst_root / task_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
PY

# build the per-language benchmark_results.json the frontend renders
cd eval
uv run python <<'PY'
from pathlib import Path
from balkanbench.leaderboard.export import write_leaderboard_export
RANKED = ["boolq", "cb", "copa", "rte"]   # add multirc/wsc once SR-only HR/MNE are filled
PRIMARY = {"boolq": "accuracy", "cb": "f1_macro", "copa": "accuracy", "rte": "accuracy"}
for lang in ["hr", "mne"]:
    write_leaderboard_export(
        benchmark="superglue", language=lang,
        results_root=Path("results/v1") / f"superglue-{lang}",
        ranked_tasks=RANKED, task_primary_metrics=PRIMARY,
        benchmark_version="0.1.0",
        out_path=Path(f"../frontend/public/leaderboards/superglue-{lang}/benchmark_results.json"),
        seeds=5,
    )
PY
```

## Troubleshooting

- **PENDING for >20 min on A100**: capacity contention is normal in
  `us-central1`. Cancel and resubmit on `ACCELERATOR=l4` (more headroom,
  cheaper, ~3x slower training).
- **Job state RUNNING but logs go silent**: high-volume stdout (5+ epoch
  log lines per second) can hit Cloud Logging's per-job rate limit;
  check `gs://$GCS_BUCKET/runs/.../results/` for the artifact - if it
  appeared the job actually finished. If no artifact after 30 min of
  silence, treat as a real hang.
- **NameError / pickle errors mid-eval on XLM-R + COPA**: the
  `CLSPoolMultipleChoice` wrapper is module-level so it pickles
  cleanly. If you see new variants, rebuild the image after the fix
  with `gcloud builds submit eval ...`.
- **`AccessDeniedException` on push**: the Cloud Build runtime SA
  (`<num>-compute@developer.gserviceaccount.com` on projects created
  after April 2024) needs `roles/artifactregistry.writer` on the AR
  repo. Granting only the legacy `cloudbuild` SA is not enough.

See [`./costs.md`](costs.md) for expected GPU-hour budgets and
[`./security.md`](security.md) for the token + audit-log contract.
