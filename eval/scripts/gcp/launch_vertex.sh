#!/usr/bin/env bash
# Launch `balkanbench run` as a Vertex AI custom training job.
#
# Why Vertex AI over raw Compute Engine: one command picks the GPU + the
# container, mounts a GCS output dir, streams logs to Cloud Logging, and
# tears the worker down on completion. No idle billing, no startup script.
#
# Usage:
#   export PROJECT_ID=my-gcp-project
#   export GCS_BUCKET=balkanbench-artifacts
#   export AR_HOST=us-central1-docker.pkg.dev
#   export AR_REPO=balkanbench
#   export IMAGE_TAG=v0.1.0
#   export HF_TOKEN=hf_...                 # passed to the container as env
#
#   export MODE=run MODEL=bertic BENCHMARK=superglue LANGUAGE=hr TASKS=cb
#   export N_TRIALS=20 EVAL_SPLIT=test DATASET_REVISION=main
#   export ACCELERATOR=a100                # or l4
#   bash eval/scripts/gcp/launch_vertex.sh
#
# Requires: `gcloud` >= 458.0.0 with the `ai` component installed.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

default_env

require_env PROJECT_ID GCS_BUCKET MODEL BENCHMARK LANGUAGE HF_TOKEN

: "${REGION:=us-central1}"
: "${AR_HOST:=${REGION}-docker.pkg.dev}"
: "${AR_REPO:=balkanbench}"
: "${IMAGE_NAME:=balkanbench-gpu}"
: "${IMAGE_TAG:=v0.1.0}"
: "${ACCELERATOR:=a100}"
: "${MODE:=run}"

if [[ "${MODE}" != "run" ]]; then
  die "launch_vertex.sh only supports MODE=run today (got ${MODE!r}); use launch_a100/launch_l4 for legacy modes"
fi

case "${ACCELERATOR}" in
  a100)
    MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
    ACCELERATOR_TYPE="NVIDIA_TESLA_A100"
    ;;
  l4)
    MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
    ACCELERATOR_TYPE="NVIDIA_L4"
    ;;
  *)
    die "unknown ACCELERATOR=${ACCELERATOR} (want a100 or l4)"
    ;;
esac

CONTAINER_URI="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

# Build the balkanbench command via the shared helper, then strip "balkanbench"
# from the front - the container's ENTRYPOINT is already `balkanbench`, so
# Vertex AI's args[] gets just the subcommand and flags.
FULL_CMD="$(pick_balkanbench_cmd)"
SUBCMD="${FULL_CMD#balkanbench }"

# Vertex AI runs are scoped under one base output directory; the CLI mounts
# it inside the container at $AIP_MODEL_DIR. We point --out at that env var
# so artifacts land in GCS without an explicit upload step.
TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
JOB_NAME="bb-${BENCHMARK}-${LANGUAGE}-${MODEL}-${TIMESTAMP}"
BASE_OUTPUT_DIR="gs://${GCS_BUCKET}/runs/${BENCHMARK}-${LANGUAGE}/${MODEL}/${TIMESTAMP}"

# Replace the local /workspace/results target with $AIP_MODEL_DIR so the
# orchestrator writes straight to GCS (FUSE-mounted by Vertex AI).
SUBCMD="${SUBCMD//\/workspace\/results/\$AIP_MODEL_DIR}"

# args list: shell-split SUBCMD into one element per token. The Vertex AI
# config block builds a worker_pool_spec with one machine + one accelerator
# and runs the container's ENTRYPOINT (`balkanbench`) with these args.
ARGS_JSON="$(python3 -c '
import json, shlex, sys
print(",".join(json.dumps(t) for t in shlex.split(sys.argv[1])))
' "${SUBCMD}")"

CONFIG_FILE="$(mktemp -t balkanbench-vertex.XXXXXX.json)"
cat > "${CONFIG_FILE}" <<EOF
{
  "workerPoolSpecs": [{
    "machineSpec": {
      "machineType": "${MACHINE_TYPE}",
      "acceleratorType": "${ACCELERATOR_TYPE}",
      "acceleratorCount": 1
    },
    "replicaCount": 1,
    "containerSpec": {
      "imageUri": "${CONTAINER_URI}",
      "command": ["balkanbench"],
      "args": [${ARGS_JSON}],
      "env": [
        {"name": "HF_TOKEN", "value": "${HF_TOKEN}"}
      ]
    },
    "diskSpec": {
      "bootDiskType": "pd-ssd",
      "bootDiskSizeGb": 200
    }
  }]
}
EOF

log "submitting Vertex AI job ${JOB_NAME}"
log "  image:   ${CONTAINER_URI}"
log "  args:    balkanbench ${SUBCMD}"
log "  output:  ${BASE_OUTPUT_DIR}"

gcloud ai custom-jobs create \
  --project "${PROJECT_ID}" \
  --region "${REGION}" \
  --display-name "${JOB_NAME}" \
  --config "${CONFIG_FILE}" \
  --args "" \
  > /tmp/balkanbench-job.txt 2>&1 || {
    cat /tmp/balkanbench-job.txt >&2
    die "gcloud ai custom-jobs create failed"
  }

cat /tmp/balkanbench-job.txt
JOB_RESOURCE="$(grep -oE 'projects/[^ ]+/customJobs/[0-9]+' /tmp/balkanbench-job.txt | head -1)"

log "submitted: ${JOB_RESOURCE}"
log "stream logs:"
log "  gcloud ai custom-jobs stream-logs ${JOB_RESOURCE} --region ${REGION}"
log "fetch the result.json once the job is done:"
log "  gsutil ls ${BASE_OUTPUT_DIR}/"

rm -f "${CONFIG_FILE}"
