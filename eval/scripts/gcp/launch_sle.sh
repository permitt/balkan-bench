#!/usr/bin/env bash
# Submit the SLE (Serbian LLM Evaluation) sweep: every (model, task) pair
# for the fixed sr language gets one Vertex AI custom job on an L4 GPU.
# SLE tasks are generative (no training, no HP search, no multi-seed), so
# each job runs `balkanbench eval --benchmark sle ...` directly and uploads
# its result.json to a per-(model, task) GCS prefix, mirroring how
# launch_batch.sh keeps partial fleets independent.
#
# Cost: 5 open-weights models * 9 tasks = 45 jobs on L4. Generative eval has
# no training step, so jobs are short relative to the SuperGLUE HP-search
# sweep - runtime scales with dataset size and each task's max_gen_tokens.
#
# API models (sle-gpt-4-1, sle-claude-sonnet-5, etc.) are out of scope here:
# they call out to their provider APIs and are scored off-GCP instead. The
# default MODELS list below only contains the 5 open-weights models.
#
# Usage:
#   export PROJECT_ID=my-gcp-project
#   export GCS_BUCKET=balkanbench-artifacts
#   bash eval/scripts/gcp/launch_sle.sh              # submit all 45 jobs
#   bash eval/scripts/gcp/launch_sle.sh --dry-run     # print, don't submit
#   MODELS=sle-qwen3-5-4b SLE_TASKS=boolq bash eval/scripts/gcp/launch_sle.sh
#
# Override scope via env: MODELS, SLE_TASKS are space-separated.
#
# Required env: PROJECT_ID, GCS_BUCKET, AR_HOST, AR_REPO, IMAGE_NAME,
# IMAGE_TAG, REGION. HF_TOKEN is fetched from Secret Manager (HF_SECRET_NAME)
# via common.sh's fetch_hf_token if not already exported.
#
# launch_vertex.sh only supports MODE=run today, and `balkanbench run`
# requires --seeds even for generative tasks that never use them (the model
# YAMLs here don't declare any). So this script builds and submits its own
# Vertex AI worker pool spec for MODE=sle-eval (mirroring launch_vertex.sh's
# shape) instead of delegating to it the way launch_batch.sh does.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

DRY_RUN=0
for arg in "$@"; do
  case "${arg}" in
    --dry-run)
      DRY_RUN=1
      ;;
    *)
      die "unknown argument: ${arg} (only --dry-run is supported)"
      ;;
  esac
done

# Pin the sle-track dataset/benchmark defaults before default_env fills in
# the generic (superglue-era) fallbacks - default_env only assigns a var if
# it is still unset, so anything set here wins.
: "${DATASET_REVISION:=v1.0.0-sle-data}"
: "${BENCHMARK_VERSION:=1.0.0}"
: "${RUN_TYPE:=official}"

default_env

require_env PROJECT_ID GCS_BUCKET

: "${MODELS:=sle-qwen3-5-4b sle-qwen3-5-9b sle-gemma-4-e2b-it sle-gemma-4-e4b-it sle-granite-4-1-8b sle-ministral-8b sle-olmo3-7b sle-smollm3-3b sle-phi4-mini sle-yugogpt}"
: "${SLE_TASKS:=arc_challenge arc_easy boolq hellaswag nq_open openbookqa piqa triviaqa winogrande}"
: "${REGION:=us-central1}"
: "${AR_HOST:=${REGION}-docker.pkg.dev}"
: "${AR_REPO:=balkanbench}"
: "${IMAGE_NAME:=balkanbench-gpu}"
: "${IMAGE_TAG:=v0.1.0}"
# Default is the L4 tier; override BOTH for bigger models, e.g. the A100
# 80GB pairing for >10B-param models:
#   ACCELERATOR_TYPE=NVIDIA_A100_80GB MACHINE_TYPE=a2-ultragpu-1g
: "${MACHINE_TYPE:=g2-standard-8}"
: "${ACCELERATOR_TYPE:=NVIDIA_L4}"

if [[ "${DRY_RUN}" -eq 1 ]]; then
  : "${HF_TOKEN:=dry-run-placeholder-token}"
else
  : "${HF_TOKEN:=$(fetch_hf_token)}"
fi

CONTAINER_URI="${AR_HOST}/${PROJECT_ID}/${AR_REPO}/${IMAGE_NAME}:${IMAGE_TAG}"

count=0
for model in ${MODELS}; do
  for task in ${SLE_TASKS}; do
    count=$((count + 1))
  done
done

printf '[sle] %s %d Vertex AI custom jobs\n' \
  "$([[ "${DRY_RUN}" -eq 1 ]] && echo "previewing" || echo "dispatching")" "${count}"
printf '[sle]   models: %s\n' "${MODELS}"
printf '[sle]   tasks:  %s\n' "${SLE_TASKS}"
printf '[sle]   gpu:    %s (%s), revision=%s, benchmark-version=%s, run-type=%s\n' \
  "${ACCELERATOR_TYPE}" "${MACHINE_TYPE}" "${DATASET_REVISION}" "${BENCHMARK_VERSION}" "${RUN_TYPE}"

i=0
for model in ${MODELS}; do
  for task in ${SLE_TASKS}; do
    i=$((i + 1))
    printf '\n[sle] (%d/%d) %s/%s\n' "${i}" "${count}" "${model}" "${task}"

    MODE=sle-eval
    MODEL="${model}"
    BENCHMARK=sle
    LANGUAGE=sr
    TASK="${task}"
    FULL_CMD="$(pick_balkanbench_cmd)"
    SUBCMD="${FULL_CMD#balkanbench }"

    TIMESTAMP="$(date -u +%Y%m%d-%H%M%S)"
    JOB_NAME="bb-sle-sr-${model}-${task}-${TIMESTAMP}"
    BASE_OUTPUT_DIR="gs://${GCS_BUCKET}/runs/sle-sr/${model}/${task}/${TIMESTAMP}"

    # Same local-write-then-upload pattern as launch_vertex.sh: balkanbench
    # writes to /workspace/results inside the container, then
    # `balkanbench gcs-upload` syncs the tree to $BALKANBENCH_GCS_OUT before
    # the worker tears down.
    LOCAL_OUT=/workspace/results
    SUBCMD="${SUBCMD//\/workspace\/results/${LOCAL_OUT}}"
    WRAPPED_CMD="balkanbench ${SUBCMD} && balkanbench gcs-upload ${LOCAL_OUT} \"\$BALKANBENCH_GCS_OUT\""
    ARGS_JSON="$(python3 -c '
import json, sys
print(json.dumps(sys.argv[1]))
' "${WRAPPED_CMD}")"

    CONFIG_FILE="$(mktemp -t balkanbench-sle.XXXXXX.json)"
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
      "command": ["sh", "-c"],
      "args": [${ARGS_JSON}],
      "env": [
        {"name": "HF_TOKEN", "value": "${HF_TOKEN}"},
        {"name": "BALKANBENCH_GCS_OUT", "value": "${BASE_OUTPUT_DIR}"}
      ]
    },
    "diskSpec": {
      "bootDiskType": "pd-ssd",
      "bootDiskSizeGb": 200
    }
  }],
  "baseOutputDirectory": {
    "outputUriPrefix": "${BASE_OUTPUT_DIR}"
  }
}
EOF

    if [[ "${DRY_RUN}" -eq 1 ]]; then
      log "[dry-run] would submit ${JOB_NAME}"
      log "[dry-run]   args:   balkanbench ${SUBCMD}"
      log "[dry-run]   output: ${BASE_OUTPUT_DIR}"
      printf '[dry-run] gcloud ai custom-jobs create --project %s --region %s --display-name %s --config %s --args ""\n' \
        "${PROJECT_ID}" "${REGION}" "${JOB_NAME}" "${CONFIG_FILE}"
      rm -f "${CONFIG_FILE}"
      continue
    fi

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
      > /tmp/balkanbench-sle-job.txt 2>&1 || {
        cat /tmp/balkanbench-sle-job.txt >&2
        rm -f "${CONFIG_FILE}"
        die "gcloud ai custom-jobs create failed for ${model}/${task}"
      }

    cat /tmp/balkanbench-sle-job.txt
    JOB_RESOURCE="$(grep -oE 'projects/[^ ]+/customJobs/[0-9]+' /tmp/balkanbench-sle-job.txt | head -1)"
    log "submitted: ${JOB_RESOURCE}"

    rm -f "${CONFIG_FILE}"
  done
done

printf '\n[sle] %s %d jobs. Watch them: gcloud ai custom-jobs list --region %s --filter="state:JOB_STATE_RUNNING" --limit 80\n' \
  "$([[ "${DRY_RUN}" -eq 1 ]] && echo "previewed" || echo "submitted")" "${count}" "${REGION}"
