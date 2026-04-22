#!/usr/bin/env bash
# Launch a BalkanBench run on a GCP VM with a single NVIDIA L4 24GB GPU.
#
# The L4 is the reference hardware for `balkanbench throughput` (24GB VRAM
# fits every v0.1 launch model). Typical usage: sweep throughput for every
# model on every ranked task, then upload the artifacts for the leaderboard
# export.
#
# Usage:
#   export PROJECT_ID=my-gcp-project
#   export GCS_BUCKET=balkanbench-artifacts
#   export MODEL=bertic MODE=throughput
#   bash eval/scripts/gcp/launch_l4.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

default_env
require_env PROJECT_ID GCS_BUCKET MODEL MODE

MACHINE_TYPE="${MACHINE_TYPE:-g2-standard-8}"
ACCELERATOR="${ACCELERATOR:-type=nvidia-l4,count=1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_SIZE="${DISK_SIZE:-100}"

# For throughput mode we sweep all ranked tasks; TASK is optional.
if [[ "${MODE}" == "throughput" ]]; then
  TASK="${TASK:-all}"
fi
require_env TASK

VM_NAME="$(vm_name_for "${MODEL}" "${TASK}" "l4")"
COMMAND="$(pick_balkanbench_cmd)"

log "creating VM ${VM_NAME} with L4 in ${ZONE}"

gcloud compute instances create "${VM_NAME}" \
  --project "${PROJECT_ID}" \
  --zone "${ZONE}" \
  --machine-type "${MACHINE_TYPE}" \
  --accelerator "${ACCELERATOR}" \
  --maintenance-policy TERMINATE \
  --image-family "${IMAGE_FAMILY}" \
  --image-project "${IMAGE_PROJECT}" \
  --boot-disk-size "${DISK_SIZE}GB" \
  --metadata "install-nvidia-driver=True,balkanbench-image=${IMAGE},balkanbench-command=${COMMAND},balkanbench-gcs-bucket=${GCS_BUCKET},balkanbench-secret=${HF_SECRET_NAME}" \
  --scopes "cloud-platform"

log "VM ${VM_NAME} is up. Pull artifacts from gs://${GCS_BUCKET}/runs/... after the run completes."
log "Tear the VM down with: gcloud compute instances delete ${VM_NAME} --zone ${ZONE}"
