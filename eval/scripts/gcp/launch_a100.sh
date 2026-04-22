#!/usr/bin/env bash
# Launch a BalkanBench run on a GCP VM with a single NVIDIA A100 40GB GPU.
#
# Usage:
#   export PROJECT_ID=my-gcp-project
#   export GCS_BUCKET=balkanbench-artifacts
#   export MODEL=bertic MODE=eval TASK=boolq SEED=42
#   bash eval/scripts/gcp/launch_a100.sh
#
# Requires gcloud + gsutil on the caller's PATH and IAM to create Compute
# Engine instances + read the HF_OFFICIAL_TOKEN secret.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${HERE}/common.sh"

default_env
require_env PROJECT_ID GCS_BUCKET MODEL MODE TASK

MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-1g}"
ACCELERATOR="${ACCELERATOR:-type=nvidia-tesla-a100,count=1}"
IMAGE_FAMILY="${IMAGE_FAMILY:-pytorch-latest-gpu}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
DISK_SIZE="${DISK_SIZE:-200}"

VM_NAME="$(vm_name_for "${MODEL}" "${TASK}" "a100")"
COMMAND="$(pick_balkanbench_cmd)"

log "creating VM ${VM_NAME} with A100 in ${ZONE}"

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
log "Remember to tear the VM down when done: gcloud compute instances delete ${VM_NAME} --zone ${ZONE}"
