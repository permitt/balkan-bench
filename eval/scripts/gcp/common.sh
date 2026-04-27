#!/usr/bin/env bash
# Shared helpers for the BalkanBench GCP launchers.
#
# Intentional design: these scripts are thin wrappers around gcloud; the
# official scoring environment is a VM that boots the pinned balkanbench
# Docker image, fetches HF_OFFICIAL_TOKEN from Secret Manager, runs the
# requested subcommand, uploads artifacts to GCS, and shuts down.

set -euo pipefail

log() {
  printf '[balkanbench-gcp %s] %s\n' "$(date -u +%H:%M:%SZ)" "$*"
}

die() {
  printf >&2 '[balkanbench-gcp ERROR] %s\n' "$*"
  exit 1
}

require_env() {
  local name
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      die "required env var is unset: ${name}"
    fi
  done
}

default_env() {
  : "${PROJECT_ID:=}"
  : "${ZONE:=us-central1-a}"
  : "${MODE:=eval}"
  : "${MODEL:=bertic}"
  : "${BENCHMARK:=superglue}"
  : "${LANGUAGE:=sr}"
  : "${DATASET_REVISION:=v0.1.0-data}"
  : "${IMAGE:=gcr.io/${PROJECT_ID}/balkanbench:v0.1.0}"
  : "${GCS_BUCKET:=}"
  : "${HF_SECRET_NAME:=balkanbench-hf-official-token}"
  : "${SEED:=42}"
  : "${TASK:=}"
}

vm_name_for() {
  # Arguments: model, task (optional), gpu
  local model="${1}"
  local task="${2:-all}"
  local gpu="${3}"
  local stamp
  stamp="$(date -u +%Y%m%d-%H%M%S)"
  printf 'bb-%s-%s-%s-%s-%s' \
    "${BENCHMARK}" "${LANGUAGE}" "${model}" "${task}" "${gpu}-${stamp}" \
    | tr '_' '-' \
    | cut -c1-63
}

fetch_hf_token() {
  require_env PROJECT_ID HF_SECRET_NAME
  gcloud secrets versions access latest \
    --project "${PROJECT_ID}" \
    --secret "${HF_SECRET_NAME}"
}

upload_artifacts() {
  local src="${1}"
  local dest
  dest="gs://${GCS_BUCKET}/runs/${BENCHMARK}-${LANGUAGE}/${MODEL}/${MODE}/$(date -u +%Y%m%d-%H%M%S)"
  require_env GCS_BUCKET
  log "uploading ${src} -> ${dest}"
  gsutil -m cp -r "${src}" "${dest}"
}

pick_balkanbench_cmd() {
  # Translate MODE + MODEL + TASK + SEED into the balkanbench subcommand.
  local out
  out="/workspace/results"
  case "${MODE}" in
    eval)
      printf 'balkanbench eval --model %s --benchmark %s --task %s --language %s --seeds %s --out %s' \
        "${MODEL}" "${BENCHMARK}" "${TASK}" "${LANGUAGE}" "${SEED}" "${out}"
      ;;
    predict)
      printf 'balkanbench predict --model %s --benchmark %s --task %s --language %s --seed %s --out %s' \
        "${MODEL}" "${BENCHMARK}" "${TASK}" "${LANGUAGE}" "${SEED}" "${out}"
      ;;
    score)
      printf 'balkanbench score --model %s --benchmark %s --task %s --language %s --predictions %s --out %s' \
        "${MODEL}" "${BENCHMARK}" "${TASK}" "${LANGUAGE}" \
        "/workspace/predictions.jsonl" "${out}"
      ;;
    hp-search)
      printf 'balkanbench hp-search --model %s --benchmark %s --task %s --language %s --n-trials %s --out %s' \
        "${MODEL}" "${BENCHMARK}" "${TASK}" "${LANGUAGE}" "${N_TRIALS:-20}" "${out}"
      ;;
    throughput)
      printf 'balkanbench throughput --model %s --benchmark %s --language %s --hardware %s --out %s' \
        "${MODEL}" "${BENCHMARK}" "${LANGUAGE}" "${HARDWARE:-NVIDIA L4 24GB}" "${out}"
      ;;
    run)
      # End-to-end: HP search per task -> 5-seed eval on the chosen split ->
      # leaderboard export (the latter is auto-skipped if TASKS is a strict
      # subset of the language's ranked tasks). TASKS is space-separated and
      # repeats --tasks; empty TASKS means "every ranked task for LANGUAGE".
      local cmd
      cmd="balkanbench run --model ${MODEL} --benchmark ${BENCHMARK} --language ${LANGUAGE}"
      if [[ "${SKIP_HP_SEARCH:-0}" == "1" || "${SKIP_HP_SEARCH:-0}" == "true" ]]; then
        cmd+=" --skip-hp-search"
      else
        cmd+=" --n-trials ${N_TRIALS:-20}"
      fi
      cmd+=" --eval-split ${EVAL_SPLIT:-test}"
      cmd+=" --dataset-revision ${DATASET_REVISION}"
      cmd+=" --benchmark-version ${BENCHMARK_VERSION:-0.1.0}"
      cmd+=" --run-type ${RUN_TYPE:-official}"
      if [[ -n "${TASKS:-}" ]]; then
        for t in ${TASKS}; do
          cmd+=" --tasks ${t}"
        done
      fi
      if [[ -n "${SEEDS:-}" ]]; then
        for s in ${SEEDS}; do
          cmd+=" --seeds ${s}"
        done
      fi
      cmd+=" --out ${out}"
      printf '%s' "${cmd}"
      ;;
    *)
      die "unknown MODE=${MODE} (want one of eval, predict, score, hp-search, throughput, run)"
      ;;
  esac
}
