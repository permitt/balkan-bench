#!/usr/bin/env bash
# Submit the v0.1 official sweep: every (model, task, language) gets one
# Vertex AI custom job with HP search + 5-seed test eval. Each job
# uploads its result.json + sweep state to a per-(lang, model, task)
# GCS prefix so partial fleets stay independent.
#
# Cost: 9 models * 4 tasks * 2 languages = 72 jobs, ~$0.13 each on L4.
# Bigger models (galton_v3_large, xlm_r_bertic, tesla_xlm) are 2-3x
# slower; budget ~$15-20 total.
#
# Override scope via env: MODELS, TASKS, LANGUAGES are space-separated.
#
# Required env (validated by launch_vertex.sh): PROJECT_ID, GCS_BUCKET,
# AR_HOST, AR_REPO, IMAGE_NAME, IMAGE_TAG, REGION, HF_TOKEN.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${MODELS:=bertic jerteh_355 mmbert mmbert_small tesla_xlm xlm_r_bertic crosloengual_bert galton_v3_base galton_v3_large}"
: "${BATCH_TASKS:=boolq cb rte copa}"
: "${LANGUAGES:=hr mne}"
: "${ACCELERATOR:=l4}"
: "${N_TRIALS:=20}"
: "${EVAL_SPLIT:=test}"
: "${DATASET_REVISION:=v0.1.0-data}"

count=0
for model in ${MODELS}; do
  for lang in ${LANGUAGES}; do
    for task in ${BATCH_TASKS}; do
      count=$((count + 1))
    done
  done
done

printf '[batch] dispatching %d Vertex AI custom jobs\n' "${count}"
printf '[batch]   models:    %s\n' "${MODELS}"
printf '[batch]   tasks:     %s\n' "${BATCH_TASKS}"
printf '[batch]   languages: %s\n' "${LANGUAGES}"
printf '[batch]   gpu:       %s, n_trials=%s, eval=%s, revision=%s\n' \
  "${ACCELERATOR}" "${N_TRIALS}" "${EVAL_SPLIT}" "${DATASET_REVISION}"

i=0
for model in ${MODELS}; do
  for lang in ${LANGUAGES}; do
    for task in ${BATCH_TASKS}; do
      i=$((i + 1))
      printf '\n[batch] (%d/%d) %s/%s/%s\n' "${i}" "${count}" "${model}" "${lang}" "${task}"
      MODE=run \
      MODEL="${model}" \
      BENCHMARK=superglue \
      LANGUAGE="${lang}" \
      TASKS="${task}" \
      ACCELERATOR="${ACCELERATOR}" \
      N_TRIALS="${N_TRIALS}" \
      EVAL_SPLIT="${EVAL_SPLIT}" \
      DATASET_REVISION="${DATASET_REVISION}" \
      bash "${HERE}/launch_vertex.sh"
    done
  done
done

printf '\n[batch] submitted %d jobs. Watch them: gcloud ai custom-jobs list --region %s --filter="state:JOB_STATE_RUNNING" --limit 80\n' \
  "${count}" "${REGION:-us-central1}"
