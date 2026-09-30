#!/usr/bin/env bash
export MODEL_FAMILY="${MODEL_FAMILY:-Gemma}"
export MODEL_NAME="${MODEL_NAME:-google/gemma-7b}"
export SAVE_STEM="${SAVE_STEM:-dbw-gemma-7b}"
export USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-1}"
E2E_BATCH_SIZE="${E2E_BATCH_SIZE:-1}"
E2E_ACCUM="${E2E_ACCUM:-32}"
E2E_LR="${E2E_LR:-2e-6}"
source "$(dirname "$0")/common/all.sh" "$@"
