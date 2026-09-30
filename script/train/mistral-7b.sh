#!/usr/bin/env bash
export MODEL_FAMILY="${MODEL_FAMILY:-Mistral}"
export MODEL_NAME="${MODEL_NAME:-mistralai/Mistral-7B-v0.1}"
export SAVE_STEM="${SAVE_STEM:-dbw-mistral-7b}"
export USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-0}"
E2E_BATCH_SIZE="${E2E_BATCH_SIZE:-1}"
E2E_ACCUM="${E2E_ACCUM:-32}"
source "$(dirname "$0")/common/all.sh" "$@"
