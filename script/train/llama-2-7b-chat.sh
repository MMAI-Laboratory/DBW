#!/usr/bin/env bash
export MODEL_FAMILY="${MODEL_FAMILY:-Llama-2}"
export MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-2-7b-chat-hf}"
export SAVE_STEM="${SAVE_STEM:-dbw-Llama-2-7b-chat}"
export USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-1}"
E2E_LR="${E2E_LR:-2e-5}"
source "$(dirname "$0")/common/all.sh" "$@"
