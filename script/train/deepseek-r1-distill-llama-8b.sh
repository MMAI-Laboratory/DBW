#!/usr/bin/env bash
export MODEL_FAMILY="${MODEL_FAMILY:-Llama-3}"
export MODEL_NAME="${MODEL_NAME:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
export SAVE_STEM="${SAVE_STEM:-dbw-DeepSeek-R1-Distill-Llama-8B}"
export USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-0}"
E2E_LR="${E2E_LR:-2e-6}"
source "$(dirname "$0")/common/all.sh" "$@"
