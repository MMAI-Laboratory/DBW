#!/usr/bin/env bash
export MODEL_FAMILY="${MODEL_FAMILY:-Qwen2.5}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B}"
export SAVE_STEM="${SAVE_STEM:-dbw-qwen2.5-7b}"
export USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-0}"
export GBIT_LR=1e-5
source "$(dirname "$0")/common/all.sh" "$@"
