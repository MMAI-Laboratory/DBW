#!/usr/bin/env bash
set -euo pipefail

CUDA_DEVICES="${1:-0}"
CUDA_DEVICES="${CUDA_DEVICES%,}"
STAGE="${2:-all}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

: "${MODEL_FAMILY:?MODEL_FAMILY must be set by the model-specific wrapper}"
: "${MODEL_NAME:?MODEL_NAME must be set by the model-specific wrapper}"
: "${SAVE_STEM:?SAVE_STEM must be set by the model-specific wrapper}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
WANDB="${WANDB:-1}"

WBITS="${WBITS:-1.61}"
GROUP_SIZE="${GROUP_SIZE:-128}"
TRAIN_SIZE="${TRAIN_SIZE:-4096}"
VAL_SIZE="${VAL_SIZE:-64}"
TRAINING_SEQLEN="${TRAINING_SEQLEN:-4096}"
BLOCK_BATCH_SIZE="${BLOCK_BATCH_SIZE:-2}"
BLOCK_EPOCHS="${BLOCK_EPOCHS:-2}"
QUANT_LR="${QUANT_LR:-1e-5}"
NORM_LR="${NORM_LR:-2e-5}"
U_LR="${U_LR:-2e-5}"
V_LR="${V_LR:-2e-5}"
BIT_LR="${BIT_LR:-1e-5}"
GBIT_LR="${GBIT_LR:-1e-4}"
BLOCK_EVAL_TASKS="${BLOCK_EVAL_TASKS:-piqa,arc_easy,arc_challenge,hellaswag,winogrande}"
OFF_LOAD_TO_DISK="${OFF_LOAD_TO_DISK:-1}"
GLOBAL_BIT="${GLOBAL_BIT:-1}"
PROGRESSIVE="${PROGRESSIVE:-1}"
PRE_LR="${PRE_LR:-1e-5}"
LAMBDA_SV="${LAMBDA_SV:-1e-6}"
SAVE_STATISTICS="${SAVE_STATISTICS:-1}"

PT_CONTEXT_LEN="${PT_CONTEXT_LEN:-4096}"
E2E_BATCH_SIZE="${E2E_BATCH_SIZE:-2}"
E2E_ACCUM="${E2E_ACCUM:-16}"
LM_EVAL_BS="${LM_EVAL_BS:-2}"
E2E_LR="${E2E_LR:-2e-5}"
E2E_MAX_TRAIN_SAMPLES="${E2E_MAX_TRAIN_SAMPLES:-5120}"
E2E_EPOCHS="${E2E_EPOCHS:-1}"
E2E_EVAL_STEPS="${E2E_EVAL_STEPS:-64}"
E2E_EVAL_TASKS="${E2E_EVAL_TASKS:-piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,lambada_openai,lambada_standard}"
PREPROCESSING_NUM_WORKERS="${PREPROCESSING_NUM_WORKERS:-32}"
E2E_PRECISION="${E2E_PRECISION:-bf16}"
FAST_DATA_SAMPLING="${FAST_DATA_SAMPLING:-1}"
STREAMING_BUFFER_SIZE="${STREAMING_BUFFER_SIZE:-10000}"

as_py_bool() {
  case "$1" in
    1|true|True|TRUE|yes|Yes|YES|on|On|ON) echo True ;;
    *) echo False ;;
  esac
}

FAST_DATA_SAMPLING_BOOL="$(as_py_bool "$FAST_DATA_SAMPLING")"
BLOCK_DATA_ARGS=(--dataset_streaming_buffer_size "$STREAMING_BUFFER_SIZE")
if [[ "$FAST_DATA_SAMPLING_BOOL" == "True" ]]; then
  BLOCK_DATA_ARGS+=(--fast_calib_sampling --redpajama_streaming)
fi

BLOCK_TUNING_ARGS=(--pre_lr "$PRE_LR" --lambda_sv "$LAMBDA_SV")
if [[ "$(as_py_bool "$GLOBAL_BIT")" == "False" ]]; then
  BLOCK_TUNING_ARGS+=(--no_global_bit)
fi
if [[ "$(as_py_bool "$PROGRESSIVE")" == "False" ]]; then
  BLOCK_TUNING_ARGS+=(--no_progressive)
fi
if [[ "$(as_py_bool "$SAVE_STATISTICS")" == "False" ]]; then
  BLOCK_TUNING_ARGS+=(--no_save_statistics)
fi
if [[ "$(as_py_bool "$WANDB")" == "True" ]]; then
  BLOCK_TUNING_ARGS+=(--wandb)
fi
PAD_TOKEN_RESIZE="${PAD_TOKEN_RESIZE:-False}"
E2E_DATA_ARGS=(
  --pad_token_resize "$PAD_TOKEN_RESIZE"
  --fast_pt_sample "$FAST_DATA_SAMPLING_BOOL"
  --redpajama_streaming "$FAST_DATA_SAMPLING_BOOL"
  --streaming_buffer_size "$STREAMING_BUFFER_SIZE"
)

TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
USE_AUTH_TOKEN="${USE_AUTH_TOKEN:-0}"
AUTO_SVD="${AUTO_SVD:-1}"
SVD_DEVICE="${SVD_DEVICE:-auto}"
SVD_TORCH_DTYPE="${SVD_TORCH_DTYPE:-float16}"

BLOCK_MODEL_DIR="${BLOCK_MODEL_DIR:-./output/block_ap_models/${SAVE_STEM}-w${WBITS}g${GROUP_SIZE}}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-${BLOCK_MODEL_DIR}.pt}"
BLOCK_LOG_DIR="${BLOCK_LOG_DIR:-./output/block_ap_log/${SAVE_STEM}-w${WBITS}g${GROUP_SIZE}}"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-./output/e2e-qp-output/${SAVE_STEM}-w${WBITS}g${GROUP_SIZE}-mix-redpajama-wiki-${E2E_MAX_TRAIN_SAMPLES}}"
SVD_CACHE_DIR="./svd/${MODEL_NAME//\//_}_fp16"

BLOCK_HF_ARGS=()
E2E_HF_ARGS=()
SVD_HF_ARGS=()
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
  BLOCK_HF_ARGS+=(--trust_remote_code)
  E2E_HF_ARGS+=(--trust_remote_code True)
  SVD_HF_ARGS+=(--trust-remote-code)
fi
if [[ "$USE_AUTH_TOKEN" == "1" ]]; then
  BLOCK_HF_ARGS+=(--use_auth_token)
  E2E_HF_ARGS+=(--use_auth_token True)
  SVD_HF_ARGS+=(--use-auth-token)
fi

OFFLOAD_ARGS=()
if [[ "$OFF_LOAD_TO_DISK" == "1" ]]; then
  OFFLOAD_ARGS+=(--off_load_to_disk)
fi

PRECISION_ARGS=()
case "$E2E_PRECISION" in
  bf16) PRECISION_ARGS+=(--bf16) ;;
  fp16) PRECISION_ARGS+=(--fp16) ;;
  fp32|none) ;;
  *) echo "Unknown E2E_PRECISION=$E2E_PRECISION. Use bf16, fp16, or fp32." >&2; exit 1 ;;
esac

ensure_svd() {
  if [[ "$AUTO_SVD" != "1" ]]; then
    return
  fi
  if [[ -f "${SVD_CACHE_DIR}/0.pt" ]]; then
    echo ">> Found SVD cache: ${SVD_CACHE_DIR}"
    return
  fi
  echo ">> Building SVD cache for ${MODEL_NAME}. This is needed by Block-AP SVQ."
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python dbw/utils/decompose.py \
    --model-name "$MODEL_NAME" \
    --save-dir ./svd \
    --torch-dtype "$SVD_TORCH_DTYPE" \
    --svd-device "$SVD_DEVICE" \
    "${SVD_HF_ARGS[@]}"
}

run_block() {
  ensure_svd
  echo ">> Block QAT: ${MODEL_NAME} -> ${BLOCK_MODEL_DIR}"
  if [[ "$OFF_LOAD_TO_DISK" == "1" ]]; then
    echo ">> activation cache mode: OFFLINE (--off_load_to_disk, activations on disk under ./cache)"
  else
    echo ">> activation cache mode: ONLINE (no --off_load_to_disk, activations in CPU RAM)"
  fi
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python train_block_ap.py \
    --model "$MODEL_NAME" \
    --output_dir "$BLOCK_LOG_DIR" \
    --net "$MODEL_FAMILY" \
    --wbits "$WBITS" \
    --epochs "$BLOCK_EPOCHS" \
    --group_size "$GROUP_SIZE" \
    --quant_lr "$QUANT_LR" \
    --norm_lr "$NORM_LR" \
    --u_lr "$U_LR" \
    --v_lr "$V_LR" \
    --bit_lr "$BIT_LR" \
    --gbit_lr "$GBIT_LR" \
    --min_lr_factor 20 \
    --eval_ppl \
    "${OFFLOAD_ARGS[@]}" \
    --train_size "$TRAIN_SIZE" \
    --val_size "$VAL_SIZE" \
    "${BLOCK_DATA_ARGS[@]}" \
    "${BLOCK_TUNING_ARGS[@]}" \
    --eval_tasks "$BLOCK_EVAL_TASKS" \
    --save_quant_dir "$BLOCK_MODEL_DIR" \
    --real_quant \
    "${BLOCK_HF_ARGS[@]}"
}

run_e2e() {
  if [[ ! -f "$QUANT_MODEL_PATH" ]]; then
    echo "Missing quantized model: ${QUANT_MODEL_PATH}" >&2
    echo "Run the block stage first or set QUANT_MODEL_PATH to an existing .pt checkpoint." >&2
    exit 1
  fi
  echo ">> End-to-end QP tuning: ${QUANT_MODEL_PATH} -> ${E2E_OUTPUT_DIR}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python train_e2e_qp.py \
    --quant_model_path "$QUANT_MODEL_PATH" \
    --model_family "$MODEL_FAMILY" \
    --model_name "$MODEL_NAME" \
    --wbits "$WBITS" \
    --group_size "$GROUP_SIZE" \
    --learning_rate "$E2E_LR" \
    --dataset mix_redpajama_wiki \
    --dataset_format pt \
    --output_dir "$E2E_OUTPUT_DIR" \
    --do_train True \
    --pt_context_len "$PT_CONTEXT_LEN" \
    --per_device_train_batch_size "$E2E_BATCH_SIZE" \
    --per_device_eval_batch_size "$E2E_BATCH_SIZE" \
    --gradient_accumulation_steps "$E2E_ACCUM" \
    --lm_eval_batch_size "$LM_EVAL_BS" \
    --logging_steps 1 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps "$E2E_EVAL_STEPS" \
    --max_train_samples "$E2E_MAX_TRAIN_SAMPLES" \
    --num_train_epochs "$E2E_EPOCHS" \
    --eval_dataset_size 64 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    "${PRECISION_ARGS[@]}" \
    --eval_tasks "$E2E_EVAL_TASKS" \
    --preprocessing_num_workers "$PREPROCESSING_NUM_WORKERS" \
    "${E2E_DATA_ARGS[@]}" \
    --do_ppl_eval \
    "${E2E_HF_ARGS[@]}"
}

run_eval_block() {
  if [[ ! -d "$BLOCK_MODEL_DIR" ]]; then
    echo "Missing Block-AP directory: ${BLOCK_MODEL_DIR}" >&2
    exit 1
  fi
  echo ">> Block checkpoint eval: ${BLOCK_MODEL_DIR}"
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" python train_block_ap.py \
    --model "$MODEL_NAME" \
    --output_dir "$BLOCK_LOG_DIR-eval" \
    --net "$MODEL_FAMILY" \
    --wbits "$WBITS" \
    --group_size "$GROUP_SIZE" \
    --eval_ppl \
    --eval_tasks "$BLOCK_EVAL_TASKS" \
    --test_only "$BLOCK_MODEL_DIR" \
    --real_quant \
    "${BLOCK_HF_ARGS[@]}"
}

case "$STAGE" in
  all) run_block; run_e2e ;;
  block) run_block ;;
  e2e) run_e2e ;;
  eval-block) run_eval_block ;;
  *)
    echo "Usage: $0 [cuda_visible_devices] [all|block|e2e|eval-block]" >&2
    exit 1
    ;;
esac
