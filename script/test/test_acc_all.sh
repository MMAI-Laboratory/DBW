#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

RESULT=${RESULT:-output/acc_results.jsonl}
LOGDIR=${LOGDIR:-output/acc_logs}
TASKS=${TASKS:-piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,lambada_openai,lambada_standard}
BS=${BS:-8}
mkdir -p "$LOGDIR"

PY=${PY:-python}

WAVE1=(
  "0|dbw-Llama-7b-w1.61g128|huggyllama/llama-7b|$PY"
  "1|dbw-Llama-2-7b-w1.61g128|meta-llama/Llama-2-7b-hf|$PY"
  "2|dbw-Llama-3-8b-w1.61g128|meta-llama/Meta-Llama-3-8B|$PY"
  "3,4|dbw-gemma-7b-w1.61g128|google/gemma-7b|$PY"
  "5|dbw-qwen2.5-7b-w1.61g128|Qwen/Qwen2.5-7B|$PY"
)
WAVE2=(
  "0|dbw-mistral-7b-w1.61g128|mistralai/Mistral-7B-v0.1|$PY"
  "1|dbw-Llama-2-7b-chat-w1.61g128|meta-llama/Llama-2-7b-chat-hf|$PY"
  "2,3|dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|$PY"
)
WAVE3=(
  "0,1,2,3|dbw-Llama-65b-w1.61g128|huggyllama/llama-65b|$PY|2"
  "4,5,6,7|dbw-Llama-2-70b-w1.61g128|meta-llama/Llama-2-70b-hf|$PY|2"
)

launch() {
  local gpus="$1" tag="$2" base="$3" py="$4" bs="${5:-$BS}"
  local bundle="output/release/$tag.pt"

  if [ ! -f "$bundle" ]; then echo "!! SKIP $tag — no $bundle"; return; fi
  if [ "${FORCE:-0}" != "1" ] && [ -s "$LOGDIR/$tag.json" ]; then
    echo "== done  $tag (cached)"; return
  fi

  echo ">> [gpu $gpus] $tag  (bs=$bs)"
  CUDA_VISIBLE_DEVICES="$gpus" TOKENIZERS_PARALLELISM=false HF_DATASETS_TRUST_REMOTE_CODE=1 \
  "$py" test.py --mode acc \
      --bundle "$bundle" \
      --tasks "$TASKS" \
      --batch_size "$bs" \
      --tag "$tag" \
      --result_json "$LOGDIR/$tag.json" \
      > "$LOGDIR/$tag.log" 2>&1 &
}

run_wave() {
  local -n jobs=$1
  for entry in "${jobs[@]}"; do
    IFS='|' read -r gpus tag base py bs <<< "$entry"
    launch "$gpus" "$tag" "$base" "$py" "${bs:-$BS}"
  done
  wait
}

echo "########## accuracy wave 1 ##########"
run_wave WAVE1
echo "########## accuracy wave 2 ##########"
run_wave WAVE2
echo "########## accuracy wave 3 — 65B / 70B, multi-GPU ##########"
run_wave WAVE3

bash script/test/summarize_acc.sh
