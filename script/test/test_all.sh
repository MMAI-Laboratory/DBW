#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/../.."

RESULT=${RESULT:-output/ppl_results.jsonl}
DATASETS=${DATASETS:-wikitext2,c4}
LOGDIR=${LOGDIR:-output/eval_logs}
mkdir -p "$LOGDIR"

PY=${PY:-python}


WAVE1=(
  "0|dbw-Llama-2-7b-w0.5g128|meta-llama/Llama-2-7b-hf"
  "1|dbw-Llama-2-7b-w1g128|meta-llama/Llama-2-7b-hf"
  "2|dbw-Llama-2-7b-w1.61g128|meta-llama/Llama-2-7b-hf"
  "3|dbw-Llama-7b-w1.61g128|huggyllama/llama-7b"
  "4|dbw-Llama-3-8b-w1.61g128|meta-llama/Meta-Llama-3-8B"
  "5|dbw-Llama-2-13b-w1.61g128|meta-llama/Llama-2-13b-hf"
  "6|dbw-Llama-13b-w1.61g128|huggyllama/llama-13b"
)
WAVE2=(
  "0|dbw-gemma-7b-w1.61g128|google/gemma-7b|$PY"
  "1|dbw-mistral-7b-w1.61g128|mistralai/Mistral-7B-v0.1|$PY"
  "2|dbw-qwen2.5-7b-w1.61g128|Qwen/Qwen2.5-7B|$PY"
  "3|dbw-Llama-2-7b-chat-w1.61g128|meta-llama/Llama-2-7b-chat-hf|$PY"
  "6,7|dbw-Llama-30b-w1.61g128|huggyllama/llama-30b"
  "4,5|dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128|deepseek-ai/DeepSeek-R1-Distill-Llama-8B|$PY"
)
WAVE3=(
  "0,1,2,3|dbw-Llama-65b-w1.61g128|huggyllama/llama-65b"
  "4,5,6,7|dbw-Llama-2-70b-w1.61g128|meta-llama/Llama-2-70b-hf"
)

launch() {
  local gpus="$1" tag="$2" base="$3" py="${4:-$PY}"
  local bundle="output/release/$tag.pt"

  if [ ! -f "$bundle" ]; then echo "!! SKIP $tag — no $bundle"; return; fi

  if [ "${FORCE:-0}" != "1" ] && [ -f "$LOGDIR/$tag.json" ] && \
     "$PY" -c "
import json,sys
want={d for d in '$DATASETS'.split(',') if d}
have=set()
for l in open('$LOGDIR/$tag.json'):
    if l.strip(): have |= set(json.loads(l)['ppl'])
sys.exit(0 if want <= have else 1)" 2>/dev/null; then
    echo "== done  $tag (cached)"
    return
  fi

  echo ">> [gpu $gpus] $tag  ($(basename "$(dirname "$(dirname "$py")")"))"
  CUDA_VISIBLE_DEVICES="$gpus" TOKENIZERS_PARALLELISM=false \
  "$py" test.py --mode ppl \
      --bundle "$bundle" \
      --datasets "$DATASETS" \
      --tag "$tag" \
      --result_json "$LOGDIR/$tag.json" \
      > "$LOGDIR/$tag.log" 2>&1 &
}

run_wave() {
  local -n jobs=$1
  for entry in "${jobs[@]}"; do
    IFS='|' read -r gpus tag base py <<< "$entry"
    launch "$gpus" "$tag" "$base" "${py:-$PY}"
  done
  wait
}

echo "########## wave 1 — Llama family, one GPU each ##########"
run_wave WAVE1
echo "########## wave 2 — other families + instruct  ##########"
run_wave WAVE2
echo "########## wave 3 — 65B / 70B, multi-GPU       ##########"
run_wave WAVE3

bash script/test/summarize_ppl.sh
