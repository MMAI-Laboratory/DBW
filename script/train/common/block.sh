net="${2:-Llama-2}"
model="${3:-meta-llama/Llama-2-7b-hf}"
save_dir="${4:-Llama-2-7b}"
mode="${5:-train}"
qlr="${6:-1e-4}"
normlr="${7:-2e-5}"
ulr="${8:-2e-5}"
vlr="${9:-2e-5}"
blr="${10:-1e-4}"
gblr="${11:-1e-4}"
wbits="${12:-2}"
train_size="${13:-4096}"
epochs="${14:-2}"

GLOBAL_BIT="${GLOBAL_BIT:-1}"
PROGRESSIVE="${PROGRESSIVE:-1}"
PRE_LR="${PRE_LR:-1e-5}"
LAMBDA_SV="${LAMBDA_SV:-1e-6}"
SAVE_STATISTICS="${SAVE_STATISTICS:-1}"
WANDB="${WANDB:-0}"

TUNING_ARGS=(--pre_lr "$PRE_LR" --lambda_sv "$LAMBDA_SV")
[[ "$GLOBAL_BIT" == "1" ]] || TUNING_ARGS+=(--no_global_bit)
[[ "$PROGRESSIVE" == "1" ]] || TUNING_ARGS+=(--no_progressive)
[[ "$SAVE_STATISTICS" == "1" ]] || TUNING_ARGS+=(--no_save_statistics)
[[ "$WANDB" == "1" ]] && TUNING_ARGS+=(--wandb)

if [[ "$mode" == "train" ]]; then
  echo ">> Training mode"
  CUDA_VISIBLE_DEVICES=$1 python train_block_ap.py \
  --model "$model"  \
  --output_dir "./output/block_ap_log/${save_dir}-w2g128" \
  --net "$net" \
  --wbits "$wbits" \
  --epochs "$epochs" \
  --group_size 128 \
  --quant_lr "$qlr" \
  --norm_lr "$normlr" \
  --u_lr "$ulr" \
  --v_lr "$vlr" \
  --bit_lr "$blr" \
  --gbit_lr "$gblr" \
  --min_lr_factor 20 \
  --eval_ppl \
  --off_load_to_disk \
  --train_size "$train_size" \
  "${TUNING_ARGS[@]}" \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --save_quant_dir "./output/block_ap_models/${save_dir}-w${wbits}g128" \
  --real_quant
elif [[ "$mode" == "eval" ]]; then
  echo ">> Eval mode"
  CUDA_VISIBLE_DEVICES=$1 python train_block_ap.py \
  --model "$model"  \
  --output_dir "./output/block_ap_log/${save_dir}-w2g128" \
  --net "$net" \
  --wbits 2 \
  --group_size 128 \
  --quant_lr 1e-4 \
  --weight_lr 2e-5 \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --test_only "./output/block_ap_models/${save_dir}-w2g128" \
  --real_quant
else
  echo ">> Unknown mode: $mode"
  echo "usage: $0 [train|eval]"
  exit 1
fi