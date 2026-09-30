model="${2:-Llama-2}"
model_name="${3:-meta-llama/Llama-2-7b-hf}"
load_dir="${4:-Llama-2-7b}"
wbits="${5:-1.61}"
bs="${6:-4}"
accum="${7:-8}"
lm_eval_bs="${8:-16}"
lr="${9:-2e-5}"

CUDA_VISIBLE_DEVICES=$1 python train_e2e_qp.py \
    --quant_model_path "./output/block_ap_models/${load_dir}-w${wbits}g128.pt" \
    --model_family "$model" \
    --model_name "$model_name" \
    --wbits "$wbits" \
    --group_size 128 \
    --learning_rate "$lr" \
    --dataset mix_redpajama_wiki \
    --dataset_format pt \
    --output_dir "./output/e2e-qp-output/${load_dir}-w${wbits}g128-mix-redpajama-wiki-5120" \
    --do_train True \
    --pt_context_len 4096 \
    --per_device_train_batch_size "$bs" \
    --per_device_eval_batch_size "$bs" \
    --gradient_accumulation_steps "$accum" \
    --lm_eval_batch_size "$lm_eval_bs" \
    --logging_steps 1 \
    --save_strategy epoch \
    --evaluation_strategy steps \
    --eval_steps 64 \
    --max_train_samples 5120 \
    --num_train_epochs 1 \
    --eval_dataset_size 64 \
    --data_seed 42 \
    --max_grad_norm 0.3 \
    --bf16 \
    --eval_tasks  piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,lambada_openai,lambada_standard \
    --preprocessing_num_workers 32 \
    --do_ppl_eval
