bash script/train/common/block.sh $1 Llama-2 meta-llama/Llama-2-7b-hf dbw-Llama-2-7b train 1e-5 2e-5 2e-5 2e-5 1e-5 1e-4 2 4096 2
bash script/train/common/e2e.sh $1 Llama-2 meta-llama/Llama-2-7b-hf dbw-Llama-2-7b 2
