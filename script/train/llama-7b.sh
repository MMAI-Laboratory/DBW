bash script/train/common/block.sh $1 Llama huggyllama/llama-7b dbw-Llama-7b train 1e-5 2e-5 2e-5 2e-5 1e-5 1e-4 1.61 4096 2
bash script/train/common/e2e.sh $1 Llama huggyllama/llama-7b dbw-Llama-7b 1.61
