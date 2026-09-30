export SPLIT=8
bash script/train/common/block.sh $1 Llama huggyllama/llama-65b dbw-Llama-65b train 5e-6 1e-5 1e-5 1e-5 5e-6 5e-5 1.61 4096 2
bash script/train/common/e2e.sh $1 Llama huggyllama/llama-65b dbw-Llama-65b 1.61 2 16 2 1e-6
