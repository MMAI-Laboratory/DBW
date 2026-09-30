export SPLIT=4
bash script/train/common/block.sh $1 Llama huggyllama/llama-30b dbw-Llama-30b train 1e-5 2e-5 2e-5 2e-5 1e-5 1e-4 1.61 4096 2
bash script/train/common/e2e.sh $1 Llama huggyllama/llama-30b dbw-Llama-30b 1.61 2 16 2 6e-6
