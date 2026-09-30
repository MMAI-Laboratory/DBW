bash script/train/common/block.sh $1 Llama-3 meta-llama/Meta-Llama-3-8B dbw-Llama-3-8b train 1e-5 2e-5 2e-5 2e-5 1e-5 1e-4 1.61 4096 2
bash script/train/common/e2e.sh $1 Llama-3 meta-llama/Meta-Llama-3-8B dbw-Llama-3-8b 1.61 2 16 8
