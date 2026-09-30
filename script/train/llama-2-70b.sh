export SPLIT=8
bash script/train/common/block.sh $1 Llama-2 meta-llama/Llama-2-70b-hf dbw-Llama-2-70b train 1e-5 2e-5 2e-5 2e-5 1e-5 1e-4 1.61 4096 2
bash script/train/common/e2e.sh $1 Llama-2 meta-llama/Llama-2-70b-hf dbw-Llama-2-70b 1.61 2 16 2 5e-7
