export CUDA_VISIBLE_DEVICES=6,7
python dbw/utils/decompose.py --model-name google/gemma-7b
python dbw/utils/decompose.py --model-name Qwen/Qwen2.5-7B
python dbw/utils/decompose.py --model-name mistralai/Mistral-7B-v0.1
