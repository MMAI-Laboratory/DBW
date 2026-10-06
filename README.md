# DBW 

This repository contains PyTorch-based official implementations for *"Differentiable Bit-Widths: Co-optimizing Pruning and Quantization via SVD for Ultra-Efficient LLM Compression (NeurIPS'26)."* This paper introduces a co-optimization method of pruning and quantization using differentiable bit-widths (DBW) to compress large language models (LLMs) into ultra-efficient scale. See [our paper](https://arxiv.org/abs/2610.06026) for more details.

<p align="center">
    <img width="850px" src="https://github.com/user-attachments/assets/585066e7-a0e2-46d6-bd6c-c5f6a4653568"/>
    <br/>
  <h4 align="center">Our compression concept illustration</h4>
</p>


## 1. Install

```bash
conda create -n dbw python=3.11 -y
conda activate dbw
pip install -r requirements.txt
```

## 2. Train and Evaluate

We provide training scripts for each model in `script/train/`. Each script contains the hyper-parameters used for training and evaluation. In the below, we provide the training and evaluation commands for each model, where each command is executed in the root directory of this repository to reproduce the results in our paper. Specifically, the training and evaluation commands are as follows: **The training command** downloads the original model and dataset from Hugging Face, decompose its weights using SVD, and perform quantization-aware training employing the proposed DBW method. **The evaluation command** downloads the released model from Hugging Face and assesses its perpelxity and accuracy. `$GPU` is a `CUDA_VISIBLE_DEVICES` string (`0`, or `0,1` for the large models).

<details>
<summary><b>LLaMA-7B</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/llama-7b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-7b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-7b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-7b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-13B</b></summary>

Requires 2 GPUs to train, 1 to test.

```bash
# 1. train
bash script/train/llama-13b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-13b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-13b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-13b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-30B</b></summary>

Requires 4 GPUs to train, 2 to test.

```bash
# 1. train
bash script/train/llama-30b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-30b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-30b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-30b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-65B</b></summary>

Requires 8 GPUs to train, 4 to test.

```bash
# 1. train
bash script/train/llama-65b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-65b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-65b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-65b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-2-7B</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/llama-2-7b.sh $GPU
bash script/train/llama-2-7b-1bit.sh $GPU
bash script/train/llama-2-7b-0.5bit.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-2-7b-w1.61g128.pt --local-dir output/release
huggingface-cli download sw24/dbw dbw-Llama-2-7b-w1g128.pt --local-dir output/release
huggingface-cli download sw24/dbw dbw-Llama-2-7b-w0.5g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-2-7b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-7b-w1.61g128.pt
python test.py --mode ppl --bundle output/release/dbw-Llama-2-7b-w1g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-7b-w1g128.pt
python test.py --mode ppl --bundle output/release/dbw-Llama-2-7b-w0.5g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-7b-w0.5g128.pt
```

</details>

<details>
<summary><b>LLaMA-2-13B</b></summary>

Requires 2 GPUs to train, 1 to test.

```bash
# 1. train
bash script/train/llama-2-13b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-2-13b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-2-13b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-13b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-2-70B</b></summary>

Requires 8 GPUs to train, 4 to test.

```bash
# 1. train
bash script/train/llama-2-70b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-2-70b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-2-70b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-70b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-3-8B</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/llama-3-8b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-3-8b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-3-8b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-3-8b-w1.61g128.pt
```

</details>

<details>
<summary><b>Gemma-7B</b></summary>

Requires 1 GPU to train, 2 to test.

```bash
# 1. train
bash script/train/gemma-7b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-gemma-7b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-gemma-7b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-gemma-7b-w1.61g128.pt
```

</details>

<details>
<summary><b>Mistral-7B-v0.1</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/mistral-7b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-mistral-7b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-mistral-7b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-mistral-7b-w1.61g128.pt
```

</details>

<details>
<summary><b>Qwen2.5-7B</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/qwen2.5-7b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-qwen2.5-7b-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-qwen2.5-7b-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-qwen2.5-7b-w1.61g128.pt
```

</details>

<details>
<summary><b>LLaMA-2-7B-chat</b></summary>

Requires 1 GPU.

```bash
# 1. train
bash script/train/llama-2-7b-chat.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-Llama-2-7b-chat-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-Llama-2-7b-chat-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-Llama-2-7b-chat-w1.61g128.pt
```

</details>

<details>
<summary><b>DeepSeek-R1-Distill-LLaMA-8B</b></summary>

Requires 1 GPU to train, 2 to test.

```bash
# 1. train
bash script/train/deepseek-r1-distill-llama-8b.sh $GPU
# 2. download the released model
huggingface-cli download sw24/dbw dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128.pt --local-dir output/release
# 3. test the released model
export CUDA_VISIBLE_DEVICES=$GPU
python test.py --mode ppl --bundle output/release/dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128.pt
python test.py --mode acc --bundle output/release/dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128.pt
```

</details>

<details>
<summary><b>Training script details</b></summary>

We provide a detailed description of the training scripts. Our training consists of two steps: stage-1 block-wise quantization-aware training and stage-2 end-to-end quantization hyper-parameter tuning, which is borrowed from the [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT) repostory. The stage-1 training script is `train_block_ap.py`, and the stage-2 training script is `train_e2e_qp.py`. The stage-1 script trains a block-wise quantized model, and the stage-2 script tunes the quantization hyper-parameters of the stage-1 model. For example, we provide the training and evaluation commands of Llama-2-7B model for each stage. Each command is executed in the root directory of this repository to reproduce the results in our paper. The exact hyper-parameters used for every released model live in **`script/train/`**.

**Step 1. block-wise quantization-aware training and evaluation.**

```bash
# ---- train -------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python train_block_ap.py \
  --model meta-llama/Llama-2-7b-hf \
  --net Llama-2 \
  --wbits 1.61 \
  --group_size 128 \
  --epochs 2 \
  --train_size 4096 \
  --quant_lr 1e-5 --norm_lr 2e-5 --u_lr 2e-5 --v_lr 2e-5 \
  --bit_lr 1e-5 --gbit_lr 1e-4 --min_lr_factor 20 \
  --off_load_to_disk \
  --real_quant \
  --eval_ppl \
  --output_dir      ./output/block_ap_log/dbw-Llama-2-7b-w2g128 \
  --save_quant_dir  ./output/block_ap_models/dbw-Llama-2-7b-w1.61g128
# -> writes ./output/block_ap_models/dbw-Llama-2-7b-w1.61g128.pt

# ---- evaluate an already-trained stage-1 model (no training) -----------
CUDA_VISIBLE_DEVICES=0 python train_block_ap.py \
  --model meta-llama/Llama-2-7b-hf \
  --net Llama-2 \
  --wbits 1.61 \
  --group_size 128 \
  --real_quant \
  --eval_ppl \
  --eval_tasks piqa,arc_easy,arc_challenge,hellaswag,winogrande \
  --output_dir ./output/block_ap_log/dbw-Llama-2-7b-w2g128-eval \
  --test_only  ./output/block_ap_models/dbw-Llama-2-7b-w1.61g128
```

`--test_only <dir>` skips training entirely and loads the saved quantized model for evaluation. `--off_load_to_disk` keeps block activations on disk under `./cache` instead of in CPU RAM.

**Step 2. end-to-end quantization hyper-parameter tuning and evaluation.**

```bash
# ---- train -------------------------------------------------------------
CUDA_VISIBLE_DEVICES=0 python train_e2e_qp.py \
  --quant_model_path ./output/block_ap_models/dbw-Llama-2-7b-w1.61g128.pt \
  --model_family Llama-2 \
  --model_name meta-llama/Llama-2-7b-hf \
  --wbits 1.61 --group_size 128 \
  --dataset mix_redpajama_wiki --dataset_format pt \
  --learning_rate 2e-5 --max_grad_norm 0.3 \
  --pt_context_len 4096 --max_train_samples 5120 --num_train_epochs 1 \
  --per_device_train_batch_size 4 --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --eval_dataset_size 64 --eval_steps 64 --evaluation_strategy steps \
  --save_strategy epoch --logging_steps 1 --data_seed 42 --bf16 \
  --do_train True --do_ppl_eval \
  --lm_eval_batch_size 16 \
  --eval_tasks piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,lambada_openai,lambada_standard \
  --output_dir ./output/e2e-qp-output/dbw-Llama-2-7b-w1.61g128-mix-redpajama-wiki-5120

# ---- evaluate an already-trained stage-2 checkpoint (no training) ------
CUDA_VISIBLE_DEVICES=0 python train_e2e_qp.py \
  --quant_model_path ./output/block_ap_models/dbw-Llama-2-7b-w1.61g128.pt \
  --model_family Llama-2 \
  --model_name meta-llama/Llama-2-7b-hf \
  --wbits 1.61 --group_size 128 \
  --test_only True \
  --eval_ckpt ./output/e2e-qp-output/dbw-Llama-2-7b-w1.61g128-mix-redpajama-wiki-5120/checkpoint-160 \
  --do_ppl_eval \
  --lm_eval_batch_size 16 \
  --eval_tasks piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,lambada_openai,lambada_standard \
  --output_dir ./output/e2e-qp-output/dbw-Llama-2-7b-w1.61g128-eval
```

`--test_only True` rebuilds the stage-1 model, applies the stage-2 checkpoint given by `--eval_ckpt`, evaluates, and exits. Results are written to `<output_dir>/test_only_metrics.json`.

</details>

## 3. Experimental Results

We evaluate the proposed DBW on the two language modeling tasks: `wikitext2`, `c4`, and on the eight commonsense reasoning tasks: `piqa`, `arc_easy`, `hellaswag`, `winogrande`, `race`, `arc_challenge`, `lambada_openai`, `lambada_standard`.
All models are quantized to 1.61 bits with group size 128. `ld-0` and `ld-s` are `lambada_openai` and `lambada_standard`. The evaluation results are summarized in the following table. The results are slightly different from the results in our paper due to the randomness of training and evaluation.

| model | weight | script | wiki2 | c4 | piqa | arc-e | hella | wino | race | arc-c | ld-0 | ld-s |
|---|:-:|:-:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LLaMA-7B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-7b-w1.61g128.pt) | [📜](script/train/llama-7b.sh) | 9.13 | 12.16 | 69.86 | 57.24 | 41.47 | 60.38 | 35.50 | 28.58 | 54.10 | 45.27 |
| LLaMA-13B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-13b-w1.61g128.pt) | [📜](script/train/llama-13b.sh) | 7.44 | 10.23 | 72.25 | 63.47 | 46.76 | 62.67 | 36.36 | 31.40 | 57.60 | 50.96 |
| LLaMA-30B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-30b-w1.61g128.pt) | [📜](script/train/llama-30b.sh) | 6.71 | 8.93 | 74.27 | 70.83 | 51.81 | 66.46 | 40.29 | 37.71 | 64.54 | 56.41 |
| LLaMA-2-7B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-2-7b-w1.61g128.pt) | [📜](script/train/llama-2-7b.sh) | 9.00 | 12.24 | 67.85 | 57.32 | 41.80 | 60.93 | 35.41 | 27.73 | 53.83 | 44.81 |
| LLaMA-2-13B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-2-13b-w1.61g128.pt) | [📜](script/train/llama-2-13b.sh) | 7.37 | 10.42 | 72.31 | 66.25 | 47.46 | 61.80 | 39.04 | 33.70 | 60.64 | 51.91 |
| LLaMA-2-70B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-2-70b-w1.61g128.pt) | [📜](script/train/llama-2-70b.sh) | 6.10 | 8.15 | 76.77 | 76.56 | 55.66 | 70.48 | 41.82 | 44.28 | 70.79 | 63.73 |
| LLaMA-3-8B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-3-8b-w1.61g128.pt) | [📜](script/train/llama-3-8b.sh) | 15.96 | 22.79 | 66.10 | 55.30 | 37.97 | 56.04 | 34.45 | 25.85 | 44.75 | 35.73 |
| Gemma-7B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-gemma-7b-w1.61g128.pt) | [📜](script/train/gemma-7b.sh) | 64.49 | 67.35 | 58.05 | 36.87 | 28.57 | 50.43 | 25.55 | 17.75 | 21.02 | 12.73 |
| Mistral-7B-v0.1 | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-mistral-7b-w1.61g128.pt) | [📜](script/train/mistral-7b.sh) | 12.25 | 16.36 | 67.95 | 56.14 | 41.11 | 57.77 | 33.21 | 27.47 | 46.28 | 35.28 |
| Qwen2.5-7B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-qwen2.5-7b-w1.61g128.pt) | [📜](script/train/qwen2.5-7b.sh) | 17.30 | 24.66 | 65.67 | 58.25 | 37.82 | 55.80 | 33.40 | 27.30 | 42.73 | 36.41 |
| LLaMA-2-7B-chat | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-Llama-2-7b-chat-w1.61g128.pt) | [📜](script/train/llama-2-7b-chat.sh) | 11.15 | 15.39 | 66.27 | 51.56 | 38.01 | 56.75 | 32.25 | 26.37 | 44.65 | 32.68 |
| DeepSeek-R1-Distill-LLaMA-8B | [🤗](https://huggingface.co/sw24/dbw/blob/main/dbw-DeepSeek-R1-Distill-Llama-8B-w1.61g128.pt) | [📜](script/train/deepseek-r1-distill-llama-8b.sh) | 18.16 | 24.03 | 67.79 | 54.34 | 38.68 | 56.43 | 36.08 | 25.17 | 46.57 | 34.99 |


## 4. Released Models

We provide ready-to-use model files on [Hugging Face](https://huggingface.co/sw24/dbw).

Each released `.pt` file is a single `torch.save` containing:

```python
{
    "model": model,
    "tokenizer": tokenizer,
    "meta": metadata,
}
```

After downloading a released model, you can load it with the following code snippet:

```python
# huggingface-cli download sw24/dbw dbw-Llama-2-7b-w1.61g128.pt --local-dir output/release
import torch

obj = torch.load(
    "output/release/dbw-Llama-2-7b-w1.61g128.pt",
    weights_only=False,
)
model = obj["model"].to("cuda").eval()
tokenizer = obj["tokenizer"]
prompt = "The three main advantages of low-bit quantization for large language models are"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=44, do_sample=False)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

which prints:

```text
The three main advantages of low-bit quantization for large language models are:
  * The model is smaller than the original one.
  * The model is more efficient than the original one.
  * The model is more flexible than the original one.
```

<details>
<summary><b>Released Model Usage with Detailed Instructions</b></summary>


**1. Install DBW**
       
The `dbw` package must be importable when loading a released model.

If you cloned this repository, install it from the project root:

```bash
pip install -e .
```

Once installed, the released model can be loaded from anywhere.

**2. Download a released model**

Download the desired model file from: **https://huggingface.co/sw24/dbw**

For example:

```text
huggingface-cli download sw24/dbw dbw-Llama-2-7b-w1.61g128.pt --local-dir output/release
```

The downloaded file can be placed anywhere on your machine.

**3. Load the model**

```python
import torch

obj = torch.load(
    "output/release/dbw-Llama-2-7b-w1.61g128.pt",
    weights_only=False,
)

model = obj["model"].to("cuda").eval()
tokenizer = obj["tokenizer"]
```

The model and tokenizer are now ready for inference.

For example:

```python
prompt = "The three main advantages of low-bit quantization for large language models are"

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=44, do_sample=False)

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

which prints:

```text
The three main advantages of low-bit quantization for large language models are:
  * The model is smaller than the original one.
  * The model is more efficient than the original one.
  * The model is more flexible than the original one.
```

The 65B and 70B models do not fit on a single GPU; spread them over several with
`accelerate`:

```python
from accelerate import dispatch_model
from dbw.utils.model import balanced_device_map_for_decoder_model

model = dispatch_model(obj["model"], device_map=balanced_device_map_for_decoder_model(obj["model"]))
model.eval()
inputs = tokenizer(prompt, return_tensors="pt").to(model.get_input_embeddings().weight.device)
```

</details>

<details>
<summary><b>Released Model FAQ</b></summary>

**Do I need to download the original Hugging Face model separately?**

No. After downloading a released `.pt` file, you do **not** need to separately download:

- the original Hugging Face model weights,
- model configuration files,
- tokenizer files,
- stage-1 artifacts, or
- stage-2 artifacts.

The released `.pt` file and an importable `dbw` installation are sufficient for inference.

**Does the released model depend on the original file paths?**

No. The released model does not depend on the directory structure or absolute paths of the machine on which it was created.

You can move the `.pt` file to any location and load it by specifying its new path.

**Why does `dbw` need to be installed?**

The released models are loaded with:

```python
torch.load(..., weights_only=False)
```

The serialized model contains references to DBW-specific Python classes, such as:

```text
dbw.real_dbw.realDBW
MultiQuantLinear
Unpacker
```

These class references are stored in the serialized file, but their Python source code is not embedded in it. Therefore, the `dbw` package must be available in the current Python environment when `torch.load` is called.

If you encounter an error such as:

```text
ModuleNotFoundError: No module named 'dbw'
```

install this repository in your environment:

```bash
pip install -e .
```

</details>

## Acknowledgements

This project is heavily based on [EfficientQAT](https://github.com/OpenGVLab/EfficientQAT). We sincerely thank the authors of the mentioned works for sharing such great libraries as open-source project.
