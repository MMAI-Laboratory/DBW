import argparse, json, math, importlib, os
from os.path import exists, join, isdir
from pathlib import Path
from typing import Optional, Dict
from dataclasses import dataclass, field
from packaging import version
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import set_seed, Seq2SeqTrainer, LlamaTokenizer
from accelerate import dispatch_model
from bitsandbytes.optim import AdamW
import dbw.utils as utils
from dbw.utils.data_block import test_ppl
from dbw.utils.data_e2e import make_data_module


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return str(version.parse(full_version).major) + "." + str(version.parse(full_version).minor)

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    quant_model_path: Optional[str] = field(
        default="",
        metadata={"help": "path of the quantization model by Block-AP."}
    )
    model_family: Optional[str] = field(
        default="llama-2",
        metadata={"help": "for the saving of dataset cache for faster experiments"}
    )
    model_name: Optional[str] = field(default="", metadata={"help": "model name."})
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."}
    )


@dataclass
class DataArguments:
    eval_dataset_size: int = field(
        default=1024, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    source_max_len: int = field(
        default=1024,
        metadata={"help": "Maximum source sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    target_max_len: int = field(
        default=256,
        metadata={"help": "Maximum target sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    dataset: str = field(
        default='alpaca',
        metadata={"help": "Which dataset to finetune on. See datamodule for options."}
    )
    eval_tasks: str = field(
        default='',
        metadata={"help": "evaluation tasks for lm eval, example:piqa,arc_easy,arc_challenge,hellaswag,winogrande"}
    )
    mask_use: bool = field(
        default=True, metadata={"help": "mask the loss to role in dialogue datas"}
    )
    dataset_format: Optional[str] = field(
        default=None,
        metadata={"help": "Which dataset format is used. [alpaca|redpajama]"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=32,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(
        default=None
    )
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to train on the input in addition to the target text."}
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the MMLU evaluation."}
    )
    do_ppl_eval: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to run the PPL evaluation."}
    )
    pt_context_len: int = field(
        default=1024,
        metadata={"help": "language modeling length."}
    )
    full_finetune: bool = field(
        default=False,
        metadata={"help": "Finetune the entire model without adapters."}
    )
    wbits: float = field(
        default=4,
        metadata={"help": "How many bits to use."}
    )
    group_size: int = field(
        default=64,
        metadata={"help": "How many group size to use."}
    )
    max_memory_MB: int = field(
        default=80000,
        metadata={"help": "Free memory per gpu."}
    )
    report_to: str = field(
        default='none',
        metadata={"help": "To use wandb or something else for reporting."}
    )
    output_dir: str = field(default='./output', metadata={"help": 'The output dir for logs and checkpoints'})
    resume_from_checkpoint: str = field(default=None, metadata={"help": 'The output dir for logs and checkpoints'})
    optim: str = field(default='paged_adamw_32bit', metadata={"help": 'The optimizer to be used'})
    per_device_train_batch_size: int = field(default=1, metadata={
        "help": 'The training batch size per GPU. Increase for better speed.'})
    gradient_accumulation_steps: int = field(default=16, metadata={
        "help": 'How many gradients to accumulate before to perform an optimizer step'})
    max_steps: int = field(default=0, metadata={"help": 'How many optimizer update steps to take'})
    weight_decay: float = field(default=0.0, metadata={
        "help": 'The L2 weight decay rate of AdamW'})
    learning_rate: float = field(default=2e-5, metadata={"help": 'The learnign rate'})
    remove_unused_columns: bool = field(default=False,
                                        metadata={"help": 'Removed unused columns. Needed to make this codebase work.'})
    max_grad_norm: float = field(default=0.3, metadata={
        "help": 'Gradient clipping max norm. This is tuned and works well for all models tested.'})
    gradient_checkpointing: bool = field(default=True,
                                         metadata={"help": 'Use gradient checkpointing. You want to use this.'})
    do_train: bool = field(default=True, metadata={"help": 'To train or not to train, that is the question?'})
    lr_scheduler_type: str = field(default='cosine', metadata={
        "help": 'Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis'})
    warmup_ratio: float = field(default=0.03, metadata={"help": 'Fraction of steps to do a warmup for'})
    logging_steps: int = field(default=10,
                               metadata={"help": 'The frequency of update steps after which to log the loss'})
    group_by_length: bool = field(default=False, metadata={
        "help": 'Group sequences into batches with same length. Saves memory and speeds up training considerably.'})
    save_strategy: str = field(default='epoch', metadata={"help": 'When to save checkpoints'})
    save_steps: int = field(default=250, metadata={"help": 'How often to save a model'})
    save_total_limit: int = field(default=5,
                                  metadata={"help": 'How many checkpoints to save before the oldest is overwritten'})
    lm_eval_batch_size: int = field(default=16, metadata={"help": "batch size of lm eval"})
    test_only: bool = field(
        default=False,
        metadata={"help": "Skip training entirely: build the model, optionally lay an e2e-qp "
                          "checkpoint on top with --eval_ckpt, evaluate, and exit."}
    )
    eval_ckpt: Optional[str] = field(
        default=None,
        metadata={"help": "--test_only: checkpoint dir holding model*.safetensors to load "
                          "on top of --quant_model_path before evaluating."}
    )
    fast_pt_sample: bool = field(
        default=False,
        metadata={"help": "For PT datasets, build only the requested sampled blocks instead of "
                          "tokenizing/chunking the full corpus. False keeps the original Llama path."},
    )
    redpajama_streaming: bool = field(
        default=False,
        metadata={"help": "Load RedPajama with HF streaming in fast_pt_sample mode."},
    )
    streaming_buffer_size: int = field(
        default=10000,
        metadata={"help": "Shuffle buffer size for streaming fast sampling."},
    )
    mix_redpajama_blocks: int = field(
        default=4096,
        metadata={"help": "Nominal RedPajama block count used by mix_redpajama_wiki before max_train_samples."},
    )
    mix_wiki_blocks: int = field(
        default=256,
        metadata={"help": "Nominal Wikitext block count before repetition in mix_redpajama_wiki."},
    )
    mix_wiki_repeats: int = field(
        default=4,
        metadata={"help": "Number of Wikitext repetitions in mix_redpajama_wiki."},
    )
    prebuilt_dataloader_cache: Optional[str] = field(
        default=None,
        metadata={"help": "If set, torch.load this pre-built {'train','validation'} dataset dict directly "
                          "and skip all dataset loading/building."},
    )
    pad_token_resize: bool = field(
        default=True,
        metadata={"help": "Add a new [PAD] token and grow the embedding (32000 -> 32001). "
                          "True reproduces every released Llama checkpoint. Set False for "
                          "models that already have a usable pad/eos token (Gemma, Qwen, "
                          "Mistral, the instruction-tuned models)."}
    )


@dataclass
class GenerationArguments:
    max_new_tokens: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum number of new tokens to be generated in evaluation or prediction loops"
                          "if predict_with_generate is set."}
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate."}
    )

    do_sample: Optional[bool] = field(default=False)
    num_beams: Optional[int] = field(default=1)
    num_beam_groups: Optional[int] = field(default=1)
    penalty_alpha: Optional[float] = field(default=None)
    use_cache: Optional[bool] = field(default=True)

    temperature: Optional[float] = field(default=1.0)
    top_k: Optional[int] = field(default=50)
    top_p: Optional[float] = field(default=1.0)
    typical_p: Optional[float] = field(default=1.0)
    diversity_penalty: Optional[float] = field(default=0.0)
    repetition_penalty: Optional[float] = field(default=1.0)
    length_penalty: Optional[float] = field(default=1.0)
    no_repeat_ngram_size: Optional[int] = field(default=0)


import math
import torch
from accelerate import dispatch_model


def get_balanced_device_map_from_model(model):
    num_gpus = torch.cuda.device_count()
    num_layers = len(model.model.layers)
    layers_per_gpu = math.ceil(num_layers / num_gpus)
    device_map = {}
    device_map["model.embed_tokens"] = 0
    for i in range(num_layers):
        target_gpu = i // layers_per_gpu
        if target_gpu >= num_gpus:
            target_gpu = num_gpus - 1
        device_map[f"model.layers.{i}"] = target_gpu
    last_gpu = num_gpus - 1
    device_map["model.norm"] = last_gpu
    device_map["lm_head"] = last_gpu
    return device_map


def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()

    max_memory = f'{args.max_memory_MB}MB'
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
        max_memory = {'': max_memory[local_rank]}

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, legacy=False)
    model = torch.load(args.quant_model_path)


    device_map = get_balanced_device_map_from_model(model)
    model = dispatch_model(model, device_map=device_map)


    tokenizer.model_max_length = args.pt_context_len

    compute_dtype = (torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    if compute_dtype == torch.float16 and (is_ipex_available() and torch.xpu.is_available()):
        compute_dtype = torch.bfloat16
        print('Intel XPU does not support float16 yet, so switching to bfloat16')

    setattr(model, 'model_parallel', True)
    setattr(model, 'is_parallelizable', True)

    model.config.torch_dtype = (torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32))
    model.train()

    if args.pad_token_resize:
        if tokenizer._pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token=DEFAULT_PAD_TOKEN),
                tokenizer=tokenizer,
                model=model,
            )
    else:
        from dbw.utils.model import ensure_tokenizer_has_pad
        ensure_tokenizer_has_pad(tokenizer, model, resize_embeddings=False)

    if isinstance(tokenizer, LlamaTokenizer):
        print('Pass: adding special tokens.')

    for name, param in model.named_parameters():
        param.requires_grad = False

    for param in model.parameters():
        if (param.dtype == torch.float16) or (param.dtype == torch.bfloat16):
            param.data = param.data.to(torch.float32)

    if args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        model.gradient_checkpointing_enable()

    for name, module in model.named_modules():
        if 'norm' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
        if 'lm_head' in name or 'embed_tokens' in name:
            if hasattr(module, 'weight'):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print('trainable module')
    print('*' * 80)
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print('*' * 80)
    if args.wbits == 4: trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def smart_tokenizer_and_embedding_resize(
        special_tokens_dict: Dict,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings_data = model.get_input_embeddings().weight.data
        output_embeddings_data = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings_data[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings_data[-num_new_tokens:] = input_embeddings_avg
        output_embeddings_data[-num_new_tokens:] = output_embeddings_avg


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, 'completed'))
        if is_completed: return None, True
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith('checkpoint'):
                max_step = max(max_step, int(filename.replace('checkpoint-', '')))
        if max_step == 0: return None, is_completed
        checkpoint_dir = join(checkpoint_dir, f'checkpoint-{max_step}')
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed
    return None, False


def load_ckpt_state_dict(ckpt_dir):
    """Read model.safetensors, or every shard listed by the index for sharded ckpts."""
    from safetensors.torch import load_file
    single = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(single):
        return load_file(single)
    index = os.path.join(ckpt_dir, "model.safetensors.index.json")
    if not os.path.exists(index):
        raise FileNotFoundError(f"no model.safetensors / index.json under {ckpt_dir}")
    state_dict = {}
    for shard in sorted(set(json.load(open(index))["weight_map"].values())):
        state_dict.update(load_file(os.path.join(ckpt_dir, shard)))
    return state_dict


def run_test_only(model, tokenizer, args, logger):
    if args.eval_ckpt:
        logger.info(f"*** loading e2e-qp weights from {args.eval_ckpt} ***")
        missing, unexpected = model.load_state_dict(load_ckpt_state_dict(args.eval_ckpt), strict=False)
        logger.info(f"missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    metrics = {}
    if args.do_ppl_eval:
        logger.info("*** PPL ***")
        with torch.cuda.amp.autocast():
            metrics["ppl"] = test_ppl(model, tokenizer, datasets=['wikitext2', 'c4'], ppl_seqlen=2048)
        logger.info(metrics["ppl"])

    if args.eval_tasks:
        logger.info("*** lm-eval ***")
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        tasks = [t.strip() for t in args.eval_tasks.split(',') if t.strip()]
        device = str(model.get_input_embeddings().weight.device)
        results = lm_eval.simple_evaluate(
            model=HFLM(pretrained=model, tokenizer=tokenizer,
                       batch_size=args.lm_eval_batch_size, device=device),
            tasks=tasks, num_fewshot=0, task_manager=lm_eval.tasks.TaskManager(),
        )
        logger.info(make_table(results))
        acc = {t: results['results'][t]['acc,none'] for t in tasks}
        metrics["acc"] = acc
        metrics["acc_avg"] = sum(acc.values()) / len(acc)
        logger.info(f"Average Acc: {metrics['acc_avg'] * 100:.2f}%")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(args.output_dir, "test_only_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"saved -> {os.path.join(args.output_dir, 'test_only_metrics.json')}")
    return metrics


def train():
    hfparser = transformers.HfArgumentParser((
        ModelArguments, DataArguments, TrainingArguments, GenerationArguments
    ))
    model_args, data_args, training_args, generation_args, extra_args = \
        hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    training_args.generation_config = transformers.GenerationConfig(**vars(generation_args))
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger = utils.create_logger(args.output_dir)
    logger.info(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print('Detected that training was already completed!')

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    model.config.use_cache = False
    print('loaded model')
    set_seed(args.seed)

    if args.test_only:
        run_test_only(model, tokenizer, args, logger)
        return

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    optimizer_grouped_parameters = []
    for name, module in model.named_modules():
        from dbw.real_dbw import MultiQuantLinear, realDBW
        if isinstance(module, MultiQuantLinear):
            module.scales.requires_grad = True
            module.zeros.requires_grad = True
        if isinstance(module, realDBW):
            module.sigma.requires_grad = True
    learnable_param_checker = lambda x: ('scales' in x) or ('zeros' in x) or ('sigma' in x)
    optimizer_grouped_parameters.append(
        {'params': [p for n, p in model.named_parameters() if learnable_param_checker(n)], 'weight_decay': 0.0,
         'lr': args.learning_rate})
    optimizer = AdamW(optimizer_grouped_parameters)

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        optimizers=(optimizer, None),
        **{k: v for k, v in data_module.items() if k != 'predict_dataset'},
    )

    if args.do_ppl_eval:
        class PPLvalCallback(transformers.TrainerCallback):
            @torch.no_grad()
            def on_evaluate(self, args=None, state=None, control=None, model=None, **kwargs):
                with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                    results = test_ppl(trainer.model, trainer.tokenizer, datasets=['wikitext2', 'c4'], ppl_seqlen=2048)
                    logger.info(results)
                    trainer.log(results)

        trainer.add_callback(PPLvalCallback)

    print_trainable_parameters(args, model)
    dtypes = {}
    for _, p in model.named_parameters():
        dtype = p.dtype
        if dtype not in dtypes: dtypes[dtype] = 0
        dtypes[dtype] += p.numel()
    total = 0
    for k, v in dtypes.items(): total += v
    for k, v in dtypes.items():
        print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}

    print(args.output_dir)
    if args.do_train:
        logger.info("*** Train ***")
        train_result = trainer.train(args.resume_from_checkpoint)
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)

    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(test_dataset=data_module['predict_dataset'], metric_key_prefix="predict")
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        with open(os.path.join(args.output_dir, 'predictions.jsonl'), 'w') as fout:
            for i, example in enumerate(data_module['predict_dataset']):
                example['prediction_with_input'] = predictions[i].strip()
                example['prediction'] = predictions[i].replace(example['input'], '').strip()
                fout.write(json.dumps(example) + '\n')
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if (args.do_train or args.do_eval or args.do_predict):
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    if args.eval_tasks != "" or args.do_mmlu_eval:
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

    if args.eval_tasks != "":
        task_list = args.eval_tasks.split(',')
        lm_eval_model = HFLM(pretrained=model, batch_size=args.lm_eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')

    if args.do_mmlu_eval:
        lm_eval_model = HFLM(pretrained=model, batch_size=16)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=lm_eval_model,
            tasks=['mmlu'],
            num_fewshot=5,
            task_manager=task_manager,
            cache_requests=True,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in results['results']:
            total_acc += results['results'][task]['acc,none']
        logger.info(f"Average MMLU Acc: {total_acc / len(results['results']) * 100:.2f}%")


if __name__ == "__main__":
    train()