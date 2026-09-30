import argparse, random, os, sys, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
import dbw.utils as utils
from dbw.block_ap import block_ap
from dbw.utils.data_block import get_loaders, test_ppl
from dbw.utils.model import load_tokenizer, ensure_tokenizer_has_pad
from dbw.utils.model import load_fake_quantized_model, load_real_quantized_model, fake_to_real
from dbw.utils.decompose import build_svd_cache, cache_dir_for, missing_svd_layers


torch.backends.cudnn.benchmark = True
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


@torch.no_grad()
def evaluate(model, tokenizer, args, logger):
    '''
    Note: evaluation simply move model to single GPU. 
    Therefor, to evaluate large model such as Llama-2-70B on single A100-80GB,
    please activate '--real_quant'.
    '''
    block_class_name = model.model.layers[0].__class__.__name__
    device_map = infer_auto_device_map(
        model, 
        max_memory={i: args.max_memory for i in range(torch.cuda.device_count())}, 
        no_split_module_classes=[block_class_name]
    )
    model = dispatch_model(model, device_map=device_map)
    results = {}

    if args.eval_ppl:
        datasets = ["wikitext2", "c4"]
        ppl_results = test_ppl(model, tokenizer, datasets, args.ppl_seqlen)
        for dataset in ppl_results:
            logger.info(f'{dataset} perplexity: {ppl_results[dataset]:.2f}')

    if args.eval_tasks != "":
        import lm_eval
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table
        task_list = args.eval_tasks.split(',')
        model = HFLM(pretrained=model, batch_size=args.eval_batch_size)
        task_manager = lm_eval.tasks.TaskManager()
        results = lm_eval.simple_evaluate(
            model=model,
            tasks=task_list,
            num_fewshot=0,
            task_manager=task_manager,
        )
        logger.info(make_table(results))
        total_acc = 0
        for task in task_list:
            total_acc += results['results'][task]['acc,none']
        logger.info(f'Average Acc: {total_acc / len(task_list) * 100:.2f}%')
    return results


def ensure_svd_cache(args, logger):
    """Block-AP reads per-layer SVD factors from svd/<model>_fp16/<i>.pt. Build any that are absent."""
    hf_kwargs = {"trust_remote_code": args.trust_remote_code}
    if args.use_auth_token:
        hf_kwargs["use_auth_token"] = True
    num_layers = AutoConfig.from_pretrained(args.model, **hf_kwargs).num_hidden_layers
    missing = missing_svd_layers(args.model, num_layers, args.svd_dir)
    target = cache_dir_for(args.model, args.svd_dir)
    if not missing:
        logger.info(f"SVD cache complete: {target} ({num_layers} layers)")
        return
    if args.no_auto_svd:
        raise FileNotFoundError(
            f"{len(missing)}/{num_layers} SVD files missing under {target} "
            f"(first missing: {missing[0]}.pt). Build them with "
            f"`python dbw/utils/decompose.py --model-name {args.model} --save-dir {args.svd_dir}` "
            f"or drop --no_auto_svd.")
    logger.info(f"SVD cache incomplete: {len(missing)}/{num_layers} missing under {target} -> building")
    build_svd_cache(args.model, args.svd_dir,
                    trust_remote_code=args.trust_remote_code, use_auth_token=args.use_auth_token,
                    svd_device=args.svd_device, log=logger.info)
    still = missing_svd_layers(args.model, num_layers, args.svd_dir)
    if still:
        raise RuntimeError(f"SVD cache still incomplete after building: {len(still)} missing")
    logger.info(f"SVD cache ready: {target}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="model name of model path")
    parser.add_argument("--net", type=str, default=None, help="model (family) name, for the easier saving of data cache")
    parser.add_argument("--cache_dir", default="./cache", type=str, help="direction of cached dataset")
    parser.add_argument("--output_dir", default="./log/", type=str, help="direction of logging file")
    parser.add_argument("--save_quant_dir", default=None, type=str, help="direction for saving quantization model")
    parser.add_argument("--real_quant", default=False, action="store_true", help="use real quantization to store weight")
    parser.add_argument("--test_only", type=str, default=None, 
    help="path to a saved quantized model: skip training and only evaluate it")
    parser.add_argument("--fake_to_real", type=str, default=None, help="save path for real model")
    parser.add_argument("--calib_dataset", type=str, default="redpajama",
                        choices=["wikitext2", "ptb", "c4", "mix", "redpajama"],
                        help="Where to extract calibration data from.")
    parser.add_argument("--train_size", type=int, default=4096, help="Number of training data samples.")
    parser.add_argument("--val_size", type=int, default=64, help="Number of validation data samples.")
    parser.add_argument("--training_seqlen", type=int, default=2048, help="lenth of the training sequence.")
    parser.add_argument("--batch_size", type=int, default=2, help="batch size.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--ppl_seqlen", type=int, default=2048, help="input sequence length for evaluating perplexity")
    parser.add_argument("--seed", type=int, default=46, help="Seed for sampling the calibration data.")
    parser.add_argument("--eval_ppl", action="store_true", help="evaluate perplexity on wikitext2 and c4")
    parser.add_argument("--eval_tasks", type=str, default="piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--max_memory", type=str, default="20GiB", help="The maximum memory of each GPU")
    parser.add_argument("--early_stop", type=int, default=0, help="early stoping after validation loss do not decrease")
    parser.add_argument("--wbits", type=float, default=4, help="weights quantization bits")
    parser.add_argument("--group_size", type=int, default=128, help="weights quantization group size")
    parser.add_argument("--quant_lr", type=float, default=1e-4, help="lr of quantization parameters (s and z)")
    parser.add_argument("--weight_lr", type=float, default=1e-5, help="lr of full-precision weights")
    parser.add_argument("--min_lr_factor", type=float, default=20, help="min_lr = lr/min_lr_factor")
    parser.add_argument("--bit_lr", type=float, default=0, help="bit learning rate")
    parser.add_argument("--gbit_lr", type=float, default=0, help="bit learning rate")
    parser.add_argument("--u_lr", type=float, default=0, help="bit learning rate")
    parser.add_argument("--v_lr", type=float, default=0, help="bit learning rate")
    parser.add_argument("--norm_lr", type=float, default=0, help="bit learning rate")
    parser.add_argument("--clip_grad", type=float, default=0.3, help="gradient clipping")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--off_load_to_disk", action="store_true", default=False,
                        help="save training dataset to disk, saving CPU memory but may reduce training speed")
    parser.add_argument("--trust_remote_code", action="store_true", default=False,
                        help="allow custom modelling code from the Hub (Qwen, some Gemma repos)")
    parser.add_argument("--use_auth_token", action="store_true", default=False,
                        help="use the cached HF token; required for gated repos such as Gemma")
    parser.add_argument("--fast_calib_sampling", action="store_true", default=False,
                        help="streaming/reservoir calibration sampling; off keeps the original Llama behaviour")
    parser.add_argument("--redpajama_streaming", action="store_true", default=False,
                        help="stream RedPajama instead of downloading the full split")
    parser.add_argument("--dataset_streaming_buffer_size", type=int, default=10000,
                        help="shuffle buffer used when --redpajama_streaming is on")
    parser.add_argument("--no_global_bit", dest="global_bit", action="store_false", default=True,
                        help="allocate bits within each linear layer instead of globally across the block")
    parser.add_argument("--no_progressive", dest="progressive", action="store_false", default=True,
                        help="skip the per-block pre-training pass governed by --pre_lr and --lambda_sv")
    parser.add_argument("--pre_lr", type=float, default=1e-5,
                        help="--progressive: lr of the per-module pre-training pass")
    parser.add_argument("--lambda_sv", type=float, default=1e-6,
                        help="--progressive: weight of both singular-value regularizers")
    parser.add_argument("--no_save_statistics", dest="save_statistics", action="store_false", default=True,
                        help="do not dump per-layer bit / sigma / relu statistics while training")
    parser.add_argument("--svd_dir", type=str, default="svd",
                        help="where per-layer SVD factors live (auto-built when missing)")
    parser.add_argument("--svd_device", choices=["auto", "cpu", "cuda"], default="cpu",
                        help="device used when building the SVD cache")
    parser.add_argument("--no_auto_svd", action="store_true", default=False,
                        help="fail instead of building a missing SVD cache")
    parser.add_argument("--wandb", action="store_true", default=False,
                        help="log training to Weights & Biases")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    if args.save_quant_dir:
        Path(args.save_quant_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(args.output_dir)
    logger = utils.create_logger(output_dir)
    logger.info(args)

    if args.net is None:
        args.net = args.model.split('/')[-1]
        logger.info(f"net is None, setting as {args.net}")

    if args.test_only:
        if args.fake_to_real:
            model, tokenizer = load_fake_quantized_model(args.test_only, args.wbits, args.group_size)
            model = fake_to_real(model, group_size=args.group_size)
        elif args.real_quant:
            model, tokenizer = load_real_quantized_model(args.test_only, args.wbits, args.group_size)
        else:
            model, tokenizer = load_fake_quantized_model(args.test_only, args.wbits, args.group_size)
        logger.info(
            f"memory footprint after loading quantized model: "
            f"{torch.cuda.max_memory_allocated('cuda') / 1024 ** 3:.2f}GiB")
    else:
        if args.wbits < 16:
            ensure_svd_cache(args, logger)
        hf_kwargs = {"trust_remote_code": args.trust_remote_code}
        if args.use_auth_token:
            hf_kwargs["use_auth_token"] = True
        config = AutoConfig.from_pretrained(args.model, **hf_kwargs)
        tokenizer = load_tokenizer(args.model, trust_remote_code=args.trust_remote_code,
                                   use_auth_token=args.use_auth_token)
        model = AutoModelForCausalLM.from_pretrained(args.model, config=config, device_map='cpu',
                                                     torch_dtype=torch.float16, **hf_kwargs)
        ensure_tokenizer_has_pad(tokenizer, model, resize_embeddings=False)
        for param in model.parameters():
            param.requires_grad = False

        if args.wbits < 16:
            logger.info("=== start quantization ===")
            tick = time.time()
            cache_prefix = (f'{args.cache_dir}/dataloader_{args.net}_{args.calib_dataset}_'
                            f'{args.train_size}_{args.val_size}_{args.training_seqlen}')
            if args.fast_calib_sampling or args.redpajama_streaming:
                cache_prefix += (f'_fast{int(bool(args.fast_calib_sampling))}'
                                 f'_buf{args.dataset_streaming_buffer_size}')
            cache_trainloader = f'{cache_prefix}_train.cache'
            cache_valloader = f'{cache_prefix}_val.cache'
            if os.path.exists(cache_trainloader) and os.path.exists(cache_valloader):
                trainloader = torch.load(cache_trainloader)
                logger.info(f"load trainloader from {cache_trainloader}")
                valloader = torch.load(cache_valloader)
                logger.info(f"load valloader from {cache_valloader}")
            else:
                trainloader, valloader = get_loaders(
                    args.calib_dataset,
                    tokenizer,
                    args.train_size,
                    args.val_size,
                    seed=args.seed,
                    seqlen=args.training_seqlen,
                    fast_sample=args.fast_calib_sampling,
                    redpajama_streaming=args.redpajama_streaming,
                    streaming_buffer_size=args.dataset_streaming_buffer_size,
                )
                torch.save(trainloader, cache_trainloader)
                torch.save(valloader, cache_valloader)
            block_ap(
                model,
                args,
                trainloader,
                valloader,
                logger,
            )
            logger.info(time.time() - tick)
    torch.cuda.empty_cache()

    if args.save_quant_dir:
        logger.info("start saving model")
        torch.save(model, args.save_quant_dir + '.pt')
        tokenizer.save_pretrained(args.save_quant_dir)
        logger.info("save model success")

    evaluate(model, tokenizer, args, logger)


if __name__ == "__main__":
    print(sys.argv)
    main()
