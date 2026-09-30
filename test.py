import argparse
import json
import os
import time

import torch
from accelerate import dispatch_model

from dbw.utils.model import balanced_device_map_for_decoder_model
from dbw.utils.data_block import test_ppl as measure_ppl

DEFAULT_ACC_TASKS = ("piqa,arc_easy,hellaswag,winogrande,race,arc_challenge,"
                     "lambada_openai,lambada_standard")


def load_bundle(path):
    print(f"[load] {path}", flush=True)
    obj = torch.load(path, weights_only=False)
    model, tokenizer = obj["model"], obj["tokenizer"]
    model = dispatch_model(model, device_map=balanced_device_map_for_decoder_model(model))
    model.eval()
    tag = os.path.basename(path).removesuffix(".pt")
    return model, tokenizer, tag


def run_ppl(model, tokenizer, tag, datasets, seqlen):
    with torch.cuda.amp.autocast():
        results = measure_ppl(model, tokenizer, datasets=datasets, ppl_seqlen=seqlen)
    for dataset, ppl in results.items():
        print(f"[result] {tag}  {dataset}  ppl={ppl:.4f}", flush=True)
    return {"ppl": {k: float(v) for k, v in results.items()}}


def run_acc(model, tokenizer, tag, tasks, batch_size):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.utils import make_table

    # the model is spread across GPUs; feed inputs to whichever holds the
    # embedding and let the accelerate hooks move activations onward
    device = str(model.get_input_embeddings().weight.device)
    results = lm_eval.simple_evaluate(
        model=HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device),
        tasks=tasks,
        num_fewshot=0,
        task_manager=lm_eval.tasks.TaskManager(),
    )
    print(make_table(results), flush=True)

    acc = {t: results["results"][t]["acc,none"] for t in tasks}
    avg = sum(acc.values()) / len(acc)
    for t in tasks:
        print(f"[acc] {tag}  {t}  {acc[t] * 100:.2f}", flush=True)
    print(f"[acc-avg] {tag}  {avg * 100:.2f}", flush=True)
    return {"acc": acc, "acc_avg": avg}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True, help="output/release/<tag>.pt")
    p.add_argument("--mode", default="ppl", choices=("ppl", "acc"),
                   help="ppl: wikitext2/c4 perplexity. acc: 0-shot lm-eval accuracy "
                        "reported as `acc` (acc,none), never acc_norm")
    p.add_argument("--datasets", default="wikitext2,c4", help="ppl mode: comma separated")
    p.add_argument("--ppl_seqlen", type=int, default=2048)
    p.add_argument("--tasks", default=DEFAULT_ACC_TASKS, help="acc mode: comma separated")
    p.add_argument("--batch_size", type=int, default=8, help="acc mode batch size")
    p.add_argument("--tag", default=None, help="label used in the JSON output")
    p.add_argument("--result_json", default=None, help="append one JSON line here")
    args = p.parse_args()

    t0 = time.time()
    model, tokenizer, tag = load_bundle(args.bundle)
    tag = args.tag or tag

    if args.mode == "ppl":
        datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
        measured = run_ppl(model, tokenizer, tag, datasets, args.ppl_seqlen)
    else:
        tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        measured = run_acc(model, tokenizer, tag, tasks, args.batch_size)

    if args.result_json:
        record = {"tag": tag, "source": args.bundle,
                  "elapsed_sec": round(time.time() - t0, 1), **measured}
        with open(args.result_json, "a") as f:
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
