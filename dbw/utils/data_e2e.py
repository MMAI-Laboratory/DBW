import torch
from typing import Dict, Sequence
from datasets import load_dataset, Dataset, concatenate_datasets

try:
    from datasets import IterableDataset as HFIterableDataset
except Exception:
    HFIterableDataset = ()
import os
from itertools import chain
from pathlib import Path
from transformers import default_data_collator
import transformers
from dataclasses import dataclass
from torch.nn.utils.rnn import pad_sequence
import copy
import numpy as np
import hashlib
from dbw.utils.data_block import REDPAJAMA_HUB_ID, redpajama_source

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _get_arg(args, name, default=None):
    return getattr(args, name, default)


def _is_iterable_dataset(dataset):
    return HFIterableDataset != () and isinstance(dataset, HFIterableDataset)


def _dataset_train_split(dataset):
    if isinstance(dataset, dict):
        return dataset["train"]
    if hasattr(dataset, "keys") and "train" in dataset.keys():
        return dataset["train"]
    return dataset


def _iter_examples(dataset, seed=0, buffer_size=10000):
    if _is_iterable_dataset(dataset):
        if buffer_size and hasattr(dataset, "shuffle"):
            dataset = dataset.shuffle(seed=seed, buffer_size=buffer_size)
        for row in dataset:
            yield row
    else:
        dataset = dataset.shuffle(seed=seed)
        for row in dataset:
            yield row


def _extract_text(row, preferred_column=None):
    if preferred_column and isinstance(row, dict) and preferred_column in row:
        return row[preferred_column]
    if isinstance(row, dict):
        if "text" in row:
            return row["text"]
        for value in row.values():
            if isinstance(value, str):
                return value
    return str(row)


def _empty_pt_dataset():
    return Dataset.from_dict({"input_ids": [], "attention_mask": [], "labels": []})


def _build_pt_blocks_from_examples(dataset, tokenizer, block_size, num_blocks, seed=0,
                                   buffer_size=10000, text_column=None, desc="dataset"):
    """Build only the requested number of LM blocks from a shuffled raw dataset."""
    num_blocks = int(num_blocks or 0)
    if num_blocks <= 0:
        return _empty_pt_dataset()

    input_ids = []
    attention_mask = []
    labels = []
    token_buffer = []
    rows_seen = 0

    print(f"[fast-data] building {num_blocks} block(s) from {desc}; block_size={block_size}")
    for row in _iter_examples(dataset, seed=seed, buffer_size=buffer_size):
        rows_seen += 1
        text = _extract_text(row, text_column)
        if not text:
            continue
        encoded = tokenizer(text, add_special_tokens=True, truncation=False)
        ids = encoded.get("input_ids", [])
        if not ids:
            continue
        token_buffer.extend(ids)
        while len(token_buffer) >= block_size and len(input_ids) < num_blocks:
            chunk = token_buffer[:block_size]
            del token_buffer[:block_size]
            input_ids.append(chunk)
            attention_mask.append([1] * block_size)
            labels.append(chunk.copy())
        if len(input_ids) >= num_blocks:
            break

    if len(input_ids) < num_blocks:
        print(
            f"[WARN] requested {num_blocks} block(s) from {desc}, "
            f"but built only {len(input_ids)} after reading {rows_seen} row(s)."
        )
    else:
        print(f"[fast-data] built {len(input_ids)} block(s) from {rows_seen} raw row(s).")

    return Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    })


def _repeat_or_trim_dataset(dataset, target_len):
    target_len = int(target_len or 0)
    if target_len <= 0:
        return _empty_pt_dataset()
    if len(dataset) == 0:
        return _empty_pt_dataset()
    indices = [i % len(dataset) for i in range(target_len)]
    return dataset.select(indices)


def _fast_mix_counts(total, rp_pool=4096, wiki_pool=1024, seed=0):
    """Approximate the old two-shuffle mix source composition for small max_train_samples."""
    total = int(total)
    pool = rp_pool + wiki_pool
    if total >= pool:
        return rp_pool, wiki_pool
    labels = np.array([0] * rp_pool + [1] * wiki_pool, dtype=np.int8)
    for _ in range(2):
        rng = np.random.default_rng(seed)
        labels = labels[rng.permutation(len(labels))]
    picked = labels[:total]
    n_rp = int((picked == 0).sum())
    n_wiki = int((picked == 1).sum())
    return n_rp, n_wiki


def _cache_fingerprint(*parts):
    raw = "|".join(str(part) for part in parts if part is not None)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()[:10]


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        bos_token = self.tokenizer.bos_token or ""
        eos_token = self.tokenizer.eos_token or ""
        sources = [f"{bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{eos_token}" for example in instances]
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
                tokenized_sources_with_prompt['input_ids'],
                tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True,
                              padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask': input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {'input': prompt_format.format(**example)}


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning or continue pre-train.
    """

    def load_data(dataset_name):
        if dataset_name == 'alpaca':
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == 'oasst1':
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == 'c4':
            dataset = load_dataset(
                "allenai/c4", "allenai--c4", 
                data_files={
                    "train": "en/c4-train.00000-of-01024.json.gz", 
                    "validation": "en/c4-validation.00000-of-00008.json.gz", 
                },
            )
            return dataset
        elif dataset_name == 'redpajama':
            data_files = None
            use_streaming = _as_bool(_get_arg(args, "redpajama_streaming", False))
            if use_streaming:
                return {"train": load_dataset(redpajama_source(), split="train", streaming=True)}

            source = redpajama_source()
            try:
                dataset = load_dataset(source)
            except Exception as local_err:
                if source == REDPAJAMA_HUB_ID:
                    raise
                print(f"[data] {source} failed ({local_err}); falling back to the Hub")
                try:
                    dataset = load_dataset(REDPAJAMA_HUB_ID)
                except Exception as hub_err:
                    raise RuntimeError(
                        f"could not load RedPajama from {source} ({local_err}) "
                        f"nor from the Hub ({hub_err})") from local_err
            return dataset
        elif dataset_name == 'wikitext2':
            return load_dataset('wikitext', 'wikitext-2-raw-v1')
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    def format_dataset(dataset, dataset_format):
        is_pt_format = dataset_format == 'pt' or (
                dataset_format is None and args.dataset in ['c4', 'redpajama', 'wikitext2', 'mix_redpajama_wiki']
        )
        if (
                dataset_format == 'alpaca' or dataset_format == 'alpaca-clean' or
                (dataset_format is None and args.dataset in ['alpaca', 'alpaca-clean'])
        ):
            dataset = dataset.map(extract_alpaca_dataset, remove_columns=['instruction'])
        elif dataset_format == 'oasst1' or (dataset_format is None and args.dataset == 'oasst1'):
            dataset = dataset.map(lambda x: {
                'input': '',
                'output': x['text'],
            })
        elif is_pt_format:
            block_size = args.pt_context_len
            column_names = list(dataset["train"].features)
            text_column_name = "text" if "text" in column_names else column_names[0]

            def tokenize_function(examples):
                output = tokenizer(examples[text_column_name])
                return output

            tokenized_datasets = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Running tokenizer on dataset",
            )

            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                if total_length >= block_size:
                    total_length = (total_length // block_size) * block_size
                result = {
                    k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            dataset = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        if not is_pt_format:
            dataset = dataset.remove_columns(
                [col for col in dataset.column_names['train'] if col not in ['input', 'output']]
            )
        return dataset

    def build_fast_mix_redpajama_wiki():
        """Fast path for the e2e PT mix without tokenizing the full corpora."""
        seed = 0
        block_size = args.pt_context_len
        full_rp = int(_get_arg(args, "mix_redpajama_blocks", 4096))
        full_wiki_base = int(_get_arg(args, "mix_wiki_blocks", 256))
        wiki_repeats = int(_get_arg(args, "mix_wiki_repeats", 4))
        full_wiki = full_wiki_base * wiki_repeats
        full_pool = full_rp + full_wiki
        target_train = _get_arg(args, "max_train_samples", None)
        target_train = full_pool if target_train is None else min(int(target_train), full_pool)
        target_eval = _get_arg(args, "max_eval_samples", None)
        target_eval = int(target_eval if target_eval is not None else _get_arg(args, "eval_dataset_size", 64))
        buffer_size = int(_get_arg(args, "streaming_buffer_size", 10000))

        n_rp, n_wiki = _fast_mix_counts(target_train, rp_pool=full_rp, wiki_pool=full_wiki, seed=seed)
        print(
            f"[fast-data] mix_redpajama_wiki target_train={target_train}: "
            f"redpajama={n_rp}, wikitext={n_wiki}"
        )

        train_parts = []
        if n_rp > 0:
            rp = _dataset_train_split(load_data('redpajama'))
            rp_train = _build_pt_blocks_from_examples(
                rp, tokenizer, block_size, n_rp, seed=seed,
                buffer_size=buffer_size, desc="redpajama train"
            )
            if len(rp_train) > 0:
                train_parts.append(rp_train)

        wk = load_data('wikitext2')
        if n_wiki > 0:
            wk_base_needed = min(full_wiki_base, n_wiki)
            wk_base = _build_pt_blocks_from_examples(
                wk['train'], tokenizer, block_size, wk_base_needed, seed=seed,
                buffer_size=buffer_size, desc="wikitext2 train"
            )
            wk_train = _repeat_or_trim_dataset(wk_base, n_wiki)
            if len(wk_train) > 0:
                train_parts.append(wk_train)

        if not train_parts:
            raise RuntimeError("fast mix_redpajama_wiki produced an empty training dataset")
        mixed_train = concatenate_datasets(train_parts).shuffle(seed=seed)

        validation = _build_pt_blocks_from_examples(
            wk['validation'], tokenizer, block_size, target_eval, seed=seed,
            buffer_size=buffer_size, desc="wikitext2 validation"
        )
        return {"train": mixed_train, "validation": validation}

    print(f"loading {args.dataset}")
    prebuilt_cache = _get_arg(args, "prebuilt_dataloader_cache", None)
    if prebuilt_cache:
        print(f"[prebuilt] loading dataloader cache from {prebuilt_cache}")
        dataset = torch.load(prebuilt_cache, weights_only=False)
        print(f"[prebuilt] splits={list(dataset.keys())} "
              f"train={len(dataset.get('train', []))} "
              f"validation={len(dataset.get('validation', []))}")
    elif args.dataset in ['c4', 'redpajama', 'wikitext2']:
        cache_dir = './cache'
        cache_dataloader = f'{cache_dir}/e2e_dataloader_{args.model_family}_{args.dataset}_{args.pt_context_len}.cache'
        if os.path.exists(cache_dataloader):
            dataset = torch.load(cache_dataloader)
            print(f"load dataset from {cache_dataloader}")
        else:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            dataset = load_data(args.dataset)
            dataset = format_dataset(dataset, args.dataset_format)
            torch.save(dataset, cache_dataloader)
    elif args.dataset == 'mix_redpajama_wiki':
        cache_dir = './cache'
        fast_pt_sample = _as_bool(_get_arg(args, "fast_pt_sample", False))
        if fast_pt_sample:
            target_train = _get_arg(args, "max_train_samples", None)
            target_train = "all" if target_train is None else str(target_train)
            target_eval = _get_arg(args, "max_eval_samples", None)
            target_eval = str(target_eval if target_eval is not None else _get_arg(args, "eval_dataset_size", 64))
            fp = _cache_fingerprint(
                args.model_family,
                args.dataset,
                args.pt_context_len,
                target_train,
                target_eval,
                _get_arg(args, "streaming_buffer_size", 10000),
                _get_arg(args, "redpajama_streaming", True),
            )
            cache_dataloader = (
                f'{cache_dir}/dataloader_{args.model_family}_{args.dataset}_'
                f'{args.pt_context_len}_fast_n{target_train}_e{target_eval}_{fp}.cache'
            )
            if os.path.exists(cache_dataloader) and not args.overwrite_cache:
                dataset = torch.load(cache_dataloader)
                print(f"load dataset from {cache_dataloader}")
            else:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)
                dataset = build_fast_mix_redpajama_wiki()
                torch.save(dataset, cache_dataloader)
                print(f"save fast dataset cache to {cache_dataloader}")
        else:
            cache_dataloader = f'{cache_dir}/dataloader_{args.model_family}_{args.dataset}_{args.pt_context_len}.cache'

            if os.path.exists(cache_dataloader):
                dataset = torch.load(cache_dataloader)
                print(f"load dataset from {cache_dataloader}")
            else:
                Path(cache_dir).mkdir(parents=True, exist_ok=True)

                rp = load_data('redpajama')
                wk = load_data('wikitext2')

                rp_pt = format_dataset(rp, 'pt')
                wk_pt = format_dataset(wk, 'pt')

                rp_train = rp_pt["train"]
                wk_train = wk_pt["train"]

                n_rp = 4096
                n_wk = 256
                wk_repeats = 4
                seed = 0

                rp_shuf = rp_train.shuffle(seed=seed)
                rp_subset = rp_shuf.select(range(min(n_rp, len(rp_shuf))))
                if len(rp_train) < n_rp:
                    print(f"[WARN] redpajama train size {len(rp_train)} < {n_rp}, using {len(rp_subset)} instead.")

                wk_shuf = wk_train.shuffle(seed=seed)

                if len(wk_shuf) >= n_wk:
                    wk_256 = wk_shuf.select(range(n_wk))
                else:
                    repeat = (n_wk + len(wk_shuf) - 1) // len(wk_shuf)
                    wk_256 = concatenate_datasets([wk_shuf] * repeat).select(range(n_wk))

                wk_subset = concatenate_datasets([wk_256] * wk_repeats)

                mixed_train = concatenate_datasets([rp_subset, wk_subset]).shuffle(seed=seed)

                validation = rp_pt['validation'] if 'validation' in rp_pt else wk_pt['validation']

                dataset = {"train": mixed_train, "validation": validation}
                torch.save(dataset, cache_dataloader)
    print(f"loading {args.dataset} successfully")

    if args.do_eval or args.do_predict:
        if 'eval' in dataset:
            eval_dataset = dataset['eval']
        elif 'validation' in dataset:
            eval_dataset = dataset['validation']
        else:
            print('Splitting train dataset in train and validation according to `eval_dataset_size`')
            dataset = dataset["train"].train_test_split(
                test_size=args.eval_dataset_size, shuffle=True, seed=42
            )
            eval_dataset = dataset['test']
        if args.max_eval_samples is not None and len(eval_dataset) > args.max_eval_samples:
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.do_train:
        train_dataset = dataset['train']
        train_dataset = train_dataset.shuffle(seed=0)
        if args.max_train_samples is not None and len(train_dataset) > args.max_train_samples:
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(lambda x: {'length': len(x['input']) + len(x['output'])})
    if args.dataset in ['c4', 'redpajama', 'wikitext2', 'mix_redpajama_wiki']:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForCausalLM(
            tokenizer=tokenizer,
            source_max_len=args.source_max_len,
            target_max_len=args.target_max_len,
            train_on_source=args.train_on_source,
            predict_with_generate=args.predict_with_generate,
        )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator
    )
