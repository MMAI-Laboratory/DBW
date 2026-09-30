import os
import random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset

try:
    from datasets import IterableDataset as HFIterableDataset
except Exception:
    HFIterableDataset = ()

from dbw.utils.model import get_input_device


def _as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _is_iterable_dataset(dataset):
    return HFIterableDataset != () and isinstance(dataset, HFIterableDataset)


def _extract_text(row):
    if isinstance(row, dict):
        if "text" in row:
            return row["text"]
        for value in row.values():
            if isinstance(value, str):
                return value
    return str(row)


def _make_lm_sample(input_ids, seqlen):
    i = random.randint(0, input_ids.shape[1] - seqlen - 1)
    j = i + seqlen
    inp = input_ids[:, i:j]
    tar = inp.clone()
    tar[:, :-1] = -100
    return inp, tar


REDPAJAMA_HUB_ID = "togethercomputer/RedPajama-Data-1T-Sample"


def redpajama_source():
    """A pre-downloaded snapshot if one is reachable, else the Hub id.

    DBW_REDPAJAMA_PATH overrides the location. Otherwise the standard
    Hugging Face datasets cache is probed, so a machine that already holds the
    dataset keeps using it and a fresh machine falls back to the Hub.
    """
    override = os.environ.get("DBW_REDPAJAMA_PATH")
    if override:
        return override if os.path.isdir(override) else REDPAJAMA_HUB_ID
    cache = os.environ.get("HF_DATASETS_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "datasets")
    snapshot = os.path.join(cache, "togethercomputer___red_pajama-data-1_t-sample")
    return snapshot if os.path.isdir(snapshot) else REDPAJAMA_HUB_ID


def _load_redpajama_dataset(streaming=False):
    return load_dataset(redpajama_source(), split="train", streaming=_as_bool(streaming))


def _sample_redpajama_stream(stream, tokenizer, train_size, val_size, seed, seqlen, buffer_size):
    random.seed(seed)
    if hasattr(stream, "shuffle") and buffer_size:
        stream = stream.shuffle(seed=seed, buffer_size=buffer_size)
    trainloader = []
    valloader = []
    rows_seen = 0
    for row in stream:
        rows_seen += 1
        text = _extract_text(row)
        if not text:
            continue
        enc = tokenizer(text, return_tensors="pt")
        if enc.input_ids.shape[1] < seqlen + 1:
            continue
        sample = _make_lm_sample(enc.input_ids, seqlen)
        if len(trainloader) < train_size:
            trainloader.append(sample)
        else:
            valloader.append(sample)
        if len(trainloader) >= train_size and len(valloader) >= val_size:
            break
    print(f"[fast-data] redpajama block calibration used {rows_seen} streamed row(s).")
    if len(trainloader) < train_size or len(valloader) < val_size:
        print(
            f"[WARN] requested train={train_size}, val={val_size}, "
            f"but built train={len(trainloader)}, val={len(valloader)}."
        )
    return trainloader, valloader


def get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_wikitext2")
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')

    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    if test_only:
        return testenc
    trainenc = tokenizer("\n\n".join(traindata['text']), return_tensors='pt')

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        i = random.randint(0, int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    valloader = []
    for _ in range(val_size):
        i = random.randint(int(trainenc.input_ids.shape[1] * val_sample_ratio) - seqlen - 1,
                           trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))
    return trainloader, valloader


def get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only):
    print("get_c4")
    try:
        traindata = load_dataset(
            "arrow",
            data_files={
                "train": [
                    "./tmp/c4_cache_fixed/train/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train-00000-of-00002.arrow",
                    "./tmp/c4_cache_fixed/train/allenai___c4/default-b04fc8a0b8562884/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-train-00001-of-00002.arrow"
                ]
            },
            split='train'
        )
        valdata = load_dataset(
            "arrow",
            data_files={
                "validation": "./tmp/c4_cache_fixed/val/allenai___c4/default-c7bc8b0aefc5e48f/0.0.0/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/c4-validation.arrow",
            },
            split='validation'
        )
    except:
        traindata = load_dataset(
            'allenai/c4', data_files={'train': 'en/c4-train.00000-of-01024.json.gz'}, split='train',
            cache_dir="./tmp/c4_cache_fixed/train"
        )
        valdata = load_dataset(
            'allenai/c4', data_files={'validation': 'en/c4-validation.00000-of-00008.json.gz'},
            split='validation', cache_dir="./tmp/c4_cache_fixed/val",
        )

    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            if tmp.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)
    if test_only:
        return valenc

    random.seed(seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        valloader.append((inp, tar))

    return trainloader, valloader


def get_redpajama(tokenizer, train_size, val_size, seed, seqlen,
                  fast_sample=False, redpajama_streaming=False,
                  streaming_buffer_size=10000):
    print("get_redpajama")
    if _as_bool(fast_sample) and _as_bool(redpajama_streaming):
        stream = _load_redpajama_dataset(streaming=True)
        return _sample_redpajama_stream(
            stream, tokenizer, train_size, val_size, seed, seqlen, streaming_buffer_size
        )

    traindata = _load_redpajama_dataset(streaming=False)
    random.seed(seed)
    traindata = traindata.shuffle(seed=seed)
    trainloader = []
    val_sample_ratio = 0.9
    for _ in range(train_size):
        while True:
            i = random.randint(0, int(len(traindata) * val_sample_ratio) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        inp, tar = _make_lm_sample(trainenc.input_ids, seqlen)
        trainloader.append((inp, tar))

    valloader = []
    for _ in range(val_size):
        while True:
            i = random.randint(int(len(traindata) * val_sample_ratio), len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] >= seqlen + 1:
                break
        inp, tar = _make_lm_sample(trainenc.input_ids, seqlen)
        valloader.append((inp, tar))
    return trainloader, valloader


def get_loaders(
        name, tokenizer, train_size=128, val_size=64, seed=0, seqlen=2048, test_only=False,
        fast_sample=False, redpajama_streaming=False, streaming_buffer_size=10000,
):
    if 'wikitext2' in name:
        return get_wikitext2(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'c4' in name:
        return get_c4(tokenizer, train_size, val_size, seed, seqlen, test_only)
    elif 'redpajama' in name:
        return get_redpajama(
            tokenizer, train_size, val_size, seed, seqlen,
            fast_sample=fast_sample,
            redpajama_streaming=redpajama_streaming,
            streaming_buffer_size=streaming_buffer_size,
        )
    else:
        raise NotImplementedError


@torch.no_grad()
def test_ppl(model, tokenizer, datasets=['wikitext2'], ppl_seqlen=2048):
    results = {}
    for dataset in datasets:
        testloader = get_loaders(
            dataset,
            tokenizer,
            seed=0,
            seqlen=ppl_seqlen,
            test_only=True
        )
        if "c4" in dataset:
            testenc = testloader
        else:
            testenc = testloader.input_ids

        seqlen = ppl_seqlen
        nsamples = testenc.numel() // seqlen
        use_cache = model.config.use_cache
        model.config.use_cache = False
        model.eval()
        nlls = []
        if hasattr(model, 'lm_head') and isinstance(model.lm_head, nn.Linear):
            classifier = model.lm_head
        elif hasattr(model.model, 'lm_head'):
            classifier = None
        elif hasattr(model, 'output'):
            classifier = model.output
        else:
            raise NotImplementedError
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen): ((i + 1) * seqlen)].to(get_input_device(model))
            with torch.cuda.amp.autocast():
                outputs = model.model(batch)
            if classifier is not None:
                hidden_states = outputs[0]
                logits = classifier(hidden_states.to(classifier.weight.dtype))
            else:
                logits = outputs[0]
            shift_logits = logits[:, :-1, :]
            shift_labels = testenc[:, (i * seqlen): ((i + 1) * seqlen)][
                :, 1:
            ].to(shift_logits.device)
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset}:{ppl}')
        results[dataset] = ppl.item()
    model.config.use_cache = use_cache
    return results


class BlockTrainDataset(Dataset):
    def __init__(self, size, seqlen, hidden_size, batch_size, dtype, cache_path='./cache/block_training_data',
                 off_load_to_disk=False):
        self.size = size
        self.seqlen = seqlen
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache_path = cache_path
        self.off_load_to_disk = off_load_to_disk
        self.batch_size = batch_size
        assert size % batch_size == 0

        if self.off_load_to_disk:
            if not os.path.exists(self.cache_path):
                os.makedirs(self.cache_path)
                self._initialize_data_on_disk()
        else:
            self.data = torch.zeros((self.size // self.batch_size, self.batch_size, self.seqlen, self.hidden_size),
                                    dtype=self.dtype)

    def _initialize_data_on_disk(self):
        for idx in range(self.size // self.batch_size):
            tensor = torch.zeros((self.batch_size, self.seqlen, self.hidden_size), dtype=self.dtype)
            filepath = self._get_file_path(idx)
            torch.save(tensor, filepath)

    def _get_file_path(self, idx):
        return os.path.join(self.cache_path, f"data_{idx}.pt")

    def __len__(self):
        return self.size // self.batch_size

    def __getitem__(self, idx):
        if idx >= self.__len__():
            raise IndexError("Index out of range")
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            tensor = torch.load(filepath)
        else:
            tensor = self.data[idx]
        return tensor

    def update_data(self, idx, new_data):
        if self.off_load_to_disk:
            filepath = self._get_file_path(idx)
            torch.save(new_data.to(self.dtype), filepath)
        else:
            self.data[idx] = new_data
