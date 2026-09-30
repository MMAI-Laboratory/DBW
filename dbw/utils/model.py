from __future__ import annotations

import inspect
from typing import Any, Dict, Mapping, Optional

import torch
from transformers import AutoTokenizer


_TOKENIZER_LOAD_ATTEMPTS = (
    {"use_fast": False, "legacy": False},
    {"use_fast": False},
    {"use_fast": True, "legacy": False},
    {"use_fast": True},
    {},
)


def _with_auth(kwargs: Dict[str, Any], use_auth_token: bool = False) -> Dict[str, Any]:
    kwargs = dict(kwargs)
    if use_auth_token:
        kwargs.setdefault("use_auth_token", True)
    return kwargs


def load_tokenizer(model_name_or_path: str, trust_remote_code: bool = False, use_auth_token: bool = False):
    """Load a tokenizer with fallbacks across LLaMA/Gemma/Qwen/Mistral.

    Some tokenizers do not accept ``legacy`` and some model families are fast-only
    in newer transformers versions.  We first preserve the original slow-tokenizer
    behavior and then fall back to safer alternatives.
    """

    last_error: Optional[BaseException] = None
    base_kwargs = _with_auth({"trust_remote_code": trust_remote_code}, use_auth_token)
    for attempt in _TOKENIZER_LOAD_ATTEMPTS:
        kwargs = {**base_kwargs, **attempt}
        try:
            return AutoTokenizer.from_pretrained(model_name_or_path, **kwargs)
        except TypeError as exc:
            last_error = exc
        except ValueError as exc:
            last_error = exc
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Failed to load tokenizer for {model_name_or_path}: {last_error}") from last_error


def ensure_tokenizer_has_pad(tokenizer, model=None, resize_embeddings: bool = False) -> None:
    """Ensure a decoder-only tokenizer has a pad token without changing vocab size.

    For quantized checkpoints, adding a brand new [PAD] token after Block-AP can
    invalidate embedding/lm_head shapes.  Prefer reusing eos/unk/bos.  Only add a
    new token when explicitly requested and no existing special token is available.
    """

    config = getattr(model, "config", None)
    for token_name, id_name in (("eos_token", "eos_token_id"), ("bos_token", "bos_token_id"), ("unk_token", "unk_token_id")):
        if getattr(tokenizer, token_name, None) is None and config is not None:
            token_id = getattr(config, id_name, None)
            if isinstance(token_id, (list, tuple)):
                token_id = token_id[0] if token_id else None
            if token_id is not None and token_id >= 0:
                try:
                    setattr(tokenizer, token_name, tokenizer.convert_ids_to_tokens(token_id))
                except Exception:
                    pass

    if tokenizer.pad_token_id is None:
        for token in (tokenizer.eos_token, tokenizer.unk_token, tokenizer.bos_token):
            if token is not None:
                tokenizer.pad_token = token
                break
        else:
            if not resize_embeddings:
                raise ValueError(
                    "Tokenizer has no pad/eos/unk/bos token. Refusing to add a new token because "
                    "that would resize quantized embeddings. Pass resize_embeddings=True before quantization if needed."
                )
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if model is not None:
                model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "right"
    if config is not None:
        config.pad_token_id = tokenizer.pad_token_id
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config.pad_token_id = tokenizer.pad_token_id


def safe_torch_load(path: str, map_location: Optional[str] = None):
    """Load pickled model checkpoints across PyTorch 2.2 and 2.6+ defaults."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def get_causal_lm_base_model(model, return_name: bool = False):
    """Return the inner decoder model and, optionally, its attribute name."""

    for name in ("model", "transformer", "gpt_neox"):
        if hasattr(model, name):
            base = getattr(model, name)
            return (base, name) if return_name else base
    return (model, "") if return_name else model


def get_decoder_layers(model):
    base = get_causal_lm_base_model(model)
    for name in ("layers", "h", "decoder"):
        if hasattr(base, name):
            value = getattr(base, name)
            if name == "decoder" and hasattr(value, "layers"):
                return value.layers
            if isinstance(value, torch.nn.ModuleList) or isinstance(value, (list, tuple)):
                return value
    raise AttributeError("Could not find decoder layers. Expected model.model.layers or a similar decoder-only layout.")


def get_input_device(model) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def move_common_model_parts(model, device) -> None:
    """Move embeddings, final norm, and model-level rotary embeddings when present."""

    base = get_causal_lm_base_model(model)
    for attr in ("embed_tokens", "wte", "word_embeddings", "norm", "final_layernorm", "ln_f", "rotary_emb"):
        if hasattr(base, attr):
            module = getattr(base, attr)
            if hasattr(module, "to"):
                setattr(base, attr, module.to(device))


def no_split_module_classes(model) -> list[str]:
    layers = get_decoder_layers(model)
    if len(layers) == 0:
        return []
    return [layers[0].__class__.__name__]


def _weights_are_tied(module_a, module_b) -> bool:
    """Return True when two modules share the same weight parameter/storage."""

    weight_a = getattr(module_a, "weight", None)
    weight_b = getattr(module_b, "weight", None)
    if weight_a is None or weight_b is None:
        return False
    if weight_a is weight_b:
        return True
    try:
        return weight_a.data_ptr() == weight_b.data_ptr()
    except Exception:
        return False


def balanced_device_map_for_decoder_model(model) -> Dict[str, int] | Dict[str, str]:
    """Balanced map for LLaMA-like causal LMs used by e2e QP training.

    Keep tied input/output embeddings on the same device. Gemma ties
    ``model.embed_tokens.weight`` and ``lm_head.weight`` by default; if those
    modules are split across GPUs, the tied parameter can be materialized on the
    lm_head device while Trainer leaves ``input_ids`` on cuda:0, causing
    ``F.embedding`` to fail with cuda:0/cuda:1 mismatch.
    """

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        return {"": "cpu"}

    base, base_name = get_causal_lm_base_model(model, return_name=True)
    prefix = f"{base_name}." if base_name else ""
    layers = get_decoder_layers(model)
    layers_per_gpu = max(1, (len(layers) + num_gpus - 1) // num_gpus)

    device_map: Dict[str, int] = {}
    input_embedding_module = None
    input_embedding_device = 0
    if hasattr(base, "embed_tokens"):
        input_embedding_module = getattr(base, "embed_tokens")
        device_map[f"{prefix}embed_tokens"] = input_embedding_device
    elif hasattr(base, "wte"):
        input_embedding_module = getattr(base, "wte")
        device_map[f"{prefix}wte"] = input_embedding_device

    for i in range(len(layers)):
        target_gpu = min(i // layers_per_gpu, num_gpus - 1)
        device_map[f"{prefix}layers.{i}"] = target_gpu

    last_gpu = num_gpus - 1
    for attr in ("norm", "final_layernorm", "ln_f", "rotary_emb"):
        if hasattr(base, attr):
            device_map[f"{prefix}{attr}"] = last_gpu

    if hasattr(model, "lm_head"):
        output_embedding_module = getattr(model, "lm_head")
        tied_by_config = bool(getattr(getattr(model, "config", None), "tie_word_embeddings", False))
        tied_by_storage = input_embedding_module is not None and _weights_are_tied(input_embedding_module, output_embedding_module)
        device_map["lm_head"] = input_embedding_device if (tied_by_config or tied_by_storage) else last_gpu
    return device_map
def clone_to_cpu(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().clone().cpu()
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(v) for v in value)
    if isinstance(value, list):
        return [clone_to_cpu(v) for v in value]
    if isinstance(value, dict):
        return {k: clone_to_cpu(v) for k, v in value.items()}
    return value


def nested_to(value: Any, device=None) -> Any:
    if torch.is_tensor(value):
        return value.to(device) if device is not None else value
    if isinstance(value, tuple):
        return tuple(nested_to(v, device=device) for v in value)
    if isinstance(value, list):
        return [nested_to(v, device=device) for v in value]
    if isinstance(value, dict):
        return {k: nested_to(v, device=device) for k, v in value.items()}
    return value


_BATCHED_LAYER_KWARGS = {"attention_mask", "position_ids", "token_type_ids", "position_embeddings"}


def _match_tensor_batch(key: str, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    if key == "cache_position" or tensor.dim() == 0 or key not in _BATCHED_LAYER_KWARGS:
        return tensor
    if key == "position_embeddings" and tensor.dim() < 3:
        return tensor
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1 and batch_size > 1:
        repeat_shape = [batch_size] + [1] * (tensor.dim() - 1)
        return tensor.repeat(*repeat_shape)
    if tensor.shape[0] > batch_size:
        return tensor[:batch_size]
    return tensor


def _match_batch(key: str, value: Any, batch_size: int) -> Any:
    if torch.is_tensor(value):
        return _match_tensor_batch(key, value, batch_size)
    if isinstance(value, tuple):
        return tuple(_match_batch(key, v, batch_size) for v in value)
    if isinstance(value, list):
        return [_match_batch(key, v, batch_size) for v in value]
    if isinstance(value, dict):
        return {k: _match_batch(k, v, batch_size) for k, v in value.items()}
    return value


def _filter_kwargs_for_forward(module, kwargs: Mapping[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(module.forward)
    except (TypeError, ValueError):
        return dict(kwargs)
    parameters = signature.parameters
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in parameters}


def prepare_layer_kwargs(module, layer_kwargs: Optional[Mapping[str, Any]], hidden_states: torch.Tensor) -> Dict[str, Any]:
    if not layer_kwargs:
        return {}
    batch_size = hidden_states.shape[0]
    device = hidden_states.device
    kwargs = {}
    for key, value in layer_kwargs.items():
        if value is None:
            continue
        value = _match_batch(key, value, batch_size)
        kwargs[key] = nested_to(value, device=device)
    return _filter_kwargs_for_forward(module, kwargs)


def call_decoder_layer(module, hidden_states: torch.Tensor, layer_kwargs: Optional[Mapping[str, Any]] = None):
    kwargs = prepare_layer_kwargs(module, layer_kwargs, hidden_states)
    return module(hidden_states, **kwargs)


def load_fake_quantized_model(model_path, wbits, group_size):
    import gc
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoConfig
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
    from dbw.fake_dbw import fakeDBW
    from dbw.utils.quantize import set_op_by_name

    print(f"Loading quantized model from {model_path}")

    tokenizer = load_tokenizer(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)
    ensure_tokenizer_has_pad(tokenizer, model, resize_embeddings=False)
    layers = get_decoder_layers(model)
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                quantlinear = fakeDBW(None, None, None, module.weight.data, wbits // 2, group_size, True, 'LBQ_inference', bias=(module.bias is not None))
                quantlinear.to(next(layer.parameters()).device)
                quantlinear.set_quant_state(False)
                set_op_by_name(layer, name, quantlinear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model, checkpoint=model_path, device_map=device_map, offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")
    return model, tokenizer

def load_real_quantized_model(model_path, wbits, group_size, ver='LBQ'):
    import gc, json, os
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoConfig
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
    from dbw.real_dbw import realDBW
    from dbw.utils.quantize import get_named_linears, set_op_by_name
    print(f"Loading quantized model from {model_path}")

    tokenizer = load_tokenizer(model_path, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config=config, torch_dtype=torch.float16, trust_remote_code=True)
    ensure_tokenizer_has_pad(tokenizer, model, resize_embeddings=False)
    layers = get_decoder_layers(model)
    for i in tqdm(range(len(layers))):
        load_file_name = os.path.join(model_path, 'wbit', f'{i}.json')
        with open(load_file_name, "r", encoding="utf-8") as f:
            wbit_dict = json.load(f)
        layer = layers[i]
        for name, module in layer.named_modules():
            if isinstance(module, torch.nn.Linear):
                wbit_count = wbit_dict[name]
                wbit_count = {int(k): int(v) for k, v in wbit_count.items()}
                rank = sum(list(wbit_count.values()))
                N, M = module.weight.data.shape
                quantlinear = realDBW(wbit_count, group_size, M, N, rank, module.bias is not None, trainable=False, ver=ver)
                quantlinear.name = f"{i}.{name}"
                quantlinear.to(next(layer.parameters()).device)
                set_op_by_name(layer, name, quantlinear)
    torch.cuda.empty_cache()
    gc.collect()
    model.tie_weights()
    device_map = infer_auto_device_map(model)
    print("Loading pre-computed quantized weights...")
    load_checkpoint_in_model(model, checkpoint=model_path, device_map=device_map, offload_state_dict=True)
    print("Loading pre-computed quantized weights Successfully")
    return model, tokenizer


@torch.no_grad()
def fake_to_real(fake_model, group_size=128, ver='LBQ'):
    import gc, json, os
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoConfig
    from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
    from dbw.real_dbw import realDBW
    from dbw.utils.quantize import get_named_linears, set_op_by_name
    from accelerate.hooks import remove_hook_from_module
    for m in fake_model.modules():
        remove_hook_from_module(m, recurse=True)
        for p in m.parameters(recurse=False):
            if p.is_cuda:
                p.data = p.data.cpu()
                if p.grad is not None:
                    p.grad = p.grad.cpu()
        for name, b in m._buffers.items():
            if b is not None and b.is_cuda:
                m._buffers[name] = b.cpu()
    gc.collect()
    torch.cuda.empty_cache()

    wbit_dict = {}
    from dbw.fake_dbw import fakeDBW as FakeDBW
    layers = get_decoder_layers(fake_model)
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        layer.cuda()
        named_linears = get_named_linears(layer, FakeDBW)
        for name, module in named_linears.items():
            org_w, org_s, org_z, qmax, sigma, sigma_total = module.get_int()
            wbit_max = int(qmax.max().item())
            wbit_min = int(qmax.min().item())
            wbit_count = {wbit: int((qmax == wbit).float().sum().item()) for wbit in
                          range(wbit_max, wbit_min - 1, -1)}
            wbit_dict[name] = wbit_count
            M = org_w[1].shape[1]
            N = org_w[0].shape[1]
            rank = sum(list(wbit_count.values()))
            module_bias = getattr(module, 'bias', None)
            if isinstance(module_bias, torch.Tensor):
                module_bias = module_bias.detach().clone().cpu()
            q_linear = realDBW(wbit_count, group_size, M, N, rank, module_bias, trainable=False, ver=ver)
            w = [item.detach().clone().cpu() for item in org_w]
            s = [item.detach().clone().cpu() for item in org_s]
            z = [item.detach().clone().cpu() for item in org_z]
            sigma = sigma.data.detach().clone().cpu()
            sigma_total = sigma_total.data.detach().clone().cpu()
            qmax = qmax.detach().clone().cpu()
            q_linear.pack(w, s, z, qmax, sigma, sigma_total)
            set_op_by_name(layer, name, q_linear)
            logger.info(f"pack quantized {name} finished")
            del module
    return fake_model
