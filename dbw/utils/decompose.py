import argparse, gc, os, sys
from collections import OrderedDict

import torch
from torch import nn
from transformers import AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dbw.utils.model import get_decoder_layers

parser = argparse.ArgumentParser(description='save-weight-svd', add_help=True,
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model-name', type=str, default='huggyllama/llama-7b', help='model-name')
parser.add_argument('--save-dir', type=str, default='svd')
parser.add_argument('--trust-remote-code', action='store_true', help='trust custom HF model code when required')
parser.add_argument('--use-auth-token', action='store_true', help='use the logged-in Hugging Face token')
parser.add_argument('--torch-dtype', choices=['auto', 'float16', 'bfloat16', 'float32'], default='float16')
parser.add_argument('--svd-device', choices=['auto', 'cpu', 'cuda'], default='cpu',
                    help="device for the SVD; 'cpu' reproduces the original Llama caches")
parser.add_argument('--force', action='store_true', help='overwrite existing per-layer SVD files')


def cache_dir_for(model_name, save_dir='svd'):
    return os.path.join(save_dir, model_name.replace('/', '_') + "_fp16")


def build_svd_cache(model_name, save_dir='svd', trust_remote_code=False, use_auth_token=False,
                    torch_dtype='float16', svd_device='cpu', force=False, log=print):
    dtype_map = {
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
        'float32': torch.float32,
    }
    resolved_dtype = 'auto' if torch_dtype == 'auto' else dtype_map[torch_dtype]
    if svd_device == 'auto':
        svd_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if svd_device == 'cuda' and not torch.cuda.is_available():
        log('[WARN] CUDA was requested for SVD but is not available; falling back to CPU.')
        svd_device = 'cpu'

    hf_kwargs = {'trust_remote_code': trust_remote_code}
    if use_auth_token:
        hf_kwargs['use_auth_token'] = True

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='cpu',
        torch_dtype=resolved_dtype,
        **hf_kwargs,
    )
    save_dir = cache_dir_for(model_name, save_dir)
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, layer in enumerate(get_decoder_layers(model)):
            output_path = os.path.join(save_dir, f'{i}.pt')
            if os.path.exists(output_path) and not force:
                log(f'skip existing {output_path}')
                continue

            feature_dict = OrderedDict()
            for name, module in layer.named_modules():
                if isinstance(module, nn.Linear):
                    weight = module.weight.detach().to(torch.float32).to(svd_device)
                    U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                    U, S, Vh = map(lambda x: x.to(torch.float16).cpu(), [U, S, Vh])
                    feature_dict[name] = [U, S, Vh]
                    del weight, U, S, Vh
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            log(f'save weight into {output_path}')
            torch.save(feature_dict, output_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return save_dir


def missing_svd_layers(model_name, num_layers, save_dir='svd'):
    d = cache_dir_for(model_name, save_dir)
    return [i for i in range(num_layers) if not os.path.exists(os.path.join(d, f'{i}.pt'))]


def main(args):
    build_svd_cache(args.model_name, args.save_dir, args.trust_remote_code, args.use_auth_token,
                    args.torch_dtype, args.svd_device, args.force)


if __name__ == '__main__':
    main(parser.parse_args())
