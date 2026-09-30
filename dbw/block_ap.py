import json, time, shutil, os, copy, math, pdb, gc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import bitsandbytes as bnb
from dbw.fake_dbw import fakeDBW
from dbw.real_dbw import realDBW
import dbw.utils as utils
from dbw.utils.data_block import BlockTrainDataset
from dbw.utils.quantize import (
    quant_parameters, weight_parameters, trainable_parameters,
    set_quant_state, quant_inplace, set_quant_parameters,
    set_weight_parameters, trainable_parameters_num, get_named_linears, set_op_by_name,
    set_bit_parameters, bit_parameters,
    set_global_bit_parameters, global_bit_parameters,
)
from dbw.utils.model import (
    call_decoder_layer,
    clone_to_cpu,
    get_causal_lm_base_model,
    get_decoder_layers,
    move_common_model_parts,
)

DEBUG = bool(int(os.environ.get('DEBUG', 0)) == 1)
SPLIT = int(os.environ.get('SPLIT', 1))
VER = 'LBQ'


class SVDLinear(nn.Module):
    def __init__(self, linear: nn.Linear):
        super().__init__()
        W = linear.weight.data.float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        self.U = nn.Parameter(U)
        self.s = nn.Parameter(S / S.abs().sum())
        self.V = nn.Parameter(Vh.T)
        self.register_buffer('s_total_const', S.sum())
        self.register_buffer('weight', W)
        if linear.bias is not None:
            self.register_buffer('bias', linear.bias.data.float())
        else:
            self.bias = None

    def forward(self, x):
        x = x @ F.normalize(self.V, dim=0)
        s = torch.relu(self.s)
        p = s * self.s_total_const
        x = x * p
        x = x @ F.normalize(self.U.T, dim=1)
        if self.bias is not None:
            x = x + self.bias.to(device=x.device, dtype=x.dtype)
        return x


def replace_linear_with_svd(module: nn.Module):
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            print(f'SVD: {name}')
            setattr(module, name, SVDLinear(child))
        else:
            replace_linear_with_svd(child)


def fake_to_real(qlayer, block_index, args, logger):
    wbit_dict = {}
    named_linears = get_named_linears(qlayer, fakeDBW)
    for name, module in named_linears.items():
        with torch.cuda.amp.autocast():
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
            q_linear = realDBW(wbit_count, args.group_size, M, N, rank, module_bias, trainable=False, ver=VER)
            w = [item.detach().clone().cpu() for item in org_w]
            s = [item.detach().clone().cpu() for item in org_s]
            z = [item.detach().clone().cpu() for item in org_z]
            sigma = sigma.data.detach().clone().cpu()
            sigma_total = sigma_total.data.detach().clone().cpu()
            qmax = qmax.detach().clone().cpu()
            q_linear.pack(w, s, z, qmax, sigma, sigma_total)
            set_op_by_name(qlayer, name, q_linear)
            logger.info(f"pack quantized {name} finished")
            del module
    save_dir = os.path.join(args.save_quant_dir, 'wbit')
    os.makedirs(save_dir, exist_ok=True)
    save_file_name = os.path.join(save_dir, f'{block_index}.json')
    with open(save_file_name, 'wt', encoding="utf-8") as f:
        json.dump(wbit_dict, f, ensure_ascii=False, indent=2)


def to_2gpus(qlayer):
    qlayer.input_layernorm.to('cuda:0')
    qlayer.self_attn.q_proj.to('cuda:0')
    qlayer.self_attn.rotary_emb.to('cuda:0')
    qlayer.self_attn.k_proj.to('cuda:0')
    qlayer.self_attn.v_proj.to('cuda:0')
    qlayer.self_attn.o_proj.to('cuda:0')

    qlayer.post_attention_layernorm.to('cuda:0')
    qlayer.mlp.gate_proj.to('cuda:1')
    qlayer.mlp.up_proj.to('cuda:1')
    qlayer.mlp.down_proj.to('cuda:1')

    return qlayer


def to_3gpus(qlayer):

    qlayer.input_layernorm.to('cuda:0')
    qlayer.self_attn.q_proj.to('cuda:0')
    qlayer.self_attn.rotary_emb.to('cuda:0')
    qlayer.self_attn.k_proj.to('cuda:0')
    qlayer.self_attn.v_proj.to('cuda:0')
    qlayer.self_attn.o_proj.to('cuda:1')

    qlayer.post_attention_layernorm.to('cuda:0')
    qlayer.mlp.gate_proj.to('cuda:1')
    qlayer.mlp.up_proj.to('cuda:2')
    qlayer.mlp.down_proj.to('cuda:2')

    return qlayer


def to_4gpus(qlayer):

    qlayer.input_layernorm.to('cuda:0')
    qlayer.self_attn.q_proj.to('cuda:0')
    qlayer.self_attn.rotary_emb.to('cuda:0')
    qlayer.self_attn.k_proj.to('cuda:0')
    qlayer.self_attn.v_proj.to('cuda:1')
    qlayer.self_attn.o_proj.to('cuda:1')

    qlayer.post_attention_layernorm.to('cuda:0')
    qlayer.mlp.gate_proj.to('cuda:2')
    qlayer.mlp.up_proj.to('cuda:2')
    qlayer.mlp.down_proj.to('cuda:3')

    return qlayer


def to_5gpus(qlayer):
    qlayer.input_layernorm.to('cuda:0')
    qlayer.self_attn.q_proj.to('cuda:0')
    qlayer.self_attn.rotary_emb.to('cuda:0')
    qlayer.self_attn.k_proj.to('cuda:1')
    qlayer.self_attn.v_proj.to('cuda:1')
    qlayer.self_attn.o_proj.to('cuda:2')

    qlayer.post_attention_layernorm.to('cuda:0')
    qlayer.mlp.gate_proj.to('cuda:2')
    qlayer.mlp.up_proj.to('cuda:3')
    qlayer.mlp.down_proj.to('cuda:4')

    return qlayer


def to_8gpus(qlayer):
    qlayer.input_layernorm.to('cuda:0')
    qlayer.self_attn.q_proj.to('cuda:1')
    qlayer.self_attn.rotary_emb.to('cuda:0')
    qlayer.self_attn.k_proj.to('cuda:2')
    qlayer.self_attn.v_proj.to('cuda:3')
    qlayer.self_attn.o_proj.to('cuda:4')

    qlayer.post_attention_layernorm.to('cuda:0')
    qlayer.mlp.gate_proj.to('cuda:5')
    qlayer.mlp.up_proj.to('cuda:6')
    qlayer.mlp.down_proj.to('cuda:7')

    return qlayer


def forward_decoder_layer(layer, hidden_states, layer_kwargs):
    return call_decoder_layer(layer, hidden_states, layer_kwargs)[0]


def update_dataset(layer, dataset, dev, layer_kwargs):
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            for index, inps in enumerate(dataset):
                inps = inps.to(dev)
                if len(inps.shape) == 2:
                    inps = inps.unsqueeze(0)
                new_data = forward_decoder_layer(layer, inps, layer_kwargs).to('cpu')
                if not DEBUG:
                    dataset.update_data(index, new_data)


def block_ap(
        model,
        args,
        trainloader,
        valloader,
        logger=None,
):
    logger.info("Starting ...")
    if args.off_load_to_disk:
        logger.info(
            "offload the training dataset to disk, saving CPU memory, but may slowdown the training due to additional I/O...")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    base_model = get_causal_lm_base_model(model)
    layers = get_decoder_layers(model)
    move_common_model_parts(model, dev)
    layers[0] = layers[0].to(dev)
    dtype = torch.float16

    if args.wandb:
        import wandb
        wandb.init(project='svq')

    if DEBUG:
        flag = '1763880717.1589737'
    else:
        flag = time.time()

    if args.off_load_to_disk:
        fp_train_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_train'
        fp_val_cache_path = f'{args.cache_dir}/{flag}/block_training_fp_val'
        quant_train_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_train'
        quant_val_cache_path = f'{args.cache_dir}/{flag}/block_training_quant_val'
        if not DEBUG:
            for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
                if os.path.exists(path):
                    shutil.rmtree(path)
    else:
        fp_train_cache_path = None
        fp_val_cache_path = None
        quant_train_cache_path = None
        quant_val_cache_path = None
    fp_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                      model.config.hidden_size, args.batch_size, dtype, cache_path=fp_train_cache_path,
                                      off_load_to_disk=args.off_load_to_disk)
    fp_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                    model.config.hidden_size, args.batch_size, dtype, cache_path=fp_val_cache_path,
                                    off_load_to_disk=args.off_load_to_disk)

    class Catcher(nn.Module):
        def __init__(self, module, dataset):
            super().__init__()
            self.module = module
            self.dataset = dataset
            self.index = 0
            self.layer_kwargs = None

        def forward(self, inp, **kwargs):
            if not DEBUG:
                self.dataset.update_data(self.index, inp.squeeze(0).to('cpu'))
            self.index += 1
            if self.layer_kwargs is None:
                self.layer_kwargs = clone_to_cpu({k: v for k, v in kwargs.items() if v is not None})
            raise ValueError

    if not DEBUG:
        layers[0] = Catcher(layers[0], fp_train_inps)
        iters = len(trainloader) // args.batch_size
        with torch.no_grad():
            for i in range(iters):
                data = torch.cat([trainloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)],
                                 dim=0)
                try:
                    model(data.to(dev))
                except ValueError:
                    pass
        layers[0] = layers[0].module

    layers[0] = Catcher(layers[0], fp_val_inps)
    iters = len(valloader) // args.batch_size
    with torch.no_grad():
        for i in range(iters):
            data = torch.cat([valloader[j][0] for j in range(i * args.batch_size, (i + 1) * args.batch_size)], dim=0)
            try:
                model(data.to(dev))
            except ValueError:
                if DEBUG:
                    break
                pass
    layer_kwargs = layers[0].layer_kwargs or {}
    layers[0] = layers[0].module
    if "attention_mask" not in layer_kwargs:
        logger.info(
            "No attention mask caught from the first layer. "
            "Seems that model's attention works without an explicit mask."
        )

    layers[0] = layers[0].cpu()
    move_common_model_parts(model, 'cpu')
    torch.cuda.empty_cache()

    if args.off_load_to_disk:
        if not DEBUG:
            shutil.copytree(fp_train_cache_path, quant_train_cache_path)
            shutil.copytree(fp_val_cache_path, quant_val_cache_path)
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype,
                                             cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype,
                                           cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
    else:
        quant_train_inps = BlockTrainDataset(args.train_size, args.training_seqlen,
                                             model.config.hidden_size, args.batch_size, dtype,
                                             cache_path=quant_train_cache_path, off_load_to_disk=args.off_load_to_disk)
        quant_val_inps = BlockTrainDataset(args.val_size, args.training_seqlen,
                                           model.config.hidden_size, args.batch_size, dtype,
                                           cache_path=quant_val_cache_path, off_load_to_disk=args.off_load_to_disk)
        for index, data in enumerate(fp_train_inps):
            quant_train_inps.update_data(index, data)
        for index, data in enumerate(fp_val_inps):
            quant_val_inps.update_data(index, data)

    loss_func = torch.nn.MSELoss()
    for block_index in range(0, len(layers)):
        logger.info(f"=== Start quantize blocks {block_index}===")

        layer = layers[block_index].to(dev)
        qlayer = copy.deepcopy(layer)

        pt = torch.load(os.path.join('svd', args.model.replace('/', '_') + '_fp16', f'{block_index}.pt'),
                        map_location='cpu')

        if args.global_bit and not args.progressive:
            total_memory = 0
            memory_list = []
            svd_size_list = []
            svq_list = []
            param_list = []

        for name, module in qlayer.named_modules():
            if isinstance(module, torch.nn.Linear):
                module_bias = module.bias.detach().clone() if getattr(module, 'bias', None) is not None else None
                u, s, v = pt[name]
                org_m, org_n = u.size(0), v.size(1)
                org_size = org_m * org_n
                svd_size = u.size(0) * u.size(1) + v.size(0) * v.size(1)
                new_bit = org_size / svd_size * args.wbits
                print(f'new bit: {new_bit}')
                if SPLIT > 1:
                    quantlinear = fakeDBW(u, s, v, module.weight.data, new_bit, args.group_size, True, VER,
                                      'cuda:0', bias=module_bias)
                else:
                    quantlinear = fakeDBW(u, s, v, module.weight.data, new_bit, args.group_size, True, VER,
                                      bias=module_bias)
                if args.global_bit and not args.progressive:
                    total_memory += org_size * args.wbits / 1024.0 / 1024.0
                    memory_list.append(org_size * args.wbits / 1024.0 / 1024.0)
                    svd_size_list.append(svd_size / 1024.0 / 1024.0)
                    svq_list.append(quantlinear)
                set_op_by_name(qlayer, name, quantlinear)
                del module

        if args.global_bit and not args.progressive:
            for param_idx, memory in enumerate(memory_list):
                svq_list[param_idx].n_bits_param = nn.Parameter(torch.tensor(memory / total_memory).to(torch.float16))
                param_list.append(svq_list[param_idx].n_bits_param)

        del pt

        qlayer.cpu()
        layer.to(dev)
        if not DEBUG:
            if args.epochs > 0:
                update_dataset(layer, fp_train_inps, dev, layer_kwargs)
                update_dataset(layer, fp_val_inps, dev, layer_kwargs)
        set_quant_state(qlayer, weight_quant=True)
        if SPLIT == 1:
            qlayer.to(dev)
        elif SPLIT == 2:
            qlayer = to_2gpus(qlayer)
        elif SPLIT == 3:
            qlayer = to_3gpus(qlayer)
        elif SPLIT == 4:
            qlayer = to_4gpus(qlayer)
        elif SPLIT == 5:
            qlayer = to_5gpus(qlayer)
        elif SPLIT == 8:
            qlayer = to_8gpus(qlayer)
        layer.cpu()

        if args.epochs > 0:
            if args.progressive:
                start_time = time.time()

                def sv_l1_loss(modules):
                    loss = 0.0
                    count = 0
                    for m in modules:
                        if isinstance(m, fakeDBW) and m.use_quant == False:
                            s = torch.relu(m.s_scale)
                            loss = loss + s.mean().to(dev)
                            count += 1
                    return loss / max(count, 1)

                def rel_sv_l1_loss(modules):
                    loss = 0.0
                    count = 0
                    for m in modules:
                        if isinstance(m, fakeDBW) and m.use_quant == False:
                            s = torch.relu(m.s_scale)
                            cur_sum = s.sum()
                            p = s / cur_sum
                            p = p.clamp(min=1e-6)
                            loss = loss - (torch.log(p) * p).sum().to(dev)
                            count += 1
                    return loss / max(count, 1)

                with torch.no_grad():
                    qlayer.float()
                qlayer.train()

                svd_modules = []
                singular_values = []
                for m in qlayer.modules():
                    if isinstance(m, (fakeDBW,)):
                        svd_modules.append(m)
                        p = torch.relu(m.s_scale) * m.s_total_const
                        singular_values.append(p.detach().clone().cpu())
                        m.use_quant = False

                max_iter = 2048
                optimizer = torch.optim.AdamW(qlayer.parameters(), lr=args.pre_lr)
                loss_scaler = utils.NativeScalerWithGradNormCount()
                scheduler = CosineAnnealingLR(optimizer, T_max=max_iter, eta_min=1e-6)
                loss_list = []
                norm_list = []
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    with torch.cuda.amp.autocast(enabled=True):
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        _quant_out = forward_decoder_layer(qlayer, input, layer_kwargs)
                        reconstruction_loss = loss_func(label, _quant_out)
                        sv_loss = rel_sv_l1_loss(svd_modules)
                        sv_loss2 = sv_l1_loss(svd_modules)
                    lam = args.lambda_sv
                    lam2 = args.lambda_sv
                    loss = reconstruction_loss + lam * sv_loss + lam2 * sv_loss2
                    optimizer.zero_grad()
                    norm = loss_scaler(loss, optimizer, parameters=trainable_parameters(qlayer)).cpu()
                    logger.info(
                        f"{index:04d} | "
                        f"rec: {reconstruction_loss:.2e} | "
                        f"sv: {sv_loss:.2e} | "
                        f"avg_sv: {sv_loss2:.2e} | "
                        f"gn: {norm:.2f} | "
                        f"lr: {scheduler.get_lr()[0]:.2e}"
                    )
                    scheduler.step()
                    loss_list.append(loss.detach().clone().cpu())
                    norm_list.append(norm.detach().clone().cpu())

                singular_values = []
                for m in qlayer.modules():
                    if isinstance(m, fakeDBW):
                        p = torch.relu(m.s_scale) * m.s_total_const
                        singular_values.append(p.detach().clone().cpu())

                val_loss_list = []
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = forward_decoder_layer(qlayer, input, layer_kwargs)
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list), 64)
                loss_mean = torch.stack(loss_list)[-(train_mean_num - 1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"recon_loss:{loss_mean} val_loss:{val_loss_mean} lr:{scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024 ** 2} time {time.time() - start_time} ")

                del optimizer
                del svd_modules
                del singular_values

                if args.global_bit:
                    total_memory = 0
                    memory_list = []
                    svd_size_list = []
                    svq_list = []
                    param_list = []
                    for name, module in qlayer.named_modules():
                        if isinstance(module, fakeDBW):
                            module.use_quant = True
                            idx = torch.argsort(module.s_scale, descending=True)
                            module.s_scale.data = module.s_scale.data[idx]
                            module.u_weight.data = module.u_weight.data[idx]
                            module.v_weight.data = module.v_weight.data[idx]
                            org_m, org_n = module.m, module.n
                            org_size = (org_m) * (org_n)
                            svd_size = (module.m + module.n) * module.r
                            new_bit = org_size / svd_size * args.wbits
                            total_memory += org_size * args.wbits / 1024.0 / 1024.0
                            memory_list.append(org_size * args.wbits / 1024.0 / 1024.0)
                            svd_size_list.append(svd_size / 1024.0 / 1024.0)
                            svq_list.append(module)

                    for param_idx, memory in enumerate(memory_list):
                        device = svq_list[param_idx].qmax.data.device
                        svq_list[param_idx].n_bits_param = nn.Parameter(
                            torch.tensor(memory / total_memory).to(torch.float16).to(device))
                        param_list.append(svq_list[param_idx].n_bits_param)

            with torch.no_grad():
                qlayer.float()

            if args.save_statistics:
                statics_list = {}
                for name, module in qlayer.named_modules():
                    if isinstance(module, fakeDBW):
                        module.save_statics = True
                        module.use_quant = False
                        statics_list[name] = 0

                count = 100
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    if index == count: break
                    with torch.cuda.amp.autocast(enabled=True), torch.no_grad():
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        _quant_out = forward_decoder_layer(qlayer, input, layer_kwargs)
                        for name, module in qlayer.named_modules():
                            if isinstance(module, fakeDBW):
                                statics_list[name] += module.statics / count

                for name, module in qlayer.named_modules():
                    if isinstance(module, fakeDBW):
                        module.save_statics = False
                        module.use_quant = True
                        qmax = statics_list[name]
                        qmax = qmax / qmax.sum()
                        dtype = module.qmax.dtype
                        device = module.qmax.device
                        shape = module.qmax.shape
                        module.qmax.data = qmax.to(dtype).to(device).reshape(shape)
                        qmax_data = qmax.detach().clone().cpu()
                        del module.statics
                del statics_list

            param = []
            param_group_index = 0
            total_training_iteration = args.epochs * args.train_size / args.batch_size
            if args.quant_lr > 0:
                set_quant_parameters(qlayer, True)
                param.append({"params": quant_parameters(qlayer), "lr": args.quant_lr})
                empty_optimizer_1 = torch.optim.AdamW([torch.tensor(0)], lr=args.quant_lr)
                quant_scheduler = CosineAnnealingLR(empty_optimizer_1, T_max=total_training_iteration,
                                                    eta_min=args.quant_lr / args.min_lr_factor)
                quant_index = param_group_index
                param_group_index += 1
            else:
                set_quant_parameters(qlayer, False)

            if args.u_lr > 0:
                set_weight_parameters(qlayer, False)

                set_weight_parameters(qlayer, True, 'norm')
                param.append({"params": weight_parameters(qlayer, 'norm'), "lr": args.norm_lr})
                empty_optimizer_2_norm = torch.optim.AdamW([torch.tensor(0)], lr=args.u_lr)
                norm_scheduler = CosineAnnealingLR(empty_optimizer_2_norm, T_max=total_training_iteration,
                                                   eta_min=args.norm_lr / args.min_lr_factor)
                norm_index = param_group_index
                param_group_index += 1

                set_weight_parameters(qlayer, True, 'u')
                param.append({"params": weight_parameters(qlayer, 'u'), "lr": args.u_lr})
                empty_optimizer_2_u = torch.optim.AdamW([torch.tensor(0)], lr=args.u_lr)
                u_scheduler = CosineAnnealingLR(empty_optimizer_2_u, T_max=total_training_iteration,
                                                eta_min=args.u_lr / args.min_lr_factor)
                u_index = param_group_index
                param_group_index += 1

                set_weight_parameters(qlayer, True, 'v')
                param.append({"params": weight_parameters(qlayer, 'v'), "lr": args.v_lr})
                empty_optimizer_2_v = torch.optim.AdamW([torch.tensor(0)], lr=args.v_lr)
                v_scheduler = CosineAnnealingLR(empty_optimizer_2_v, T_max=total_training_iteration,
                                                eta_min=args.v_lr / args.min_lr_factor)
                v_index = param_group_index
                param_group_index += 1
            elif args.weight_lr > 0:
                set_weight_parameters(qlayer, True)
                param.append({"params": weight_parameters(qlayer), "lr": args.weight_lr})
                empty_optimizer_2 = torch.optim.AdamW([torch.tensor(0)], lr=args.weight_lr)
                weight_scheduler = CosineAnnealingLR(empty_optimizer_2, T_max=total_training_iteration,
                                                     eta_min=args.weight_lr / args.min_lr_factor)
                weight_index = param_group_index
                param_group_index += 1
            else:
                set_weight_parameters(qlayer, False)

            if args.gbit_lr > 0 and args.global_bit:
                bit_lr = args.gbit_lr
                set_global_bit_parameters(qlayer, True)
                param.append({"params": global_bit_parameters(qlayer), "lr": bit_lr})
                empty_optimizer_3 = torch.optim.AdamW([torch.tensor(0)], lr=bit_lr)
                global_bit_scheduler = CosineAnnealingLR(empty_optimizer_3, T_max=total_training_iteration,
                                                         eta_min=1e-6)
                gbit_index = param_group_index
                param_group_index += 1
            else:
                set_bit_parameters(qlayer, False)

            if args.bit_lr > 0:
                bit_lr = args.bit_lr
                set_bit_parameters(qlayer, True)
                param.append({"params": bit_parameters(qlayer), "lr": bit_lr})
                empty_optimizer_3 = torch.optim.AdamW([torch.tensor(0)], lr=bit_lr)
                bit_scheduler = CosineAnnealingLR(empty_optimizer_3, T_max=total_training_iteration,
                                                  eta_min=1e-6)
                bit_index = param_group_index
                param_group_index += 1
            else:
                set_bit_parameters(qlayer, False)

            optimizer = torch.optim.AdamW(param, weight_decay=args.wd)
            loss_scaler = utils.NativeScalerWithGradNormCount()
            trainable_number = trainable_parameters_num(qlayer)
            print(f"trainable parameter number: {trainable_number / 1e6}M")

            best_val_loss = 1e6
            early_stop_flag = 0
            global_idx = 0
            w2_rel_loss = 0
            loss2 = 0
            torch.cuda.empty_cache()

            for epoch in range(args.epochs):
                loss_list = []
                norm_list = []
                start_time = time.time()

                qlayer.train()
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_train_inps, fp_train_inps)):
                    global_idx += 1
                    with torch.cuda.amp.autocast(enabled=True):
                        input = quant_inps.to(dev)
                        label = fp_inps.to(dev)
                        if args.global_bit:
                            if SPLIT > 1:
                                param_sum = sum([p.cpu() for p in param_list])
                            else:
                                param_sum = sum(param_list)
                            for param_idx, param in enumerate(param_list):
                                memory = param / param_sum.to(param.device) * total_memory
                                avg_bit = memory / svd_size_list[param_idx]
                                svq_list[param_idx].n_bits = avg_bit

                        _quant_out = forward_decoder_layer(qlayer, input, layer_kwargs)
                        reconstruction_loss = loss_func(label, _quant_out)
                        loss = reconstruction_loss

                    if not math.isfinite(loss.item()):
                        logger.info("Loss is NAN, stopping training")
                        pdb.set_trace()
                    loss_list.append(reconstruction_loss.detach().cpu())
                    optimizer.zero_grad()

                    norm = loss_scaler(loss, optimizer, parameters=trainable_parameters(qlayer)).cpu()
                    norm_list.append(norm.data)

                    param_sum = 0
                    trunc_sum = 0
                    count = 0
                    max_bit = []

                    with torch.no_grad():
                        for name, module in qlayer.named_modules():
                            if isinstance(module, fakeDBW):
                                param, trunc, bit = module.get_effective_avg_bit()
                                trunc_sum += trunc.detach().clone().cpu()
                                param_sum += param
                                count += 1
                                max_bit.append(bit.detach().clone().cpu())
                        trunc = trunc_sum / param_sum * 100
                        avg_bit = 100 / (100 - trunc) * args.wbits
                        min_bit = min(max_bit)
                        max_bit = max(max_bit)

                    if DEBUG:
                        logger.info(f"{index}: l: {reconstruction_loss:.6f} "
                                    f"t: {trunc:.2f} b: {avg_bit:.2f} n: {norm:.2f} "
                                    f"max: {max_bit:.2f} min: {min_bit:.2f}")

                    if args.wandb:
                        wandb.log({
                            "index": index,
                            f'loss/{block_index}': reconstruction_loss,
                            f'avg_bit/{block_index}': avg_bit,
                            f'trunc/{block_index}': trunc,
                            'norm': norm,
                        })

                    if args.quant_lr > 0:
                        quant_scheduler.step()
                        optimizer.param_groups[quant_index]['lr'] = quant_scheduler.get_lr()[0]

                    if args.u_lr > 0:
                        norm_scheduler.step()
                        optimizer.param_groups[norm_index]['lr'] = norm_scheduler.get_lr()[0]
                        u_scheduler.step()
                        optimizer.param_groups[u_index]['lr'] = u_scheduler.get_lr()[0]
                        v_scheduler.step()
                        optimizer.param_groups[v_index]['lr'] = v_scheduler.get_lr()[0]
                    elif args.weight_lr > 0:
                        weight_scheduler.step()
                        optimizer.param_groups[weight_index]['lr'] = weight_scheduler.get_lr()[0]

                    if args.bit_lr > 0:
                        bit_scheduler.step()
                        optimizer.param_groups[bit_index]['lr'] = bit_scheduler.get_lr()[0]

                    if args.gbit_lr > 0 and args.global_bit:
                        global_bit_scheduler.step()
                        optimizer.param_groups[gbit_index]['lr'] = global_bit_scheduler.get_lr()[0]

                qlayer.eval()
                val_loss_list = []
                for index, (quant_inps, fp_inps) in enumerate(zip(quant_val_inps, fp_val_inps)):
                    with torch.no_grad():
                        with torch.cuda.amp.autocast():
                            input = quant_inps.to(dev)
                            label = fp_inps.to(dev)
                            quant_out = forward_decoder_layer(qlayer, input, layer_kwargs)
                            reconstruction_loss = loss_func(label, quant_out)
                    val_loss_list.append(reconstruction_loss.cpu())

                train_mean_num = min(len(loss_list), 64)
                loss_mean = torch.stack(loss_list)[-(train_mean_num - 1):].mean()
                val_loss_mean = torch.stack(val_loss_list).mean()
                norm_mean = torch.stack(norm_list).mean()
                logger.info(
                    f"blocks {block_index} epoch {epoch} recon_loss:{loss_mean} val_loss:{val_loss_mean} quant_lr:{quant_scheduler.get_lr()[0]} norm:{norm_mean:.8f} max memory_allocated {torch.cuda.max_memory_allocated(dev) / 1024 ** 2} time {time.time() - start_time} ")

                if args.wandb:
                    wandb.log({
                        'loss': loss_mean,
                        'val': val_loss_mean,
                        'avg_bit': avg_bit,
                        'trunc': trunc,
                    })

                if val_loss_mean < best_val_loss:
                    best_val_loss = val_loss_mean
                else:
                    early_stop_flag += 1
                    if args.early_stop > 0 and early_stop_flag >= args.early_stop:
                        break
            optimizer.zero_grad()
            del optimizer

        if DEBUG:
            if args.wandb:
                wandb.finish()
            exit(0)

        if args.real_quant:
            if SPLIT > 1:
                for name, module in qlayer.named_modules():
                    if isinstance(module, fakeDBW):
                        module.n_bits = module.n_bits.to(dev)
            qlayer = qlayer.to(dev)
            torch.cuda.empty_cache()
            fake_to_real(qlayer, block_index, args, logger)
            qlayer = qlayer.to(dev)

            for name, parameter in qlayer.named_parameters():
                parameter.data = parameter.data.to(torch.float16)
            for name, buffer in qlayer.named_buffers():
                if 'rotary_emb' in name or 'qweight' in name:
                    continue
                buffer.data = buffer.data.to(torch.float16)
        else:
            quant_inplace(qlayer)

            for name, module in qlayer.named_modules():
                if 'rotary_emb' in name:
                    continue
                module.to(torch.float16)
            set_quant_state(qlayer, weight_quant=False)

        if args.epochs > 0:
            update_dataset(qlayer, quant_train_inps, dev, layer_kwargs)
            update_dataset(qlayer, quant_val_inps, dev, layer_kwargs)
        layers[block_index] = qlayer.to("cpu")

        del layer
        torch.cuda.empty_cache()

    if args.off_load_to_disk:
        for path in [fp_train_cache_path, fp_val_cache_path, quant_train_cache_path, quant_val_cache_path]:
            if os.path.exists(path):
                shutil.rmtree(path)

    torch.cuda.empty_cache()
    gc.collect()
    model.config.use_cache = use_cache

    if args.wandb:
        wandb.finish()

    return model
