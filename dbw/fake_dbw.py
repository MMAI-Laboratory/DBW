import torch
import torch.nn as nn
import torch.nn.functional as F


BIT_MAX = 5

def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def floor_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.floor() - x).detach() + x


def apportion_softmax(theta, rank, bit, rand=False, clip=None):
    org_shape = theta.shape
    theta = theta.squeeze()
    p = F.relu(theta)
    p = p / p.sum()
    r = p * rank * bit

    n = floor_ste(r)
    n = n.clamp(0, BIT_MAX)

    leftover = int(rank * bit - n.sum())

    fractional = (r - n)

    while leftover > 0:
        can_add = (n < BIT_MAX)
        idx = torch.nonzero(can_add, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            break

        scores = fractional[idx]
        k = min(leftover, idx.numel())

        top_idx = torch.topk(scores, k=k, largest=True).indices
        chosen = idx[top_idx]

        n[chosen] += 1
        leftover -= k

    return n.reshape(org_shape)


class fakeDBW(nn.Module):
    def __init__(self, u, s, v, w, wbits=4, group_size=128, learn_n_bits=False, ver='LBQ', main_dev=None, bias=None):
        super().__init__()
        self.ver = ver
        self.use_weight_quant = False
        self.use_quant = True
        self.save_statics = False
        self.n_rows = v.shape[0] if v is not None else None
        self.n_bits = wbits
        self.group_size = group_size
        self.main_dev = main_dev
        if isinstance(bias, torch.Tensor):
            self.register_buffer('bias', bias.detach().clone())
        elif bias and w is not None:
            self.register_buffer('bias', torch.zeros((w.shape[0],), dtype=w.dtype, device=w.device))
        else:
            self.bias = None
        
        if ver == 'LBQ':
            u = u.mT
            qmax = torch.normal(
                mean=0, std=1, size=(u.size(0), 1)
            ).abs().sort(dim=0, descending=True)[0].to(u.dtype).to(u.device)
            qmax /= qmax.sum()
            qmax, u, v, s, w = self.unify_compute_type(qmax, u, v, s, w)
            s = torch.relu(s)
            self.s_scale = nn.Parameter((s / s.sum()).detach().clone())
            self.register_buffer('s_total_const', s.sum().detach().clone())
            self.qmax = nn.Parameter(qmax.detach().clone())
            self.bit_qmax = None
            self.n = u.size(-1)
            self.m = v.size(-1)
            self.r = min(self.n, self.m)
            self.u_weight = nn.Parameter(u.detach().clone().contiguous())
            self.v_weight = nn.Parameter(v.detach().clone().contiguous())
            torch.cuda.empty_cache()
        elif ver == 'LBQ_inference':
            m, n = self.w.shape
            rank = min(m, n)
            self.n_rows = rank
            self.s_scale = nn.Parameter(torch.rand((rank,), dtype=torch.float16))
            self.qmax = nn.Parameter(torch.rand((rank, 1), dtype=torch.float16))
            self.u_weight = nn.Parameter(torch.rand((rank, m), dtype=torch.float16))
            self.v_weight = nn.Parameter(torch.rand((rank, n), dtype=torch.float16))

    def unify_compute_type(self, qmax, u, v, s, w):
        dtype = torch.float16
        device = 'cpu'
        qmax = qmax.to(dtype).to(device)
        u = u.to(dtype).to(device)
        v = v.to(dtype).to(device)
        s = s.to(dtype).to(device)
        w = w.to(dtype).to(device)
        return qmax, u, v, s, w

    def sorted_normal_dist(self, size, std):
        return torch.normal(mean=0, std=std, size=(size, 1)).sort(dim=0, descending=True)[0]

    def quant(self, v):
        v = v.reshape(-1, 128)
        v_min = torch.amin(v, dim=-1, keepdim=True)
        v_max = torch.amax(v, dim=-1, keepdim=True)
        scale = v_max - v_min + 1e-6
        bias = v_min 
        v_quant = (v - bias) / scale
        return v_quant, scale, bias

    def dequant(self, v_quant, v_scale, v_min, qmax, size, repeat):
        qmax = (2 ** qmax - 1).repeat_interleave(repeat, 0)
        mask = (qmax != 0.0).squeeze()
        zeros = torch.zeros_like(qmax)
        v_dequant = round_ste(v_quant * qmax).clamp(zeros, qmax)
        v_dequant[mask] = (v_dequant[mask] / qmax[mask]) * v_scale[mask] + v_min[mask]
        v_dequant = v_dequant.reshape(size, -1)
        return v_dequant

    def get_raw_bit(self, mask=None):
        if mask is not None:
            return self.qmax[mask]
        else:
            return self.qmax

    def get_bit(self, mask=None):
        return apportion_softmax(self.get_raw_bit(mask), self.n_rows, self.n_bits, False, self.bit_qmax)

    def get_quv(self, mask=None):
        qmax = self.get_bit(mask)
        size = self.r
        u_repeat = int(self.n // 128)
        v_repeat = int(self.m // 128)
        def get_uv(u, v, qmax, size):
            u_quant, u_scale, u_min = self.quant(u)
            v_quant, v_scale, v_min = self.quant(v)
            u = self.dequant(u_quant, u_scale, u_min, qmax, size, u_repeat)
            v = self.dequant(v_quant, v_scale, v_min, qmax, size, v_repeat)
            u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-6).to(torch.float16)
            v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6).to(torch.float16)
            return u, v
        mask = (qmax > 0).squeeze()
        qmax1 = qmax[mask]
        size = qmax1.size(0)
        u_in = self.u_weight[mask]
        v_in = self.v_weight[mask]
        u, v = get_uv(u_in, v_in, qmax1, size)
        return u, v, mask

    def get_w(self, mask=None):
        if self.use_quant:
            u, v, mask = self.get_quv()
            v = torch.einsum("...rj, ...r -> ...rj", v, self.get_scale()[mask])
            return u, v
        else:
            u, v = self.u_weight, self.v_weight
            u = F.normalize(u, dim=-1)
            v = F.normalize(v, dim=-1)
            v = torch.einsum("...rj, ...r -> ...rj", v, self.get_scale())
            return torch.matmul(u.mT, v)

    def get_scale(self):
        s = torch.relu(self.s_scale)
        p = s * self.s_total_const
        return p

    def _apply_bias(self, out):
        bias = getattr(self, 'bias', None)
        if bias is not None:
            out = out + bias.to(device=out.device, dtype=out.dtype)
        return out

    def forward(self, input):
        if self.main_dev:
            input = input.to(self.u_weight.device)

        if self.use_weight_quant:
            if self.save_statics:
                u = self.u_weight
                v = self.v_weight
                out = F.linear(input, v)
                out = F.linear(out, torch.diag(self.get_scale()))
                self.statics = out.detach().clone().abs().float().mean(dim=[0, 1]).cpu()
                out = F.linear(out, u.mT)
                if self.main_dev:
                    out = out.to(self.main_dev)
                return self._apply_bias(out)
            if self.use_quant:
                u, v = self.get_w()
                out = F.linear(input, v)
                out = F.linear(out, u.mT)
                if self.main_dev:
                    out = out.to(self.main_dev)
                return self._apply_bias(out)
            out = F.linear(input, self.get_w())
            if self.main_dev:
                out = out.to(self.main_dev)
            return self._apply_bias(out)
        else:
            w = self.w
            out = F.linear(input, w)
            if self.main_dev:
                out = out.to(self.main_dev)
            return self._apply_bias(out)
        
    # ----------- functions below this lines are utilities ------------ # 
    def quant_int_as_fp16(self, v, qmax, repeat):
        v = v.reshape(-1, 128)
        v_max = v.amax(dim=-1, keepdim=True)
        v_min = v.amin(dim=-1, keepdim=True)
        scale = (v_max - v_min) + 1e-6
        v_quant = (v - v_min) / scale
        qmax = (2 ** qmax - 1).repeat_interleave(repeat, 0)
        zeros = torch.zeros_like(qmax)
        v_quant = round_ste(v_quant * qmax).clamp(zeros, qmax)
        return v_quant, scale.to(torch.float16), v_min.to(torch.float16)

    def uvq_fp16(self):
        qmax = self.get_bit()
        self.qmax_fp16 = qmax.to(torch.float16)
        u_repeat = int(self.n // 128)
        v_repeat = int(self.m // 128)
        self.u_weight_fp16 = self.quant_int_as_fp16(self.u_weight, qmax, u_repeat)
        self.v_weight_fp16 = self.quant_int_as_fp16(self.v_weight, qmax, v_repeat)

    def dequant_as_fp16(self, v_quant, v_scale, v_min, qmax, size, repeat):
        qmax = (2 ** qmax - 1).repeat_interleave(repeat, 0)
        mask = (qmax != 0.0).squeeze()
        v_quant[mask] = (v_quant[mask] / qmax[mask])
        v_quant = v_quant.half()
        v_quant[mask] = v_quant[mask] * v_scale[mask] + v_min[mask]
        v_dequant = v_quant.reshape(size, -1)
        return v_dequant.half()

    def uv_fp16(self):
        qmax = self.qmax_fp16
        size = self.r
        u_repeat = int(self.n // 128)
        v_repeat = int(self.m // 128)
        u_quant, u_scale, u_min = self.u_weight_fp16
        v_quant, v_scale, v_min = self.v_weight_fp16
        u = self.dequant_as_fp16(u_quant, u_scale, u_min, qmax, size, u_repeat)
        v = self.dequant_as_fp16(v_quant, v_scale, v_min, qmax, size, v_repeat)
        u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-6).to(torch.float16)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6).to(torch.float16)
        return u, v

    def quant_inplace(self):
        with torch.cuda.amp.autocast():
            self.uvq_fp16()
            self.to(torch.float16)
            u, v = self.uv_fp16()
            v = torch.einsum("...rj, ...r -> ...rj", v, self.get_scale())
            w = torch.matmul(u.mT, v)
            self.w.copy_(w)
            del self.qmax_fp16
            del self.u_weight_fp16
            del self.v_weight_fp16

    def change_n_bits(self, n_bits):
        self.n_bits = n_bits

    def set_quant_state(self, weight_quant: bool = False):
        self.use_weight_quant = weight_quant

    def get_avg_bit(self):
        qmax = self.get_bit()
        avg_bit = qmax.mean()
        return avg_bit

    def get_effective_avg_bit(self):
        bit = self.get_bit()
        param = self.n * self.m
        new_param = self.n * self.r + self.m * self.r
        trunc = param - (bit != 0.0).float().mean() * new_param
        max_bit = bit.max()
        return param, trunc, max_bit

    def dequant_int(self, v_quant, qmax, size, repeat):
        qmax = (2 ** qmax - 1).repeat_interleave(repeat, 0)
        zeros = torch.zeros_like(qmax)
        v_dequant = round_ste(v_quant * qmax).clamp(zeros, qmax)
        v_dequant = v_dequant.reshape(size, -1)
        return v_dequant

    def get_int(self):
        qmax = self.get_bit()
        order = torch.argsort(qmax.squeeze(), descending=True)
        qmax = qmax[order]
        mask = (qmax != 0).squeeze()
        qmax = qmax[mask]
        size = qmax.size(0)
        u = self.u_weight.data.detach().clone()[order][mask]
        u_repeat = int(u.shape[-1] // 128)
        u_quant, u_scale, u_min = self.quant(u)
        u_w = self.dequant_int(u_quant, qmax, size, u_repeat)
        v = self.v_weight.data.detach().clone()[order][mask]
        v_repeat = int(v.shape[-1] // 128)
        v_quant, v_scale, v_min = self.quant(v)
        v_w = self.dequant_int(v_quant, qmax, size, v_repeat)
        u_scale, u_min = u_scale.to(torch.float16), u_min.to(torch.float16)
        v_scale, v_min = v_scale.to(torch.float16), v_min.to(torch.float16)
        s_scale = torch.relu(self.s_scale[order][mask]).to(torch.float16)
        s_total_const = self.s_total_const.to(torch.float16)
        return [u_w, v_w], [u_scale, v_scale], [u_min, v_min], qmax, s_scale, s_total_const
