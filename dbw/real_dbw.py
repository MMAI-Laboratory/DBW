import math
from logging import getLogger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dbw.triton_kernel.gemm import dequant_dim0


logger = getLogger(__name__)


class TritonModuleMixin:
    @classmethod
    def warmup(cls, model, transpose=False, seqlen=2048):
        pass


class Unpacker(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
            self,
            bits,
            group_size,
            infeatures,
            outfeatures,
            **kwargs
    ):
        super().__init__()
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.group_size = group_size if group_size != -1 else infeatures
        self.maxq = 2 ** self.bits - 1
        self.register_buffer(
            'qweight',
            torch.zeros((math.ceil(infeatures / (32 // self.bits)), outfeatures), dtype=torch.int32)
        )

    def pack(self, W):
        intweight = W
        intweight = intweight.t().contiguous()
        intweight = intweight.numpy().astype(np.uint32)

        i = 0
        row = 0
        qweight = np.zeros((math.ceil(intweight.shape[0] / (32 // self.bits)), intweight.shape[1]), dtype=np.uint32)
        while row < qweight.shape[0]:
            for j in range(i, min(i + (32 // self.bits), intweight.shape[0])):
                qweight[row] |= intweight[j] << (self.bits * (j - i))
            i += 32 // self.bits
            row += 1

        qweight = qweight.astype(np.int32)
        self.qweight = torch.from_numpy(qweight)

    def forward(self):
        if not self.qweight.data.is_contiguous():
            self.qweight.data = self.qweight.data.contiguous()

        z = dequant_dim0(self.qweight, self.bits, self.maxq, self.infeatures, self.outfeatures)
        return z.transpose(0, 1)


class MultiQuantLinear(nn.Module, TritonModuleMixin):
    QUANT_TYPE = "triton"

    def __init__(
            self,
            bits_count,
            group_size,
            infeatures,
            outfeatures,
            bias,
            trainable=False,
            **kwargs
    ):
        super().__init__()
        self.quant_list = nn.ModuleList([])
        self.bits_count = bits_count
        for wbit, dim in bits_count.items():
            if wbit != 0 and dim != 0:
                self.quant_list.append(
                    Unpacker(wbit, group_size, infeatures, dim)
                )

        self.group_size = group_size
        self.register_parameter(
            'scales',
            torch.nn.Parameter(
                torch.zeros((math.ceil(infeatures / self.group_size) * outfeatures, 1), dtype=torch.float16))
        )
        self.register_parameter(
            'zeros',
            torch.nn.Parameter(
                torch.zeros((math.ceil(infeatures / self.group_size) * outfeatures, 1), dtype=torch.float16))
        )
        if bias:
            self.register_buffer('bias', torch.zeros((outfeatures,), dtype=torch.float16))
        else:
            self.bias = None
        self.repeat = infeatures // group_size
        self.zeros_dim0, self.zeros_dim1 = self.scales.shape
        self.trainable = trainable
        self.scales.requires_grad = True
        self.zeros.requires_grad = True
        self.use_fake = False

    def pack(self, Ws, scales, zeros, qmax):
        qmax = (2 ** qmax - 1).repeat_interleave(self.repeat, 0)
        self.scales = nn.Parameter(scales / qmax)
        self.zeros = nn.Parameter(zeros)
        sum_count = 0
        index = 0
        for i in range(len(list(self.bits_count.values()))):
            end_count = list(self.bits_count.values())[i]
            if end_count != 0:
                self.quant_list[index].pack(Ws[sum_count:sum_count + end_count])
                sum_count += end_count
                index += 1

    def forward(self):
        z = []
        for layer in self.quant_list:
            z.append(layer())
        z = torch.cat(z, dim=0)
        dim0, dim1 = z.shape
        w = z.reshape(-1, self.group_size)
        w = w * self.scales + self.zeros
        return w.reshape(dim0, dim1), z


class realDBW(nn.Module):
    QUANT_TYPE = "triton"

    def __init__(self, bits, group_size, infeatures, outfeatures, rank, bias, trainable=False, ver='LBQ'):
        super().__init__()
        self.ver = ver
        if isinstance(bias, torch.Tensor):
            self.register_buffer('bias', bias.detach().clone().to(torch.float16))
        elif bias:
            self.register_buffer('bias', torch.zeros((outfeatures,), dtype=torch.float16))
        else:
            self.bias = None

        self.u = MultiQuantLinear(bits, group_size, outfeatures, rank, False, trainable)
        self.v = MultiQuantLinear(bits, group_size, infeatures, rank, False, trainable)
        self.register_parameter('sigma', nn.Parameter(torch.rand((rank,), dtype=torch.float16)))
        self.sigma.requires_grad = True

    def pack(self, linear, scales, zeros, qmax=None, sigma=None, sigma_total=None):
        self.u.pack(linear[0], scales[0], zeros[0], qmax)
        self.v.pack(linear[1], scales[1], zeros[1], qmax)
        self.sigma = nn.Parameter(sigma)
        self.register_buffer('sigma_total', sigma_total)

    def use_fake_quantization(self, del_quant=False, transpose=False):
        self.u.use_fake_quantization(del_quant, transpose)
        self.v.use_fake_quantization(del_quant, transpose)

    def get_uv(self):
        u, u_z = self.u()
        v, v_z = self.v()
        u = u / (torch.norm(u, dim=-1, keepdim=True) + 1e-6)
        v = v / (torch.norm(v, dim=-1, keepdim=True) + 1e-6)
        v = torch.einsum("...rj, ...r -> ...rj", v, torch.relu(self.sigma) * self.sigma_total)
        return [u, v], [u_z, v_z], [0, 0]

    def get_w(self):
        [u, v], [u_z, v_z], [u_zero, v_zero] = self.get_uv()
        w = torch.matmul(u.mT, v)
        return w, [u_z, v_z], [u_zero, v_zero]

    def forward(self, x):
        [u, v], _, _ = self.get_uv()
        out = F.linear(x, v.to(x.dtype))
        out = F.linear(out, u.mT.to(x.dtype))
        bias = getattr(self, 'bias', None)
        if bias is not None:
            out = out + bias.to(device=out.device, dtype=out.dtype)
        return out
