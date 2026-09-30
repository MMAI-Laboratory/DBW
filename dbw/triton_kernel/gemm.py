import math
from logging import getLogger
import torch
import triton
import triton.language as tl
from dbw.triton_kernel import custom_autotune
logger = getLogger(__name__)


@custom_autotune.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
            },
            num_stages=4,
            num_warps=4
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=2,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=4,
            num_warps=4
        ),
    ],
    key=['M', 'N'],
    nearest_power_of_two=True,
    prune_configs_by={
        'early_config_prune': custom_autotune.hadamard248_kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
@triton.jit
def dequant_kernel_dim0(
        b_ptr, c_ptr,
        M, N,
        bits, maxq,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        B_M, B_N,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    dequant the quantized tensor to fp tensor
    B is of shape (M/(32//bits), N) int32
    C is of shape (M, N) float16
    """

    bits_per_feature = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    row_idx = offs_m[:, None] // bits_per_feature
    col_idx = offs_n[None, :]
    b_mask = (row_idx < B_M) & (col_idx < B_N)

    mask = c_mask & b_mask
    safe_row_idx = tl.where(b_mask, row_idx, 0)
    safe_col_idx = tl.where(b_mask, col_idx, 0)

    b_ptrs = b_ptr + safe_row_idx * stride_bk + safe_col_idx * stride_bn
    b = tl.load(b_ptrs, mask=mask, other=0)

    shifter = (offs_m[:, None] % bits_per_feature) * bits
    b = (b >> shifter) & maxq

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, b, mask=mask)


@custom_autotune.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N': 128,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N': 64,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N': 32,
            },
            num_stages=8,
            num_warps=8
        ),
        triton.Config(
            {
                'BLOCK_SIZE_M': 2,
                'BLOCK_SIZE_N': 2,
            },
            num_stages=8,
            num_warps=8
        ),
    ],
    key=['M', 'N'],
    nearest_power_of_two=True,
    prune_configs_by={
        'early_config_prune': custom_autotune.hadamard248_kernel_config_pruner,
        'perf_model': None,
        'top_k': None,
    },
)
@triton.jit
def dequant_kernel_dim1(
        b_ptr, c_ptr,
        M, N,
        bits, maxq,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    """
    dequant the quantized tensor to fp tensor
    B is of shape (M, N/(32//bits)) int32
    C is of shape (M, N) float16
    """

    bits_per_feature = 32 // bits

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    b_ptrs = b_ptr + (offs_am[:, None] * stride_bk + (offs_bn[None, :] // bits_per_feature) * stride_bn)

    shifter = (offs_bn[None, :] % bits_per_feature) * bits

    b = tl.load(b_ptrs)
    b = (b >> shifter) & maxq

    c = b

    c_ptrs = c_ptr + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)


def dequant_dim0(qweight, bits, maxq, infeatures, outfeatures):
    bits_per_feature = 32 // bits
    q_infeatures, q_outfeatures = qweight.shape

    expected = math.ceil(infeatures / bits_per_feature)
    if q_infeatures != expected:
        raise RuntimeError(
            f"qweight rows ({q_outfeatures}) does not match to the expected ({expected}) "
            f"(M={infeatures}, bits={bits}, bits_per_feature={bits_per_feature})"
        )

    with torch.cuda.device(qweight.device):
        output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=torch.float16).contiguous()
        grid = lambda META: (
            triton.cdiv(output.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output.shape[1], META['BLOCK_SIZE_N']),
        )
        dequant_kernel_dim0[grid](
            qweight, output,
            output.shape[0], output.shape[1],
            bits, maxq,
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
            qweight.shape[0], qweight.shape[1]
        )
        return output


def dequant_dim1(qweight, bits, maxq, infeatures, outfeatures):
    with torch.cuda.device(qweight.device):
        output = torch.empty((infeatures, outfeatures), device=qweight.device, dtype=torch.float16)
        grid = lambda META: (
            triton.cdiv(output.shape[0], META['BLOCK_SIZE_M']) * triton.cdiv(output.shape[1], META['BLOCK_SIZE_N']),
        )
        dequant_kernel_dim1[grid](
            qweight, output,
            output.shape[0], output.shape[1],
            bits, maxq,
            qweight.stride(0), qweight.stride(1),
            output.stride(0), output.stride(1),
        )
        return output


if __name__ == '__main__':
    pass
