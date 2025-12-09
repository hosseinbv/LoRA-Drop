# triton_lora.py
import math
import torch
import triton
import triton.language as tl
from torch.autograd import Function
from torch import nn

import triton
import triton.language as tl
@triton.jit
def _lora_forward_kernel(
    X, A, B, Y,
    M, D,                                   # runtime ints
    stride_xm, stride_xd,
    stride_ar, stride_ad,
    stride_bd, stride_br,
    stride_ym, stride_yd,
    scale,                                   # runtime float
    R: tl.constexpr,                         # constexpr rank
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    USE_BF16: tl.constexpr,                  # constexpr: 1 => bf16, 0 => fp16
):
    pid_m = tl.program_id(0)
    if pid_m >= M:
        return

    # ----- s[r] = <x, A[r,:]> for r=0..R-1 (vectorized) -----
    s = tl.zeros((R,), dtype=tl.float32)

    offs_k = tl.arange(0, BLOCK_K)
    x_row_ptr = X + pid_m * stride_xm
    k0 = 0
    while k0 < D:
        k_idx = k0 + offs_k                                 # [BLOCK_K]
        x_tile = tl.load(
            x_row_ptr + k_idx * stride_xd,
            mask=k_idx < D, other=0.0
        ).to(tl.float32)                                    # [BLOCK_K]

        # A_tile: [R, BLOCK_K]
        a_ptr  = A + (tl.arange(0, R)[:, None] * stride_ar) + (k_idx[None, :] * stride_ad)
        a_tile = tl.load(
            a_ptr,
            mask=(k_idx[None, :] < D), other=0.0
        ).to(tl.float32)                                    # [R, BLOCK_K]

        s += tl.sum(a_tile * x_tile[None, :], axis=1)       # [R]
        k0 += BLOCK_K

    s = s * scale                                           # [R]

    # ----- Y row in N-sized blocks: out = sum_r B[:,r] * s[r]
    nblocks = (D + BLOCK_N - 1) // BLOCK_N
    for nb in range(nblocks):
        n0 = nb * BLOCK_N
        offs_n = n0 + tl.arange(0, BLOCK_N)                 # [BLOCK_N]

        # B_block: [BLOCK_N, R]
        b_ptr = B + (offs_n[:, None] * stride_bd) + (tl.arange(0, R)[None, :] * stride_br)
        b_blk = tl.load(
            b_ptr,
            mask=(offs_n[:, None] < D), other=0.0
        ).to(tl.float32)                                    # [BLOCK_N, R]

        out = tl.sum(b_blk * s[None, :], axis=1)            # [BLOCK_N]

        y_ptr = Y + pid_m * stride_ym + offs_n * stride_yd
        if USE_BF16:
            tl.store(y_ptr, out.to(tl.bfloat16), mask=offs_n < D)
        else:
            tl.store(y_ptr, out.to(tl.float16),  mask=offs_n < D)




def _is_triton_eligible(x: torch.Tensor, A: torch.Tensor, B: torch.Tensor):
    return (
        x.is_cuda and A.is_cuda and B.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and A.dtype == x.dtype and B.dtype == x.dtype
    )

class _TritonLoRAFunction(Function):
    @staticmethod
    def forward(ctx, x, A, B, scale: float):
        orig_shape = x.shape
        D = orig_shape[-1]
        x2d = x.reshape(-1, D).contiguous()
        A = A.contiguous()
        B = B.contiguous()
        M, D = x2d.shape
        R = int(A.shape[0])

        # define BEFORE kernel launch
        y2d = torch.empty_like(x2d)

        if _is_triton_eligible(x2d, A, B):
            BLOCK_K = 128 if D >= 128 else 64
            BLOCK_N = 128 if D >= 128 else 64
            grid = (M,)

            use_bf16 = int(x2d.dtype == torch.bfloat16)

            _lora_forward_kernel[grid](
                X=x2d, A=A, B=B, Y=y2d,
                M=M, D=D,
                stride_xm=x2d.stride(0), stride_xd=x2d.stride(1),
                stride_ar=A.stride(0),  stride_ad=A.stride(1),
                stride_bd=B.stride(0),  stride_br=B.stride(1),
                stride_ym=y2d.stride(0), stride_yd=y2d.stride(1),
                scale=float(scale),
                R=R,
                BLOCK_K=BLOCK_K,
                BLOCK_N=BLOCK_N,
                USE_BF16=use_bf16,          # constexpr flag for store dtype
            )
        else:
            y2d = (x2d @ A.t()).matmul(B.t()) * scale

        ctx.save_for_backward(x2d, A, B)
        ctx.scale = scale
        return y2d.view(orig_shape)

    @staticmethod
    def backward(ctx, grad_out):
        x2d, A, B = ctx.saved_tensors
        scale = ctx.scale
        D = x2d.shape[1]
        go = grad_out.reshape(-1, D)

        Z0 = x2d.matmul(A.t())              # [M, R]
        dB = go.t().matmul(Z0) * scale      # [D, R]
        gZ = go.matmul(B) * scale           # [M, R]
        dA = gZ.t().matmul(x2d)             # [R, D]
        dX = gZ.matmul(A)                   # [M, D]
        return dX.reshape_as(grad_out), dA, dB, None


class TritonLoRAAdapter(torch.nn.Module):
    """
    y = (x @ A^T) @ B^T * (alpha / r)
    A: [r, d_model], B: [d_model, r]
    """
    def __init__(self, d_model: int, rank: int = 8, alpha: float = 16.0, dtype=torch.float16):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.alpha = float(alpha)
        self.scale = self.alpha / float(rank)
        self.A = torch.nn.Parameter(torch.empty(rank, d_model, dtype=dtype))
        self.B = torch.nn.Parameter(torch.empty(d_model, rank, dtype=dtype))
        torch.nn.init.normal_(self.A, mean=0.0, std=1.0 / math.sqrt(d_model))
        torch.nn.init.zeros_(self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _TritonLoRAFunction.apply(x, self.A, self.B, self.scale)


class LoRADropAdapter(nn.Module):
    def __init__(self, hidden_size: int, rank: int = 8, alpha: float = 16.0, dtype=torch.float16):
        super().__init__()
        self.A = nn.Linear(hidden_size, rank, bias=False, dtype=dtype)
        self.B = nn.Linear(rank, hidden_size, bias=False, dtype=dtype)
        self.scaling = alpha / max(1, rank)

        # init like LoRA
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.B(self.A(x)) * self.scaling


B, T, D, r = 2, 3, 4094, 8
x = torch.randn(B, T, D, device="cuda", dtype=torch.float16, requires_grad=True)

lora = TritonLoRAAdapter(D, rank=r, alpha=16.0, dtype=torch.float16).cuda()

orig_lora = LoRADropAdapter(hidden_size=D, rank=r).cuda()


y_triton = lora(x)
y_lora = orig_lora(x)

print("max abs diff:", (y_triton - y_lora).abs().max().item())  # ~1e-3–1e-4 for fp16

# A, Bm, s = lora.A, lora.B, lora.scale
# y_ref = (x @ A.t()) @ Bm.t() * s
# print("max abs diff:", (y_triton - y_ref).abs().max().item())  # ~1e-3–1e-4 for fp16

# (y_triton.sum()).backward()
# print("grad norms:", x.grad.norm().item(), lora.A.grad.norm().item(), lora.B.grad.norm().item())


# import math, torch
# torch.nn.init.normal_(lora.B, mean=0.0, std=1.0/math.sqrt(D))  # make B ≠ 0
# # (optional) also reinit A if you want different stats:
# # torch.nn.init.normal_(lora.A, mean=0.0, std=1.0/math.sqrt(D))

# x.grad = None
# y_triton = lora(x)
# y_ref = (x @ lora.A.t()) @ lora.B.t() * lora.scale
# print("max abs diff:", (y_triton - y_ref).abs().max().item())   # expect ~1e-3 for fp16/bf16

# (y_triton.sum()).backward()
# print("grad norms:", x.grad.norm().item(), lora.A.grad.norm().item(), lora.B.grad.norm().item())
