import torch
from torch.nn import functional as F

def _kernels():
    global triton, tl
    import triton
    import triton.language as tl

    @triton.jit
    def norm(X, Y, W, R, S, D: tl.constexpr, XD: tl.constexpr, RD: tl.constexpr, EPS: tl.constexpr, ADD: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        col = tl.arange(0, BLOCK)
        x = tl.load(X + row * XD + col, col < D, 0).to(tl.float32)
        if ADD:
            r = tl.load(R + row * RD + col, col < D, 0).to(tl.float32)
            x = (x + r).to(X.dtype.element_ty).to(tl.float32)
            tl.store(S + row * D + col, x, col < D)
        variance = tl.sum(x * x, 0) / D
        scale = tl.rsqrt(variance + EPS)
        w = tl.load(W + col, col < D, 0).to(tl.float32)
        tl.store(Y + row * D + col, x * scale * (1.0 + w), col < D)

    @triton.jit
    def _v7kuxsp4ygwdk6j967lqw(G, U, Y, N):
        i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
        g = tl.load(G + i, i < N, 0).to(tl.float32)
        u = tl.load(U + i, i < N, 0).to(tl.float32)
        activated = (g / (1.0 + tl.exp(-g))).to(G.dtype.element_ty).to(tl.float32)
        tl.store(Y + i, activated * u, i < N)
    return (triton, norm, _v7kuxsp4ygwdk6j967lqw)
_loaded = None

def rms_norm(x, weight, eps, residual=None):
    if not x.is_cuda:
        value = x if residual is None else x + residual
        xf = value.float()
        y = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps) * (1 + weight.float())
        y = y.to(x.dtype)
        return y if residual is None else (y, value)
    if x.stride(-1) != 1:
        raise ValueError('fused RMSNorm requires unit feature stride')
    flat = x.view(-1, x.shape[-1])
    residual_flat = flat if residual is None else residual.view_as(flat)
    global _loaded
    if _loaded is None:
        _loaded = _kernels()
    triton, kernel, _ = _loaded
    y = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    summed = torch.empty_like(y) if residual is not None else y
    d = x.shape[-1]
    kernel[x.numel() // d,](x, y, weight, x if residual is None else residual, summed, d, flat.stride(0), residual_flat.stride(0), eps, residual is not None, triton.next_power_of_2(d), enable_fp_fusion=False)
    return y if residual is None else (y, summed)

def swi_glu(gate, up):
    if not gate.is_cuda:
        return F.silu(gate) * up
    if not gate.is_contiguous() or not up.is_contiguous():
        raise ValueError('fused gate-up kernel requires dense projections')
    global _loaded
    if _loaded is None:
        _loaded = _kernels()
    triton, _, kernel = _loaded
    y = torch.empty_like(gate)
    kernel[triton.cdiv(gate.numel(), 1024),](gate, up, y, gate.numel(), num_warps=4, enable_fp_fusion=False)
    return y