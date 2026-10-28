"""Triton source of the three GPU kernels the INT8 runtime ships as compiled cubins (lib/kernels/k<cc>.pt).

act_quant  per-row INT8 quantization of activations (optional smoothing multiply, absmax scale, round half away)
mm_i8      INT8 x INT8 -> INT32 GEMM with per-row and per-column scales, BF16 output (W8A8 linear layers)
attn_fwd   causal grouped-query attention forward for the 8 full-attention layers (16 query / 4 K/V heads, dim 256)

build.py compiles these functions ahead of time for every entry of specs/sm<cc>.json. The runtime launches the cubins
through the CUDA driver (lib/int8_kernels.py); the same math in PyTorch is in lib/int8_quant.py.
"""
import triton
import triton.language as tl


@triton.jit
def act_quant(X, INV, Q, S, K, s_xm, s_qm, HAS_INV: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row: smoothing multiply, absmax, round half away from zero, clamp to +-127, INT8 store."""
    m = tl.program_id(0)
    amax = tl.zeros((BLOCK,), dtype=tl.float32)
    for k0 in range(0, K, BLOCK):
        rk = k0 + tl.arange(0, BLOCK)
        ok = rk < K
        x = tl.load(X + m * s_xm + rk, mask=ok, other=0.).to(tl.float32)
        if HAS_INV:
            x = x * tl.load(INV + rk, mask=ok, other=0.)
        amax = tl.maximum(amax, tl.abs(x))
    scale = tl.maximum(tl.max(amax, 0), 1e-8) / 127.
    tl.store(S + m, scale)
    for k0 in range(0, K, BLOCK):
        rk = k0 + tl.arange(0, BLOCK)
        ok = rk < K
        x = tl.load(X + m * s_xm + rk, mask=ok, other=0.).to(tl.float32)
        if HAS_INV:
            x = x * tl.load(INV + rk, mask=ok, other=0.)
        v = x / scale
        r = tl.where(v >= 0, v + 0.5, v - 0.5).to(tl.int32)
        r = tl.minimum(tl.maximum(r, -127), 127)
        tl.store(Q + m * s_qm + rk, r.to(tl.int8), mask=ok)


@triton.jit
def mm_i8(A, B, SA, SB, BIAS, C, M, N, K, s_am, s_ak, s_bn, s_bk, s_cm, s_cn,
          HAS_BIAS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr):
    """C[M, N] = (A_i8 @ B_i8^T)_i32 * SA[m] * SB[n] (+ BIAS[n]); B is stored [N, K]."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    width = GROUP_M * num_n
    first = (pid // width) * GROUP_M
    size = tl.minimum(num_m - first, GROUP_M)
    pid_m = first + (pid % width) % size
    pid_n = (pid % width) // size
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    n_ok = rn < N
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        k_ok = (k0 + rk) < K
        a = tl.load(A + rm[:, None] * s_am + (k0 + rk)[None, :] * s_ak, mask=m_ok[:, None] & k_ok[None, :], other=0)
        b = tl.load(B + rn[None, :] * s_bn + (k0 + rk)[:, None] * s_bk, mask=n_ok[None, :] & k_ok[:, None], other=0)
        acc += tl.dot(a, b, out_dtype=tl.int32)
    sa = tl.load(SA + rm, mask=m_ok, other=0.)
    sb = tl.load(SB + rn, mask=n_ok, other=0.)
    y = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    if HAS_BIAS:
        y = y + tl.load(BIAS + rn, mask=n_ok, other=0.).to(tl.float32)[None, :]
    tl.store(C + rm[:, None] * s_cm + rn[None, :] * s_cn, y.to(C.dtype.element_ty), mask=m_ok[:, None] & n_ok[None, :])


@triton.jit
def attn_fwd(Q, K, V, O, n_q, n_k,
             s_qb, s_qh, s_qm, s_kb, s_kh, s_kn, s_vb, s_vh, s_vn, s_ob, s_oh, s_om,
             HQ: tl.constexpr, REP: tl.constexpr, D: tl.constexpr, QK_SCALE: tl.constexpr,
             BM: tl.constexpr, BN: tl.constexpr):
    """Causal attention for n_q new queries over n_k keys (the last n_q keys are the queries' own positions).
    One program per (query block, batch, head); K/V head = head // REP. QK_SCALE includes log2(e) (exp2 softmax)."""
    pid = tl.program_id(0)
    num_m = tl.cdiv(n_q, BM)
    mb = pid % num_m
    bh = pid // num_m
    b = (bh // HQ).to(tl.int64)
    h = bh % HQ
    hk = h // REP
    offs_m = mb * BM + tl.arange(0, BM)
    offs_n = tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    row_ok = offs_m < n_q
    q = tl.load(Q + b * s_qb + h * s_qh + offs_m[:, None] * s_qm + offs_d[None, :], mask=row_ok[:, None], other=0.)
    k_ptr = K + b * s_kb + hk * s_kh + offs_n[None, :] * s_kn + offs_d[:, None]     # K^T tile [D, BN]
    v_ptr = V + b * s_vb + hk * s_vh + offs_n[:, None] * s_vn + offs_d[None, :]     # V tile [BN, D]
    shift = n_k - n_q
    m_i = tl.full([BM], float('-inf'), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    # every row of this block sees keys < mb * BM + shift + 1: whole BN tiles below that need no mask
    lo = tl.minimum(((mb * BM + shift + 1) // BN) * BN, n_k)
    hi = tl.minimum((mb + 1) * BM + shift, n_k)
    for start in range(0, lo, BN):
        kt = tl.load(k_ptr + start * s_kn)
        qk = tl.dot(q, kt) * QK_SCALE
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + start * s_vn)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    for start in range(lo, hi, BN):
        cols = start + offs_n
        kt = tl.load(k_ptr + start * s_kn, mask=cols[None, :] < n_k, other=0.)
        qk = tl.dot(q, kt) * QK_SCALE
        visible = (cols[None, :] <= offs_m[:, None] + shift) & (cols[None, :] < n_k)
        qk = tl.where(visible, qk, float('-inf'))
        m_new = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        v = tl.load(v_ptr + start * s_vn, mask=cols[:, None] < n_k, other=0.)
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        m_i = m_new
    acc = acc / l_i[:, None]
    tl.store(O + b * s_ob + h * s_oh + offs_m[:, None] * s_om + offs_d[None, :], acc.to(O.dtype.element_ty),
             mask=row_ok[:, None])


KERNELS = {'act_quant': act_quant, 'mm_i8': mm_i8, 'attn_fwd': attn_fwd}
