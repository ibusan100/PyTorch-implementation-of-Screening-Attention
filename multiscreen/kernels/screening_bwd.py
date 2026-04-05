"""
Triton backward kernel for Screening Attention.

Computes dQ, dK, dV without materializing the T×T alpha matrix.

Math recap (causal, per head):
    sim_ij   = q_i · k_j
    a_ij     = relu(1 - r*(1 - sim_ij))          # pre-square
    alpha_ij = a_ij^2 * softmask_ij
    out_i    = sum_j alpha_ij * v_j

Gradients:
    dV_j   = sum_i alpha_ij * dOut_i
    d_alpha_ij = (dOut_i · v_j) * softmask_ij    (elementwise before alpha)
    d_a_ij     = 2 * a_ij * d_alpha_ij
    d_sim_ij   = d_a_ij * r * (a_ij > 0)         (relu gate)
    dQ_i  += sum_j d_sim_ij * k_j
    dK_j  += sum_i d_sim_ij * q_i
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# dV kernel
# dV_j = sum_i alpha_ij * dOut_i
# Grid: (ceildiv(T, BLOCK_T), B, H)  — one program per key tile
# ---------------------------------------------------------------------------

@triton.jit
def _screening_bwd_dv(
    Q_ptr, K_ptr, V_ptr, dOut_ptr, dV_ptr,
    R_ptr, W_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_dvb, stride_dvh, stride_dvt, stride_dvd,
    B, H, T, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    tile_k = tl.program_id(0)
    b      = tl.program_id(1)
    h      = tl.program_id(2)

    r = tl.load(R_ptr + h)
    w = tl.load(W_ptr + h)

    k_start = tile_k * BLOCK_T
    offs_k  = k_start + tl.arange(0, BLOCK_T)
    offs_d  = tl.arange(0, D)

    q_base  = b * stride_qb + h * stride_qh
    k_base  = b * stride_kb + h * stride_kh
    o_base  = b * stride_ob + h * stride_oh
    dv_base = b * stride_dvb + h * stride_dvh

    # Load K tile: (BLOCK_T, D)
    k_ptrs = K_ptr + k_base + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd
    k = tl.load(k_ptrs, mask=offs_k[:, None] < T, other=0.0)

    # Accumulator for dV: (BLOCK_T, D)
    dv_acc = tl.zeros((BLOCK_T, D), dtype=tl.float32)

    # Iterate over query tiles that can attend to this key tile
    # Window pruning: query tile is active only if q_start - k_end < w
    #   k_end = tile_k * BLOCK_T + BLOCK_T - 1
    #   q_start < k_end + w  =>  tile_q < (k_end + w) / BLOCK_T
    w_int = w.to(tl.int32)
    k_end  = tile_k * BLOCK_T + BLOCK_T - 1
    max_tile_q = tl.minimum(tl.cdiv(T, BLOCK_T), (k_end + w_int) // BLOCK_T + 1)
    if CAUSAL:
        q_tile_start = tile_k   # causal: query >= key
    else:
        q_tile_start = 0
    n_q_tiles = max_tile_q

    for tile_q in range(q_tile_start, n_q_tiles):
        q_start = tile_q * BLOCK_T
        offs_q  = q_start + tl.arange(0, BLOCK_T)

        # Load Q tile
        q_ptrs = Q_ptr + q_base + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
        q = tl.load(q_ptrs, mask=offs_q[:, None] < T, other=0.0)

        # Load dOut tile: (BLOCK_T_q, D)
        do_ptrs = dOut_ptr + o_base + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od
        do = tl.load(do_ptrs, mask=offs_q[:, None] < T, other=0.0)

        # Cosine sim: (BLOCK_T_q, BLOCK_T_k)
        sim = tl.dot(q, tl.trans(k))

        # Softmask
        rel = offs_k[None, :].to(tl.float32) - offs_q[:, None].to(tl.float32)
        cos_mask = 0.5 * (tl.cos(math.pi * rel / w) + 1.0)
        if CAUSAL:
            in_window = (rel > -w) & (rel <= 0.0)
        else:
            in_window = tl.abs(rel) < w
        softmask = tl.where(in_window, cos_mask, 0.0)

        # alpha: (BLOCK_T_q, BLOCK_T_k)
        a = tl.maximum(1.0 - r * (1.0 - sim), 0.0)
        alpha = a * a * softmask
        alpha = tl.where(offs_k[None, :] < T, alpha, 0.0)
        alpha = tl.where(offs_q[:, None] < T, alpha, 0.0)

        # dV_j += sum_i alpha_ij * dOut_i  =>  alpha^T (BLOCK_T_k, BLOCK_T_q) @ do (BLOCK_T_q, D)
        dv_acc += tl.dot(tl.trans(alpha).to(tl.float32), do.to(tl.float32))

    # Store dV
    dv_ptrs = dV_ptr + dv_base + offs_k[:, None] * stride_dvt + offs_d[None, :] * stride_dvd
    tl.store(dv_ptrs, dv_acc.to(dV_ptr.dtype.element_ty), mask=offs_k[:, None] < T)


# ---------------------------------------------------------------------------
# dQ + dK kernel
# For each query tile, iterate over key tiles to accumulate dQ;
# use atomic adds to accumulate dK.
# Grid: (ceildiv(T, BLOCK_T), B, H)  — one program per query tile
# ---------------------------------------------------------------------------

@triton.jit
def _screening_bwd_dqk(
    Q_ptr, K_ptr, V_ptr, dOut_ptr, dQ_ptr, dK_ptr,
    R_ptr, W_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_dqb, stride_dqh, stride_dqt, stride_dqd,
    stride_dkb, stride_dkh, stride_dkt, stride_dkd,
    B, H, T, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    tile_q = tl.program_id(0)
    b      = tl.program_id(1)
    h      = tl.program_id(2)

    r = tl.load(R_ptr + h)
    w = tl.load(W_ptr + h)

    q_start = tile_q * BLOCK_T
    offs_q  = q_start + tl.arange(0, BLOCK_T)
    offs_d  = tl.arange(0, D)

    q_base  = b * stride_qb + h * stride_qh
    k_base  = b * stride_kb + h * stride_kh
    v_base  = b * stride_vb + h * stride_vh
    o_base  = b * stride_ob + h * stride_oh
    dq_base = b * stride_dqb + h * stride_dqh
    dk_base = b * stride_dkb + h * stride_dkh

    # Load Q tile and dOut tile
    q_ptrs  = Q_ptr   + q_base + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    do_ptrs = dOut_ptr + o_base + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od
    q  = tl.load(q_ptrs,  mask=offs_q[:, None] < T, other=0.0)
    do = tl.load(do_ptrs, mask=offs_q[:, None] < T, other=0.0)

    # Accumulator for dQ
    dq_acc = tl.zeros((BLOCK_T, D), dtype=tl.float32)

    # Number of key tiles to process (causal + window pruning)
    w_int = w.to(tl.int32)
    if CAUSAL:
        n_k_tiles = tile_q + 1
        min_tile_k = tl.maximum(0, (tile_q * BLOCK_T - w_int) // BLOCK_T)
    else:
        n_k_tiles = tl.cdiv(T, BLOCK_T)
        min_tile_k = 0

    for tile_k in range(min_tile_k, n_k_tiles):
        k_start = tile_k * BLOCK_T
        offs_k  = k_start + tl.arange(0, BLOCK_T)

        k_ptrs = K_ptr + k_base + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd
        v_ptrs = V_ptr + v_base + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd
        k = tl.load(k_ptrs, mask=offs_k[:, None] < T, other=0.0)
        v = tl.load(v_ptrs, mask=offs_k[:, None] < T, other=0.0)

        # Cosine sim and softmask
        sim = tl.dot(q, tl.trans(k))
        rel = offs_k[None, :].to(tl.float32) - offs_q[:, None].to(tl.float32)
        cos_mask = 0.5 * (tl.cos(math.pi * rel / w) + 1.0)
        if CAUSAL:
            in_window = (rel > -w) & (rel <= 0.0)
        else:
            in_window = tl.abs(rel) < w
        softmask = tl.where(in_window, cos_mask, 0.0)

        # Forward quantities
        a     = tl.maximum(1.0 - r * (1.0 - sim), 0.0)      # (BT_q, BT_k)
        alpha = a * a * softmask
        valid = (offs_k[None, :] < T) & (offs_q[:, None] < T)
        alpha = tl.where(valid, alpha, 0.0)

        # d_alpha_ij = dOut_i · v_j  (outer product via matmul)
        # do: (BT_q, D),  v: (BT_k, D)  =>  (BT_q, BT_k)
        d_alpha = tl.dot(do, tl.trans(v)) * softmask
        d_alpha = tl.where(valid, d_alpha, 0.0)

        # d_a_ij = 2 * a_ij * d_alpha_ij
        # d_sim_ij = r * (a > 0) * d_a_ij
        d_a   = 2.0 * a * d_alpha
        d_sim = r * (a > 0).to(tl.float32) * d_a   # (BT_q, BT_k)

        # dQ_i += sum_j d_sim_ij * k_j   =>  d_sim (BT_q, BT_k) @ k (BT_k, D) -> (BT_q, D)
        dq_acc += tl.dot(d_sim.to(tl.float32), k.to(tl.float32))

        # dK_j += sum_i d_sim_ij * q_i   =>  d_sim^T (BT_k, BT_q) @ q (BT_q, D) -> (BT_k, D)
        dk_delta = tl.dot(tl.trans(d_sim).to(tl.float32), q.to(tl.float32))

        # Atomic add to dK (multiple query tiles write to same key tile)
        dk_ptrs = dK_ptr + dk_base + offs_k[:, None] * stride_dkt + offs_d[None, :] * stride_dkd
        tl.atomic_add(dk_ptrs, dk_delta.to(dK_ptr.dtype.element_ty),
                      mask=offs_k[:, None] < T)

    # Store dQ
    dq_ptrs = dQ_ptr + dq_base + offs_q[:, None] * stride_dqt + offs_d[None, :] * stride_dqd
    tl.store(dq_ptrs, dq_acc.to(dQ_ptr.dtype.element_ty), mask=offs_q[:, None] < T)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def screening_attention_bwd(
    q: torch.Tensor,     # (B, H, T, D)
    k: torch.Tensor,
    v: torch.Tensor,
    d_out: torch.Tensor, # (B, H, T, D)
    r: torch.Tensor,     # (H,)
    w: torch.Tensor,     # (H,)
    causal: bool = True,
    block_t: int = 32,   # smaller than forward to fit in shared memory
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (dQ, dK, dV), each (B, H, T, D) float32.
    """
    B, H, T, D = q.shape
    assert D in (16, 32, 64, 128)

    q    = q.contiguous().float()
    k    = k.contiguous().float()
    v    = v.contiguous().float()
    d_out = d_out.contiguous().float()
    r    = r.contiguous().float()
    w    = w.contiguous().float()

    dQ = torch.zeros_like(q)
    dK = torch.zeros_like(k)
    dV = torch.zeros_like(v)

    grid = (triton.cdiv(T, block_t), B, H)

    # dV kernel
    _screening_bwd_dv[grid](
        q, k, v, d_out, dV,
        r, w,
        q.stride(0),  q.stride(1),  q.stride(2),  q.stride(3),
        k.stride(0),  k.stride(1),  k.stride(2),  k.stride(3),
        v.stride(0),  v.stride(1),  v.stride(2),  v.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        dV.stride(0), dV.stride(1), dV.stride(2), dV.stride(3),
        B, H, T, D,
        BLOCK_T=block_t,
        CAUSAL=causal,
    )

    # dQ + dK kernel
    _screening_bwd_dqk[grid](
        q, k, v, d_out, dQ, dK,
        r, w,
        q.stride(0),  q.stride(1),  q.stride(2),  q.stride(3),
        k.stride(0),  k.stride(1),  k.stride(2),  k.stride(3),
        v.stride(0),  v.stride(1),  v.stride(2),  v.stride(3),
        d_out.stride(0), d_out.stride(1), d_out.stride(2), d_out.stride(3),
        dQ.stride(0), dQ.stride(1), dQ.stride(2), dQ.stride(3),
        dK.stride(0), dK.stride(1), dK.stride(2), dK.stride(3),
        B, H, T, D,
        BLOCK_T=block_t,
        CAUSAL=causal,
    )

    return dQ, dK, dV
