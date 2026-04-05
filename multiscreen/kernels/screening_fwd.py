"""
Triton fused forward kernel for Screening Attention.

Fuses: cosine sim → Trim-and-Square → softmask → value accumulation
into a single tiled kernel that never materializes the T×T alpha matrix.

Based on arXiv:2604.01178, Section 3.
"""

import math
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _screening_fwd_kernel(
    # Pointers
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    # Per-head scalars (passed as pointer to 1-D arrays of length H)
    R_ptr,   # r = exp(s_r) + 1, shape (H,)
    W_ptr,   # w = exp(s_v) + 1, shape (H,)
    # Strides:  Q/K/V/Out are (B, H, T, D) contiguous
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    # Dimensions
    T: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,   # tile size along T (both Q and K axes)
    CAUSAL: tl.constexpr,
):
    """
    Each kernel instance handles one (batch, head, query-tile) triplet.

    Grid: (ceildiv(T, BLOCK_T), B*H)
    """
    # ---- identify this instance ----
    tile_q = tl.program_id(0)          # which query tile
    bh     = tl.program_id(1)          # flattened (batch, head) index
    B_val  = tl.num_programs(1)        # total B*H (not used beyond indexing)

    # Decode batch and head from bh (assumes H is known at runtime via strides)
    # We pass B*H as grid dim; we need H to split.  Instead, pass H as a
    # constexpr so we can do: b = bh // H, h = bh % H
    # NOTE: H is derived from stride_qh / stride_qt (since stride_qt = D and
    #       stride_qh = T*D).  However it's cleanest to pass it explicitly.
    # We'll derive H from the pointer strides at compile time via constexpr.
    # To keep the signature clean we instead accept H_val as a separate arg.
    pass  # placeholder — real implementation below uses the correct args


@triton.jit
def _screening_fwd_inner(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    R_ptr, W_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T, D: tl.constexpr,
    BLOCK_T: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    """
    Grid: (ceildiv(T, BLOCK_T), B, H)
    program_id: (tile_q, b, h)
    """
    tile_q = tl.program_id(0)
    b      = tl.program_id(1)
    h      = tl.program_id(2)

    # Per-head r and w
    r = tl.load(R_ptr + h)  # scalar
    w = tl.load(W_ptr + h)  # scalar

    # Row (query) indices for this tile
    q_start = tile_q * BLOCK_T
    offs_q  = q_start + tl.arange(0, BLOCK_T)        # (BLOCK_T,)
    offs_d  = tl.arange(0, D)                         # (D,)

    # Base pointer offsets for this (b, h)
    q_base = b * stride_qb + h * stride_qh
    k_base = b * stride_kb + h * stride_kh
    v_base = b * stride_vb + h * stride_vh
    o_base = b * stride_ob + h * stride_oh

    # Load Q tile: (BLOCK_T, D)
    q_mask = offs_q[:, None] < T  # (BLOCK_T, 1)
    q_ptrs = Q_ptr + q_base + offs_q[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)  # (BLOCK_T, D)

    # Accumulator for output: (BLOCK_T, D)
    acc = tl.zeros((BLOCK_T, D), dtype=tl.float32)

    # ---------------------------------------------------------------------------
    # Key-tile loop
    #
    # For causal attention we split into two phases:
    #   Phase A — "full" tiles  (tile_k < tile_q): all j <= i guaranteed,
    #             so the causal check is always satisfied.  Skip it entirely.
    #   Phase B — "diagonal" tile (tile_k == tile_q): j may be > i, apply mask.
    #
    # For non-causal we keep a single loop over all tiles.
    # ---------------------------------------------------------------------------

    if CAUSAL:
        # --- Phase A: fully-past tiles (no causal mask needed) ---
        for tile_k in range(0, tile_q):
            k_start = tile_k * BLOCK_T
            offs_k  = k_start + tl.arange(0, BLOCK_T)

            k_ptrs = K_ptr + k_base + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=offs_k[:, None] < T, other=0.0)
            sim = tl.dot(q, tl.trans(k))

            rel = offs_k[None, :].to(tl.float32) - offs_q[:, None].to(tl.float32)
            # For tile_k < tile_q: max(rel) = tile_k*BT + BT-1 - tile_q*BT = -(tile_q-tile_k)*BT + BT-1 <= -1
            # So rel <= 0 always; only need to check rel > -w
            in_window = rel > -w
            cos_mask  = 0.5 * (tl.cos(math.pi * rel / w) + 1.0)
            softmask  = tl.where(in_window, cos_mask, 0.0)

            alpha = tl.maximum(1.0 - r * (1.0 - sim), 0.0)
            alpha = alpha * alpha * softmask
            alpha = tl.where(offs_k[None, :] < T, alpha, 0.0)

            v_ptrs = V_ptr + v_base + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_k[:, None] < T, other=0.0)
            acc += tl.dot(alpha.to(tl.float32), v.to(tl.float32))

        # --- Phase B: diagonal tile (tile_k == tile_q, full causal mask) ---
        k_start = tile_q * BLOCK_T
        offs_k  = k_start + tl.arange(0, BLOCK_T)

        k_ptrs = K_ptr + k_base + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=offs_k[:, None] < T, other=0.0)
        sim = tl.dot(q, tl.trans(k))

        rel = offs_k[None, :].to(tl.float32) - offs_q[:, None].to(tl.float32)
        in_window = (rel > -w) & (rel <= 0.0)
        cos_mask  = 0.5 * (tl.cos(math.pi * rel / w) + 1.0)
        softmask  = tl.where(in_window, cos_mask, 0.0)

        alpha = tl.maximum(1.0 - r * (1.0 - sim), 0.0)
        alpha = alpha * alpha * softmask
        alpha = tl.where(offs_k[None, :] < T, alpha, 0.0)

        v_ptrs = V_ptr + v_base + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=offs_k[:, None] < T, other=0.0)
        acc += tl.dot(alpha.to(tl.float32), v.to(tl.float32))

    else:
        # --- Non-causal: single loop over all tiles ---
        n_k_tiles = tl.cdiv(T, BLOCK_T)
        for tile_k in range(0, n_k_tiles):
            k_start = tile_k * BLOCK_T
            offs_k  = k_start + tl.arange(0, BLOCK_T)

            k_ptrs = K_ptr + k_base + offs_k[:, None] * stride_kt + offs_d[None, :] * stride_kd
            k = tl.load(k_ptrs, mask=offs_k[:, None] < T, other=0.0)
            sim = tl.dot(q, tl.trans(k))

            rel      = offs_k[None, :].to(tl.float32) - offs_q[:, None].to(tl.float32)
            in_window = tl.abs(rel) < w
            cos_mask  = 0.5 * (tl.cos(math.pi * rel / w) + 1.0)
            softmask  = tl.where(in_window, cos_mask, 0.0)

            alpha = tl.maximum(1.0 - r * (1.0 - sim), 0.0)
            alpha = alpha * alpha * softmask
            alpha = tl.where(offs_k[None, :] < T, alpha, 0.0)

            v_ptrs = V_ptr + v_base + offs_k[:, None] * stride_vt + offs_d[None, :] * stride_vd
            v = tl.load(v_ptrs, mask=offs_k[:, None] < T, other=0.0)
            acc += tl.dot(alpha.to(tl.float32), v.to(tl.float32))

    # Write output
    o_ptrs = Out_ptr + o_base + offs_q[:, None] * stride_ot + offs_d[None, :] * stride_od
    tl.store(o_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=offs_q[:, None] < T)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

def screening_attention_fwd(
    q: torch.Tensor,   # (B, H, T, D) unit-normalized
    k: torch.Tensor,   # (B, H, T, D) unit-normalized
    v: torch.Tensor,   # (B, H, T, D) unit-normalized
    r: torch.Tensor,   # (H,)  acceptance sharpness per head
    w: torch.Tensor,   # (H,)  window size per head
    causal: bool = True,
    block_t: int = 64,
) -> torch.Tensor:
    """
    Fused Screening Attention forward pass.

    Computes:
        alpha_ij = relu(1 - r*(1 - q_i·k_j))^2 * softmask(i, j, w)
        out_i    = sum_j alpha_ij * v_j

    without materializing the full T×T alpha matrix.

    Args:
        q, k, v:  (B, H, T, D) float16 or float32, already L2-normalized
        r:        (H,) float32
        w:        (H,) float32
        causal:   whether to apply causal masking
        block_t:  tile size (must be a power of 2, >= 16)

    Returns:
        out: (B, H, T, D) float32
    """
    B, H, T, D = q.shape
    assert D in (16, 32, 64, 128), f"D={D} must be a power of 2 between 16 and 128"
    assert r.shape == (H,) and w.shape == (H,)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    r = r.contiguous().float()
    w = w.contiguous().float()

    out = torch.empty(B, H, T, D, dtype=torch.float32, device=q.device)

    grid = (triton.cdiv(T, block_t), B, H)

    _screening_fwd_inner[grid](
        q, k, v, out,
        r, w,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, T, D,
        BLOCK_T=block_t,
        CAUSAL=causal,
    )

    return out
