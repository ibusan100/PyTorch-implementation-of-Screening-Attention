"""
Screening Attention: drop-in replacement for softmax attention.

Based on "Screening Is Enough" (arXiv:2604.01178).

Key differences from standard attention:
- Q, K, V are L2-normalized to unit vectors (similarities stay in [-1, 1])
- Relevance is computed independently per key (no global softmax competition)
- Trim-and-Square thresholding: alpha = [max(1 - r*(1 - sim), 0)]^2
- Distance-aware cosine softmask gates attention by causal window
- Value aggregation has NO normalization (can represent absent context)
- Output is normalized with TanhNorm
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import TanhNorm


class ScreeningAttention(nn.Module):
    """
    Multi-head Screening Attention.

    Args:
        d_model:    Total model dimension.
        num_heads:  Number of attention heads.
        dropout:    Dropout probability on attention weights.
        causal:     If True, apply causal masking (autoregressive decoding).
        eps:        Small value for numerical stability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.0,
        causal: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.eps = eps

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable log-scale parameters for window and acceptance width
        # w = exp(s_v) + 1  (screening window size, > 1)
        # r = exp(s_r) + 1  (acceptance sharpness, > 1)
        self.s_v = nn.Parameter(torch.zeros(num_heads))  # log(w - 1)
        self.s_r = nn.Parameter(torch.zeros(num_heads))  # log(r - 1)

        self.attn_dropout = nn.Dropout(dropout)
        self.tahn_norm = TanhNorm(eps=eps)

    # ------------------------------------------------------------------
    # Core screening components
    # ------------------------------------------------------------------

    def _trim_and_square(self, sim: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        """
        Compute absolute-threshold relevance scores (Trim-and-Square).

            alpha_ij = [max(1 - r * (1 - s_ij), 0)]^2

        Args:
            sim: (..., T_q, T_k) cosine similarities in [-1, 1].
            r:   (num_heads,) acceptance sharpness per head, shape broadcastable.

        Returns:
            alpha: (..., T_q, T_k) relevance scores in [0, 1].
        """
        # r shape: (H,) -> (1, H, 1, 1) for broadcast with (B, H, T_q, T_k)
        r = r.view(1, -1, 1, 1)
        return F.relu(1.0 - r * (1.0 - sim)).pow(2)

    def _softmask(self, T_q: int, T_k: int, w: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Compute the distance-aware cosine softmask for causal attention.

            m_ij(w) = 0.5 * (cos(pi * (j - i) / w) + 1)  if -w < j-i <= 0
                    = 0                                     otherwise

        Args:
            T_q: Query sequence length.
            T_k: Key sequence length.
            w:   (num_heads,) window sizes.
            device: Target device.

        Returns:
            mask: (num_heads, T_q, T_k)
        """
        # Relative position: j - i (negative = past, positive = future)
        i_idx = torch.arange(T_q, device=device).unsqueeze(1)  # (T_q, 1)
        j_idx = torch.arange(T_k, device=device).unsqueeze(0)  # (1, T_k)
        rel = (j_idx - i_idx).float()  # (T_q, T_k)

        # Per-head window: (H, 1, 1)
        w = w.view(-1, 1, 1)

        # Cosine mask: valid only for -w < rel <= 0
        cos_mask = 0.5 * (torch.cos(math.pi * rel.unsqueeze(0) / w) + 1.0)
        in_window = (rel.unsqueeze(0) > -w) & (rel.unsqueeze(0) <= 0)
        return cos_mask * in_window.float()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            query:            (B, T_q, d_model)
            key:              (B, T_k, d_model)
            value:            (B, T_k, d_model)
            key_padding_mask: (B, T_k) bool mask; True = position to ignore.

        Returns:
            out: (B, T_q, d_model)
        """
        B, T_q, _ = query.shape
        T_k = key.shape[1]
        H = self.num_heads
        D = self.head_dim

        # Project and reshape to (B, H, T, D)
        def project_and_split(x, proj):
            return proj(x).view(x.shape[0], x.shape[1], H, D).transpose(1, 2)

        q = project_and_split(query, self.q_proj)
        k = project_and_split(key,   self.k_proj)
        v = project_and_split(value,  self.v_proj)

        # Normalize to unit vectors
        q = F.normalize(q, dim=-1, eps=self.eps)
        k = F.normalize(k, dim=-1, eps=self.eps)
        v = F.normalize(v, dim=-1, eps=self.eps)

        # Cosine similarity: (B, H, T_q, T_k)
        sim = torch.matmul(q, k.transpose(-2, -1))

        # Learnable per-head parameters
        w = torch.exp(self.s_v) + 1.0  # (H,) window size
        r = torch.exp(self.s_r) + 1.0  # (H,) acceptance sharpness

        # Relevance scores via Trim-and-Square
        alpha = self._trim_and_square(sim, r)  # (B, H, T_q, T_k)

        # Distance-aware softmask
        if self.causal:
            softmask = self._softmask(T_q, T_k, w, query.device)  # (H, T_q, T_k)
            alpha = alpha * softmask.unsqueeze(0)
        else:
            # Non-causal: use symmetric softmask (|j - i|)
            i_idx = torch.arange(T_q, device=query.device).unsqueeze(1)
            j_idx = torch.arange(T_k, device=query.device).unsqueeze(0)
            rel = (j_idx - i_idx).float().abs()
            w_ = w.view(-1, 1, 1)
            cos_mask = 0.5 * (torch.cos(math.pi * rel.unsqueeze(0) / w_) + 1.0)
            in_window = (rel.unsqueeze(0) < w_).float()
            softmask = cos_mask * in_window
            alpha = alpha * softmask.unsqueeze(0)

        # Key padding mask
        if key_padding_mask is not None:
            # (B, 1, 1, T_k) -> broadcast
            alpha = alpha.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), 0.0)

        alpha = self.attn_dropout(alpha)

        # Aggregate values (no normalization)
        out = torch.matmul(alpha, v)  # (B, H, T_q, D)

        # TanhNorm per head
        out = self.tahn_norm(out)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T_q, self.d_model)
        return self.out_proj(out)
