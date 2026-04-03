"""
Multiscreen Transformer blocks.

Replaces standard Multi-Head Attention with ScreeningAttention in
standard pre-norm Transformer decoder/encoder layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ScreeningAttention


class MultiscreenBlock(nn.Module):
    """
    Pre-norm Transformer block with Screening Attention.

    Architecture:
        x = x + ScreeningAttn(RMSNorm(x))
        x = x + FFN(RMSNorm(x))

    Args:
        d_model:     Model dimension.
        num_heads:   Number of attention heads.
        ffn_dim:     Feed-forward hidden dimension (default: 4 * d_model).
        dropout:     Dropout probability.
        causal:      Causal (decoder) or bidirectional (encoder) attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
        causal: bool = True,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)

        self.attn = ScreeningAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=causal,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                (B, T, d_model)
            key_padding_mask: (B, T) bool — True = ignore this position.

        Returns:
            x: (B, T, d_model)
        """
        normed = self.norm1(x)
        x = x + self.attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class MultiscreenDecoderLayer(nn.Module):
    """
    Pre-norm Transformer decoder layer with cross-attention support.

    Architecture:
        x = x + SelfScreeningAttn(RMSNorm(x))
        x = x + CrossScreeningAttn(RMSNorm(x), memory)
        x = x + FFN(RMSNorm(x))

    Args:
        d_model:       Model dimension.
        num_heads:     Number of attention heads.
        ffn_dim:       Feed-forward hidden dimension.
        dropout:       Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model

        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.norm3 = nn.RMSNorm(d_model)

        self.self_attn = ScreeningAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=True,
        )
        self.cross_attn = ScreeningAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            causal=False,
        )

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model, bias=False),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.self_attn(normed, normed, normed, key_padding_mask=tgt_key_padding_mask)

        normed = self.norm2(x)
        x = x + self.cross_attn(normed, memory, memory, key_padding_mask=memory_key_padding_mask)

        x = x + self.ffn(self.norm3(x))
        return x
