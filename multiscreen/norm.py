"""TanhNorm: direction-preserving, magnitude-bounding normalization."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhNorm(nn.Module):
    """
    TanhNorm normalizes a vector by preserving direction while bounding magnitude.

        TanhNorm(x) = (tanh(||x||) / ||x||) * x

    This ensures the output magnitude is in (0, 1) regardless of input scale,
    while keeping the direction unchanged. Used to normalize aggregated values
    in the screening attention output.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(dim=-1, keepdim=True).clamp(min=self.eps)
        return (torch.tanh(norm) / norm) * x
