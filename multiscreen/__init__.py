"""
multiscreen: PyTorch implementation of the Screening attention mechanism.

Based on "Screening Is Enough" (arXiv:2604.01178) by Ken M. Nakanishi.

The screening mechanism replaces softmax attention with independent, absolute
relevance evaluation via a learnable threshold, enabling:
- ~40% fewer parameters than Transformer baselines at comparable loss
- Stable training at higher learning rates
- Reduced inference latency at long contexts
"""

from .attention import ScreeningAttention
from .layers import MultiscreenBlock, MultiscreenDecoderLayer
from .model import MultiscreenLM
from .norm import TanhNorm

__all__ = [
    "ScreeningAttention",
    "MultiscreenBlock",
    "MultiscreenDecoderLayer",
    "MultiscreenLM",
    "TanhNorm",
]

__version__ = "0.1.0"
