# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable, with dev deps)
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test
pytest tests/test_attention.py::TestScreeningAttention::test_causal_masking

# Run the training example
python examples/train_tiny_lm.py
```

## Architecture

This library implements **Screening Attention** from arXiv:2604.01178 as a drop-in PyTorch replacement for softmax attention. The dependency graph is strictly layered:

```
norm.py (TanhNorm)
  └── attention.py (ScreeningAttention)
        └── layers.py (MultiscreenBlock, MultiscreenDecoderLayer)
              └── model.py (MultiscreenLM)
```

### Core mechanism (`attention.py`)

`ScreeningAttention` diverges from standard attention in four ways that must be kept consistent:

1. **Q/K/V are unit-normalized** — similarities are always in `[-1, 1]`. This is a precondition for Trim-and-Square to produce meaningful bounded scores.

2. **Trim-and-Square** (`_trim_and_square`) — replaces softmax with an absolute threshold: `α = relu(1 - r*(1 - sim))²`. The learnable `s_r` parameter (per-head) controls `r = exp(s_r) + 1 > 1`, enforced by the `+1` to prevent collapse.

3. **Cosine Softmask** (`_softmask`) — a causal windowed mask with `m_ij = 0.5*(cos(π*(j-i)/w)+1)` for `-w < j-i ≤ 0`. The learnable `s_v` controls `w = exp(s_v) + 1 > 1`. Non-causal mode uses `|j-i| < w` symmetrically.

4. **No normalization on aggregation** — `h = Σ α·v` with no softmax denominator. The zero vector legitimately encodes "no relevant context". TanhNorm (`norm.py`) is applied *after* aggregation per head to bound magnitude while preserving direction.

### `key_padding_mask` convention

`True` = **ignore** this position (padding). Applied by zeroing `alpha` at masked key positions before aggregation. This is the opposite of PyTorch's `src_key_padding_mask` in some older APIs — be careful when integrating with external code.

### `MultiscreenLM` positional encoding

Uses learned absolute positional embeddings (`pos_emb`), not RoPE or ALiBi. Sequences longer than `max_seq_len` will index out of bounds — truncate inputs before passing to the model.
