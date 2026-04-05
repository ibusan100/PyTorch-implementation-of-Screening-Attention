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

# Benchmarks
python benchmarks/bench_efficiency.py       # Latency/throughput/memory vs MHA and SDPA
python benchmarks/bench_sparsity.py         # Cosine sim distribution and sparsity analysis
python benchmarks/bench_wikitext2_v2.py     # WikiText-2 perplexity (GPT-2 BPE, 10K steps)
python benchmarks/bench_r_evolution.py      # r parameter and sparsity evolution over training
```

## Architecture

This library implements **Screening Attention** from arXiv:2604.01178 as a drop-in PyTorch replacement for softmax attention. The dependency graph is strictly layered:

```
norm.py (TanhNorm)
  └── attention.py (ScreeningAttention)          ← pure PyTorch, dense O(T²)
  └── kernels/screening_fwd.py                   ← Triton fused forward (WIP)
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

---

## Benchmark results (as of 2026-04-05)

Hardware: RTX 4060 Ti 8GB, CUDA 12.1, PyTorch 2.5.1, Windows 11

### Latency (torch.utils.benchmark, B=4, H=8, D=64)

| seq_len | Screening | nn.MHA | F.SDPA | Screening/SDPA |
|--------:|----------:|-------:|-------:|---------------:|
| 128 | 1.69ms | 0.54ms | 0.36ms | 4.7× |
| 256 | 1.77ms | 0.58ms | 0.35ms | 5.1× |
| 512 | 2.75ms | 0.92ms | 0.69ms | 4.0× |
| 1024 | 11.2ms | 2.65ms | 1.75ms | 6.4× |
| 2048 | 43.1ms | 8.4ms | 4.8ms | 8.9× |
| 4096 | **1956ms** | 30ms | 15ms | **131×** |

seq_len=4096 はピークメモリ 7,126MB が CUDA context + 他プロセス分と合わさり 8GB を超え、Windows WDDM shared memory にスピルしたことによる異常値。純粋な演算遅延ではない。

**なぜ遅いか（設計上の問題）：** PyTorch の `relu`/`pow` は dense。T×T の alpha 行列と softmask 行列を両方フルサイズでメモリ確保してから乗算している。95%+ がゼロなのにそのコストを払っている。FlashAttention 的なタイリングで T×T を出さなければこの問題は消える。→ Triton カーネルで解決予定。

### WikiText-2 perplexity (GPT-2 BPE tokenizer, d_model=128, H=4, L=4, ~7.3M params, 10K steps)

| model | test PPL | train time |
|-------|----------:|----------:|
| TransformerLM | 221.6 | 481s |
| MultiscreenLM | **191.3** | 608s |

MultiscreenLM が 14% 低い test PPL。学習時間は 26% 増（dense alpha 行列をバックプロパゲーション用に保持するため）。

### Sparsity (random unit vectors, head_dim=64)

cosine sim の std ≈ 0.125、P(sim > 0.5) = 0.001%。r=2（初期値）で 100% sparse、r=1.5 でも 99.9% sparse。

### r evolution (10K-step training)

平均 r: 2.0 → 1.93、sparsity: 100% → ~95%。attention はほとんど「開かない」。モデルはキーの 5% 程度に選択的に attend することを学習する。

---

## Triton kernel plan (`multiscreen/kernels/`) — WIP

**目標：** T×T の alpha 行列をメモリに出さずに forward を fuse する。

```
Q (B,H,T,D) --normalize--> q_hat
K (B,H,T,D) --normalize--> k_hat
                 ↓ dot product per tile
              sim (tile)
                 ↓ trim_and_square + softmask (fused)
              alpha (tile, never materialized)
                 ↓ accumulate into
V (B,H,T,D) --normalize--> v_hat
              out_tile (B,H,T,D)
                 ↓ TanhNorm (separate kernel or fused)
```

- `kernels/screening_fwd.py` : causal fused forward
- `kernels/screening_bwd.py` : backward (dV, dK, dQ)
- `attention_triton.py` : PyTorch autograd Function でラップし、既存 `ScreeningAttention` の drop-in 差し替えとして提供

**制約：** triton-windows 3.6.0 (CUDA 12.1)。Windows では `triton.autotune` の一部制約あり。
