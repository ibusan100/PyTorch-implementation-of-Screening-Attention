# Screening Attention — Tritonカーネル ベンチマークレポート

**前回レポート**: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — PyTorch実装の計算効率・WikiText-2品質
**実装**: `multiscreen` v0.1.0 + Triton fused kernels (fwd + bwd)
**論文**: "Screening Is Enough" (arXiv:[2604.01178](https://arxiv.org/abs/2604.01178)) — Ken M. Nakanishi
**実行環境**: NVIDIA GeForce RTX 4060 Ti (8GB VRAM) / PyTorch 2.5.1+cu121 / Python 3.11 / triton-windows 3.6.0

---

## 背景

前回レポートでは、PyTorchのナイーブな実装がnn.MHAより**3〜66倍遅く**、メモリも**5〜21倍多く消費**するという結果を報告した。

原因は明確だった：

```
PyTorch実装の問題
  ├── T×T の alpha 行列を密に確保（95%以上がゼロなのに）
  ├── T×T の softmask 行列も別途確保
  └── T=4096 で 7.1GB → VRAMあふれ → システムRAMスピル → 2秒超
```

前回の結論：「カスタムカーネルがあれば高速化余地は極めて大きい」。今回はそれを実装した。

---

## 実装内容

### カーネル構成

| ファイル | 内容 |
|---------|------|
| `kernels/screening_fwd.py` | Triton fused forward kernel |
| `kernels/screening_bwd.py` | Triton fused backward kernels (dV, dQ+dK) |
| `attention_triton.py` | `ScreeningAttention` と同一 API の差し替えモジュール |

### 最適化フェーズ

**Phase 1-a: Causal loop bound**
causal 時、将来タイルをループから完全に除外。対角タイル（tile_k == tile_q）とそれ以前のタイルのみ処理。

**Phase 1-b: Triton backward kernel**
dV・dQ・dK をすべて Triton で実装。backward でも T×T alpha 行列を生成しない。

**Phase 2-a: Window-based block pruning**
ソフトマスクウィンドウ w に基づき、ウィンドウ外のタイルをループ開始前に除外。
`min_tile_k = max(0, (q_start - w) // BLOCK_T)` でループ下限を引き上げる。

**Phase 2-b: Trim-and-Square block skip**
K をロードして sim を計算後、タイル内の `max_sim` を Trim-and-Square 閾値 `1 - 1/r` と比較。
全ペアが閾値以下なら V のロードと acc 更新を丸ごとスキップ。

---

## ベンチマーク結果

### 測定条件

`d_model=512, num_heads=8, batch=4, causal=True, r≈1.93 (post-training相当)`

### レイテンシ累積改善 (ms)

| seq_len | PyTorch実装 | Phase1-a | Phase1-b+2-a | Phase2-b(最終) | vs PyTorch |
|--------:|------------:|---------:|-------------:|---------------:|-----------:|
| 512 | 2.783 | 1.221 | 0.737 | **0.681** | **4.1×** |
| 1,024 | 11.385 | 1.808 | 1.475 | **1.389** | **8.2×** |
| 2,048 | 43.348 | 4.391 | 2.970 | **2.825** | **15.3×** |
| 4,096 | 2,089.8 | 12.534 | 6.202 | **5.858** | **357×** |

### ピークメモリ — forward のみ (eval)

| seq_len | PyTorch実装 | Triton最終 | 削減率 |
|--------:|------------:|-----------:|-------:|
| 512 | 146.9 MB | 52.6 MB | **2.8×** |
| 1,024 | 495.1 MB | 88.2 MB | **5.6×** |
| 2,048 | 1,845.6 MB | 159.5 MB | **11.6×** |
| 4,096 | 7,164.0 MB | 302.1 MB | **23.7×** |

### ピークメモリ — backward込み (学習時)

| seq_len | PyTorch実装 | Triton最終 | 削減率 |
|--------:|------------:|-----------:|-------:|
| 512 | 296.4 MB | 98.2 MB | **3.0×** |
| 1,024 | 1,045.4 MB | 168.8 MB | **6.2×** |
| 2,048 | 3,944.5 MB | 303.8 MB | **13.0×** |

### 数値精度（PyTorch参照実装との差分、T=64, r=1.1）

| 指標 | 値 |
|------|-----|
| Forward max diff | 9.8 × 10⁻⁵ |
| Grad max diff | 1.4 × 10⁻⁴ |

---

## 考察

### 各フェーズの効果

| フェーズ | T=2048 での寄与 |
|---------|----------------|
| Phase 1-a causal loop | 43ms → 4.4ms（**9.9×**、最大の改善） |
| Phase 1-b backward Triton | forward 速度に影響なし、学習メモリ 13× 削減 |
| Phase 2-a window pruning | 4.4ms → 2.97ms（**1.48×**） |
| Phase 2-b Trim-and-Square skip | 2.97ms → 2.83ms（**1.05×**） |

Phase 1-a（causal ループ上限）が最も効いた。将来タイルを計算しないだけで約10× 改善したのは、元の実装がそれだけ無駄な計算をしていたことを示す。

Phase 2-b（Trim-and-Square skip）の効果が限定的なのは、ブロック内に 1 ペアでも `sim > threshold` があればスキップできないため。ランダム入力では BLOCK_T=64 ブロック内に該当ペアが含まれやすく、スキップ率が低かった。実際の学習済みモデルでは Q/K が align しているためさらに低くなる可能性がある。

### T=4096 の 357× について

PyTorch版は T=4096 で VRAM（8GB）をあふれてシステム RAM にスピルしており、2 秒超の異常値。純粋な演算効率差は T=2048 の **15×** が実態に近い。

### F.SDPA との比較

| seq_len | F.SDPA | Triton最終 |
|--------:|-------:|-----------:|
| 512 | 0.69ms | 0.68ms ≈ 同等 |
| 1,024 | 1.75ms | 1.39ms **1.3× 速い** |
| 2,048 | 4.85ms | 2.83ms **1.7× 速い** |
| 4,096 | 14.84ms | 5.86ms **2.5× 速い** |

T=512 以上で F.SDPA（FlashAttention-2 相当）を上回った。これは window pruning の効果によるもの — FlashAttention は全タイルを処理するが、Screening Attention のウィンドウ制約で処理タイル数が大幅に減るため。

---

## 論文の主張との照合

| 主張 | 論文 | 本実装（最終） |
|------|------|--------------|
| 計算速度 | 最大 3.2× 高速化 | F.SDPA 比 最大 **2.5×**（T=4096） |
| モデル品質 | ~40% 削減で同等 loss | 同規模で **14% 改善**（test PPL） |
| メモリ | 削減を示唆 | forward **23.7×**、学習時 **13×** 削減 |

スピードについては論文の主張（3.2×）に届いていないが、これは論文がより大きい T（100K コンテキスト）での数字であるためと考えられる。方向性としては一致している。

---

## 再現方法

```bash
pip install -e ".[dev]"
pip install triton-windows  # Windows の場合

# Triton vs PyTorch 効率ベンチマーク
python benchmarks/bench_triton.py
```

結果JSON: `benchmarks/results_triton.json`
