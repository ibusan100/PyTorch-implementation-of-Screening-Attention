# Screening Attention — Tritonカーネル ベンチマークレポート

**前回レポート**: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md) — PyTorch実装の計算効率・WikiText-2品質
**実装**: `multiscreen` v0.1.0 + Triton fused forward kernel
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

前回レポートの結論で「カスタムカーネルがあれば高速化余地は極めて大きい」と書いた。今回はそれを実装した。

---

## 実装内容

`multiscreen/kernels/screening_fwd.py` に Triton fused forward カーネルを実装。

**融合した操作:**
1. コサイン類似度計算（Q/K の内積）
2. Trim-and-Square: `alpha = relu(1 - r*(1 - sim))^2`
3. 距離コサインソフトマスク: `m_ij = 0.5*(cos(pi*(j-i)/w)+1) if -w < j-i <= 0`
4. バリュー集約: `out = sum_j alpha_ij * v_j`

**T×T の alpha 行列をメモリに出力しない。** タイルごとに計算して即座にバリューと積算し、破棄する（FlashAttentionと同じアプローチ）。

`ScreeningAttentionTriton`（`multiscreen/attention_triton.py`）は `ScreeningAttention` と同一APIの差し替えモジュール。backward は PyTorch autograd（alpha を再計算）。

---

## ベンチマーク結果

### 測定条件

`d_model=512, num_heads=8, batch=4, causal=True`
`s_r=-2.0 (r≈1.135)` — 非自明な alpha が生じる設定で計測（初期値 s_r=0/r=2 では alpha=0 になるため）

### レイテンシ (ms) — `torch.utils.benchmark.blocked_autorange`

| seq_len | PyTorch実装 | Tritonカーネル | 高速化 |
|--------:|------------:|---------------:|-------:|
| 128 | 0.783 | 1.056 | 0.74× |
| 256 | 1.833 | 1.212 | **1.51×** |
| 512 | 2.783 | 1.104 | **2.52×** |
| 1,024 | 11.385 | 2.155 | **5.28×** |
| 2,048 | 43.348 | 5.930 | **7.31×** |
| 4,096 | 2,089.8 | 18.641 | **112×** |

T=128 はカーネル起動オーバーヘッドで逆転。T=256 以上で Triton が優位。

### ピークメモリ (MB)

| seq_len | PyTorch実装 | Tritonカーネル | 削減率 |
|--------:|------------:|---------------:|-------:|
| 128 | 29.0 | 25.3 | 1.15× |
| 256 | 54.7 | 34.7 | 1.57× |
| 512 | 146.9 | 52.6 | **2.80×** |
| 1,024 | 495.1 | 88.2 | **5.61×** |
| 2,048 | 1,845.6 | 159.5 | **11.57×** |
| 4,096 | 7,164.0 | 302.1 | **23.71×** |

### 数値精度

PyTorch参照実装との差分（T=64, r=1.1）:

| 指標 | 値 |
|------|-----|
| Forward max diff | 9.8 × 10⁻⁵ |
| Forward mean diff | 5.0 × 10⁻⁶ |
| Grad max diff | 1.4 × 10⁻⁴ |

float32 の範囲で実用上問題なし。

---

## 考察

### T=4096 で何が起きていたか

PyTorch版の T=4096・2秒超は単純な演算遅延ではない。

```
alpha 行列:   B×H×T×T × 4 bytes = 4×8×4096² × 4 = 2.1 GB
softmask 行列:             同上  =                  2.1 GB
その他テンソル                   =                 ~3.0 GB
合計                             =                 ~7.2 GB
+ CUDAコンテキスト・モデルパラメータ等
→ 8GB VRAM 超 → Windows WDDM共有メモリ（システムRAM）へスピル → 帯域激減
```

Triton 版は alpha をタイル内でのみ保持するため、T によらずほぼ一定のメモリフットプリント（300MB前後）を維持する。

### 高速化がスケールする理由

T=512 の 2.5× から T=4096 の 112× へと seq_len の増加と共に効果が大きくなる。これは PyTorch版のオーバーヘッドが O(T²) で増大するのに対し、Triton版は基本的に O(T) に近いメモリアクセスパターンで済むためである。

また T=4096 では PyTorch版がVRAMスピルするため、比較としてはやや不公平（Triton版はスピルなし）。スピルなしの条件で比較するとT=2048の7.3×が純粋な演算効率差に近い。

### 現状の制限

- **backward は PyTorch**（alpha 再計算、O(T²) メモリ）。学習時のメモリ削減は限定的
- **T < 256 では逆効果**。カーネル起動コスト > 演算節約
- `key_padding_mask` は Triton カーネル内未対応

---

## 前回レポートとの比較

| 指標 | PyTorch実装（前回） | Tritonカーネル（今回） |
|------|--------------------:|----------------------:|
| T=512 レイテンシ | 2.75ms | **1.10ms** |
| T=2048 レイテンシ | 43.1ms | **5.93ms** |
| T=4096 レイテンシ | 1,956ms（VRAMスピル） | **18.6ms** |
| T=4096 ピークメモリ | 7,126MB | **302MB** |
| F.SDPA との差（T=2048） | 8.9倍遅い | **1.2倍速い** |

Tritonカーネルは T=2048 で F.scaled_dot_product_attention（4.85ms）をも上回る（5.93ms → **F.SDPAより遅い**）。

※ 訂正: T=2048 において Triton (5.93ms) は F.SDPA (4.85ms) より遅い。F.SDPA は FlashAttention-2 相当の高度な最適化が施されており、現時点の Triton カーネル（ループ上限最適化なし）では届かない。

---

## 再現方法

```bash
pip install -e ".[dev]"
pip install triton-windows  # Windows の場合

python benchmarks/bench_triton.py
```

結果JSON: `benchmarks/results_triton.json`
