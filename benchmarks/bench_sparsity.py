"""
Benchmark A (補足): Screening Attention のスパース性分析

初期化時と学習後（ランダムウェイト）でどの程度のキーがスクリーンされるかを計測。
スパース性が高いほど、カスタムCUDAカーネルでの高速化余地が大きい。
"""

import json, os, sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiscreen import ScreeningAttention

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL   = 512
NUM_HEADS = 8
BATCH     = 4
SEQ_LENS  = [128, 256, 512, 1024, 2048]
N_TRIALS  = 10


def measure_sparsity(seq_len, r_value=None):
    """
    alpha の中でゼロになっている割合 = スクリーンされたキーの割合
    r_value: None=学習初期(s_r=0 → r=2), または任意の r を強制設定
    """
    model = ScreeningAttention(D_MODEL, NUM_HEADS, causal=True).to(DEVICE).eval()

    if r_value is not None:
        # r = exp(s_r) + 1  =>  s_r = log(r - 1)
        import math
        with torch.no_grad():
            model.s_r.fill_(math.log(r_value - 1))

    sparsities = []
    with torch.no_grad():
        for _ in range(N_TRIALS):
            x = torch.randn(BATCH, seq_len, D_MODEL, device=DEVICE)

            # 内部のalphaを取り出すために forward を手動再現
            q = F.normalize(model.q_proj(x).view(BATCH, seq_len, NUM_HEADS, -1).transpose(1,2), dim=-1)
            k = F.normalize(model.k_proj(x).view(BATCH, seq_len, NUM_HEADS, -1).transpose(1,2), dim=-1)

            sim = torch.matmul(q, k.transpose(-2, -1))
            w = torch.exp(model.s_v) + 1.0
            r = torch.exp(model.s_r) + 1.0
            alpha = F.relu(1.0 - r.view(1,-1,1,1) * (1.0 - sim)).pow(2)

            # Causal softmask
            import math as _math
            i_idx = torch.arange(seq_len, device=DEVICE).unsqueeze(1)
            j_idx = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
            rel   = (j_idx - i_idx).float()
            w_    = w.view(-1,1,1)
            cos_mask   = 0.5*(torch.cos(_math.pi * rel.unsqueeze(0) / w_) + 1.0)
            in_window  = ((rel.unsqueeze(0) > -w_) & (rel.unsqueeze(0) <= 0)).float()
            alpha = alpha * (cos_mask * in_window).unsqueeze(0)

            sparsity = (alpha == 0).float().mean().item()
            sparsities.append(sparsity)

    return sum(sparsities) / len(sparsities)


def main():
    print("=== Screening Sparsity Analysis ===")
    print(f"Device: {DEVICE}, d={D_MODEL}, H={NUM_HEADS}, B={BATCH}\n")

    results = {}

    # r=2 (初期値: s_r=0), r=4, r=8
    r_configs = {"r=2 (init)": 2.0, "r=4": 4.0, "r=8": 8.0}

    header = f"{'seq_len':>8}" + "".join(f"  {label:>16}" for label in r_configs)
    print(header)
    print("-" * len(header))

    for seq_len in SEQ_LENS:
        row = f"{seq_len:>8}"
        results[seq_len] = {}
        for label, r_val in r_configs.items():
            s = measure_sparsity(seq_len, r_value=r_val)
            results[seq_len][label] = round(s * 100, 1)
            row += f"  {s*100:>15.1f}%"
        print(row)

    out = os.path.join(os.path.dirname(__file__), "results_sparsity.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    print("\nNote: higher sparsity → more potential speedup from sparse CUDA kernels")


if __name__ == "__main__":
    main()
