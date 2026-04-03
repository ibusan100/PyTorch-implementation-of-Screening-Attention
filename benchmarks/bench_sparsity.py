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


def measure_sim_stats(seq_len):
    """初期化時のコサイン類似度分布を計測"""
    model = ScreeningAttention(D_MODEL, NUM_HEADS, causal=True).to(DEVICE).eval()
    with torch.no_grad():
        x = torch.randn(BATCH, seq_len, D_MODEL, device=DEVICE)
        q = F.normalize(model.q_proj(x).view(BATCH, seq_len, NUM_HEADS, -1).transpose(1,2), dim=-1)
        k = F.normalize(model.k_proj(x).view(BATCH, seq_len, NUM_HEADS, -1).transpose(1,2), dim=-1)
        sim = torch.matmul(q, k.transpose(-2, -1))
        return {
            "mean": round(sim.mean().item(), 4),
            "std":  round(sim.std().item(), 4),
            "max":  round(sim.max().item(), 4),
            "frac_gt_0.5": round((sim > 0.5).float().mean().item(), 6),
        }


def main():
    print("=== Screening Sparsity Analysis ===")
    print(f"Device: {DEVICE}, d={D_MODEL}, H={NUM_HEADS}, B={BATCH}\n")

    # --- コサイン類似度の分布 ---
    print("--- Cosine similarity distribution at init (seq_len=512) ---")
    stats = measure_sim_stats(512)
    print(f"  mean={stats['mean']}  std={stats['std']}  max={stats['max']}")
    print(f"  P(sim > 0.5) = {stats['frac_gt_0.5']:.6f}")
    print(f"  -> r=2 threshold (sim > 0.5): almost no pairs exceed it, so init sparsity ~100%\n")

    results = {"sim_stats_seq512": stats}

    # --- Trim-and-Square + softmask 統合スパース率 ---
    r_configs = {"r=2 (init)": 2.0, "r=1.5": 1.5, "r=1.2": 1.2}

    header = f"{'seq_len':>8}" + "".join(f"  {label:>16}" for label in r_configs)
    print("--- Total sparsity (Trim-and-Square × softmask) ---")
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

    print(f"\n--- Softmask-only zero fraction per head (seq=512, init w linspace) ---")
    model = ScreeningAttention(D_MODEL, NUM_HEADS, causal=True).to(DEVICE).eval()
    w = torch.exp(model.s_v) + 1.0
    import math as _math
    i_idx = torch.arange(512, device=DEVICE).unsqueeze(1)
    j_idx = torch.arange(512, device=DEVICE).unsqueeze(0)
    rel   = (j_idx - i_idx).float()
    w_    = w.view(-1,1,1)
    cos_m = 0.5*(torch.cos(_math.pi * rel.unsqueeze(0) / w_) + 1.0)
    in_w  = ((rel.unsqueeze(0) > -w_) & (rel.unsqueeze(0) <= 0)).float()
    softmask = cos_m * in_w
    head_ws = w.tolist()
    for h in range(NUM_HEADS):
        frac = (softmask[h] == 0).float().mean().item()
        print(f"  head {h} (w={head_ws[h]:.1f}): mask_zero={frac*100:.1f}%")

    out = os.path.join(os.path.dirname(__file__), "results_sparsity.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")
    print("\nNote: 100% sparsity at init is by design, not a bug.")
    print("      During training, s_r decreases (r->1), lowering the threshold so attention opens up.")


if __name__ == "__main__":
    main()
