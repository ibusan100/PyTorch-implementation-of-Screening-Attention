"""
Benchmark A: Computational efficiency
Compares ScreeningAttention vs nn.MultiheadAttention vs F.scaled_dot_product_attention
using torch.utils.benchmark (PyTorch's official benchmarking tool).

Metrics:
  - Latency (ms) across sequence lengths
  - Throughput (tokens/sec)
  - Peak GPU memory (MB)
  - Parameter count
"""

import json
import math
import gc
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.benchmark as benchmark

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiscreen import ScreeningAttention


# ---------------------------------------------------------------------------
# Baseline: standard nn.MultiheadAttention wrapper (same interface)
# ---------------------------------------------------------------------------
class StdMHA(nn.Module):
    def __init__(self, d_model, num_heads, causal=True):
        super().__init__()
        self.causal = causal
        self.mha = nn.MultiheadAttention(d_model, num_heads, bias=False, batch_first=True)

    def forward(self, query, key, value, key_padding_mask=None):
        T = query.shape[1]
        attn_mask = None
        if self.causal:
            attn_mask = torch.triu(
                torch.ones(T, T, device=query.device, dtype=torch.bool), diagonal=1
            )
        out, _ = self.mha(query, key, value,
                          attn_mask=attn_mask,
                          key_padding_mask=key_padding_mask,
                          need_weights=False)
        return out


# ---------------------------------------------------------------------------
# Baseline: F.scaled_dot_product_attention wrapper
# ---------------------------------------------------------------------------
class SDPA(nn.Module):
    def __init__(self, d_model, num_heads, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.causal = causal
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, key_padding_mask=None):
        B, T, C = query.shape
        H, D = self.num_heads, self.head_dim
        def split(x, proj):
            return proj(x).view(B, -1, H, D).transpose(1, 2)
        q, k, v = split(query, self.q), split(key, self.k), split(value, self.v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out(out.transpose(1, 2).contiguous().view(B, -1, C))


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D_MODEL    = 512
NUM_HEADS  = 8
BATCH_SIZE = 4
SEQ_LENS   = [128, 256, 512, 1024, 2048, 4096]
N_WARMUP   = 10
N_REPEAT   = 50


@dataclass
class Result:
    model: str
    seq_len: int
    latency_ms: float
    latency_iqr: float
    throughput_tok_per_sec: float
    peak_memory_mb: float
    param_count: int


def measure(model, seq_len, name):
    model = model.to(DEVICE).eval()
    x = torch.randn(BATCH_SIZE, seq_len, D_MODEL, device=DEVICE)

    # --- latency via torch.utils.benchmark ---
    t = benchmark.Timer(
        stmt="model(x, x, x)",
        globals={"model": model, "x": x},
        num_threads=1,
        label=name,
        sub_label=f"T={seq_len}",
    )
    m = t.blocked_autorange(min_run_time=2.0)
    latency_ms  = m.median * 1e3
    latency_iqr = (m.iqr or 0.0) * 1e3
    throughput  = (BATCH_SIZE * seq_len) / m.median

    # --- peak GPU memory ---
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats(DEVICE)
        with torch.no_grad():
            model(x, x, x)
        peak_mb = torch.cuda.max_memory_allocated(DEVICE) / 1e6
    else:
        peak_mb = float("nan")

    param_count = sum(p.numel() for p in model.parameters())

    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return Result(
        model=name,
        seq_len=seq_len,
        latency_ms=round(latency_ms, 4),
        latency_iqr=round(latency_iqr, 4),
        throughput_tok_per_sec=round(throughput, 1),
        peak_memory_mb=round(peak_mb, 2),
        param_count=param_count,
    )


def main():
    print(f"Device : {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU    : {torch.cuda.get_device_name(DEVICE)}")
    print(f"d_model={D_MODEL}, heads={NUM_HEADS}, batch={BATCH_SIZE}")
    print()

    results = []

    for seq_len in SEQ_LENS:
        print(f"=== seq_len={seq_len} ===")

        for name, cls, kwargs in [
            ("ScreeningAttention", ScreeningAttention, {"d_model": D_MODEL, "num_heads": NUM_HEADS, "causal": True}),
            ("nn.MultiheadAttention", StdMHA,           {"d_model": D_MODEL, "num_heads": NUM_HEADS, "causal": True}),
            ("F.scaled_dot_product_attention", SDPA,    {"d_model": D_MODEL, "num_heads": NUM_HEADS, "causal": True}),
        ]:
            try:
                model = cls(**kwargs)
                r = measure(model, seq_len, name)
                results.append(r)
                print(f"  {name:<38} latency={r.latency_ms:8.3f}ms  "
                      f"throughput={r.throughput_tok_per_sec:>10,.0f} tok/s  "
                      f"mem={r.peak_memory_mb:7.1f}MB  params={r.param_count:,}")
            except Exception as e:
                print(f"  {name}: ERROR — {e}")
        print()

    # Save JSON
    out_path = os.path.join(os.path.dirname(__file__), "results_efficiency.json")
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"Results saved to {out_path}")

    # Summary table
    print("\n=== Speedup vs nn.MultiheadAttention (latency) ===")
    print(f"{'seq_len':>8}  {'Screening':>12}  {'SDPA':>12}")
    for seq_len in SEQ_LENS:
        def lat(name):
            for r in results:
                if r.seq_len == seq_len and r.model == name:
                    return r.latency_ms
            return None
        base = lat("nn.MultiheadAttention")
        scr  = lat("ScreeningAttention")
        sdpa = lat("F.scaled_dot_product_attention")
        if base and scr and sdpa:
            print(f"{seq_len:>8}  {base/scr:>11.2f}x  {base/sdpa:>11.2f}x")


if __name__ == "__main__":
    main()
