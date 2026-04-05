"""
Benchmark: ScreeningAttentionTriton vs ScreeningAttention (PyTorch dense)

Measures latency and peak memory across seq_len values.
Results saved to benchmarks/results_triton.json.
"""

import sys
import json
import torch
import torch.utils.benchmark as bench

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
sys.stdout.reconfigure(line_buffering=True)

from multiscreen.attention import ScreeningAttention
from multiscreen.attention_triton import ScreeningAttentionTriton

B, H, D_MODEL = 4, 8, 512
# s_r=-2.0 -> r=exp(-2)+1~1.135: produces non-trivial (non-zero) alpha values
S_R_INIT = -2.0

print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"B={B}, H={H}, d_model={D_MODEL}, s_r={S_R_INIT} (r~1.135), causal=True")
print()

attn_pt = ScreeningAttention(D_MODEL, num_heads=H, causal=True).cuda().eval()
attn_tri = ScreeningAttentionTriton(D_MODEL, num_heads=H, causal=True).cuda().eval()
with torch.no_grad():
    attn_tri.load_state_dict(attn_pt.state_dict())
    attn_pt.s_r.fill_(S_R_INIT)
    attn_tri.s_r.fill_(S_R_INIT)

SEQ_LENS = [128, 256, 512, 1024, 2048, 4096]
results = []

print(f"{'seq_len':>8} {'PyTorch(ms)':>12} {'Triton(ms)':>11} {'speedup':>8} "
      f"{'memPT(MB)':>10} {'memTri(MB)':>11} {'memRedux':>9}")
print("-" * 75)

for T in SEQ_LENS:
    x = torch.randn(B, T, D_MODEL, device="cuda")

    t_pt = bench.Timer(
        "f(x, x, x)", globals={"f": attn_pt, "x": x}
    ).blocked_autorange(min_run_time=1)

    t_tri = bench.Timer(
        "f(x, x, x)", globals={"f": attn_tri, "x": x}
    ).blocked_autorange(min_run_time=1)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = attn_tri(x, x, x)
    mem_tri = torch.cuda.max_memory_allocated() / 1e6

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = attn_pt(x, x, x)
    mem_pt = torch.cuda.max_memory_allocated() / 1e6

    speedup = t_pt.median / t_tri.median
    mem_redux = mem_pt / mem_tri

    row = {
        "seq_len": T,
        "pytorch_ms": round(t_pt.median * 1e3, 3),
        "pytorch_iqr": round(t_pt.iqr * 1e3, 4),
        "triton_ms": round(t_tri.median * 1e3, 3),
        "triton_iqr": round(t_tri.iqr * 1e3, 4),
        "speedup": round(speedup, 2),
        "mem_pytorch_mb": round(mem_pt, 1),
        "mem_triton_mb": round(mem_tri, 1),
        "mem_reduction": round(mem_redux, 2),
    }
    results.append(row)

    print(f"{T:>8} {row['pytorch_ms']:>12.3f} {row['triton_ms']:>11.3f} "
          f"{speedup:>8.2f}x {mem_pt:>10.1f} {mem_tri:>11.1f} {mem_redux:>8.2f}x")

out_path = __import__("pathlib").Path(__file__).parent / "results_triton.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out_path}")
