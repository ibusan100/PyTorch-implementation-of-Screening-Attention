"""
r値とAttention sparsityの学習推移を追跡・可視化する。

生成物:
  benchmarks/r_evolution.png  -- r値 (per head, per layer) の学習曲線
  benchmarks/attention_maps.png -- step 0/500/2000/5000 のalpha行列ヒートマップ
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset
from transformers import GPT2TokenizerFast
from torch.utils.data import Dataset, DataLoader

from multiscreen import MultiscreenLM
from multiscreen.attention import ScreeningAttention

DEVICE    = torch.device("cuda")
SEQ_LEN   = 128
BATCH     = 8
D_MODEL   = 128
NUM_HEADS = 4
NUM_LAYERS= 4
MAX_STEPS = 5000
LOG_EVERY = 250
SNAP_STEPS= {0, 500, 2000, 5000}  # attention map snapshots

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, t, s): self.t=t; self.s=s
    def __len__(self): return max(0,len(self.t)-self.s)
    def __getitem__(self,i): c=self.t[i:i+self.s+1]; return c[:-1],c[1:]

print("Loading data...")
ds  = load_dataset("Salesforce/wikitext","wikitext-2-raw-v1")
tok = GPT2TokenizerFast.from_pretrained("gpt2")
full= " ".join(t for t in ds["train"]["text"] if t.strip())
ids = torch.tensor(tok.encode(full), dtype=torch.long)
loader = DataLoader(TokenDataset(ids,SEQ_LEN),batch_size=BATCH,shuffle=True,
                    drop_last=True,num_workers=0)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MultiscreenLM(vocab_size=tok.vocab_size, d_model=D_MODEL,
                      num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
                      ffn_dim=512, max_seq_len=SEQ_LEN, dropout=0.1).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_r_values():
    """全layerの全headのr値を返す: shape (num_layers, num_heads)"""
    rs = []
    for block in model.blocks:
        r = (torch.exp(block.attn.s_r) + 1.0).detach().cpu().tolist()
        rs.append(r)
    return rs  # list[list[float]]

def get_attention_map(x):
    """最初のlayerのhead=0のalphaマップを返す (T, T) numpy array"""
    model.eval()
    with torch.no_grad():
        attn = model.blocks[0].attn
        normed = model.blocks[0].norm1(model.token_emb(x) + model.pos_emb(
            torch.arange(x.shape[1], device=DEVICE).unsqueeze(0)))
        q = F.normalize(attn.q_proj(normed).view(x.shape[0],-1,NUM_HEADS,D_MODEL//NUM_HEADS).transpose(1,2), dim=-1)
        k = F.normalize(attn.k_proj(normed).view(x.shape[0],-1,NUM_HEADS,D_MODEL//NUM_HEADS).transpose(1,2), dim=-1)
        sim = torch.matmul(q, k.transpose(-2,-1))
        r = (torch.exp(attn.s_r)+1.0).view(1,-1,1,1)
        alpha = F.relu(1.0 - r*(1.0-sim)).pow(2)
        w = (torch.exp(attn.s_v)+1.0)
        T = x.shape[1]
        i_ = torch.arange(T,device=DEVICE).unsqueeze(1)
        j_ = torch.arange(T,device=DEVICE).unsqueeze(0)
        rel= (j_-i_).float()
        w_ = w.view(-1,1,1)
        cm = 0.5*(torch.cos(math.pi*rel.unsqueeze(0)/w_)+1.0)
        iw = ((rel.unsqueeze(0)>-w_)&(rel.unsqueeze(0)<=0)).float()
        alpha = alpha * (cm*iw).unsqueeze(0)
        # head=0, batch=0
        return alpha[0,0].cpu().numpy()
    model.train()

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
log_steps   = []
log_r_mean  = []   # mean r across all heads/layers
log_sparsity= []
snap_maps   = {}

x_fixed = next(iter(loader))[0][:1].to(DEVICE)  # fixed input for snapshots

step = 0
loader_iter = iter(loader)

print("Training...")
while step < MAX_STEPS:
    try: x,y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        x,y = next(loader_iter)

    if step in SNAP_STEPS:
        snap_maps[step] = get_attention_map(x_fixed)

    loss = model(x.to(DEVICE), labels=y.to(DEVICE))["loss"]
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); step += 1

    if step % LOG_EVERY == 0:
        rs = get_r_values()
        flat_r = [r for layer in rs for r in layer]
        mean_r = sum(flat_r)/len(flat_r)

        # sparsity on a random batch
        model.eval()
        with torch.no_grad():
            attn = model.blocks[0].attn
            xb = x.to(DEVICE)
            normed = model.blocks[0].norm1(
                model.token_emb(xb) + model.pos_emb(torch.arange(xb.shape[1],device=DEVICE).unsqueeze(0)))
            q = F.normalize(attn.q_proj(normed).view(xb.shape[0],-1,NUM_HEADS,D_MODEL//NUM_HEADS).transpose(1,2),dim=-1)
            k = F.normalize(attn.k_proj(normed).view(xb.shape[0],-1,NUM_HEADS,D_MODEL//NUM_HEADS).transpose(1,2),dim=-1)
            sim = torch.matmul(q,k.transpose(-2,-1))
            r_t = (torch.exp(attn.s_r)+1.0).view(1,-1,1,1)
            alpha = F.relu(1.0-r_t*(1.0-sim)).pow(2)
            sparsity = (alpha==0).float().mean().item()
        model.train()

        log_steps.append(step)
        log_r_mean.append(mean_r)
        log_sparsity.append(sparsity*100)
        print(f"  step {step:5d}  loss={loss.item():.3f}  mean_r={mean_r:.3f}  sparsity={sparsity*100:.1f}%")

if MAX_STEPS in SNAP_STEPS:
    snap_maps[MAX_STEPS] = get_attention_map(x_fixed)

# ---------------------------------------------------------------------------
# Plot 1: r evolution + sparsity
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,6), sharex=True)
ax1.plot(log_steps, log_r_mean, color="steelblue", linewidth=2)
ax1.axhline(2.0, color="gray", linestyle="--", linewidth=1, label="r=2 (init)")
ax1.set_ylabel("Mean r across heads/layers")
ax1.set_title("Screening threshold r during training (r=2 at init, decreases as attention opens)")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.plot(log_steps, log_sparsity, color="tomato", linewidth=2)
ax2.set_ylabel("Alpha sparsity (%) — layer 0")
ax2.set_xlabel("Training step")
ax2.set_title("Attention sparsity (% of zero alpha values)")
ax2.grid(alpha=0.3)

plt.tight_layout()
out1 = os.path.join(os.path.dirname(__file__), "r_evolution.png")
plt.savefig(out1, dpi=150)
plt.close()
print(f"Saved {out1}")

# ---------------------------------------------------------------------------
# Plot 2: Attention map snapshots
# ---------------------------------------------------------------------------
snap_keys = sorted(snap_maps.keys())
fig, axes = plt.subplots(1, len(snap_keys), figsize=(4*len(snap_keys), 4))
for ax, step_k in zip(axes, snap_keys):
    m = snap_maps[step_k]
    im = ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max()+1e-6, aspect="auto")
    ax.set_title(f"step {step_k}\nsparsity={100*(m==0).mean():.0f}%")
    ax.set_xlabel("Key position"); ax.set_ylabel("Query position")
    plt.colorbar(im, ax=ax, fraction=0.046)

fig.suptitle("Attention alpha maps (layer 0, head 0) — from dead-init to learned patterns", fontsize=12)
plt.tight_layout()
out2 = os.path.join(os.path.dirname(__file__), "attention_maps.png")
plt.savefig(out2, dpi=150)
plt.close()
print(f"Saved {out2}")

# ---------------------------------------------------------------------------
# Final r values
# ---------------------------------------------------------------------------
print("\nFinal r values per layer/head:")
for li, layer_rs in enumerate(get_r_values()):
    print(f"  layer {li}: " + "  ".join(f"h{hi}={r:.3f}" for hi,r in enumerate(layer_rs)))
