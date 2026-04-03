"""
Minimal working example: train MultiscreenLM on WikiText-2.

Usage:
    pip install -e ".[dev]"
    pip install datasets transformers
    python train.py
"""

import sys, os, math, time
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

from multiscreen import MultiscreenLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 256
BATCH_SIZE = 16
D_MODEL    = 128
NUM_HEADS  = 4
NUM_LAYERS = 4
FFN_DIM    = 512
LR         = 3e-4
MAX_STEPS  = 10_000
EVAL_EVERY = 1_000

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens = tokens; self.seq_len = seq_len
    def __len__(self): return max(0, len(self.tokens) - self.seq_len)
    def __getitem__(self, i):
        c = self.tokens[i: i + self.seq_len + 1]
        return c[:-1], c[1:]

print("Downloading WikiText-2 and GPT-2 tokenizer...")
ds  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
tok = GPT2TokenizerFast.from_pretrained("gpt2")

def encode(split):
    full = " ".join(t for t in ds[split]["text"] if t.strip())
    return torch.tensor(tok.encode(full), dtype=torch.long)

train_ids = encode("train")
valid_ids = encode("validation")
vocab_size = tok.vocab_size

train_loader = DataLoader(TokenDataset(train_ids, SEQ_LEN), batch_size=BATCH_SIZE,
                          shuffle=True, drop_last=True, num_workers=0)
valid_loader = DataLoader(TokenDataset(valid_ids, SEQ_LEN), batch_size=BATCH_SIZE,
                          shuffle=False, drop_last=True, num_workers=0)
print(f"vocab={vocab_size:,}  train={len(train_ids):,} tokens")

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
model = MultiscreenLM(
    vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, max_seq_len=SEQ_LEN, dropout=0.1,
).to(DEVICE)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate(max_batches=100):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(valid_loader):
        if i >= max_batches: break
        total += model(x.to(DEVICE), labels=y.to(DEVICE))["loss"].item()
        n += 1
    model.train()
    return math.exp(total / n) if n else float("inf")

step = 0
loader_iter = iter(train_loader)
t0 = time.time()

print("\nTraining...")
while step < MAX_STEPS:
    try: x, y = next(loader_iter)
    except StopIteration:
        loader_iter = iter(train_loader); x, y = next(loader_iter)

    loss = model(x.to(DEVICE), labels=y.to(DEVICE))["loss"]
    opt.zero_grad(); loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); step += 1

    if step % EVAL_EVERY == 0 or step == MAX_STEPS:
        vppl = evaluate()
        print(f"  step {step:5d}/{MAX_STEPS}  "
              f"train_ppl={math.exp(loss.item()):7.2f}  "
              f"valid_ppl={vppl:7.2f}  "
              f"elapsed={time.time()-t0:.0f}s")

print("Done.")
