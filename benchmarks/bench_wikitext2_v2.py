"""
Benchmark B v2: WikiText-2 Language Modeling — 修正版

変更点 (v1からの修正):
  1. GPT-2 BPEトークナイザー使用 (vocab_size=50,257) — char-levelは不適切
  2. モデル規模を ~8M params に拡大 (論文の最小実験規模)
  3. 学習ステップを20,000に延長
  4. 学習曲線をJSONに保存

注意: 論文はSlimPajama 628Bトークンで学習。WikiText-2は約200Kトークンのみのため
スケールは全く異なる。これは「同じアーキテクチャを同条件で比較する」実験。
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import GPT2TokenizerFast

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiscreen import MultiscreenLM


# ---------------------------------------------------------------------------
# Baseline: 標準TransformerLM
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 ffn_dim=None, max_seq_len=512, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.blocks   = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self._init()

    def _init(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def forward(self, input_ids, labels=None, key_padding_mask=None):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        x    = self.blocks(x, mask=mask, src_key_padding_mask=key_padding_mask, is_causal=True)
        logits = self.lm_head(self.norm(x))
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, tokens, seq_len):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def load_wikitext2_bpe(seq_len, batch_size):
    print("Downloading WikiText-2 & GPT-2 tokenizer ...")
    ds  = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def encode_split(split_name):
        texts = [t for t in ds[split_name]["text"] if t.strip()]
        full  = " ".join(texts)
        ids   = tok.encode(full)
        return torch.tensor(ids, dtype=torch.long)

    train_tok = encode_split("train")
    valid_tok = encode_split("validation")
    test_tok  = encode_split("test")

    vocab_size = tok.vocab_size
    print(f"vocab_size={vocab_size:,}  train={len(train_tok):,} tokens  "
          f"valid={len(valid_tok):,}  test={len(test_tok):,}")

    make = lambda t, shuf: DataLoader(
        TokenDataset(t, seq_len), batch_size=batch_size,
        shuffle=shuf, drop_last=True, num_workers=0)

    return make(train_tok, True), make(valid_tok, False), make(test_tok, False), vocab_size


# ---------------------------------------------------------------------------
# Config — ~8M params
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 256
BATCH_SIZE = 16
D_MODEL    = 512
NUM_HEADS  = 8
NUM_LAYERS = 6
FFN_DIM    = 1024
LR         = 3e-4
MAX_STEPS  = 20_000
EVAL_EVERY = 1_000
GRAD_CLIP  = 1.0


@torch.no_grad()
def evaluate(model, loader, device, max_batches=200):
    model.eval()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        loss = model(x.to(device), labels=y.to(device))["loss"]
        total += loss.item(); n += 1
    model.train()
    return math.exp(total / n) if n else float("inf")


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def train(model, name, train_loader, valid_loader, device):
    model = model.to(device).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

    log = {"name": name, "param_count": count_params(model),
           "steps": [], "train_ppl": [], "valid_ppl": [], "elapsed_s": []}
    t0, step = time.time(), 0
    loader_iter = iter(train_loader)

    print(f"\n{'='*65}")
    print(f"Training: {name}  ({log['param_count']:,} params)")
    print(f"{'='*65}")

    while step < MAX_STEPS:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        loss = model(x.to(device), labels=y.to(device))["loss"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step(); sched.step()
        step += 1

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            t_ppl = math.exp(loss.item())
            v_ppl = evaluate(model, valid_loader, device)
            elapsed = time.time() - t0
            log["steps"].append(step)
            log["train_ppl"].append(round(t_ppl, 3))
            log["valid_ppl"].append(round(v_ppl, 3))
            log["elapsed_s"].append(round(elapsed, 1))
            print(f"  step {step:5d}/{MAX_STEPS}  train_ppl={t_ppl:7.3f}  "
                  f"valid_ppl={v_ppl:7.3f}  elapsed={elapsed:.0f}s")

    return model, log


def main():
    train_loader, valid_loader, test_loader, vocab_size = \
        load_wikitext2_bpe(SEQ_LEN, BATCH_SIZE)

    cfg = dict(vocab_size=vocab_size, d_model=D_MODEL, num_heads=NUM_HEADS,
               num_layers=NUM_LAYERS, ffn_dim=FFN_DIM, max_seq_len=SEQ_LEN, dropout=0.1)

    results = {}

    baseline = TransformerLM(**cfg)
    _, log_b = train(baseline, "TransformerLM (softmax)", train_loader, valid_loader, DEVICE)
    log_b["test_ppl"] = round(evaluate(baseline, test_loader, DEVICE, max_batches=500), 3)
    print(f"  → test_ppl = {log_b['test_ppl']}")
    results["TransformerLM"] = log_b
    del baseline; torch.cuda.empty_cache()

    screening = MultiscreenLM(**cfg)
    _, log_m = train(screening, "MultiscreenLM (screening)", train_loader, valid_loader, DEVICE)
    log_m["test_ppl"] = round(evaluate(screening, test_loader, DEVICE, max_batches=500), 3)
    print(f"  → test_ppl = {log_m['test_ppl']}")
    results["MultiscreenLM"] = log_m
    del screening; torch.cuda.empty_cache()

    print("\n" + "="*65)
    print("FINAL RESULTS — WikiText-2, GPT-2 BPE tokenizer")
    print("="*65)
    for nm, log in results.items():
        print(f"  {nm:<35}  test_ppl={log['test_ppl']:7.3f}  "
              f"params={log['param_count']:,}  elapsed={log['elapsed_s'][-1]:.0f}s")

    out = os.path.join(os.path.dirname(__file__), "results_wikitext2_v2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
