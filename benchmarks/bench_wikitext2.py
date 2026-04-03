"""
Benchmark B: WikiText-2 Language Modeling (perplexity)

WikiText-2 は言語モデリングの標準ベンチマーク。
MultiscreenLM と同規模の標準TransformerLM を同一条件で学習し、
検証perplexityで品質を比較する。

Usage:
    python benchmarks/bench_wikitext2.py
"""

import json, math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from multiscreen import MultiscreenLM

# ---------------------------------------------------------------------------
# Baseline: 同規模の標準TransformerLM（softmax attention）
# ---------------------------------------------------------------------------
class TransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers,
                 ffn_dim=None, max_seq_len=256, dropout=0.1):
        super().__init__()
        ffn_dim = ffn_dim or 4 * d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_seq_len, d_model)
        self.drop      = nn.Dropout(dropout)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=ffn_dim,
            dropout=dropout, batch_first=True, norm_first=True,  # pre-norm
        )
        self.blocks   = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm     = nn.LayerNorm(d_model)
        self.lm_head  = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.max_seq_len = max_seq_len
        self._init()

    def _init(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    def _causal_mask(self, T, device):
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, input_ids, labels=None, key_padding_mask=None):
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x    = self.drop(self.token_emb(input_ids) + self.pos_emb(pos))
        mask = self._causal_mask(T, input_ids.device)
        x    = self.blocks(x, mask=mask, src_key_padding_mask=key_padding_mask,
                           is_causal=True)
        x    = self.norm(x)
        logits = self.lm_head(x)
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return result

    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None):
        self.eval()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                ids = input_ids[:, -self.max_seq_len:]
                logits = self(ids)["logits"][:, -1, :] / temperature
                if top_k:
                    v, _ = torch.topk(logits, top_k)
                    logits[logits < v[:, -1:]] = float("-inf")
                next_tok = torch.multinomial(F.softmax(logits, -1), 1)
                input_ids = torch.cat([input_ids, next_tok], dim=1)
        return input_ids


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self):
        return max(0, len(self.tokens) - self.seq_len)

    def __getitem__(self, idx):
        chunk = self.tokens[idx: idx + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_vocab_and_tokenize(texts):
    """Character-level tokenizer (no external dependency)."""
    chars = sorted(set("".join(texts)))
    stoi = {c: i+1 for i, c in enumerate(chars)}  # 0 = pad
    stoi["<unk>"] = len(stoi) + 1
    vocab_size = len(stoi) + 2

    def encode(text):
        return [stoi.get(c, stoi["<unk>"]) for c in text]

    return stoi, vocab_size, encode


def load_wikitext2(seq_len, batch_size, device):
    print("Downloading WikiText-2 ...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")

    train_text = " ".join(ds["train"]["text"])
    valid_text = " ".join(ds["validation"]["text"])
    test_text  = " ".join(ds["test"]["text"])

    stoi, vocab_size, encode = build_vocab_and_tokenize([train_text, valid_text, test_text])

    def to_tensor(text):
        return torch.tensor(encode(text), dtype=torch.long)

    train_tok = to_tensor(train_text)
    valid_tok = to_tensor(valid_text)
    test_tok  = to_tensor(test_text)

    train_loader = DataLoader(
        TokenDataset(train_tok, seq_len), batch_size=batch_size,
        shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(
        TokenDataset(valid_tok, seq_len), batch_size=batch_size,
        shuffle=False, drop_last=True, num_workers=0)
    test_loader  = DataLoader(
        TokenDataset(test_tok,  seq_len), batch_size=batch_size,
        shuffle=False, drop_last=True, num_workers=0)

    print(f"vocab_size={vocab_size}, train={len(train_tok):,} chars, "
          f"valid={len(valid_tok):,}, test={len(test_tok):,}")
    return train_loader, valid_loader, test_loader, vocab_size


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEQ_LEN    = 256
BATCH_SIZE = 32
D_MODEL    = 256
NUM_HEADS  = 4
NUM_LAYERS = 4
LR         = 3e-4
MAX_STEPS  = 5000
EVAL_EVERY = 500
GRAD_CLIP  = 1.0


@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval()
    total_loss, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        loss = model(x, labels=y)["loss"]
        total_loss += loss.item()
        n += 1
    ppl = math.exp(total_loss / n) if n > 0 else float("inf")
    model.train()
    return ppl


def train(model, name, train_loader, valid_loader, device):
    model = model.to(device).train()
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_STEPS)

    log = {"name": name, "steps": [], "train_ppl": [], "valid_ppl": [], "elapsed_s": []}
    step = 0
    t0   = time.time()

    print(f"\n{'='*60}")
    print(f"Training: {name}")
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    print(f"{'='*60}")

    loader_iter = iter(train_loader)
    while step < MAX_STEPS:
        try:
            x, y = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            x, y = next(loader_iter)

        x, y = x.to(device), y.to(device)
        loss = model(x, labels=y)["loss"]
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        sched.step()
        step += 1

        if step % EVAL_EVERY == 0 or step == MAX_STEPS:
            train_ppl = math.exp(loss.item())
            valid_ppl = evaluate(model, valid_loader, device)
            elapsed   = time.time() - t0
            log["steps"].append(step)
            log["train_ppl"].append(round(train_ppl, 2))
            log["valid_ppl"].append(round(valid_ppl, 2))
            log["elapsed_s"].append(round(elapsed, 1))
            print(f"  step {step:5d}/{MAX_STEPS}  "
                  f"train_ppl={train_ppl:7.2f}  valid_ppl={valid_ppl:7.2f}  "
                  f"elapsed={elapsed:.0f}s")

    return model, log


def main():
    train_loader, valid_loader, test_loader, vocab_size = load_wikitext2(
        SEQ_LEN, BATCH_SIZE, DEVICE)

    model_configs = dict(
        vocab_size=vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        ffn_dim=D_MODEL * 4,
        max_seq_len=SEQ_LEN,
        dropout=0.1,
    )

    results = {}

    # --- Baseline: Standard Transformer ---
    baseline = TransformerLM(**model_configs)
    _, log_b = train(baseline, "TransformerLM (softmax)", train_loader, valid_loader, DEVICE)
    test_ppl_b = evaluate(baseline, test_loader, DEVICE, max_batches=200)
    log_b["test_ppl"] = round(test_ppl_b, 2)
    print(f"  → test_ppl = {test_ppl_b:.2f}")
    results["TransformerLM"] = log_b

    # --- MultiscreenLM ---
    multiscreen = MultiscreenLM(**model_configs)
    _, log_m = train(multiscreen, "MultiscreenLM (screening)", train_loader, valid_loader, DEVICE)
    test_ppl_m = evaluate(multiscreen, test_loader, DEVICE, max_batches=200)
    log_m["test_ppl"] = round(test_ppl_m, 2)
    print(f"  → test_ppl = {test_ppl_m:.2f}")
    results["MultiscreenLM"] = log_m

    # --- Summary ---
    print("\n" + "="*60)
    print("FINAL RESULTS (WikiText-2, character-level)")
    print("="*60)
    for model_name, log in results.items():
        print(f"  {model_name:<35} test_ppl={log['test_ppl']:7.2f}  "
              f"elapsed={log['elapsed_s'][-1]:.0f}s")

    out = os.path.join(os.path.dirname(__file__), "results_wikitext2.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
