"""
Minimal example: train a tiny MultiscreenLM on random token sequences.

Run:
    python examples/train_tiny_lm.py
"""

import torch
import torch.optim as optim
from multiscreen import MultiscreenLM


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Tiny model for demonstration
    model = MultiscreenLM(
        vocab_size=256,
        d_model=128,
        num_heads=4,
        num_layers=3,
        max_seq_len=128,
        dropout=0.1,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    model.train()
    for step in range(50):
        # Random token sequences (batch_size=4, seq_len=64)
        ids = torch.randint(0, 256, (4, 64), device=device)
        labels = ids.clone()

        out = model(ids, labels=labels)
        loss = out["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 10 == 0:
            print(f"Step {step:3d} | loss: {loss.item():.4f}")

    print("\nGeneration example:")
    model.eval()
    prompt = torch.tensor([[1, 2, 3, 4]], device=device)
    generated = model.generate(prompt, max_new_tokens=16, temperature=0.8, top_k=50)
    print("Token ids:", generated[0].tolist())


if __name__ == "__main__":
    main()
