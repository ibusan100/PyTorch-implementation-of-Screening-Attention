"""
MultiscreenLM: a causal language model built with Screening Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import MultiscreenBlock


class MultiscreenLM(nn.Module):
    """
    Causal language model using Multiscreen (Screening Attention) blocks.

    Args:
        vocab_size:  Vocabulary size.
        d_model:     Model (embedding) dimension.
        num_heads:   Number of attention heads per layer.
        num_layers:  Number of Multiscreen blocks.
        ffn_dim:     FFN hidden dimension (default: 4 * d_model).
        max_seq_len: Maximum sequence length for positional embedding.
        dropout:     Dropout probability.
        tie_weights: Whether to tie input/output embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        ffn_dim: int | None = None,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_weights: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            MultiscreenBlock(
                d_model=d_model,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                causal=True,
            )
            for _ in range(num_layers)
        ])

        self.norm = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids:        (B, T) integer token ids.
            key_padding_mask: (B, T) bool — True = padding position.
            labels:           (B, T) integer token ids for loss computation.
                              Positions with value -100 are ignored.

        Returns:
            dict with:
                'logits': (B, T, vocab_size)
                'loss':   scalar (only if labels provided)
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)

        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        result = {"logits": logits}
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            result["loss"] = loss

        return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        Greedy / temperature sampling generation.

        Args:
            input_ids:      (B, T) prompt token ids.
            max_new_tokens: Number of tokens to generate.
            temperature:    Sampling temperature (1.0 = no change).
            top_k:          If set, restricts sampling to top-k logits.

        Returns:
            (B, T + max_new_tokens) generated token ids.
        """
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids)["logits"][:, -1, :]  # (B, vocab)
            logits = logits / temperature
            if top_k is not None:
                values, _ = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)
        return input_ids
