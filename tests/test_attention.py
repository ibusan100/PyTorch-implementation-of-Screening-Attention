"""Unit tests for the Screening Attention mechanism."""

import torch
import pytest
from multiscreen import ScreeningAttention, TanhNorm, MultiscreenBlock, MultiscreenLM


class TestTanhNorm:
    def test_zero_input(self):
        norm = TanhNorm()
        x = torch.zeros(2, 4)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.allclose(out, torch.zeros_like(out))

    def test_magnitude_bounded(self):
        norm = TanhNorm()
        x = torch.randn(8, 16) * 100  # large values
        out = norm(x)
        norms = out.norm(dim=-1)
        # tanh saturates to 1.0 in float32 for large inputs — bound is <= 1.0
        assert (norms <= 1.0 + 1e-5).all(), "TanhNorm output magnitude must be <= 1"

    def test_direction_preserved(self):
        norm = TanhNorm()
        x = torch.randn(4, 8)
        out = norm(x)
        # Normalized directions should match
        x_dir = torch.nn.functional.normalize(x, dim=-1)
        out_dir = torch.nn.functional.normalize(out, dim=-1)
        assert torch.allclose(x_dir, out_dir, atol=1e-5)


class TestScreeningAttention:
    def setup_method(self):
        self.B, self.T, self.d = 2, 16, 64
        self.attn = ScreeningAttention(d_model=self.d, num_heads=4)

    def test_output_shape(self):
        x = torch.randn(self.B, self.T, self.d)
        out = self.attn(x, x, x)
        assert out.shape == (self.B, self.T, self.d)

    def test_causal_masking(self):
        """Future positions must have zero influence in causal mode."""
        attn = ScreeningAttention(d_model=self.d, num_heads=4, causal=True)
        x = torch.randn(1, self.T, self.d)
        # Perturb only the last token
        x2 = x.clone()
        x2[0, -1] = torch.randn(self.d)
        out1 = attn(x, x, x)
        out2 = attn(x2, x2, x2)
        # The first token's output should be identical regardless of future tokens
        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5)

    def test_key_padding_mask(self):
        x = torch.randn(self.B, self.T, self.d)
        mask = torch.zeros(self.B, self.T, dtype=torch.bool)
        mask[:, -4:] = True  # mask last 4 positions
        out = self.attn(x, x, x, key_padding_mask=mask)
        assert out.shape == (self.B, self.T, self.d)

    def test_non_causal(self):
        attn = ScreeningAttention(d_model=self.d, num_heads=4, causal=False)
        x = torch.randn(self.B, self.T, self.d)
        out = attn(x, x, x)
        assert out.shape == (self.B, self.T, self.d)

    def test_different_q_k_lengths(self):
        attn = ScreeningAttention(d_model=self.d, num_heads=4, causal=False)
        q = torch.randn(self.B, 8, self.d)
        kv = torch.randn(self.B, 20, self.d)
        out = attn(q, kv, kv)
        assert out.shape == (self.B, 8, self.d)


class TestMultiscreenBlock:
    def test_output_shape(self):
        block = MultiscreenBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 16, 64)
        out = block(x)
        assert out.shape == x.shape

    def test_residual_gradient_flow(self):
        block = MultiscreenBlock(d_model=64, num_heads=4)
        x = torch.randn(2, 16, 64, requires_grad=True)
        loss = block(x).sum()
        loss.backward()
        assert x.grad is not None


class TestMultiscreenLM:
    def setup_method(self):
        self.model = MultiscreenLM(
            vocab_size=100,
            d_model=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=64,
        )

    def test_forward_logits_shape(self):
        ids = torch.randint(0, 100, (2, 16))
        out = self.model(ids)
        assert out["logits"].shape == (2, 16, 100)

    def test_loss_computation(self):
        ids = torch.randint(0, 100, (2, 16))
        labels = ids.clone()
        labels[:, :4] = -100
        out = self.model(ids, labels=labels)
        assert "loss" in out
        assert out["loss"].ndim == 0  # scalar

    def test_generate(self):
        prompt = torch.randint(0, 100, (1, 4))
        generated = self.model.generate(prompt, max_new_tokens=8)
        assert generated.shape == (1, 12)
