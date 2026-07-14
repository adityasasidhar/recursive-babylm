"""Shared building blocks: SwiGLU FFN, QK-Norm, RoPE.

Used by all 5 variants. RMSNorm comes from torch (nn.RMSNorm, torch>=2.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward: down(silu(gate(x)) * up(x)). No biases."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class QKNorm(nn.Module):
    """Per-head RMSNorm on q and k (Qwen3-style: one learnable RMSNorm of size
    head_dim each for q and k, shared across heads).

    Expects q, k of shape (B, n_heads, T, head_dim).
    """

    def __init__(self, head_dim: int, eps: float = 1e-5):
        super().__init__()
        self.q_norm = nn.RMSNorm(head_dim, eps=eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=eps)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


class RotaryEmbedding(nn.Module):
    """Standard RoPE. Caches cos/sin up to max_seq_len, extends on demand."""

    def __init__(self, head_dim: int, base: float = 10000.0, max_seq_len: int = 2048):
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q, k: (B, n_heads, T, head_dim) -> rotated q, k."""
        T = q.shape[-2]
        if T > self.cos_cached.shape[0]:
            self._build_cache(T)
        cos = self.cos_cached[:T].to(q.dtype)  # (T, head_dim/2)
        sin = self.sin_cached[:T].to(q.dtype)
        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)  # rotate-half convention
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)
