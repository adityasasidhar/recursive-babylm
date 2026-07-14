"""Shared attention layers: GQA (softmax) and Gated DeltaNet (linear, delta-rule).

GatedDeltaNet comes from `flash-linear-attention` (fla) — the reference
implementation by the Gated DeltaNet authors (Yang et al., 2024). We wrap it
rather than reimplementing; see notes/design_decisions.md for the citation and
configuration choices. NOTE: fla's kernels are Triton and require a CUDA GPU
for forward/backward — layer *construction* and param counting work on CPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.layers import QKNorm, RotaryEmbedding

# FlashAttention-3 for the GQA layers — Hopper (sm_90 / H100) only. The prebuilt
# `flash_attn_3` wheel exposes it under the module `flash_attn_interface`. Absent
# on non-Hopper GPUs (e.g. the local RTX 3050) -> fall back to torch SDPA so the
# models stay runnable. See pyproject `[fa3]` note and notes/design_decisions.md.
try:
    from flash_attn_interface import flash_attn_func as flash_attn_3_func  # type: ignore[import-not-found]

    _HAS_FA3 = True
except ImportError:  # pragma: no cover - depends on the training box
    flash_attn_3_func = None
    _HAS_FA3 = False


class GQAttention(nn.Module):
    """Grouped Query Attention with RoPE and QK-Norm, causal.

    n_heads query heads share n_kv_heads key/value heads.

    Softmax backend is selected by `attn_backend`:
      - "fa3"  : FlashAttention-3 (Hopper only); errors if unavailable.
      - "sdpa" : torch scaled_dot_product_attention.
      - "auto" : FA3 when importable + input is a CUDA half-precision tensor,
                 else SDPA. FA3 consumes GQA KV heads directly (no expansion).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
        rope_base: float = 10000.0,
        max_seq_len: int = 2048,
        norm_eps: float = 1e-5,
        attn_backend: str = "auto",
    ):
        super().__init__()
        assert n_heads % n_kv_heads == 0
        assert attn_backend in ("auto", "fa3", "sdpa")
        if attn_backend == "fa3" and not _HAS_FA3:
            raise RuntimeError(
                "attn_backend='fa3' but flash_attn_interface (FA3) is not "
                "installed. Install the FA3 extra on a Hopper GPU, or use "
                "'auto'/'sdpa'."
            )
        self.attn_backend = attn_backend
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(d_model, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.qk_norm = QKNorm(head_dim, eps=norm_eps)
        self.rope = RotaryEmbedding(head_dim, base=rope_base, max_seq_len=max_seq_len)

    def _use_fa3(self, x: torch.Tensor) -> bool:
        if self.attn_backend == "fa3":
            return True
        if self.attn_backend == "sdpa":
            return False
        return _HAS_FA3 and x.is_cuda and x.dtype in (torch.float16, torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # QK-Norm + RoPE on (B, H, T, D); head_dim is last so both are layout-safe
        q, k = self.qk_norm(q, k)
        q, k = self.rope(q, k)

        if self._use_fa3(x):
            # FA3 wants (B, T, H, D) and handles GQA internally — pass KV heads
            # un-expanded (no repeat_interleave), and causal masking is built in.
            o = flash_attn_3_func(  # type: ignore[misc]
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), causal=True
            )
            if isinstance(o, tuple):  # some FA3 builds return (out, softmax_lse)
                o = o[0]
            o = o.reshape(B, T, self.n_heads * self.head_dim)
        else:
            rep = self.n_heads // self.n_kv_heads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        return self.o_proj(o)


class GatedDeltaNetLayer(nn.Module):
    """Thin wrapper around fla's GatedDeltaNet so blocks get a plain
    tensor -> tensor interface.

    Config notes (see notes/design_decisions.md):
      - head_dim/num_heads set explicitly to mirror the GQA layers
        (fla's defaults assume much larger models).
      - expand_v=1 keeps the value/state width at d_model for param parity.
      - use_gate=True, short conv enabled = the reference Gated DeltaNet recipe.
      - q/k are L2-normalized inside the delta rule (the linear-attention
        analogue of QK-Norm); the learnable QK-Norm applies to GQA layers.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        expand_v: int = 1,
        norm_eps: float = 1e-5,
    ):
        super().__init__()
        from fla.layers.gated_deltanet import GatedDeltaNet  # GPU kernels; import here

        self.inner = GatedDeltaNet(
            hidden_size=d_model,
            head_dim=head_dim,
            num_heads=n_heads,
            expand_v=expand_v,
            mode="chunk",
            use_gate=True,
            use_short_conv=True,
            norm_eps=norm_eps,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o, _, _ = self.inner(x)
        return o
