"""Initialization ablation for recursive_2to1: unique-depth residual scaling.

Structurally identical to recursive_2to1 — same two independent super-blocks,
same R=3 tied passes, same parameter count to the digit. The ONLY difference is
_init_own_weights, which scales residual-out projections by 1/sqrt(2 * unique
depth) instead of 1/sqrt(2 * effective depth).

recursive_2to1 changes both tying and residual scaling relative to its
non-recursive twin, so the paper identifies the recursive *configuration*
rather than tying alone. This arm holds tying fixed and moves the scaling back,
isolating the two. Added for the BabyLM 2026 review response.
"""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.attention import GatedDeltaNetLayer, GQAttention
from src.common.layers import SwiGLU

Config = importlib.import_module("src.recursion_gdn.2to1_uniqinit.config").Config


class HybridBlock(nn.Module):
    """Pre-norm block whose attention is either GDN or GQA per `kind`."""

    def __init__(self, cfg: Config, kind: str):
        super().__init__()
        assert kind in ("gdn", "gqa")
        self.kind = kind
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        if kind == "gdn":
            self.attn = GatedDeltaNetLayer(
                cfg.d_model, cfg.n_heads, cfg.head_dim,
                expand_v=cfg.gdn_expand_v, norm_eps=cfg.norm_eps,
            )
        else:
            self.attn = GQAttention(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim,
                rope_base=cfg.rope_base, max_seq_len=cfg.max_seq_len,
                norm_eps=cfg.norm_eps, attn_backend=cfg.attn_backend,
            )
        self.ffn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x


class Model(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.n_layers % len(cfg.layer_pattern) == 0
        self.cfg = cfg
        kinds = cfg.layer_pattern * (cfg.n_layers // len(cfg.layer_pattern))
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        # independent super-blocks; each reused R times (weights tied within)
        self.super_blocks = nn.ModuleList(
            nn.ModuleList(HybridBlock(cfg, k) for k in kinds)
            for _ in range(cfg.n_super_blocks)
        )
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self._init_own_weights()

    def _init_own_weights(self) -> None:
        """Init embeddings + GQA/FFN linears; leave fla's GatedDeltaNet
        internals on their reference init. Residual-out projections are scaled
        by the UNIQUE depth (n_super_blocks * n_layers) — the scaling the
        non-recursive twin uses — NOT the effective depth the shipped recursive
        variant uses. This is the whole point of the arm: it separates tying
        from depth-aware initialization, which the paper's matched pairs
        change together. The factor differs by sqrt(n_recursions)."""
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        uniq = self.cfg.n_super_blocks * self.cfg.n_layers
        scale = 1 / math.sqrt(2 * uniq)
        for sb in self.super_blocks:
            for blk in sb:
                if blk.kind == "gqa":
                    for lin in (blk.attn.q_proj, blk.attn.k_proj, blk.attn.v_proj):
                        nn.init.normal_(lin.weight, std=0.02)
                    nn.init.normal_(blk.attn.o_proj.weight, std=0.02 * scale)
                for lin in (blk.ffn.gate_proj, blk.ffn.up_proj):
                    nn.init.normal_(lin.weight, std=0.02)
                nn.init.normal_(blk.ffn.down_proj.weight, std=0.02 * scale)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = self.tok_emb(idx)
        for sb in self.super_blocks:  # sequence of independent super-blocks
            for _ in range(self.cfg.n_recursions):  # R tied passes each
                for blk in sb:
                    x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
