"""Hybrid non-recursive transformer, GDN:GQA = 3:1.

Layer stack tiles the (gdn, gdn, gdn, gqa) pattern through all 8 layers, each
layer with independent weights, applied once (single pass).
"""

from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.common.attention import GatedDeltaNetLayer, GQAttention
from src.common.layers import SwiGLU

Config = importlib.import_module("src.gdn_baseline.3to1.config").Config


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
        self.blocks = nn.ModuleList(HybridBlock(cfg, k) for k in kinds)
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self._init_own_weights()

    def _init_own_weights(self) -> None:
        """Init embeddings + GQA/FFN linears; leave fla's GatedDeltaNet
        internals on their reference init."""
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        scale = 1 / math.sqrt(2 * self.cfg.n_layers)
        for blk in self.blocks:
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
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.float().view(-1, logits.size(-1)), targets.reshape(-1)
            )
        return logits, loss
