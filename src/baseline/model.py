"""Baseline: standard pre-norm transformer, GQA attention only, non-recursive.

Attention/FFN primitives come from src/common; the block and model classes are
deliberately local to this variant (each leaf folder is self-contained).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.baseline.config import Config
from src.common.attention import GQAttention
from src.common.layers import SwiGLU


class GQABlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn_norm = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
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
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(GQABlock(cfg) for _ in range(cfg.n_layers))
        self.norm_f = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # tied

        self.apply(self._init_weights)
        # GPT-2-style scaled init on residual-out projections
        scale = 1 / math.sqrt(2 * cfg.n_layers)
        for blk in self.blocks:
            nn.init.normal_(blk.attn.o_proj.weight, std=0.02 * scale)
            nn.init.normal_(blk.ffn.down_proj.weight, std=0.02 * scale)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

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
