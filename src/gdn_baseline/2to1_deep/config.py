"""Untied-depth control, GDN:GQA = 2:1 (pattern GDN,GDN,GQA repeated 6x).

The compute- and depth-matched counterpart of recursive_2to1: the SAME 18 layer
applications per token, but every layer independently parameterised instead of
6 tied re-applications of one super-block. That costs 3x the learned
parameters (~140.7M non-embedding vs gdn_2to1's), which is the point — it prices
what tying buys. Added for the BabyLM 2026 review response; NOT one of the five
variants the paper's parameter-matched claim rests on.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16384  # BabyLM-community tokenizer — shared across all variants
    d_model: int = 768
    n_layers: int = 18  # 6 pattern units; 12 GDN + 6 GQA -> ~140.7M, 3x gdn_2to1
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA layers only
    head_dim: int = 64
    d_ff: int = 2304  # 3x d_model, shared across all variants
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    attn_backend: str = "auto"  # GQA softmax kernel: "auto" | "fa3" | "sdpa"
    gdn_to_gqa: str = "2:1"
    layer_pattern: tuple[str, ...] = ("gdn", "gdn", "gqa")
    gdn_expand_v: int = 1
    n_recursions: int = 1  # non-recursive


CONFIG = Config()
