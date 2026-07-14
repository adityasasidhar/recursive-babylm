"""Hybrid non-recursive, GDN:GQA = 3:1 (pattern GDN,GDN,GDN,GQA repeated 2x).

FFN width is fixed at 3x (d_ff = 2304), matching every variant; depth is the
sizing knob. 8 layers lands ~63.5M non-embedding. Verified by
src/common/param_count.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16384  # BabyLM-community tokenizer — shared across all variants
    d_model: int = 768
    n_layers: int = 8  # sizing knob; must be a multiple of len(layer_pattern)
    n_heads: int = 12
    n_kv_heads: int = 4  # GQA layers only
    head_dim: int = 64
    d_ff: int = 2304  # 3x d_model, shared across all variants
    rope_base: float = 10000.0
    norm_eps: float = 1e-5
    max_seq_len: int = 1024
    attn_backend: str = "auto"  # GQA softmax kernel: "auto" | "fa3" | "sdpa"
    gdn_to_gqa: str = "3:1"
    layer_pattern: tuple[str, ...] = ("gdn", "gdn", "gdn", "gqa")
    gdn_expand_v: int = 1
    n_recursions: int = 1  # non-recursive


CONFIG = Config()
