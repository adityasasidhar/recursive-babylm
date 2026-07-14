"""Hybrid non-recursive, GDN:GQA = 2:1 (pattern GDN,GDN,GQA repeated 2x).

d_ff = 3x d_model (2304); depth is the sizing knob. 6 layers (4 GDN + 2 GQA)
lands 46.9M non-embedding — EXACTLY matching recursive_2to1, which has the
same 6-layer composition (2 super-blocks x one ratio unit). This makes the
non-recursive/recursive 2:1 pair a clean recursion ablation (same params,
recursion is the only difference), mirroring gdn_3to1 == recursive_3to1.
Verified by src/common/param_count.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16384  # BabyLM-community tokenizer — shared across all variants
    d_model: int = 768
    n_layers: int = 6  # 2 pattern units; 4 GDN + 2 GQA -> 46.9M, matches recursive_2to1
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
