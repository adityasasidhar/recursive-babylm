"""Config for the recursive_3to1 initialization ablation.

Byte-for-byte the same hyperparameters as recursive_3to1 — the arms differ
only in model.py's residual-out scaling (unique depth vs effective depth), so
they are parameter-identical and this file must stay in lockstep with
src/recursion_gdn/3to1/config.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16384  # BabyLM-community tokenizer — shared across all variants
    d_model: int = 768
    n_layers: int = 4  # unique layers PER super-block: one (gdn, gdn, gdn, gqa) unit
    n_super_blocks: int = 2  # independent super-blocks applied in sequence
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
    n_recursions: int = 3  # R: passes PER super-block, weights tied within it


CONFIG = Config()
