"""Weight-tied recursive hybrid, GDN:GQA = 2:1, R=3 per super-block.

TWO super-blocks in sequence, each ONE ratio unit — 3 unique layers
(gdn, gdn, gqa) — and each applied 3 times with weights shared within that
super-block (the two super-blocks are independent of each other). 6 unique
layers, effective depth 18. FFN is capped at 4x and tuned DOWN to hit the
~50M non-embedding target: d_ff=2528 (~3.3x). Verified by
src/common/param_count.py.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    vocab_size: int = 16384  # BabyLM-community tokenizer — shared across all variants
    d_model: int = 768
    n_layers: int = 3  # unique layers PER super-block: one (gdn, gdn, gqa) unit
    n_super_blocks: int = 2  # independent super-blocks applied in sequence
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
    n_recursions: int = 3  # R: passes PER super-block, weights tied within it


CONFIG = Config()
