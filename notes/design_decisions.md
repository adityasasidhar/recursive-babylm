# Design decisions

This file records the choices that define the controlled comparison. Changes to
these choices alter the experiment and should be made consistently across all
five variants.

## Model sizing

- All models use `d_model = 768`, `d_ff = 2304`, vocabulary size 16,384, tied
  input/output embeddings, pre-norm RMSNorm, and SwiGLU feed-forward layers.
- GDN 2:1 and Recursive 2:1 contain the same six unique layers and have exactly
  46,913,888 non-embedding parameters.
- GDN 3:1 and Recursive 3:1 contain the same eight unique layers and have
  exactly 63,487,504 non-embedding parameters.
- The pure-GQA baseline is a larger reference model with ten unique layers and
  68,830,208 non-embedding parameters; it is not a parameter-matched twin.
- `src/common/param_count.py` is the executable source of truth for these
  counts. Do not resize one member of a matched pair independently.

## Attention and recursion

- GQA uses 12 query heads, four KV heads, head dimension 64, RoPE, and per-head
  QK normalization. FlashAttention-3 is used on supported Hopper systems;
  PyTorch SDPA is the portability fallback.
- GDN uses the reference `flash-linear-attention` Gated DeltaNet implementation
  rather than a local approximation. It uses 12 heads of dimension 64, value
  expansion 1, output gating, and size-4 short convolutions.
- A 2:1 ratio unit is `(GDN, GDN, GQA)`; a 3:1 unit is
  `(GDN, GDN, GDN, GQA)`. Each hybrid contains two independent ratio units.
- Recursive variants apply each ratio unit three times with weights tied within
  that unit. Weights are never tied across the two units. This raises effective
  depth from 6 to 18 or from 8 to 24 without changing parameter count.
- Residual-output initialization is scaled using effective depth because a tied
  projection writes to the residual stream on every recursive application.

## Controlled training protocol

- Every variant uses the same tokenizer, tokenized corpus, seeded data order,
  optimizer, schedule, batch of 32 sequences, 4,096-token context, and
  499,908,608-token budget.
- Micro-batch size is a memory-layout setting only; gradient accumulation keeps
  the optimizer batch fixed. It must not affect evaluation weighting.
- Full training is deliberately gated in `src/common/train.py` and
  `modal_train.py`; do not launch it without project-owner approval.

## Evaluation provenance

- Paper validation losses come from `modal_train.py::ckpt_eval`, which applies
  uniform per-token weighting to every checkpoint. Training-time validation
  logs are not comparable across the historical micro-batch settings.
- `src/common/blimp_eval.py` evaluates native checkpoints with the official
  BabyLM 2026 causal BLiMP protocol. Evaluator, dataset, and tokenizer revisions
  are pinned in `paper/figures/blimp_eval.json`.
- The manuscript reports conclusions within the exact matched pairs separately
  from across-architecture rankings: recursion wins both pairs on validation
  loss and BLiMP, while the larger pure-GQA baseline leads BLiMP overall.

## Primary implementation references

- Gated DeltaNet: Yang et al. (2025), *Gated Delta Networks: Improving Mamba2
  with Delta Rule* (`arXiv:2412.06464`).
- FLA implementation: <https://github.com/fla-org/flash-linear-attention>.
- Grouped-query attention: Ainslie et al. (2023), *GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints*.
- BLiMP: Warstadt et al. (2020), *BLiMP: The Benchmark of Linguistic Minimal
  Pairs for English*.
