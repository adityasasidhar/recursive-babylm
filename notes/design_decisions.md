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
  This differs within each learned-parameter-matched pair, so the experiment
  identifies the recursive configuration (tying plus depth-aware scaling), not
  tying alone.

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
- `paper/figures/paired_uncertainty.py` and its JSON artifact report paired
  evaluation-sample bootstrap intervals over aligned validation windows and
  BLiMP paradigms. They do not estimate training-seed variance; the
  `seed_replication*` and `accuracy_seed_noise` blocks do that separately.
- **`ckpt_eval`'s 2M-token slice is not domain-balanced.** `data.prepare`
  concatenates dev files in sorted-filename order, so the first 1,738,031 tokens
  are all of BNC Spoken and the next 260,817 are CHILDES; four of six domains are
  absent. It is retained for the 255-checkpoint series because it bounds cost, and
  it is a *paired* comparison so the contrast stays valid — but final-checkpoint
  numbers the paper cites come from `domain_eval`, over all six domains, with
  windows that never straddle a domain boundary. `data.prepare` now records the
  ranges in a `.domains.json` sidecar; `val_domains` rebuilds it for an existing
  bin only after verifying a re-tokenization is byte-identical, and never
  overwrites the bin.
- **Grammatical accuracy at this scale is seed-dominated.** Retraining one
  architecture under a different seed moves BLiMP macro-accuracy by up to 5.56
  points (sd 2.80) while validation loss moves by ≤0.0020 nats. Any accuracy
  claim needs seed replication before it means anything; single-seed accuracy
  differences below ~5 points are not resolvable. This is why the paper withdrew
  its 2:1 BLiMP claim.
- `src/common/zeroshot_eval.py` adds official COMPS and entity tracking as a
  separate module. `blimp_eval.py` must stay untouched — it produced the
  published BLiMP numbers, and a refactor would put their reproducibility at
  risk for no gain.
- The manuscript reports conclusions within the exact matched pairs separately
  from across-architecture rankings: recursion wins both pairs on validation
  loss and BLiMP, while the larger pure-GQA baseline leads BLiMP overall.

## Revision arms (BabyLM 2026 review response)

The five variants above are the reported grid and their protocol stays pinned.
Four further arms were added to answer specific reviewer objections. They are
registered in `VARIANTS` but excluded from `modal_train.PAPER_VARIANTS`, so no
fan-out entrypoint trains or evaluates them by default.

- `untied_2to1_deep` (18 independent layers, 140,740,128 non-embedding) and
  `untied_3to1_deep` (24 layers, 190,460,976) are depth- and FLOPs-matched to
  `recursive_2to1` / `recursive_3to1` and deliberately parameter-UNmatched.
  They isolate the cost of tying, which the parameter-matched pairs cannot:
  those confound tying with a 3x compute difference. Their `1/sqrt(2*n_layers)`
  residual init evaluates to exactly the recursive arms' `1/sqrt(2*D_eff)` at
  these depths, so the two share an initialization scale by construction.
- `recursive_2to1_uniqinit` / `recursive_3to1_uniqinit` are parameter-identical
  to the recursive variants and differ only in `_init_own_weights`, which uses
  unique depth instead of effective depth. This separates tying from
  depth-aware scaling — the confound the "Attention and recursion" section
  above records, and which the reviewed manuscript names but does not resolve.
- Compute parity for the twins is a *token-budget* change, not a new variant:
  `gdn_2to1` / `gdn_3to1` at 1,500,000,000 tokens is ~884M words of exposure,
  under the Strict track's 1B cap but close enough that the paper must say so.
  The cosine horizon derives from `token_budget`, so these are fresh runs, not
  resumes of the 500M checkpoints.
- `--seed` sets weight init and data order together. Within a seed group all
  variants still share an identical data order, so the pinned control holds;
  across groups the spread is genuine run-to-run variance rather than
  initialization variance alone.

## Primary implementation references

- Gated DeltaNet: Yang et al. (2025), *Gated Delta Networks: Improving Mamba2
  with Delta Rule* (`arXiv:2412.06464`).
- FLA implementation: <https://github.com/fla-org/flash-linear-attention>.
- Grouped-query attention: Ainslie et al. (2023), *GQA: Training Generalized
  Multi-Query Transformer Models from Multi-Head Checkpoints*.
- BLiMP: Warstadt et al. (2020), *BLiMP: The Benchmark of Linguistic Minimal
  Pairs for English*.
