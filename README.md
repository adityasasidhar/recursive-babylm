# babylm-recursive-hybrid

Weight-tied depth-recursive transformers with **hybrid Gated DeltaNet + GQA**
attention, for the **BabyLM 2026** workshop (non-competition paper track,
deadline 2026-07-15).

**Claim under test:** can a SMALLER (~50M non-embedding) weight-tied recursive
hybrid — sequences of GDN+GQA super-blocks, each recursed — beat LARGER
(~80M) non-recursive models (pure GQA and hybrid) under the BabyLM Strict
budget (≤100M words)?

## The 5 variants

All variants: same tokenizer (BabyLM-community 100M, vocab 16384), same SwiGLU FFN, QK-Norm,
pre-norm RMSNorm, d_model=768, **d_ff = 2304 (3x) everywhere**. With FFN width
fixed, per-variant sizes are whatever the structure gives — not force-matched.

| Variant | Folder | Attention | Unique layers | Recursion | eff. depth | non-embed |
|---|---|---|---|---|---|---|
| baseline | `src/baseline/` | pure GQA | 10 | — | 10 | 68.8M |
| gdn 2:1 | `src/gdn_baseline/2to1/` | GDN:GQA 2:1 | 6 | — | 6 | **46.9M** |
| gdn 3:1 | `src/gdn_baseline/3to1/` | GDN:GQA 3:1 | 8 | — | 8 | **63.5M** |
| recursive 2:1 | `src/recursion_gdn/2to1/` | GDN:GQA 2:1 | 6 = 2 SBs × (g,g,q) | R=3 per SB | 18 | **46.9M** |
| recursive 3:1 | `src/recursion_gdn/3to1/` | GDN:GQA 3:1 | 8 = 2 SBs × (g,g,g,q) | R=3 per SB | 24 | **63.5M** |

Recursive variants are a sequence of 2 independent super-blocks, each ONE
ratio unit, each applied 3 times with weights tied within that super-block.
The design yields **two exact learned-parameter-matched recursion pairs**:
`gdn_2to1` == `recursive_2to1` at
46.9M, and `gdn_3to1` == `recursive_3to1` at 63.5M. `baseline` (68.8M, pure
GQA) sits alongside as the no-GDN reference. Within each pair, the recursive
configuration also uses effective-depth residual scaling, so the comparison
identifies the recursive configuration rather than tying alone. Each leaf folder is a
self-contained variant (`model.py` + `config.py`) by design; only the
attention/FFN primitives, tokenizer, data, and training loop are shared in
`src/common/`.

## Layout

```
src/common/    attention.py (GQA + fla GatedDeltaNet wrapper)  layers.py (SwiGLU, QK-Norm, RoPE)
               tokenizer.py  data.py  train.py  param_count.py  smoke_test.py  variants.py
               blimp_eval.py (native-checkpoint causal BLiMP evaluation)
src/<variant>/ model.py  config.py            (5 leaf folders, see table)
notes/         design_decisions.md            (sizing math, fla citation, open items)
paper/         paper.md (paper source)  figures/ (scripts + evaluation/uncertainty JSON)  latex/ (ACL-format main.tex → main.pdf)
```

`2to1`/`3to1` start with a digit, so they load via `importlib`
(`src/common/variants.py`), not plain imports.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and a CUDA GPU (the Gated DeltaNet
kernels from [`flash-linear-attention`](https://github.com/fla-org/flash-linear-attention)
are Triton-only; param counting works on CPU).

```bash
uv sync
```

**Hardware note.** Local GPU is an RTX 3050 Mobile (4GB) — fine for the smoke
tests, not for full training; plan real runs on a rented GPU (≥16GB).

## Gates & tests

```bash
uv run python -m src.common.param_count   # size report + auto-detects the recursion ablation pair
uv run python -m src.common.smoke_test    # forward shapes + dummy-batch overfit (PASS)
uv run python -m src.common.tokenizer     # snapshot shared tokenizer to tokenizer/artifacts
```

## Training (gated — do not start without sign-off)

```bash
# 1. data: official BabyLM 2026 Strict corpus (HF: BabyLM-community/BabyLM-2026-Strict
#    + BabyLM-community/BabyLM-dev); on Modal use prep_data below instead
uv run python -m src.common.data --corpus-dir <raw_dir> --out data/babylm_strict/train.bin
# 2. one shared loop trains any variant with identical data order/schedule:
uv run python -m src.common.train --variant recursive_3to1
```

### On Modal (H100)

`modal_train.py` wraps the same `src/common/train.py` loop — same recipe,
just launched on a Modal H100 with data/checkpoints on Volumes:

```bash
modal setup                                                    # one-time auth
modal run modal_train.py::prep_data                            # download + tokenize official
                                                               # corpus onto the data volume
modal run modal_train.py::check_env                            # GPU/FA3/fla/data sanity check
modal run --detach modal_train.py::train_all                   # ALL 5 variants in parallel (the paper grid)
modal run --detach modal_train.py::main --variant recursive_3to1 --use-wandb   # or one variant

modal volume get babylm-checkpoints /recursive_3to1 ./checkpoints/
```

`recursive_3to1` needs `--micro-batch-size 8` (effective depth 24 OOMs an 80GB
H100 at the default 16); `train_all` applies that override automatically. The
optimizer step is identical (32 sequences) either way.

After training, `modal run modal_train.py::ckpt_eval` re-evaluates every
checkpoint of every variant with uniform per-token weighting and writes
`/checkpoints/analysis/ckpt_eval.json` — **all validation numbers in the paper
come from this**, not from the wandb val series (the training-time logger
weighted eval batches unevenly across micro-batch settings; see paper
Appendix A).

`modal run modal_train.py::blimp_eval_all` evaluates all five final
checkpoints on the official BabyLM 2026 full BLiMP and BLiMP Supplement sets
using the native models and causal summed-log-probability protocol. It writes
the full per-paradigm artifact to `/checkpoints/analysis/blimp_eval.json`;
the paper copy is `paper/figures/blimp_eval.json`.

Copy `.env.example` to `.env` for the optional W&B settings. `--use-wandb`
reads `WANDB_API_KEY` from the repo-root `.env` (gitignored) or your shell env
and forwards it into the container. Figure regeneration also requires the full
`WANDB_ENTITY_PROJECT=entity/project` path. `--resume
/checkpoints/<run>/ckpt_00300M.pt` continues an interrupted run.

## Status

- [x] `src/common` + all 5 variants implemented
- [x] Size report + smoke tests PASS (no NaNs, loss ↓ on all 5)
- [x] Corpus pinned (`BabyLM-community/BabyLM-2026-Strict` + `BabyLM-dev`), recipe locked (500M tokens, LR 6e-4)
- [x] All 5 runs finished on Modal H100s (2026-07-12) + uniform `ckpt_eval` re-evaluation
- [x] Full official BLiMP + BLiMP Supplement evaluation (2026-07-14)
- [x] Paired evaluation-sample bootstrap intervals for loss and BLiMP (2026-07-14)
- [x] Paper written: `paper/paper.md` (source) → `paper/latex/main.pdf` (ACL format)
- [ ] Remaining official Challenge evaluation (COMPS, entity tracking, GLUE, and fast checkpoint evaluation)

**Final validation loss** (uniform per-token, 2M dev tokens): recursive 3:1
**3.0926** < recursive 2:1 3.1072 < gdn 3:1 3.1121 < baseline 3.1244 <
gdn 2:1 3.1333 — recursion wins both param-matched pairs.

**Zero-shot BLiMP** (official 2026 full sets; macro accuracy):

| Variant | BLiMP | Supplement |
|---|---:|---:|
| baseline | **71.07** | **59.59** |
| gdn 2:1 | 62.24 | 52.28 |
| gdn 3:1 | 63.27 | 53.93 |
| recursive 2:1 | 65.56 | 53.30 |
| recursive 3:1 | 67.48 | 55.17 |

Recursion improves both matched pairs on BLiMP (+3.31 and +4.21 points);
paired 95% evaluation-sample intervals are [0.38, 6.85] and [2.43, 6.11].
The larger pure-GQA baseline leads overall. These intervals do not estimate
training-seed variance.
