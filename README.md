# babylm-recursive-hybrid

Weight-tied depth-recursive transformers with **hybrid Gated DeltaNet + GQA**
attention, for the **BabyLM 2026** workshop (non-competition paper track,
deadline 2026-07-15).

**Claim under test:** can a SMALLER (~50M non-embedding) weight-tied recursive
hybrid — sequences of GDN+GQA super-blocks, each recursed — beat LARGER
(~80M) non-recursive models (pure GQA and hybrid) under the BabyLM Strict
budget (≤100M words)?

## The variants

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
| untied 2:1 deep | `src/gdn_baseline/2to1_deep/` | GDN:GQA 2:1 | 18 | — | 18 | 140.7M |
| untied 3:1 deep | `src/gdn_baseline/3to1_deep/` | GDN:GQA 3:1 | 24 | — | 24 | 190.5M |
| recursive 2:1 uniq-init | `src/recursion_gdn/2to1_uniqinit/` | GDN:GQA 2:1 | 6 = 2 SBs × (g,g,q) | R=3 per SB | 18 | **46.9M** |
| recursive 3:1 uniq-init | `src/recursion_gdn/3to1_uniqinit/` | GDN:GQA 3:1 | 8 = 2 SBs × (g,g,g,q) | R=3 per SB | 24 | **63.5M** |

Recursive variants are a sequence of 2 independent super-blocks, each ONE
ratio unit, each applied 3 times with weights tied within that super-block.
The design yields **two exact learned-parameter-matched recursion pairs**:
`gdn_2to1` == `recursive_2to1` at
46.9M, and `gdn_3to1` == `recursive_3to1` at 63.5M. `baseline` (68.8M, pure
GQA) sits alongside as the no-GDN reference. Within each pair, the recursive
configuration also uses effective-depth residual scaling; the `uniqinit` arms
isolate that choice and show it accounts for at most 0.0034 nats, so the pairs
identify tying rather than the scaling. The `*_deep` arms are depth- and
FLOP-matched to the recursive models but deliberately NOT parameter-matched —
they price what tying costs. The four revision arms are registered in
`VARIANTS` but excluded from `modal_train.PAPER_VARIANTS`, so no fan-out
entrypoint trains them by default. Each leaf folder is a
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

`modal run modal_train.py::blimp_eval_all` evaluates each run's largest-budget
checkpoint on the official BabyLM 2026 full BLiMP and BLiMP Supplement sets
using the native models and causal summed-log-probability protocol. It writes
the full per-paradigm artifact to `/checkpoints/analysis/blimp_eval.json`;
the paper copy is `paper/figures/blimp_eval.json`. Both this and `ckpt_eval`
merge into the existing analysis file, so a `--runs`-filtered call extends it
rather than replacing it with only the runs it was asked for.

`modal run modal_train.py::zeroshot_eval_all` scores the official COMPS and
entity-tracking sets through `src/common/zeroshot_eval.py`, which implements the
same causal protocol but scores only the *completion* tokens (both tasks rank
candidates that share a prefix or a suffix). Results go to
`/checkpoints/analysis/zeroshot_eval.json` and `paper/figures/zeroshot_eval.json`.

`modal run modal_train.py::val_domains` records which token range of `val.bin`
came from which dev file (writing a `.domains.json` sidecar, after verifying a
re-tokenization is byte-identical to the bin in use), and
`modal run modal_train.py::domain_eval` then scores final checkpoints on each
dev domain separately — the domain-balanced numbers, written to
`/checkpoints/analysis/domain_eval.json` and copied to
`paper/figures/domain_eval.json`.

Copy `.env.example` to `.env` for the optional W&B settings. `--use-wandb`
reads `WANDB_API_KEY` from the repo-root `.env` (gitignored) or your shell env
and forwards it into the container. Figure regeneration also requires the full
`WANDB_ENTITY_PROJECT=entity/project` path. `--resume
/checkpoints/<run>/ckpt_00300M.pt` continues an interrupted run.

## Status

- [x] `src/common` + all 9 variants implemented; size report + smoke tests PASS
- [x] Corpus pinned (`BabyLM-community/BabyLM-2026-Strict` + `BabyLM-dev`), recipe locked (500M tokens, LR 6e-4)
- [x] All 5 primary runs finished on Modal H100s (2026-07-12) + uniform `ckpt_eval` re-evaluation
- [x] Full official BLiMP + BLiMP Supplement evaluation (2026-07-14)
- [x] **Review response (2026-09-08): 16 further runs, 21 total, ~12 GPU-hours**
  - [x] Seed replication (seeds 43, 44) of both matched pairs
  - [x] Compute-matched twins at 1B (annealed) and 1.5B tokens
  - [x] Untied-depth controls (`untied_*_deep`)
  - [x] Effective- vs unique-depth initialization ablation (`*_uniqinit`)
  - [x] 25M-token checkpoint grid locating the recursion crossover
  - [x] `ckpt_eval` over all 21 runs / 255 checkpoints; BLiMP over 13 models
- [x] Paper rewritten: `PAPER.md` (source) → `paper/latex/main.tex` (ACL format)
- [x] **BLiMP seed replication (2026-09-10)**: seeds 43/44 of both pairs, 21 models
      total — the 2:1 accuracy gain does not survive it
- [x] **Domain-balanced dev evaluation (2026-09-10)**: all six dev domains,
      13 final checkpoints, `domain_eval.json`
- [x] **COMPS + entity tracking (2026-09-10)**: all four official zero-shot suites
      now reported, 13 models, full sets — both null for recursion, and every
      model is at/below chance on entity tracking
- [ ] Fine-tuning (GLUE) — blocked on `GQAttention` having no `attention_mask`

**Headline: recursion pays per word, not per FLOP.**

| Contrast | Δ loss (nats) | Reading |
|---|---:|---|
| Recursion, matched exposure — 2:1 | **+0.0252** (seed sd 0.0015) | recursion wins |
| Recursion, matched exposure — 3:1 | **+0.0195** (seed sd 0.0005) | recursion wins |
| Compute-matched twin (1B, annealed) — 2:1 | **−0.0117** | twin wins at ⅔ the FLOPs |
| Compute-matched twin (1B, annealed) — 3:1 | **−0.0088** | twin wins at ⅔ the FLOPs |
| Untied depth vs tied — 2:1 / 3:1 | +0.0467 / +0.0399 | what tying costs (3× params) |
| Effective- vs unique-depth init — 2:1 / 3:1 | +0.0034 / +0.0003 | init is not the mechanism |
| Domain-balanced recursion — 2:1 / 3:1 | **+0.0270** / **+0.0196** | holds in 36/36 domain×seed cells |

Positive favours the treatment. Seed columns are means over seeds 42/43/44.

**Per-domain recursion advantage** (full 17.4M-token dev set, mean of 3 seeds):

| Domain | 2:1 | 3:1 |
|---|---:|---:|
| BNC Spoken | +0.0267 | +0.0207 |
| CHILDES | +0.0139 | +0.0099 |
| Gutenberg | +0.0406 | +0.0310 |
| OpenSubtitles | +0.0238 | +0.0190 |
| Simple Wiki | +0.0405 | +0.0267 |
| Switchboard | +0.0164 | +0.0100 |
| **Balanced** | **+0.0270** | **+0.0196** |

Positive in every cell. The effect is ~3× larger on edited written text than on
conversational speech — the 2M-token prefix used for the checkpoint series is
87% BNC Spoken, i.e. the region where recursion helps *least*.

**Final validation loss** (uniform per-token, 2M dev tokens, seed 42): untied 3:1
deep **3.0527** < untied 2:1 deep 3.0605 < gdn 3:1 @1B 3.0838 < recursive 3:1
3.0926 < gdn 2:1 @1B 3.0955 < recursive 2:1 3.1072 < gdn 3:1 3.1121 <
baseline 3.1244 < gdn 2:1 3.1333.

**Zero-shot BLiMP** (official 2026 full sets; macro accuracy):

| Variant | BLiMP | Supplement |
|---|---:|---:|
| baseline | **71.07** | **59.59** |
| gdn 2:1 | 62.24 | 52.28 |
| gdn 3:1 | 63.27 | 53.93 |
| recursive 2:1 | 65.56 | 53.30 |
| recursive 3:1 | 67.48 | 55.17 |
| untied 2:1 deep | 68.17 | 59.26 |
| untied 3:1 deep | 67.50 | 59.53 |
| gdn 2:1 @1.5B | 67.56 | 60.68 |
| gdn 3:1 @1.5B | 66.82 | 56.32 |
| gdn 2:1 @1B | 65.58 | 57.95 |
| gdn 3:1 @1B | 67.71 | 55.95 |

Table is seed 42. **BLiMP does not survive seed replication.** Re-running both
matched pairs at seeds 43 and 44:

| Pair | Suite | s42 | s43 | s44 | mean | sd |
|---|---|---:|---:|---:|---:|---:|
| 2:1 | BLiMP | +3.31 | −1.38 | −0.18 | **+0.58** | 2.44 |
| 2:1 | Suppl. | +1.01 | −4.82 | +2.54 | **−0.42** | 3.88 |
| 3:1 | BLiMP | +4.21 | +0.71 | +0.59 | **+1.84** | 2.05 |
| 3:1 | Suppl. | +1.24 | +4.20 | −0.25 | **+1.73** | 2.26 |

**COMPS and entity tracking are also null.** All four official zero-shot suites
are now reported (13 models, full sets):

| Task | Chance | Range, 13 models | 2:1 | 3:1 |
|---|---:|---|---:|---:|
| COMPS | 50.0 | 53.50–54.67 | −0.23 (sd 0.45) | +0.21 (sd 0.27) |
| Entity tracking | 20.0 | 16.79–19.11 | −0.99 (sd 1.41) | −0.10 (sd 0.55) |

Every model is **at or below chance on entity tracking** — outside binomial noise
over 6,780 items, and not a length artifact (correct options and distractors both
average 26.5 chars). The official GPT-2 reference gets 23.58. Neither suite
discriminates at this scale.

The cause of the BLiMP spread is the benchmark's noise floor: retraining *one*
architecture under a different seed moves BLiMP by up to **5.56 points** (gdn 2:1: 62.24 / 65.60 /
67.80, sd 2.80), which is larger than any architectural contrast in the grid —
including the baseline's 3.59-point lead. The previously reported 2:1 gain of
+3.31 is withdrawn; the 3:1 gain keeps its sign in all three seeds at +1.84 ±
2.05. Validation loss separates these same models at 0.0005–0.0015 nats of seed
noise and is the metric to select on at this scale.
