# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## What this is

A finished BabyLM 2026 (paper track) experiment: five from-scratch ~47–69M
non-embedding-parameter language models trained on the 100M-word Strict corpus,
plus the paper reporting them. The whole repo exists to support **one controlled
comparison** — does weight-tied depth recursion beat spending the same learned
parameters on independent layers? Two of the five variants are *exact*
parameter-matched twins of two others (46,913,888 and 63,487,504 non-embedding
params), differing only in that the recursive one re-applies each super-block
three times and scales residual-output init by effective depth.

Because of that, most files are not free to change independently. Anything that
touches tokenizer, corpus, data order, optimizer, schedule, token budget, or
`d_model`/`d_ff`/vocab breaks the comparison unless changed for all five
variants at once. `notes/design_decisions.md` is the authority on what is
pinned and why; read it before altering a config or the training loop.

## Commands

```bash
uv sync                                          # deps (uv is the convention; .venv/)

# gates — run after touching models or shared primitives
uv run python -m src.common.param_count          # size table; prints the matched pairs (informational, always exit 0)
uv run python -m src.common.smoke_test           # forward shapes + dummy-batch overfit; CUDA REQUIRED (fla kernels are Triton)
uv run python -m src.common.smoke_test --steps 5 # quicker
uv run python -m src.common.tokenizer            # snapshot shared tokenizer -> tokenizer/artifacts/ (offline use)

# data (local path; on Modal use prep_data instead)
uv run python -m src.common.data --corpus-dir <raw_dir> --split train --out data/babylm_strict/train.bin

# training — one shared loop, variant is the only model-level knob
uv run python -m src.common.train --variant recursive_3to1
```

There is no test suite, linter config, or CI. `param_count` + `smoke_test` are
the gates; `smoke_test` returns non-zero on failure and is the only thing that
actually asserts correctness.

### Modal (H100) — how the real runs happened

```bash
modal run modal_train.py::prep_data                       # download + tokenize official corpus onto the data volume
modal run modal_train.py::check_env                       # GPU/FA3/fla/tilelang/data sanity, incl. a real GDN fwd+bwd
modal run --detach modal_train.py::train_all              # all 5 variants in parallel, one H100 each (--detach required)
modal run --detach modal_train.py::main --variant recursive_3to1 --use-wandb
modal run modal_train.py::ckpt_eval                       # uniform per-token val loss for every ckpt -> /checkpoints/analysis/ckpt_eval.json
modal run modal_train.py::blimp_eval_all                  # official BLiMP + Supplement -> /checkpoints/analysis/blimp_eval.json
modal run modal_train.py::val_domains                     # per-domain token ranges for val.bin -> /data/val.bin.domains.json (CPU)
modal run modal_train.py::domain_eval                     # per-dev-domain val loss, final ckpts -> /checkpoints/analysis/domain_eval.json
modal run modal_train.py::zeroshot_eval_all               # official COMPS + entity tracking -> /checkpoints/analysis/zeroshot_eval.json
modal volume get babylm-checkpoints /<run_name> ./checkpoints/
```

### Paper

```bash
uv run --with matplotlib python paper/figures/make_figures.py     # needs WANDB_API_KEY + WANDB_ENTITY_PROJECT=entity/project
uv run python paper/figures/paired_uncertainty.py --ckpt-eval paper/figures/ckpt_eval.json \
    --blimp-eval paper/figures/blimp_eval.json --domain-eval paper/figures/domain_eval.json \
    --zeroshot-eval paper/figures/zeroshot_eval.json --out paper/figures/paired_uncertainty.json
cd paper/latex && tectonic main.tex                               # or pdflatex + bibtex twice
```

`.env` (gitignored, repo root) holds `WANDB_API_KEY`; figure regeneration also
needs `WANDB_ENTITY_PROJECT`. `modal_train.py` loads `.env` itself with a tiny
parser (no python-dotenv) and forwards the key as a Modal Secret. The README
tells you to copy `.env.example`, but that file is not present in the repo.

## Architecture

### Five variants, one shared everything-else

`src/common/` holds every shared piece; each variant is a self-contained leaf
folder with only `config.py` (frozen dataclass `Config`, exported as `CONFIG`)
and `model.py` (exports `Model`). The duplication between leaf `model.py` files
is deliberate — a variant is meant to be readable in one file, so do not
refactor them into a shared base class.

| Variant key | Package | Layers | Recursion |
|---|---|---|---|
| `baseline` | `src/baseline` | 10 pure GQA | — |
| `gdn_2to1` | `src/gdn_baseline/2to1` | 6 (GDN,GDN,GQA ×2) | — |
| `gdn_3to1` | `src/gdn_baseline/3to1` | 8 (GDN,GDN,GDN,GQA ×2) | — |
| `recursive_2to1` | `src/recursion_gdn/2to1` | 6 = 2 super-blocks × (g,g,q) | R=3 per SB |
| `recursive_3to1` | `src/recursion_gdn/3to1` | 8 = 2 super-blocks × (g,g,g,q) | R=3 per SB |

Four more arms are registered for the BabyLM 2026 review response. They are
**not** part of the reported grid — `modal_train.PAPER_VARIANTS` is the tuple the
fan-out entrypoints default to, and the paper's parameter-matched claim rests on
the five above only.

| Variant key | Package | Layers | Purpose |
|---|---|---|---|
| `untied_2to1_deep` | `src/gdn_baseline/2to1_deep` | 18 independent (140.7M) | depth/FLOPs-matched to `recursive_2to1`, untied — prices what tying costs |
| `untied_3to1_deep` | `src/gdn_baseline/3to1_deep` | 24 independent (190.5M) | same, for the 3:1 family |
| `recursive_2to1_uniqinit` | `src/recursion_gdn/2to1_uniqinit` | 6 = 2 SB × (g,g,q), R=3 | identical to `recursive_2to1` but unique-depth residual init — separates tying from depth-aware scaling |
| `recursive_3to1_uniqinit` | `src/recursion_gdn/3to1_uniqinit` | 8 = 2 SB × (g,g,g,q), R=3 | same, for the 3:1 family |

The `_uniqinit` leaves must stay in lockstep with the recursive leaves they
ablate — they differ **only** in `_init_own_weights`, and `param_count` checks
that they remain parameter-identical.

`2to1`/`3to1` are not valid Python identifiers, so **never `import` a variant
directly** — go through `src/common/variants.py::load_variant(name)`, which uses
`importlib`. `VARIANTS` there is the single registry; adding a variant means
adding a key there, and it will automatically appear in `param_count`,
`smoke_test`, `train --variant`, `check_env`, `ckpt_eval`, `blimp_eval_all`,
`domain_eval`, and `zeroshot_eval_all`.
The leaf `model.py` files themselves resolve their own `Config` via
`importlib.import_module("src.recursion_gdn.3to1.config")`.

Recursion is *within* a super-block: two independent super-blocks in sequence,
each applied `n_recursions` times with weights tied inside it, never tied across
the two. `_init_own_weights` scales residual-output projections by
`1/sqrt(2 * n_super_blocks * n_recursions * n_layers)` — effective, not unique,
depth — which is part of what the experiment identifies, so it is not a
detachable detail.

### Shared primitives

- `attention.py` — `GQAttention` (RoPE + QK-Norm; backend `auto`/`fa3`/`sdpa`,
  where `auto` picks FlashAttention-3 only on Hopper with a CUDA half tensor and
  otherwise falls back to torch SDPA) and `GatedDeltaNetLayer`, a thin wrapper
  over `fla`'s reference `GatedDeltaNet`. GDN is imported *inside* `__init__` so
  CPU param counting works; forward/backward needs a GPU.
- `layers.py` — SwiGLU, QKNorm, RoPE. RMSNorm comes from `torch.nn.RMSNorm`.
- `tokenizer.py` — one pinned pretrained BabyLM-community BPE tokenizer, vocab
  16384, asserted against every config's `vocab_size`. Prefers the local
  `tokenizer/artifacts/` snapshot when present.
- `data.py` — tokenizes the corpus once into a flat `uint16` memmap; suffix
  routing (`.txt`/`.train` → train, `.dev` → val) is what keeps dev out of
  train. `LMChunkDataset` orders chunks by a `(DATA_SEED=42, epoch)` permutation
  so every variant sees identical tokens in identical order.
- `train.py` (440 lines) — the whole recipe: `TrainConfig` defaults are the
  locked paper settings (lr 6e-4, batch 32 seqs via grad accum, seq 4096,
  500M-token budget, cosine to 10%). Also does provenance logging (git commit,
  env, data sha256 fingerprint), per-block grad-norm and residual-RMS
  diagnostics, and resumable checkpoints every 100M tokens.
- `blimp_eval.py` — evaluates *native* `nn.Module` checkpoints with the official
  causal summed-log-prob protocol, so there is no HF conversion step. Dataset,
  evaluator, and tokenizer revisions are pinned as constants at the top. Its
  numbers are the paper's published BLiMP source of truth, so **do not refactor
  it** — extend alongside it instead.
- `zeroshot_eval.py` — the same official protocol for COMPS and entity tracking,
  deliberately a separate module for that reason. These two tasks score only the
  *completion* tokens (COMPS conditions one shared property phrase on two
  differing prefixes; entity tracking ranks five completions after one prefix),
  and entity tracking macro-averages in two levels over its three splits. Pinning
  constants are imported from `blimp_eval` so the two cannot drift.

`micro_batch_size` is a memory-layout knob only; gradient accumulation keeps the
optimizer batch at 32 sequences. `recursive_3to1` needs `--micro-batch-size 8`
(effective depth 24 OOMs an 80GB H100 at 16); `train_all` applies that override.

### Evaluation provenance (matters when touching numbers)

All validation numbers in the paper come from `modal_train.py::ckpt_eval`, not
from the wandb `val/` series — the training-time logger weighted eval batches
unevenly across micro-batch settings. The paper's copies of the artifacts live
in `paper/figures/{ckpt_eval,blimp_eval,domain_eval,zeroshot_eval,paired_uncertainty}.json` and
are the committed source of truth for every table and figure. `ckpt_eval` scores
only the first 2M tokens of `val.bin`, which sorted-file concatenation makes ~87%
BNC Spoken; `domain_eval` scores final checkpoints on all six dev domains and is
where the domain-balanced numbers the paper cites come from. `data.prepare`
writes a `.domains.json` sidecar recording each source file's token range, and
`val_domains` can rebuild it for an existing bin without overwriting it. `paired_uncertainty.py`
bootstraps over evaluation samples only; it does **not** estimate training-seed
variance, and the paper says so — do not let a claim drift past that.

**The analysis writers merge, they do not replace.** `ckpt_eval`,
`blimp_eval_all`, `domain_eval` and `zeroshot_eval_all` all seed from the
existing JSON on the volume, so a `--runs`-filtered call extends it instead of
replacing the file with only the runs it was asked for. (An earlier version did
replace, and silently dropped the five primary runs from the volume's
`blimp_eval.json`.) `zeroshot_eval_all` additionally records `limit_per_file` per
entry and drops entries evaluated at a different limit, so smoke-test rows cannot
leak into a real run.

Run directories are `{variant}` or `{variant}_{tag}`. `ckpt_eval` and
`blimp_eval_all` both take a `--runs` filter and discover runs by directory,
resolving the variant with `modal_train._resolve_variant` (longest-prefix match,
so `recursive_2to1_uniqinit` beats `recursive_2to1`); `blimp_eval_all` picks each
run's largest-budget checkpoint rather than a fixed `ckpt_00499M.pt`, and
`paired_uncertainty.final_checkpoint` does the same. `--seed` (on `train`,
`main`, and `train_all`) sets weight init **and** the data-order permutation
together, so a seed group stays internally controlled across variants while
differing from other groups.

### Paper

`PAPER.md` at the repo root is the prose source; `paper/latex/main.tex` →
`main.pdf` is the ACL-formatted deliverable, with figures duplicated into
`paper/latex/figures/`. They are maintained by hand, so a results change means
editing both plus the README's results tables. The README still refers to
`paper/paper.md`, which has moved to the root `PAPER.md`.

## Hard rules from the project

- **Do not launch real training without the project owner's sign-off.** Stated
  in `train.py`, `modal_train.py`, and `notes/design_decisions.md`. Smoke tests
  and param counting are always fine.
- Do not resize one member of a matched pair on its own; `param_count.py` is the
  executable source of truth for the counts, and the pair-match must stay at a
  0-parameter difference.
- Local GPU is an RTX 3050 (4GB) — smoke tests only; real runs go to Modal.
- `data/`, `checkpoints/`, `tokenizer/artifacts/`, `wandb/`, and
  `eval/babylm-eval/` (a vendored clone of `babylm-org/babylm-eval`, pinned at
  commit `3d57ddc`) are gitignored and regenerated, not committed.
