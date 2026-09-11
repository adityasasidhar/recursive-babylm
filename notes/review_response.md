# Response to BabyLM 2026 reviews (Submission 18)

Both reviewers scored 3 (Borderline), confidence 4. Every substantive request was
run rather than deferred: 16 new training runs (21 total, ~12 GPU-hours on H100),
`ckpt_eval` over 255 checkpoints, BLiMP over 21 models, and a domain-balanced
re-evaluation over the whole dev set. **Three results went against the original
paper, and all three are reported**: the compute-matched comparison, the BLiMP
seed replication, and (in the paper's favour) the domain breakdown.

## Reviewer BRBm

### 1. "The grid compares tied 18 layers against untied 6 layers, at 3× the FLOPs … It does not establish that tying is a good way to obtain depth."

**Accepted, and addressed with two new arms.**

- **Untied-depth control** (`untied_2to1_deep`, `untied_3to1_deep`): 18 and 24
  *independent* layers — same depth, same FLOPs, same residual-init scale as the
  recursive models, 3× the parameters. Untying is worth **+0.0467** and
  **+0.0399** nats. Tying therefore forfeits ~0.04 nats and returns a 3×
  parameter saving. Measured throughput independently confirms the FLOP match
  (172,852 vs 180,660 tok/s at 2:1; 133,451 vs 134,334 at 3:1), so the parity
  claim does not rest on the analytic 3× alone.
- The framing question is now stated explicitly as under-specified (§1): "scarce"
  can mean scarce parameters, compute, or data, and the paper measures all three.

### 2. "§6 notes that a compute-matched training comparison remains necessary but treats it as future work. Runs consume ~295M words against the Strict track's 1B allowance."

**Accepted, run, and it goes against the original claim.**

The twins were trained to 1.5B tokens (exact FLOP parity, ~884M words) *and* to
1B tokens annealed to 1B (~589M words, two thirds of parity). Both remain inside
the exposure cap. At the 1B point the twins reach 3.0955 and 3.0838, **beating
the recursive models by 0.0117 and 0.0088 nats while using two thirds of the
compute**. Recursion wins only 128/488 and 160/488 windows.

The 1.5B arm alone would have flattered recursion: both twins bottom out near
1.0B tokens and degrade thereafter as repetition passes ~6 epochs, so the
FLOP-parity endpoint (3.1081, 3.1120) understates what the twin can do. We report
both and lead with the one less favourable to us.

The paper now adopts the reviewer's own suggested framing, sharpened by the
exposure axis: **recursion converts compute into quality at fixed parameter
count, and does so per word rather than per FLOP.** At matched exposure recursion
wins (+0.0252/+0.0195); at matched compute it loses. On a fixed-corpus benchmark
the data budget is the binding one, which is why the result still belongs here —
but the compute claim the paper never made is now measured and negative.

### 3. "§3.2 identifies … the recursive configuration rather than tying alone. The paper states an alternative-scaling ablation is needed and does not run it."

**Run. The confound is real but small, and resolves in the paper's favour.**

`recursive_*_uniqinit` are parameter-identical to the recursive models and differ
only in scaling residual-output projections by unique rather than effective depth
(a √3 difference). Effect: **+0.0034** nats (2:1) and **+0.0003** (3:1). The 3:1
contrast is a clean null (interval spans zero, sign test 257/488 ≈ chance); the
2:1 effect is ~2× the seed sd and wins only 62.5% of windows. BLiMP shows nothing
either way (−0.20, −0.06). Tying does the work, not the scaling. The paper now
recommends depth-aware residual scaling as a stability measure only.

### Request: "re-evaluate the retained checkpoints on a balanced dev sample and report per-domain deltas."

**Done (2026-09-10).** `data.prepare` now records each dev file's token range in
a `.domains.json` sidecar; `modal_train.py::val_domains` recovered the manifest
for the existing `val.bin` after verifying a re-tokenization is byte-identical to
it, and `domain_eval` scored the final checkpoint of all thirteen 500M-token runs
on the whole 17,390,678-token dev set, in windows contained within one domain.

The advantage is positive in **36 of 36** domain×seed×family cells. Domain-balanced
(macro over the six domains): **+0.0270** (2:1) and **+0.0196** (3:1), against
+0.0252 / +0.0195 on the old prefix — so the narrow slice was *conservative*, not
flattering. The effect is ~3× larger on edited written text (Gutenberg +0.0406,
Simple Wiki +0.0405) than on conversational speech (CHILDES +0.0139, Switchboard
+0.0164), which is what a depth account predicts; the old prefix was 87% BNC
Spoken, i.e. the region where recursion helps least. New §5.2 table, artifact in
`paper/figures/domain_eval.json`.

The 255-checkpoint *trajectories* are still on the 2M prefix, to bound cost; that
narrower point is now the Limitation.

### Comment: related work does not contextualize the contribution within BabyLM (e.g. Haller et al. 2024 & 2025).

**Accepted.** §2 now has a "Subquadratic token mixers at BabyLM scale" paragraph
citing BabyHGRN (Haller, Golde and Akbik, CoNLL-BabyLM 2024) and BLaLM (same
authors, BabyLM Workshop 2025), and states how our setting differs: we retain a
*minority* of softmax layers rather than eliminating attention, and our object of
study is the re-application schedule of a fixed layer inventory, not the mixer.

## Reviewer NsKJ

### "Single training seed."

**Addressed.** Both matched pairs were retrained at seeds 43 and 44 (a seed sets
weight init *and* data order together, so within a seed group the control is
preserved). Paired recursion advantage:

| Pair | s42 | s43 | s44 | mean | sd |
|---|---|---|---|---|---|
| 2:1 | +0.0261 | +0.0236 | +0.0261 | +0.0252 | 0.0015 |
| 3:1 | +0.0195 | +0.0200 | +0.0190 | +0.0195 | 0.0005 |

Per-model seed sd is 0.0005–0.0020. The published single-seed values sit on the
three-seed means. Establishing this floor is what makes the compute result
(6–8σ) and the initialization null (≤2σ) interpretable.

### "Not training to 1B for a proper comparison with the GPT-2 baseline."

**Partly addressed, and a misreading corrected.** The review states "the baseline
was trained for 10 epochs / 1B words." It was not: our GQA baseline used the
*identical* 500M-token budget as every other primary run — the 1B/10-epoch figure
belongs to the official GPT-2 reference we cite for external context in §5.1.
That two careful readers could take it otherwise means the text was unclear, so
Table 2 now carries an explicit Tokens column and §5.7 states the budget in
prose. We do add 1B- and 1.5B-token arms (above), though as compute controls
rather than as a GPT-2 comparison.

### "Limited number of BabyLM zero-shot scores & no finetuning scores."

**Zero-shot: done (2026-09-10). Fine-tuning: still declined.** All four official
zero-shot suites are now reported. `src/common/zeroshot_eval.py` implements the
official causal protocol for COMPS and entity tracking (scoring only the
*completion* tokens, with entity tracking's two-level macro-average), kept
separate from `blimp_eval.py` so the published BLiMP path stays byte-identical.
All thirteen 500M-token runs were evaluated on the full sets.

Both are null for recursion — COMPS −0.23 (sd 0.45) and +0.21 (sd 0.27);
entity tracking −0.99 (sd 1.41) and −0.10 (sd 0.55), with three of four
contrasts changing sign across seeds. The more useful observation is that
**every model is at or below the 20% chance level on entity tracking**
(16.79–19.11; 6,780 items, and not a length artifact — correct options and
distractors both average 26.5 characters). The official GPT-2 reference manages
23.58. Neither suite discriminates at this scale, which is consistent with, and
strengthens, the §5.8 argument that accuracy benchmarks cannot resolve this
contrast at 100M words while loss can.

Fine-tuning remains not run: it needs a Hugging Face wrapper *and* an
`attention_mask` argument on our GQA implementation, which has none — a change
to the one shared primitive the whole controlled comparison rests on.

**BLiMP seed replication: done (2026-09-10), and it goes against us.** Both
matched pairs were evaluated at seeds 43 and 44 (21 models in `blimp_eval.json`).
The 2:1 gain of +3.31 becomes **+0.58 (sd 2.44), negative in two of three seeds**,
and −0.42 on the supplement; the 3:1 gain of +4.21 becomes **+1.84 (sd 2.05)**,
positive in all three. The cause is the benchmark's noise floor at this scale:
retraining *one* architecture under a different seed moves BLiMP macro-accuracy
by up to **5.56 points** (gdn 2:1: 62.24 / 65.60 / 67.80, sd 2.80) — larger than
any architectural contrast in the grid, including the baseline's 3.59-point lead.

We therefore **withdraw the 2:1 accuracy claim**, hold the 3:1 one weakly, and
promote the noise measurement to a contribution (new §5.8), since BabyLM-scale
papers routinely compare architectures on single-seed BLiMP differences of one to
three points. Validation loss separates the same models at 0.0005–0.0015 nats of
seed noise. The other nine models remain seed-42 only, which is now the
Limitation.

## What changed in the artifacts

| File | Change |
|---|---|
| `PAPER.md`, `paper/latex/main.tex` | Rewritten around the exposure-vs-compute distinction; §5.2–5.6 are new |
| `paper/figures/ckpt_eval.json` | 5 runs / 25 ckpts → 21 runs / 255 ckpts (original 5 reproduce bit-for-bit) |
| `paper/figures/blimp_eval.json` | 5 → 13 models, identical pinned protocol |
| `paper/figures/paired_uncertainty.py` | Generalized to 8 contrasts + a `seed_replication` block |
| `paper/figures/budget_axes.{png,pdf}` | New figure: the same intervention against both budgets |
| `src/common/variants.py` | 4 revision arms registered, excluded from `PAPER_VARIANTS` |
| `paper/figures/domain_eval.json` | New: per-domain losses, 13 models × 6 dev domains |
| `src/common/data.py` | `prepare` writes a `.domains.json` token-range sidecar |
| `modal_train.py` | `val_domains` + `domain_eval` + `zeroshot_eval_all`; the analysis writers now merge instead of overwriting a filtered file |
| `src/common/zeroshot_eval.py` | New: official COMPS + entity-tracking scorer for native checkpoints |
| `paper/figures/zeroshot_eval.json` | New: COMPS + entity tracking, 13 models, full official sets |
| `src/common/train.py` | `--seed` and `--ckpt-every-tokens` exposed |
| `modal_train.py` | Run-directory discovery, max-token checkpoint selection, `--runs`/`--variants` filters |

## Section removed

The original §5.2 ("Coarse density comparison: no identified causal effect") was
dropped. It was explicitly non-causal, concluded nothing, and its space is now
taken by four experiments that do conclude something. The per-position data
survives in `ckpt_eval.json` (`per_pos` on final checkpoints) if it is wanted
back.
