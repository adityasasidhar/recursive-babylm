"""Parameter report for the two-group design (revised 2026-07-05, user decision).

d_ff is pinned to 3x d_model for ALL variants, so there is no per-variant knob
to force a param match — sizes are whatever each variant's structure gives, and
that is intentional. This script is INFORMATIONAL (always exits 0): it prints
the size of every variant and highlights the two exact param-matched pairs the
design produces for free.

  uv run python -m src.common.param_count

The experiment's headline comparison is the smaller recursive hybrids vs the
larger non-recursive models; do not treat the spread below as a gate.
"""

from __future__ import annotations

import sys

from src.common.variants import VARIANTS, load_variant, non_embedding_params


def main() -> int:
    counts, cfgs = {}, {}
    for name in VARIANTS:
        cfg, Model = load_variant(name)
        model = Model(cfg)
        counts[name] = non_embedding_params(model)
        cfgs[name] = (cfg, sum(p.numel() for p in model.parameters()))
        del model

    w = max(len(n) for n in VARIANTS)
    print(
        f"{'variant':<{w}} {'d_model':>7} {'uniq layers':>11} {'SBs':>3} {'R':>2} "
        f"{'eff depth':>9} {'d_ff':>5} {'non-embed':>12} {'total':>12}"
    )
    for name in VARIANTS:
        cfg, total = cfgs[name]
        sbs = getattr(cfg, "n_super_blocks", 1)
        r = getattr(cfg, "n_recursions", 1)
        uniq = sbs * cfg.n_layers
        print(
            f"{name:<{w}} {cfg.d_model:>7} {uniq:>11} {sbs:>3} {r:>2} "
            f"{uniq * r:>9} {cfg.d_ff:>5} {counts[name]:>12,} {total:>12,}"
        )

    # The design produces two exact param-matched pairs for free: recursive
    # variants whose unique-layer composition equals a non-recursive variant's.
    # Each matches learned parameters and layer inventory. Recursive variants
    # additionally use effective-depth residual initialization, so this is a
    # comparison of the recursive configuration rather than tying alone.
    print("\nnatural param-matched pairs (recursion ablation, 0 param diff):")
    found = False
    for r_name in ("recursive_2to1", "recursive_3to1"):
        for nr_name in ("baseline", "gdn_2to1", "gdn_3to1"):
            if counts[r_name] == counts[nr_name]:
                print(
                    f"  {r_name} == {nr_name}  ({counts[r_name]:,} non-embed) — "
                    "learned parameters match exactly"
                )
                found = True
    if not found:
        print("  (none at current configs)")

    # Revision arms (BabyLM 2026 review response). These have invariants worth
    # checking rather than merely printing, so this section reports PASS/FAIL —
    # but the script stays informational and still exits 0 (smoke_test is the
    # gate that fails builds).
    print("\nrevision arms:")
    for base, deep, mult in (("gdn_2to1", "untied_2to1_deep", 3),
                             ("gdn_3to1", "untied_3to1_deep", 3)):
        if base not in counts or deep not in counts:
            continue
        # every non-embedding parameter is per-layer except the final norm,
        # so tripling the layer count triples everything but norm_f's d_model
        d_model = cfgs[base][0].d_model
        expect = (counts[base] - d_model) * mult + d_model
        ok = "PASS" if counts[deep] == expect else f"FAIL (expected {expect:,})"
        print(f"  {deep}: {counts[deep]:,} non-embed = {mult}x {base}'s layers  [{ok}]")
    for rec in ("recursive_2to1", "recursive_3to1"):
        abl = f"{rec}_uniqinit"
        if rec not in counts or abl not in counts:
            continue
        ok = "PASS" if counts[abl] == counts[rec] else "FAIL"
        print(f"  {abl}: parameter-identical to {rec} "
              f"({counts[abl]:,} non-embed; init-only change)  [{ok}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
