"""Emit (never launch) the LR pilot-sweep commands.

    uv run python -m src.common.pilot

6 runs x 100M tokens on the rented H100: the 3:1 recursion-ablation pair
(gdn_3to1 = non-recursive competitor, recursive_3to1 = weight-tied twin)
across peak LR {3e-4, 6e-4, 1e-3}. Selection metric: val loss at 100M tokens.
Decision rule (locked with Aditya 2026-07-06): pick the largest LR where BOTH
variants are stable (no loss spikes / grad-norm blowups) and val loss is best
or within noise; on disagreement take the LR that favors the NON-recursive
model. Launching requires explicit sign-off.
"""

from __future__ import annotations

PILOT_TOKENS = 100_000_000
PILOT_WARMUP = 60  # ~8% of the 763 pilot steps (250 would be a third of the run)
LRS = ["3e-4", "6e-4", "1e-3"]
PILOT_VARIANTS = ["gdn_3to1", "recursive_3to1"]


def commands() -> list[str]:
    return [
        (
            f"uv run python -m src.common.train --variant {v} --lr {lr} "
            f"--token-budget {PILOT_TOKENS} --warmup-steps {PILOT_WARMUP} "
            f"--tag pilot_lr{lr} --wandb"
        )
        for v in PILOT_VARIANTS
        for lr in LRS
    ]


if __name__ == "__main__":
    print(__doc__)
    for c in commands():
        print(c)
