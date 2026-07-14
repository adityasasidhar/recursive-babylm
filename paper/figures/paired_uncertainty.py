"""Paired uncertainty estimates for the paper's controlled comparisons.

The validation analysis resamples the 488 aligned 4,096-token windows.  The
BLiMP analysis resamples aligned UIDs (67 paradigms for BLiMP and five tasks
for the supplement), preserving the macro-average used by the evaluator.

These intervals quantify finite-evaluation-sample uncertainty only.  They do
not estimate variation from training a new model with a different seed.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


PAIRS = {
    "2to1": ("gdn_2to1", "recursive_2to1"),
    "3to1": ("gdn_3to1", "recursive_3to1"),
}
FINAL_CHECKPOINT = "ckpt_00499M"


def percentile_ci(
    differences: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
    batch_size: int = 2_000,
) -> tuple[float, float]:
    """Percentile CI for the mean paired difference, in bounded memory."""
    n = len(differences)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch_size):
        stop = min(start + batch_size, resamples)
        indices = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = differences[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def exact_sign_test(differences: np.ndarray) -> dict[str, float | int]:
    """Two-sided exact binomial sign test after dropping exact ties."""
    positive = int(np.count_nonzero(differences > 0))
    negative = int(np.count_nonzero(differences < 0))
    ties = int(np.count_nonzero(differences == 0))
    n = positive + negative
    tail = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return {
        "positive": positive,
        "negative": negative,
        "ties": ties,
        "p_two_sided": min(1.0, 2.0 * probability),
    }


def summarize(
    differences: np.ndarray,
    rng: np.random.Generator,
    resamples: int,
) -> dict[str, object]:
    low, high = percentile_ci(differences, rng, resamples)
    return {
        "units": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "ci_95_percentile": [low, high],
        "sign_test": exact_sign_test(differences),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-eval", type=Path, default=Path("ckpt_eval.json"))
    parser.add_argument("--blimp-eval", type=Path, default=Path("blimp_eval.json"))
    parser.add_argument("--out", type=Path, default=Path("paired_uncertainty.json"))
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    ckpt = json.loads(args.ckpt_eval.read_text())
    blimp = json.loads(args.blimp_eval.read_text())
    rng = np.random.default_rng(args.seed)

    loss_results: dict[str, object] = {}
    blimp_results: dict[str, object] = {}
    for pair, (plain_name, recursive_name) in PAIRS.items():
        plain_loss = np.asarray(
            ckpt[plain_name][FINAL_CHECKPOINT]["chunk_losses"], dtype=np.float64
        )
        recursive_loss = np.asarray(
            ckpt[recursive_name][FINAL_CHECKPOINT]["chunk_losses"],
            dtype=np.float64,
        )
        if plain_loss.shape != recursive_loss.shape:
            raise ValueError(f"unaligned validation windows for {pair}")
        loss_results[pair] = summarize(
            plain_loss - recursive_loss, rng, args.resamples
        )

        suite_results: dict[str, object] = {}
        for suite in ("blimp", "blimp_supplement"):
            plain_uids = blimp["models"][plain_name]["suites"][suite]["groups"][
                "uid"
            ]
            recursive_uids = blimp["models"][recursive_name]["suites"][suite][
                "groups"
            ]["uid"]
            if plain_uids.keys() != recursive_uids.keys():
                raise ValueError(f"unaligned {suite} UIDs for {pair}")
            differences = np.asarray(
                [
                    recursive_uids[uid]["accuracy"]
                    - plain_uids[uid]["accuracy"]
                    for uid in sorted(plain_uids)
                ],
                dtype=np.float64,
            )
            suite_results[suite] = summarize(differences, rng, args.resamples)
        blimp_results[pair] = suite_results

    output = {
        "protocol": {
            "method": "paired nonparametric percentile bootstrap of mean differences",
            "confidence": 0.95,
            "resamples": args.resamples,
            "seed": args.seed,
            "validation_unit": "aligned 4096-token window",
            "blimp_unit": "aligned UID (paradigm/task)",
            "validation_direction": "non-recursive loss minus recursive loss",
            "blimp_direction": "recursive accuracy minus non-recursive accuracy",
            "scope": "finite evaluation-sample uncertainty; excludes training-seed variance",
        },
        "validation_loss": loss_results,
        "grammatical_accuracy": blimp_results,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    for pair in PAIRS:
        loss = loss_results[pair]
        blimp_pair = blimp_results[pair]
        print(
            f"{pair} loss: {loss['mean_difference']:.4f} "
            f"[{loss['ci_95_percentile'][0]:.4f}, "
            f"{loss['ci_95_percentile'][1]:.4f}]"
        )
        for suite in ("blimp", "blimp_supplement"):
            result = blimp_pair[suite]
            print(
                f"{pair} {suite}: {result['mean_difference']:.2f} "
                f"[{result['ci_95_percentile'][0]:.2f}, "
                f"{result['ci_95_percentile'][1]:.2f}]"
            )


if __name__ == "__main__":
    main()
