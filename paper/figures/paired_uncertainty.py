"""Paired uncertainty estimates for the paper's controlled comparisons.

The validation analysis resamples the 488 aligned 4,096-token windows.  The
BLiMP analysis resamples aligned UIDs (67 paradigms for BLiMP and five tasks
for the supplement), preserving the macro-average used by the evaluator.

Two kinds of uncertainty are reported, and they answer different questions:

  * Bootstrap intervals and sign tests over evaluation units quantify
    finite-evaluation-sample uncertainty for a single pair of trained models.
  * The ``seed_replication`` block reports the same contrast recomputed from
    independently seeded training runs, which is the only estimate here of
    run-to-run training variance.  Seeds vary weight init AND data order
    together, so within one seed group every variant still sees identical
    tokens in identical order.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np


# name -> (reference run, treatment run).  Loss difference is reported as
# reference minus treatment and accuracy as treatment minus reference, so a
# POSITIVE number always means "the treatment is better".
CONTRASTS = {
    # the paper's headline: parameter-matched recursion vs its twin
    "recursion_2to1": ("gdn_2to1", "recursive_2to1"),
    "recursion_3to1": ("gdn_3to1", "recursive_3to1"),
    # what tying costs: same depth and FLOPs, 3x the learned parameters
    "untied_depth_2to1": ("recursive_2to1", "untied_2to1_deep"),
    "untied_depth_3to1": ("recursive_3to1", "untied_3to1_deep"),
    # does effective-depth residual scaling carry the recursion gain?
    "effective_depth_init_2to1": ("recursive_2to1_uniqinit", "recursive_2to1"),
    "effective_depth_init_3to1": ("recursive_3to1_uniqinit", "recursive_3to1"),
    # compute parity: the twin trained to 1B tokens, annealed to 1B, using
    # two THIRDS of the recursive model's training FLOPs
    "compute_matched_2to1": ("gdn_2to1_anneal1b", "recursive_2to1"),
    "compute_matched_3to1": ("gdn_3to1_anneal1b", "recursive_3to1"),
}

# contrast -> per-seed (reference, treatment) run names
SEED_REPLICATION = {
    f"recursion_{fam}": [
        (f"gdn_{fam}{suffix}", f"recursive_{fam}{suffix}")
        for suffix in ("", "_seed43", "_seed44")
    ]
    for fam in ("2to1", "3to1")
}
SEEDS = (42, 43, 44)


def final_checkpoint(run: dict) -> str:
    """Name of the largest-budget checkpoint in one run's ckpt_eval entry.

    Not a constant: the compute-matched arms train to 1B or 1.5B tokens and so
    end at ckpt_00999M / ckpt_01499M, while the paper grid ends at ckpt_00499M.
    """
    return max(run, key=lambda name: run[name]["tokens_M"])


def window_losses(ckpt: dict, run: str) -> np.ndarray:
    return np.asarray(
        ckpt[run][final_checkpoint(ckpt[run])]["chunk_losses"], dtype=np.float64
    )


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


def suite_differences(blimp: dict, reference: str, treatment: str, suite: str):
    """Per-UID accuracy differences (treatment minus reference), or None when
    either model has not been evaluated on this suite."""
    models = blimp["models"]
    if reference not in models or treatment not in models:
        return None
    ref = models[reference]["suites"][suite]["groups"]["uid"]
    trt = models[treatment]["suites"][suite]["groups"]["uid"]
    if ref.keys() != trt.keys():
        raise ValueError(f"unaligned {suite} UIDs for {reference} vs {treatment}")
    return np.asarray(
        [trt[uid]["accuracy"] - ref[uid]["accuracy"] for uid in sorted(ref)],
        dtype=np.float64,
    )


def across_seeds(values: list[float], per_seed: dict) -> dict[str, object]:
    """Summary of one contrast recomputed under independent training seeds."""
    return {
        "per_seed": per_seed,
        "mean_difference": statistics.mean(values),
        "sd_difference": statistics.stdev(values),
        "min_difference": min(values),
        "max_difference": max(values),
        "seeds_favouring_treatment": int(sum(v > 0 for v in values)),
        "seeds": len(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt-eval", type=Path, default=Path("ckpt_eval.json"))
    parser.add_argument("--blimp-eval", type=Path, default=Path("blimp_eval.json"))
    parser.add_argument("--domain-eval", type=Path, default=Path("domain_eval.json"))
    parser.add_argument("--zeroshot-eval", type=Path,
                        default=Path("zeroshot_eval.json"))
    parser.add_argument("--out", type=Path, default=Path("paired_uncertainty.json"))
    parser.add_argument("--resamples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260714)
    args = parser.parse_args()

    ckpt = json.loads(args.ckpt_eval.read_text())
    blimp = json.loads(args.blimp_eval.read_text())
    domain = (json.loads(args.domain_eval.read_text())
              if args.domain_eval and args.domain_eval.exists() else None)
    zeroshot = (json.loads(args.zeroshot_eval.read_text())
                if args.zeroshot_eval and args.zeroshot_eval.exists() else None)

    # One flat {suite -> {run -> macro accuracy}} view over both accuracy
    # artifacts, so BLiMP and the two extra zero-shot tasks get identical
    # seed-replication and noise-floor treatment.
    accuracy_by_suite: dict[str, dict[str, float]] = {}
    for suite in ("blimp", "blimp_supplement"):
        accuracy_by_suite[suite] = {
            run: entry["suites"][suite]["macro_accuracy"]
            for run, entry in blimp["models"].items()
            if suite in entry["suites"]
        }
    if zeroshot is not None:
        for task in ("comps", "entity_tracking"):
            accuracy_by_suite[task] = {
                run: entry["tasks"][task]["macro_accuracy"]
                for run, entry in zeroshot["models"].items()
                if task in entry["tasks"]
            }
    rng = np.random.default_rng(args.seed)

    loss_results: dict[str, object] = {}
    blimp_results: dict[str, object] = {}
    for name, (reference, treatment) in CONTRASTS.items():
        if reference not in ckpt or treatment not in ckpt:
            continue
        ref_loss = window_losses(ckpt, reference)
        trt_loss = window_losses(ckpt, treatment)
        if ref_loss.shape != trt_loss.shape:
            raise ValueError(f"unaligned validation windows for {name}")
        loss_results[name] = {
            "reference": reference,
            "treatment": treatment,
            **summarize(ref_loss - trt_loss, rng, args.resamples),
        }
        suites: dict[str, object] = {}
        for suite in ("blimp", "blimp_supplement"):
            differences = suite_differences(blimp, reference, treatment, suite)
            if differences is not None:
                suites[suite] = summarize(differences, rng, args.resamples)
        if suites:
            blimp_results[name] = {
                "reference": reference,
                "treatment": treatment,
                **suites,
            }

    # Training-seed variance: the same contrast from independently seeded runs.
    seed_results: dict[str, object] = {}
    for name, pairs in SEED_REPLICATION.items():
        deltas, per_seed = [], {}
        for seed, (reference, treatment) in zip(SEEDS, pairs):
            if reference not in ckpt or treatment not in ckpt:
                continue
            ref_mean = float(window_losses(ckpt, reference).mean())
            trt_mean = float(window_losses(ckpt, treatment).mean())
            delta = ref_mean - trt_mean
            deltas.append(delta)
            per_seed[str(seed)] = {
                "reference_loss": ref_mean,
                "treatment_loss": trt_mean,
                "difference": delta,
            }
        if len(deltas) >= 2:
            seed_results[name] = across_seeds(deltas, per_seed)

    # The same seed groups, scored on grammatical accuracy.  Validation loss is
    # stable across seeds; BLiMP is not, and reporting the two side by side is
    # the point of this block rather than an aside.
    blimp_seed_results: dict[str, object] = {}
    for name, pairs in SEED_REPLICATION.items():
        suites: dict[str, object] = {}
        for suite, accuracies in accuracy_by_suite.items():
            deltas, per_seed = [], {}
            for seed, (reference, treatment) in zip(SEEDS, pairs):
                if reference not in accuracies or treatment not in accuracies:
                    continue
                ref_acc, trt_acc = accuracies[reference], accuracies[treatment]
                deltas.append(trt_acc - ref_acc)
                per_seed[str(seed)] = {
                    "reference_accuracy": ref_acc,
                    "treatment_accuracy": trt_acc,
                    "difference": trt_acc - ref_acc,
                }
            if len(deltas) >= 2:
                suites[suite] = across_seeds(deltas, per_seed)
        if suites:
            blimp_seed_results[name] = suites

    # Noise floor: one architecture retrained under a different seed, which is
    # what the contrast above has to clear to mean anything.
    accuracy_noise: dict[str, object] = {}
    for fam in ("2to1", "3to1"):
        for prefix in ("gdn", "recursive"):
            runs = [f"{prefix}_{fam}{tag}" for tag in ("", "_seed43", "_seed44")]
            for suite, accuracies in accuracy_by_suite.items():
                if not all(r in accuracies for r in runs):
                    continue
                accs = [accuracies[r] for r in runs]
                accuracy_noise[f"{prefix}_{fam}/{suite}"] = {
                    "per_seed": dict(zip((str(x) for x in SEEDS), accs)),
                    "mean": statistics.mean(accs),
                    "sd": statistics.stdev(accs),
                    "range": max(accs) - min(accs),
                }

    # Per-domain deltas on the full dev set, answering whether the headline
    # contrast survives outside the 2M-token prefix ckpt_eval scores.
    domain_results: dict[str, object] = {}
    if domain is not None:
        entries = domain["models"]
        names = list(domain["protocol"]["windows_per_domain"]) + ["balanced_macro"]
        for name, pairs in SEED_REPLICATION.items():
            per_domain: dict[str, object] = {}
            for dom_name in names:
                deltas, per_seed = [], {}
                for seed, (reference, treatment) in zip(SEEDS, pairs):
                    if reference not in entries or treatment not in entries:
                        continue

                    def value(run: str) -> float:
                        entry = entries[run]
                        return (entry["balanced_macro"]
                                if dom_name == "balanced_macro"
                                else entry["per_domain"][dom_name]["mean_loss"])

                    delta = value(reference) - value(treatment)
                    deltas.append(delta)
                    per_seed[str(seed)] = {
                        "reference_loss": value(reference),
                        "treatment_loss": value(treatment),
                        "difference": delta,
                    }
                if len(deltas) >= 2:
                    per_domain[dom_name] = across_seeds(deltas, per_seed)
            if per_domain:
                domain_results[name] = per_domain

    output = {
        "protocol": {
            "method": "paired nonparametric percentile bootstrap of mean differences",
            "confidence": 0.95,
            "resamples": args.resamples,
            "seed": args.seed,
            "validation_unit": "aligned 4096-token window",
            "blimp_unit": "aligned UID (paradigm/task)",
            "sign_convention": "positive favours the treatment run in every contrast",
            "validation_direction": "reference loss minus treatment loss",
            "blimp_direction": "treatment accuracy minus reference accuracy",
            "scope": (
                "bootstrap intervals cover finite evaluation-sample uncertainty for "
                "one pair of trained models; seed_replication covers run-to-run "
                "training variance over three seeds"
            ),
            "domain_unit": "full-dev per-domain mean loss, windows contained "
                           "within one domain (see domain_eval.json)",
            "accuracy_seed_noise": (
                "spread of one architecture's macro-accuracy across three "
                "training seeds; the floor any single-seed accuracy contrast "
                "must clear"
            ),
        },
        "validation_loss": loss_results,
        "grammatical_accuracy": blimp_results,
        "seed_replication": seed_results,
        "seed_replication_accuracy": blimp_seed_results,
        "accuracy_seed_noise": accuracy_noise,
        "domain_breakdown": domain_results,
    }
    args.out.write_text(json.dumps(output, indent=2) + "\n")

    for name, loss in loss_results.items():
        low, high = loss["ci_95_percentile"]
        print(f"{name:<28} loss {loss['mean_difference']:+.4f} [{low:+.4f}, {high:+.4f}]"
              f"  wins {loss['sign_test']['positive']}/{loss['units']}")
        for suite in ("blimp", "blimp_supplement"):
            entry = blimp_results.get(name, {}).get(suite)
            if entry:
                lo, hi = entry["ci_95_percentile"]
                print(f"{'':<28} {suite:<17} {entry['mean_difference']:+.2f} "
                      f"[{lo:+.2f}, {hi:+.2f}]")
    for name, res in seed_results.items():
        print(f"{name:<28} seeds {res['mean_difference']:+.4f} "
              f"sd {res['sd_difference']:.4f} "
              f"[{res['min_difference']:+.4f}, {res['max_difference']:+.4f}]")
    for name, suites in blimp_seed_results.items():
        for suite, res in suites.items():
            print(f"{name:<28} {suite:<17} seeds {res['mean_difference']:+.2f} "
                  f"sd {res['sd_difference']:.2f} "
                  f"[{res['min_difference']:+.2f}, {res['max_difference']:+.2f}] "
                  f"{res['seeds_favouring_treatment']}/{res['seeds']} seeds positive")
    for key, res in accuracy_noise.items():
        print(f"{'noise floor':<28} {key:<26} sd {res['sd']:.2f} "
              f"range {res['range']:.2f}")
    for name, per_domain in domain_results.items():
        for dom_name, res in per_domain.items():
            print(f"{name:<28} {dom_name:<17} {res['mean_difference']:+.4f} "
                  f"sd {res['sd_difference']:.4f} "
                  f"{res['seeds_favouring_treatment']}/{res['seeds']} seeds positive")


if __name__ == "__main__":
    main()
