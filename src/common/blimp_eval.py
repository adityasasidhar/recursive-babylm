"""Evaluate native project checkpoints on the official BabyLM 2026 BLiMP sets.

The official Strict evaluator ranks each minimal pair by the summed causal
log-probability of every non-BOS sentence token, then macro-averages accuracy
over BLiMP UIDs (paradigms). This module implements that same scoring directly
for the project's native ``nn.Module`` checkpoints, avoiding a lossy or
duplicated Hugging Face model conversion.

The evaluation data is pinned to the official 2026 release revision below.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.common.tokenizer import TOKENIZER_ID
from src.common.variants import load_variant

EVAL_REPO_ID = "BabyLM-community/BabyLM-2026-Strict-Evals"
EVAL_REVISION = "8d52da9424a9ff30b9e8266c4f751aba9c504233"
OFFICIAL_EVALUATOR_COMMIT = "3d57ddc8c40ee795c0b5e41b3a20251a9457a593"
TOKENIZER_REVISION = "6660c0417342d99cd989e30c40f56e6e712d2a6b"

SUITES = {
    "blimp": "evaluation_data/full_eval/blimp_filtered",
    "blimp_supplement": "evaluation_data/full_eval/supplement_filtered",
}


def download_data(local_dir: str | Path) -> Path:
    """Download only the two official full BLiMP directories."""
    from huggingface_hub import snapshot_download

    root = Path(local_dir)
    snapshot_download(
        repo_id=EVAL_REPO_ID,
        repo_type="dataset",
        revision=EVAL_REVISION,
        local_dir=root,
        allow_patterns=[f"{path}/*.jsonl" for path in SUITES.values()],
    )
    return root


def load_suite(
    root: str | Path, suite: str, limit_per_file: int | None = None
) -> list[dict[str, str]]:
    """Load a suite in the same sorted-file/line order as the official data."""
    if suite not in SUITES:
        raise KeyError(f"unknown suite {suite!r}; choose from {list(SUITES)}")
    data_dir = Path(root) / SUITES[suite]
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL files found in {data_dir}")

    rows: list[dict[str, str]] = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit_per_file is not None and i >= limit_per_file:
                    break
                raw = json.loads(line)
                rows.append(
                    {
                        "good": raw["sentence_good"],
                        "bad": raw["sentence_bad"],
                        "uid": raw.get("UID", path.stem),
                        "field": raw.get("field", "supplement").replace(
                            "syntax_semantics", "syntax/semantics"
                        ),
                        "term": raw.get("linguistics_term", "supplement"),
                    }
                )
    return rows


@torch.inference_mode()
def _sentence_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[str],
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Return summed causal log-probabilities under the official protocol."""
    scores: list[torch.Tensor] = []
    for start in range(0, len(sentences), batch_size):
        batch = sentences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            add_special_tokens=True,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = encoded["input_ids"].to(device)
        attention = encoded["attention_mask"].to(device)
        inputs, targets = ids[:, :-1], ids[:, 1:]
        target_mask = attention[:, 1:].bool()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        token_log_probs = torch.gather(
            F.log_softmax(logits.float(), dim=-1),
            -1,
            targets.unsqueeze(-1),
        ).squeeze(-1)
        scores.append((token_log_probs * target_mask).sum(dim=1).cpu())
    return torch.cat(scores)


def _aggregate(rows: list[dict[str, str]], correct: torch.Tensor) -> dict[str, Any]:
    groups: dict[str, dict[str, list[int]]] = {
        "uid": defaultdict(lambda: [0, 0]),
        "field": defaultdict(lambda: [0, 0]),
        "term": defaultdict(lambda: [0, 0]),
    }
    for row, is_correct in zip(rows, correct.tolist()):
        for group in groups:
            bucket = groups[group][row[group]]
            bucket[0] += int(is_correct)
            bucket[1] += 1

    by_group: dict[str, dict[str, dict[str, float | int]]] = {}
    for group, values in groups.items():
        by_group[group] = {
            name: {
                "correct": counts[0],
                "total": counts[1],
                "accuracy": 100.0 * counts[0] / counts[1],
            }
            for name, counts in sorted(values.items())
        }

    uid_accuracies = [v["accuracy"] for v in by_group["uid"].values()]
    total_correct = int(correct.sum().item())
    return {
        "items": len(rows),
        "correct": total_correct,
        "micro_accuracy": 100.0 * total_correct / len(rows),
        "macro_accuracy": sum(uid_accuracies) / len(uid_accuracies),
        "groups": by_group,
    }


def evaluate_suite(
    model: torch.nn.Module,
    tokenizer: Any,
    rows: list[dict[str, str]],
    batch_size: int,
    device: str = "cuda",
) -> dict[str, Any]:
    good = _sentence_scores(
        model, tokenizer, [row["good"] for row in rows], batch_size, device
    )
    bad = _sentence_scores(
        model, tokenizer, [row["bad"] for row in rows], batch_size, device
    )
    ties = good == bad
    # The official evaluator randomly resolves exact ties. They are retained in
    # the output and deterministically resolved here for reproducible reruns.
    if ties.any():
        generator = torch.Generator().manual_seed(0)
        tie_choices = torch.randint(0, 2, (int(ties.sum()),), generator=generator)
        correct = good > bad
        correct[ties] = tie_choices.bool()
    else:
        correct = good > bad
    result = _aggregate(rows, correct)
    result["ties"] = int(ties.sum().item())
    result["mean_good_log_prob"] = good.mean().item()
    result["mean_bad_log_prob"] = bad.mean().item()
    return result


def evaluate_checkpoint(
    variant: str,
    checkpoint: str | Path,
    data_root: str | Path,
    batch_size: int = 64,
    limit_per_file: int | None = None,
) -> dict[str, Any]:
    """Load one final checkpoint and evaluate both full BLiMP suites."""
    assert torch.cuda.is_available(), "BLiMP evaluation requires CUDA for GDN kernels"
    from transformers import AutoTokenizer

    torch.manual_seed(0)
    cfg, Model = load_variant(variant)
    model = Model(cfg).cuda().eval()
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if ckpt.get("variant") != variant:
        raise ValueError(
            f"checkpoint variant {ckpt.get('variant')!r} does not match {variant!r}"
        )
    model.load_state_dict(ckpt["model"])
    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_ID, revision=TOKENIZER_REVISION, trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    suites: dict[str, Any] = {}
    for suite in SUITES:
        rows = load_suite(data_root, suite, limit_per_file=limit_per_file)
        suites[suite] = evaluate_suite(model, tokenizer, rows, batch_size)
        print(
            f"{variant} {suite}: {suites[suite]['macro_accuracy']:.2f} "
            f"macro ({suites[suite]['correct']}/{suites[suite]['items']}, "
            f"ties={suites[suite]['ties']})"
        )

    del model
    torch.cuda.empty_cache()
    return {
        "variant": variant,
        "checkpoint": str(checkpoint),
        "checkpoint_tokens": ckpt.get("tokens_seen"),
        "suites": suites,
    }


def result_document(models: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": {
            "name": "BabyLM 2026 Strict full causal zero-shot BLiMP",
            "scoring": "summed log-probability of all non-BOS sentence tokens",
            "aggregation": "macro-average of per-UID accuracies",
            "temperature": 1.0,
            "eval_repo": "https://github.com/babylm-org/babylm-eval",
            "eval_commit": OFFICIAL_EVALUATOR_COMMIT,
            "data_repo": EVAL_REPO_ID,
            "data_revision": EVAL_REVISION,
            "tokenizer": TOKENIZER_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        "models": models,
    }
