"""Evaluate native checkpoints on the official BabyLM 2026 COMPS and
entity-tracking sets.

Companion to ``blimp_eval.py``, which is deliberately left untouched: its
numbers are the paper's published BLiMP source of truth, so this module does not
refactor that code path. Both implement the same official causal protocol --
score every candidate by the summed log-probability of its COMPLETION tokens and
take the argmax -- but the shape of a candidate set differs:

  * BLiMP's completion is the whole sentence, and there are two candidates.
  * COMPS conditions one shared property phrase on two differing prefixes, so
    only the property-phrase tokens are scored.
  * Entity tracking ranks five completions after one shared prefix.

Aggregation follows ``evaluation_pipeline/sentence_zero_shot/run.py``: a plain
macro-average over UIDs for COMPS, and for entity tracking a two-level average --
each of the three splits averages its own per-``numops`` UIDs, then the three
split accuracies are averaged. Entity-tracking rows in which any option contains
the word "nothing" are skipped, as the official reader does.

Data revision, evaluator commit and tokenizer revision are the pinned constants
from ``blimp_eval``, so the two modules always describe the same release.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from src.common.blimp_eval import (
    EVAL_REPO_ID,
    EVAL_REVISION,
    OFFICIAL_EVALUATOR_COMMIT,
    TOKENIZER_REVISION,
)
from src.common.tokenizer import TOKENIZER_ID
from src.common.variants import load_variant

TASKS = {
    "comps": "evaluation_data/full_eval/comps",
    "entity_tracking": "evaluation_data/full_eval/entity_tracking",
}

# file stem -> official COMPS subset name (read_files.decode_comps)
COMPS_SUBSETS = {
    "comps_base": "base",
    "comps_wugs": "wugs",
    "comps_wugs_dist-before": "wugs_dist_before",
    "comps_wugs_dist-in-between": "wugs_dist_in_between",
}
# official split order for the entity-tracking two-level average
ENTITY_SPLITS = ("regular", "ambiref", "move_contents")


def download_data(local_dir: str | Path) -> Path:
    """Download only the two official task directories."""
    from huggingface_hub import snapshot_download

    root = Path(local_dir)
    snapshot_download(
        repo_id=EVAL_REPO_ID,
        repo_type="dataset",
        revision=EVAL_REVISION,
        local_dir=root,
        allow_patterns=[f"{path}/*.jsonl" for path in TASKS.values()],
    )
    return root


def load_task(
    root: str | Path, task: str, limit_per_file: int | None = None
) -> list[dict[str, Any]]:
    """Rows of {uid, split, sentences, completions, label} in official order."""
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; choose from {list(TASKS)}")
    data_dir = Path(root) / TASKS[task]
    files = sorted(data_dir.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no JSONL files found in {data_dir}")

    rows: list[dict[str, Any]] = []
    for path in files:
        kept = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            if limit_per_file is not None and kept >= limit_per_file:
                break
            raw = json.loads(line)
            if task == "comps":
                # decode_comps: " ".join([prefix, property_phrase])
                phrase = raw["property_phrase"]
                rows.append({
                    "uid": COMPS_SUBSETS[path.stem],
                    "split": COMPS_SUBSETS[path.stem],
                    "sentences": [
                        " ".join([raw["prefix_acceptable"], phrase]),
                        " ".join([raw["prefix_unacceptable"], phrase]),
                    ],
                    "completions": [phrase, phrase],
                    "label": 0,
                })
            else:
                # decode_entity_tracking: rows with a "nothing" option are dropped
                options = raw["options"]
                if any("nothing" in option for option in options):
                    continue
                prefix = raw["input_prefix"]
                rows.append({
                    "uid": f'{path.stem}_{raw["numops"]}_ops',
                    "split": path.stem,
                    "sentences": [prefix + option for option in options],
                    "completions": list(options),
                    "label": 0,
                })
            kept += 1
    return rows


def _completion_masks(
    tokenizer: Any, sentences: list[str], completions: list[str]
) -> tuple[list[list[int]], list[list[int]]]:
    """Token ids plus a 0/1 mask marking the completion tokens of each sentence.

    Mirrors ``dataset.process_causal_sentences``: a token counts as part of the
    completion when its character span *ends* past the completion's first
    character, which is how the official code handles a BPE token that swallows
    the preceding space.
    """
    encoded = tokenizer(
        sentences, add_special_tokens=True, return_offsets_mapping=True
    )
    all_ids, all_masks = [], []
    for ids, offsets, sentence, completion in zip(
        encoded["input_ids"], encoded["offset_mapping"], sentences, completions
    ):
        start_char = len(sentence) - len(completion) + offsets[0][0]
        all_ids.append(list(ids))
        all_masks.append([int(end > start_char) for _, end in offsets])
    return all_ids, all_masks


def _candidate_scores(
    model: torch.nn.Module,
    tokenizer: Any,
    sentences: list[str],
    completions: list[str],
    batch_size: int,
    device: str,
    pad_id: int,
) -> torch.Tensor:
    """Summed causal log-probability of each sentence's completion tokens."""
    ids_list, mask_list = _completion_masks(tokenizer, sentences, completions)
    # Candidate lengths vary a lot (COMPS ~10 tokens, entity tracking ~110), so
    # batch length-sorted and invert the permutation afterwards: identical
    # arithmetic, far less padding. Sort is stable, so ties keep official order.
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    ids_list = [ids_list[i] for i in order]
    mask_list = [mask_list[i] for i in order]

    scores: list[torch.Tensor] = []
    for start in range(0, len(ids_list), batch_size):
        chunk_ids = ids_list[start : start + batch_size]
        chunk_masks = mask_list[start : start + batch_size]
        width = max(len(ids) for ids in chunk_ids)
        padded = torch.full((len(chunk_ids), width), pad_id, dtype=torch.long)
        phrase = torch.zeros((len(chunk_ids), width), dtype=torch.float32)
        for i, (ids, mask) in enumerate(zip(chunk_ids, chunk_masks)):
            padded[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            phrase[i, : len(mask)] = torch.tensor(mask, dtype=torch.float32)
        padded = padded.to(device)
        # collate_fn slices inputs [:, :-1] and both targets and phrase mask
        # [:, 1:], so the mask already aligns with the predicted positions
        inputs, targets = padded[:, :-1], padded[:, 1:]
        target_phrase = phrase[:, 1:].to(device)

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = model(inputs)
        token_log_probs = torch.gather(
            F.log_softmax(logits.float(), dim=-1), -1, targets.unsqueeze(-1)
        ).squeeze(-1)
        scores.append((token_log_probs * target_phrase).sum(dim=1).cpu())
    sorted_scores = torch.cat(scores)
    restored = torch.empty_like(sorted_scores)
    restored[torch.tensor(order, dtype=torch.long)] = sorted_scores
    return restored


def _rank(rows: list[dict[str, Any]], scores: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Correctness per row, plus the number of exact ties.

    The official runner samples uniformly among tied candidates; as in
    ``blimp_eval.evaluate_suite`` we reproduce that with a fixed seed so reruns
    are identical, rather than scoring a tie as a loss (which would bias a
    two-candidate task downward by half a point per tie).
    """
    generator = torch.Generator().manual_seed(0)
    correct, cursor, ties = [], 0, 0
    for row in rows:
        n = len(row["sentences"])
        window = scores[cursor : cursor + n]
        cursor += n
        tied = (window == torch.max(window)).nonzero().flatten()
        if tied.numel() > 1:
            ties += 1
            choice = int(tied[torch.randint(tied.numel(), (1,), generator=generator)])
        else:
            choice = int(tied[0])
        correct.append(int(choice == row["label"]))
    return torch.tensor(correct, dtype=torch.long), ties


def _aggregate(task: str, rows: list[dict[str, Any]], correct: torch.Tensor) -> dict:
    uid_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for row, is_correct in zip(rows, correct.tolist()):
        bucket = uid_counts[row["uid"]]
        bucket[0] += int(is_correct)
        bucket[1] += 1

    by_uid = {
        name: {
            "correct": counts[0],
            "total": counts[1],
            "accuracy": 100.0 * counts[0] / counts[1],
        }
        for name, counts in sorted(uid_counts.items())
    }

    if task == "entity_tracking":
        # run.process_results: average within each split, then across splits
        by_split, split_accuracies = {}, []
        for split in ENTITY_SPLITS:
            keys = [k for k in by_uid if k.startswith(split)]
            if not keys:
                continue
            accuracy = sum(by_uid[k]["accuracy"] for k in keys) / len(keys)
            by_split[split] = accuracy
            split_accuracies.append(accuracy)
        macro = sum(split_accuracies) / len(split_accuracies)
    else:
        by_split = None
        macro = sum(v["accuracy"] for v in by_uid.values()) / len(by_uid)

    total_correct = int(correct.sum().item())
    result = {
        "items": len(rows),
        "correct": total_correct,
        "micro_accuracy": 100.0 * total_correct / len(rows),
        "macro_accuracy": macro,
        "groups": {"uid": by_uid},
    }
    if by_split is not None:
        result["groups"]["split"] = by_split
    return result


def evaluate_task(
    model: torch.nn.Module,
    tokenizer: Any,
    root: str | Path,
    task: str,
    batch_size: int = 64,
    limit_per_file: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    rows = load_task(root, task, limit_per_file)
    sentences = [s for row in rows for s in row["sentences"]]
    completions = [c for row in rows for c in row["completions"]]
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 3  # <pad>; only ever read at masked-out positions
    scores = _candidate_scores(
        model, tokenizer, sentences, completions, batch_size, device, pad_id
    )
    correct, ties = _rank(rows, scores)
    result = _aggregate(task, rows, correct)
    result["ties"] = ties
    return result


def evaluate_checkpoint(
    variant: str,
    checkpoint: str | Path,
    root: str | Path,
    batch_size: int = 64,
    limit_per_file: int | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    """Both tasks for one native checkpoint."""
    from src.common.tokenizer import BabyLMTokenizer

    cfg, Model = load_variant(variant)
    model = Model(cfg).to(device).eval()
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert state["variant"] == variant, \
        f"{checkpoint} holds {state['variant']!r}, not {variant!r}"
    model.load_state_dict(state["model"])
    tokenizer = BabyLMTokenizer().tok

    tasks = {}
    for task in TASKS:
        tasks[task] = evaluate_task(
            model, tokenizer, root, task, batch_size, limit_per_file, device
        )
        print(f"  {task}: {tasks[task]['macro_accuracy']:.2f} macro "
              f"({tasks[task]['correct']}/{tasks[task]['items']})")
    del model
    torch.cuda.empty_cache()
    return {"variant": variant, "checkpoint": str(checkpoint), "tasks": tasks}


def result_document(models: dict[str, Any]) -> dict[str, Any]:
    return {
        "protocol": {
            "name": "BabyLM 2026 Strict causal zero-shot COMPS + entity tracking",
            "scoring": "summed log-probability of the completion tokens only",
            "aggregation": (
                "COMPS: macro-average over the four subsets. "
                "Entity tracking: mean over per-numops UIDs within each of "
                "regular/ambiref/move_contents, then mean of the three splits."
            ),
            "temperature": 1.0,
            "skipped": "entity-tracking rows where any option contains 'nothing'",
            "ties": "resolved as incorrect (the official runner samples uniformly)",
            "eval_repo": "https://github.com/babylm-org/babylm-eval",
            "eval_commit": OFFICIAL_EVALUATOR_COMMIT,
            "data_repo": EVAL_REPO_ID,
            "data_revision": EVAL_REVISION,
            "tokenizer": TOKENIZER_ID,
            "tokenizer_revision": TOKENIZER_REVISION,
        },
        "models": models,
    }
