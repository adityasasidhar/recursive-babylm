"""Shared data pipeline: tokenize the BabyLM Strict corpus once into a flat
uint16 memmap, then serve fixed-length LM chunks in a deterministic seeded
order identical across all 5 variants.

CORPUS SOURCE — PINNED 2026-07-07: the official HF Hub repos
`BabyLM-community/BabyLM-2026-Strict` (6 *.train.txt files, 100M words) and
`BabyLM-community/BabyLM-dev` (6 *.dev files). The suffix routing below maps
them correctly as-is (.txt -> train, .dev -> val). `modal_train.py::prep_data`
downloads both and builds the bins on the Modal data volume; `prepare` also
accepts any local directory of such files.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.common.tokenizer import EOS_TOKEN_ID, VOCAB_SIZE, BabyLMTokenizer

assert VOCAB_SIZE < 2**16, "token ids must fit uint16"

DATA_SEED = 42  # single seed for chunk order — identical for all variants

SPLIT_SUFFIXES = {"train": {".txt", ".train"}, "val": {".dev"}}

def prepare(corpus_dir: str | Path, out_path: str | Path, split: str = "train") -> None:
    """Tokenize every corpus file for `split` under corpus_dir (sorted order,
    docs joined by EOS) into one flat uint16 .bin memmap. The train split
    takes *.txt/*.train only; *.dev files go to the val split so the held-out
    validation set never leaks into training."""
    corpus_dir, out_path = Path(corpus_dir), Path(out_path)
    files = sorted(
        p for p in corpus_dir.rglob("*") if p.suffix in SPLIT_SUFFIXES[split]
    )
    if not files:
        raise FileNotFoundError(f"no corpus files under {corpus_dir}")
    tok = BabyLMTokenizer()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    all_ids: list[np.ndarray] = []
    n_tokens = 0
    for f in files:
        ids = tok.encode(f.read_text(encoding="utf-8", errors="replace"))
        ids.append(EOS_TOKEN_ID)
        arr = np.asarray(ids, dtype=np.uint16)
        all_ids.append(arr)
        n_tokens += arr.size
        print(f"  {f.name}: {arr.size:,} tokens")
    flat = np.concatenate(all_ids)
    flat.tofile(out_path)
    print(f"wrote {n_tokens:,} tokens -> {out_path}")


class LMChunkDataset(Dataset):
    """Fixed-length (seq_len+1) chunks from the token memmap. Chunk order is a
    deterministic permutation of (DATA_SEED, epoch) so every variant sees the
    exact same tokens in the exact same order, and each epoch is a fresh
    shuffle. Call set_epoch() before creating a new DataLoader iterator."""

    def __init__(self, bin_path: str | Path, seq_len: int, seed: int = DATA_SEED):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.seq_len = seq_len
        self.seed = seed
        self.n_chunks = (len(self.data) - 1) // seq_len
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        g = np.random.default_rng([self.seed, epoch])
        self.order = g.permutation(self.n_chunks)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = int(self.order[i]) * self.seq_len
        buf = torch.from_numpy(
            self.data[start : start + self.seq_len + 1].astype(np.int64)
        )
        return buf[:-1], buf[1:]  # input ids, shifted targets


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-dir", required=True, help="dir of raw corpus files")
    ap.add_argument("--split", choices=list(SPLIT_SUFFIXES), default="train")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    prepare(args.corpus_dir,
            args.out or f"data/babylm_strict/{args.split}.bin", args.split)
