"""Shared tokenizer for all 5 variants.

Per project decision (2026-07-05, revised): we use the pretrained BabyLM
community BPE tokenizer (BabyLM-community/babylm-baseline-100m-gpt-bert-masked-
focus, vocab 16384) — trained on the BabyLM 100M corpus itself. It is IDENTICAL
across all variants by construction — loaded from one source, never retrained.
The repo ships custom code, so load with trust_remote_code=True. The first load
downloads from the HF Hub and caches locally; pass a local dir to load offline.
"""

from __future__ import annotations

from pathlib import Path

TOKENIZER_ID = "BabyLM-community/babylm-baseline-100m-gpt-bert-masked-focus"
VOCAB_SIZE = 16384  # must match every variant's config.vocab_size
EOS_TOKEN_ID = 2  # </s>; also has <s>=1 (auto-prepended), <pad>=3, <unk>=0, <mask>=4

_LOCAL_DIR = Path(__file__).resolve().parents[2] / "tokenizer" / "artifacts"


class BabyLMTokenizer:
    """Wrapper pinning the single shared tokenizer + the ids configs rely on."""

    def __init__(self, local_dir: str | Path | None = None):
        from transformers import AutoTokenizer

        source = local_dir or (_LOCAL_DIR if _LOCAL_DIR.exists() else TOKENIZER_ID)
        self.tok = AutoTokenizer.from_pretrained(str(source), trust_remote_code=True)
        assert len(self.tok) == VOCAB_SIZE, (
            f"tokenizer vocab {len(self.tok)} != expected {VOCAB_SIZE}; "
            "all variants must share the exact same tokenizer"
        )
        self.vocab_size = VOCAB_SIZE
        self.eos_token_id = self.tok.eos_token_id
        assert self.eos_token_id == EOS_TOKEN_ID

    def encode(self, text: str) -> list[int]:
        return self.tok.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids)

    def save_local(self, path: str | Path = _LOCAL_DIR) -> None:
        """Snapshot to the repo-local dir so training boxes can run offline."""
        self.tok.save_pretrained(str(path))


if __name__ == "__main__":
    t = BabyLMTokenizer()
    t.save_local()
    ids = t.encode("The quick brown fox jumps over the lazy dog.")
    print(f"vocab={t.vocab_size} eos={t.eos_token_id}")
    print(f"roundtrip: {ids} -> {t.decode(ids)!r}")
    print(f"saved local snapshot to {_LOCAL_DIR}")
