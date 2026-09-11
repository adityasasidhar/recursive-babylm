"""Smoke test for all 5 variants (GPU required — fla kernels are Triton-only):

  1. forward pass on a dummy batch -> correct logits shape, no NaN/Inf
  2. N training steps on a small FIXED dummy batch -> loss must decrease
     (memorization) and never NaN

    uv run python -m src.common.smoke_test [--steps 30]

This is a sanity gate, not training. Sized for the 4GB local GPU.
"""

from __future__ import annotations

import argparse
import sys

import torch

from src.common.variants import VARIANTS, load_variant, non_embedding_params

B, T = 4, 256


def smoke_one(name: str, steps: int) -> bool:
    cfg, Model = load_variant(name)
    torch.manual_seed(0)
    model = Model(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (B, T), device="cuda")
    y = torch.roll(x, -1, dims=1)

    # 1. forward: shape + finiteness
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, loss0 = model(x, targets=y)
    assert logits.shape == (B, T, cfg.vocab_size), f"bad shape {logits.shape}"
    assert torch.isfinite(logits.float()).all(), "non-finite logits"
    assert torch.isfinite(loss0), "non-finite initial loss"

    # 2. overfit the fixed batch for a few steps
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
    first = last = None
    for i in range(steps):
        opt.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, targets=y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        val = loss.item()
        assert val == val, f"NaN loss at step {i}"
        first = first if first is not None else val
        last = val

    ok = last < first
    drop = first - last
    print(
        f"{name:<16} params(non-emb) {non_embedding_params(model):>12,}  "
        f"logits {tuple(logits.shape)}  loss {first:.3f} -> {last:.3f} "
        f"({'-' if ok else '+'}{abs(drop):.3f})  {'OK' if ok else 'FAIL'}"
    )
    del model, opt
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--variants", default="",
                    help="comma-separated subset (default: all registered). The "
                         "deep revision arms may not fit the 4GB local GPU — run "
                         "those on Modal via check_env.")
    args = ap.parse_args()
    assert torch.cuda.is_available(), "CUDA GPU required for fla GDN kernels"
    chosen = [v.strip() for v in args.variants.split(",") if v.strip()] or list(VARIANTS)
    unknown = [v for v in chosen if v not in VARIANTS]
    assert not unknown, f"unknown variants {unknown}; choose from {list(VARIANTS)}"
    print(f"device: {torch.cuda.get_device_name()}  dummy batch: {B}x{T}\n")
    results = [smoke_one(name, args.steps) for name in chosen]
    print("\nALL PASS" if all(results) else "\nFAILURES — see above")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
