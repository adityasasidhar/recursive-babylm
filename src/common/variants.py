"""Registry of the 5 variants. The 2to1/3to1 package names start with a digit,
so they can't be imported with a plain `import` statement — use load_variant(),
which goes through importlib (which handles non-identifier module names)."""

from __future__ import annotations

import importlib

VARIANTS: dict[str, str] = {
    "baseline": "src.baseline",
    "gdn_2to1": "src.gdn_baseline.2to1",
    "gdn_3to1": "src.gdn_baseline.3to1",
    "recursive_2to1": "src.recursion_gdn.2to1",
    "recursive_3to1": "src.recursion_gdn.3to1",
}


def load_variant(name: str):
    """Returns (config, model_ctor) for a variant name from VARIANTS."""
    pkg = VARIANTS[name]
    config_mod = importlib.import_module(f"{pkg}.config")
    model_mod = importlib.import_module(f"{pkg}.model")
    return config_mod.CONFIG, model_mod.Model


def non_embedding_params(model) -> int:
    """Total trainable params minus the token embedding table (the LM head is
    weight-tied to it, so this excludes both)."""
    total = sum(p.numel() for p in model.parameters())
    return total - model.tok_emb.weight.numel()


def param_report(model) -> dict:
    """Size breakdown for logging/paper tables: total, embedding, unique
    non-embedding, and — for the recursive variants — the effective
    non-embedding count (recursed super-block params counted once per pass)
    plus effective depth."""
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    emb = model.tok_emb.weight.numel()
    non_emb = total - emb
    r = getattr(cfg, "n_recursions", 1)
    sb = getattr(model, "super_blocks", None)
    if sb is not None and r > 1:
        sb_params = sum(p.numel() for p in sb.parameters())
        eff_non_emb = non_emb + sb_params * (r - 1)
        eff_depth = cfg.n_super_blocks * r * cfg.n_layers
    else:
        eff_non_emb = non_emb
        eff_depth = cfg.n_layers
    return {"total": total, "embedding": emb, "non_embedding": non_emb,
            "effective_non_embedding": eff_non_emb, "effective_depth": eff_depth}
