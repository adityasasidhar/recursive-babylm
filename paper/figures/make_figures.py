"""Generate the two paper figures.

    uv run --with matplotlib python paper/figures/make_figures.py

Training-loss curves come from wandb (needs WANDB_API_KEY; source the repo
.env). Validation numbers come from ckpt_eval.json next to this file — the
uniformly-weighted per-checkpoint re-evaluation produced by
`modal run modal_train.py::ckpt_eval` (the training-time val logger weighted
batches unevenly across micro-batch settings; see paper Appendix A).
Outputs loss_curves.{png,pdf} and params_vs_val.{png,pdf}.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import wandb

OUT = Path(__file__).parent
ENTITY_PROJECT = "telikicherlaadityasasidhar-vellore-institute-of-technology/babylm-2026"
TOKENS_PER_STEP = 131_072
CKPTS = ["ckpt_00100M", "ckpt_00200M", "ckpt_00300M", "ckpt_00400M", "ckpt_00499M"]
CKPT_TOKENS_M = [100, 200, 300, 400, 499.9]

# hue = ablation pair, line style = recursion (validated: blue/green/red on white)
STYLE = {
    # name            label                 color      ls    z
    "recursive_2to1": ("Recursive 2:1 (46.9M)", "#2a78d6", "-", 5),
    "gdn_2to1":       ("GDN 2:1 (46.9M)",       "#2a78d6", "--", 4),
    "recursive_3to1": ("Recursive 3:1 (63.5M)", "#e34948", "-", 5),
    "gdn_3to1":       ("GDN 3:1 (63.5M)",       "#e34948", "--", 4),
    "baseline":       ("GQA baseline (68.8M)",  "#008300", "-.", 3),
}
ORDER = ["recursive_2to1", "gdn_2to1", "recursive_3to1", "gdn_3to1", "baseline"]

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "figure.dpi": 110, "savefig.bbox": "tight",
})


def smooth(xs: list[float], w: int = 7) -> list[float]:
    out, buf = [], []
    for x in xs:
        buf.append(x)
        if len(buf) > w:
            buf.pop(0)
        out.append(sum(buf) / len(buf))
    return out


def main() -> None:
    ev = json.loads((OUT / "ckpt_eval.json").read_text())
    val = {n: [sum(ev[n][c]["chunk_losses"]) / len(ev[n][c]["chunk_losses"])
               for c in CKPTS] for n in ORDER}

    api = wandb.Api()
    runs = {r.name: r for r in api.runs(ENTITY_PROJECT)}
    train_hist = {
        n: sorted((row["_step"], row["train/loss"]) for row in
                  runs[n].history(keys=["train/loss"], pandas=False))
        for n in ORDER
    }

    # ---- Figure 1: training loss (wandb) + corrected validation loss -------
    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    for name in ORDER:
        label, color, ls, z = STYLE[name]
        tr = train_hist[name]
        xs = [s * TOKENS_PER_STEP / 1e6 for s, _ in tr]
        ax_tr.plot(xs, smooth([v for _, v in tr]), ls, color=color, lw=1.4, zorder=z)
        ax_va.plot(CKPT_TOKENS_M, val[name], ls, color=color, lw=1.4, zorder=z,
                   marker="o", ms=3.5, label=label)
    ax_tr.set_xlabel("training tokens (M)")
    ax_tr.set_ylabel("training loss (nats)")
    ax_tr.set_ylim(2.3, 4.8)
    ax_va.set_xlabel("training tokens (M)")
    ax_va.set_ylabel("validation loss (nats)")
    ax_va.legend(frameon=False, handlelength=2.6, borderaxespad=0.2)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"loss_curves.{ext}", dpi=220)
    plt.close(fig)

    # ---- Figure 2: params vs final validation loss ------------------------
    fig, ax = plt.subplots(figsize=(4.4, 3.1))
    final = {n: val[n][-1] for n in ORDER}
    params = {"recursive_2to1": 46.9, "gdn_2to1": 46.9,
              "recursive_3to1": 63.5, "gdn_3to1": 63.5, "baseline": 68.8}
    for name in ORDER:
        label, color, ls, _ = STYLE[name]
        filled = ls == "-"  # recursive solid -> filled marker
        ax.plot(params[name], final[name], "o",
                mfc=color if filled else "white", mec=color, mew=1.4, ms=7)
    for pair, x in (("2to1", 46.9), ("3to1", 63.5)):
        lo, hi = sorted([final[f"recursive_{pair}"], final[f"gdn_{pair}"]])
        ax.plot([x, x], [lo, hi], ":", color=STYLE[f"recursive_{pair}"][1],
                lw=1.0, zorder=1)
        ax.annotate(f"$\\Delta$={hi - lo:.3f}", (x, (lo + hi) / 2),
                    textcoords="offset points", xytext=(6, 0), fontsize=8,
                    color="#52514e", va="center")
    labels = {
        "recursive_2to1": (8, -3, "Recursive 2:1"),
        "gdn_2to1": (8, -3, "GDN 2:1"),
        "recursive_3to1": (-8, -3, "Recursive 3:1"),
        "gdn_3to1": (8, -10, "GDN 3:1"),
        "baseline": (-8, 6, "GQA baseline"),
    }
    for name, (dx, dy, txt) in labels.items():
        ax.annotate(txt, (params[name], final[name]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    color="#0b0b0b", ha="right" if dx < 0 else "left")
    ax.set_xlabel("non-embedding parameters (M)")
    ax.set_ylabel("final validation loss (nats)")
    ax.set_xlim(41, 75)
    ax.set_ylim(3.086, 3.140)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"params_vs_val.{ext}", dpi=220)
    plt.close(fig)
    print("wrote figures; finals:", {n: round(v, 4) for n, v in final.items()})


if __name__ == "__main__":
    main()
