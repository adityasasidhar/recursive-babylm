"""Shared training loop for all 5 variants.

    uv run python -m src.common.train --variant baseline --data data/babylm_strict/train.bin

Everything except the model comes from here, so tokenizer, data order/seed,
LR schedule, batch size, and token budget are identical across variants by
construction. Recipe locked 2026-07-06 (seq 4096, 32x4096 tokens/step, cosine
to 10%); budget finalized 2026-07-11 at 500M tokens total (~2.9 epochs of the
~170M-token/epoch Strict corpus, ~290M words of exposure — well under the
BabyLM 1B-word cap counting repeats);
peak LR default 6e-4 is the literature prior — the
pilot sweep (src/common/pilot.py) confirms it before the finals.

Logging (stdout always; wandb with --wandb):
  once      — model config, param report (unique vs effective), git commit,
              torch/fla/GPU environment, data fingerprints
  per log   — train loss, grad norm, LR, epoch, windowed + cumulative tok/s,
              step time, peak GPU mem, loss-spike counter
  per val   — val loss/ppl/bits-per-token, per-block grad-norm breakdown,
              residual-stream RMS per block (per recursion pass for the
              recursive variants)
  per ckpt  — model/opt state, RNG states, dataloader position (resumable
              via --resume), model + train config, provenance

DO NOT launch real training without sign-off; this is wired for smoke tests
and the eventual GPU runs only.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import platform
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.common.data import DATA_SEED, LMChunkDataset
from src.common.variants import VARIANTS, load_variant, param_report

LN2 = math.log(2)


@dataclass
class TrainConfig:
    lr: float = 6e-4  # peak; pilot sweep confirms {3e-4, 6e-4, 1e-3}
    min_lr_frac: float = 0.1
    warmup_steps: int = 250  # ~6.5% of the 500M-token run (was ~3% of 1B)
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip: float = 1.0
    batch_size: int = 32  # sequences per optimizer step (via grad accum)
    micro_batch_size: int = 16  # accum 2; 32 OOMed the 80GB H100 (fp32 logits for CE)
    seq_len: int = 4096
    token_budget: int = 500_000_000  # ~2.9 epochs (epoch ≈ 170M tokens ≈ 100M words)
    log_every: int = 20
    val_every: int = 500  # optimizer steps between val-loss evals
    val_tokens: int = 2_000_000  # fixed slice of val.bin per eval
    ckpt_every_tokens: int = 100_000_000  # comparable-budget checkpoints
    seed: int = DATA_SEED
    spike_factor: float = 1.5  # train loss > factor * EMA counts as a spike
    spike_ema_decay: float = 0.98


def lr_at(step: int, total_steps: int, tc: TrainConfig) -> float:
    if step < tc.warmup_steps:
        return tc.lr * (step + 1) / tc.warmup_steps
    t = (step - tc.warmup_steps) / max(1, total_steps - tc.warmup_steps)
    return tc.lr * (tc.min_lr_frac + (1 - tc.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * t)))


# ---------------------------------------------------------------- provenance


def git_info() -> dict:
    """Commit + dirty flag; empty strings when not in a git checkout (e.g. the
    Modal container — the launcher passes them in via extra_meta instead)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip())
        return {"git_commit": commit, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": "", "git_dirty": None}


def env_info(cfg) -> dict:
    from src.common.attention import _HAS_FA3

    info = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "python": platform.python_version(),
        "attn_backend_cfg": cfg.attn_backend,
        "fa3_available": _HAS_FA3,
    }
    try:
        import importlib.metadata

        info["fla"] = importlib.metadata.version("flash-linear-attention")
    except Exception:
        info["fla"] = "unknown"
    return info


def data_fingerprint(path: str | Path, seq_len: int) -> dict:
    """Size, token count, and a head+tail sha256 — cheap identity check that
    every run trained on the same corpus build."""
    p = Path(path)
    size = p.stat().st_size
    n_tokens = size // 2  # uint16
    h, chunk = hashlib.sha256(), 16 << 20
    with open(p, "rb") as f:
        h.update(f.read(chunk))
        if size > 2 * chunk:
            f.seek(-chunk, 2)
            h.update(f.read())
    return {"path": str(p), "bytes": size, "tokens": n_tokens,
            "chunks_per_epoch": (n_tokens - 1) // seq_len,
            "sha256_head_tail": h.hexdigest()}


# ---------------------------------------------------------------- diagnostics


def grad_norm_breakdown(model) -> dict[str, float]:
    """Per-block grad norms (per super-block for the recursive variants), plus
    tok_emb (tied with lm_head) and norm_f. Call while grads are populated."""
    sq: dict[str, float] = {}
    for n, p in model.named_parameters():
        if p.grad is None:
            continue
        parts = n.split(".")
        key = ".".join(parts[:2]) if parts[0] in ("blocks", "super_blocks") else parts[0]
        sq[key] = sq.get(key, 0.0) + p.grad.float().pow(2).sum().item()
    return {k: v ** 0.5 for k, v in sq.items()}


@torch.no_grad()
def activation_rms(model, batch: torch.Tensor) -> dict[str, float]:
    """Residual-stream RMS after each block. For the recursive variants the
    hook on a super-block's last block fires once per recursion pass, giving
    per-pass RMS — the stability evidence for weight-tying at effective depth.
    Pass the raw (uncompiled) model so hooks don't trigger recompilation."""
    records: dict[str, list[float]] = {}
    hooks = []

    def mk(name: str):
        def hook(_mod, _inp, out):
            records.setdefault(name, []).append(
                out.detach().float().pow(2).mean().sqrt().item())
        return hook

    if hasattr(model, "super_blocks"):
        for i, sb in enumerate(model.super_blocks):
            hooks.append(sb[-1].register_forward_hook(mk(f"super_block{i}")))
    else:
        for i, blk in enumerate(model.blocks):
            hooks.append(blk.register_forward_hook(mk(f"block{i}")))
    was_training = model.training
    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model(batch)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()

    out: dict[str, float] = {}
    for name, vals in records.items():
        if len(vals) > 1:  # recursive: one entry per pass
            out.update({f"{name}/pass{j + 1}": v for j, v in enumerate(vals)})
        else:
            out[name] = vals[0]
    return out


@torch.no_grad()
def eval_val_loss(model, val_data: np.memmap, tc: TrainConfig, device: str) -> float:
    """Mean loss over the first val_tokens of val.bin, sequential chunks —
    the same fixed slice every eval and every variant."""
    model.eval()
    n_chunks = min(tc.val_tokens, len(val_data) - 1) // tc.seq_len
    total, n_rows = 0.0, 0
    for start in range(0, n_chunks * tc.seq_len, tc.seq_len * tc.micro_batch_size):
        rows = []
        for s in range(start, min(start + tc.seq_len * tc.micro_batch_size,
                                  n_chunks * tc.seq_len), tc.seq_len):
            rows.append(torch.from_numpy(
                val_data[s : s + tc.seq_len + 1].astype(np.int64)))
        buf = torch.stack(rows).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(buf[:, :-1], targets=buf[:, 1:])
        # weight by rows, NOT mean-of-batch-means: a ragged final batch would
        # otherwise be over-weighted, and the weighting would depend on
        # micro_batch_size — the 2026-07 paper runs logged val losses that
        # were incomparable across micro-batch settings because of this
        # (mb=16 runs biased -0.0247 vs the mb=8 run on the same slice)
        total += loss.item() * len(rows)
        n_rows += len(rows)
    model.train()
    return total / max(1, n_rows)


def _val_probe_batch(val_data: np.memmap, tc: TrainConfig, device: str) -> torch.Tensor:
    """Small fixed batch from the start of val.bin for the activation probe."""
    rows = max(1, min(4, tc.micro_batch_size, (len(val_data) - 1) // tc.seq_len))
    return torch.stack([
        torch.from_numpy(val_data[i * tc.seq_len : (i + 1) * tc.seq_len].astype(np.int64))
        for i in range(rows)
    ]).to(device)


# --------------------------------------------------------------------- train


def train(variant: str, data_path: str, out_dir: str, tc: TrainConfig,
          val_path: str | None = None, use_wandb: bool = False,
          tag: str = "", compile_model: bool = False,
          resume: str | None = None, extra_meta: dict | None = None) -> None:
    assert torch.cuda.is_available(), "fla GDN kernels require a CUDA GPU"
    device = "cuda"
    torch.manual_seed(tc.seed)
    torch.set_float32_matmul_precision("high")

    cfg, Model = load_variant(variant)
    raw_model = Model(cfg).to(device)
    model = raw_model
    if compile_model:
        # fullgraph=False: fla's Triton GDN kernels graph-break; compile the rest
        model = torch.compile(raw_model, fullgraph=False)

    params = param_report(raw_model)
    meta = {
        "variant": variant,
        "model": dataclasses.asdict(cfg),
        "params": params,
        "env": env_info(cfg),
        "data": data_fingerprint(data_path, tc.seq_len),
        "compile": compile_model,
        **git_info(),
        **(extra_meta or {}),
    }
    if val_path and Path(val_path).exists():
        meta["val_data"] = data_fingerprint(val_path, tc.seq_len)
    print(f"{variant}: {params['non_embedding']:,} non-embedding params "
          f"({params['effective_non_embedding']:,} effective, "
          f"depth {params['effective_depth']})")
    print(f"env: {meta['env']}")
    print(f"git: {meta['git_commit'][:12] or '<none>'}"
          f"{' DIRTY' if meta.get('git_dirty') else ''}")
    print(f"data: {meta['data']['tokens']:,} tokens, "
          f"{meta['data']['chunks_per_epoch']:,} chunks/epoch, "
          f"sha {meta['data']['sha256_head_tail'][:12]}")

    run_name = f"{variant}{'_' + tag if tag else ''}"
    wandb_run = None
    if use_wandb:
        import wandb

        wandb_run = wandb.init(project="babylm-2026", name=run_name,
                               config={**vars(tc), **meta})

    ds = LMChunkDataset(data_path, tc.seq_len, seed=tc.seed)
    dl = DataLoader(ds, batch_size=tc.micro_batch_size, shuffle=False, drop_last=True,
                    num_workers=2, pin_memory=True)
    val_data = (np.memmap(val_path, dtype=np.uint16, mode="r")
                if val_path and Path(val_path).exists() else None)
    if val_path and val_data is None:
        print(f"WARNING: val file {val_path} not found — skipping val evals")
    probe_batch = _val_probe_batch(val_data, tc, device) if val_data is not None else None

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (decay if p.dim() >= 2 else no_decay).append(p)
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": tc.weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=tc.lr, betas=tc.betas, fused=True,
    )

    accum = tc.batch_size // tc.micro_batch_size
    assert accum * tc.micro_batch_size == tc.batch_size, \
        "batch_size must be a multiple of micro_batch_size"
    tokens_per_step = tc.batch_size * tc.seq_len
    total_steps = tc.token_budget // tokens_per_step
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    step, tokens_seen, epoch, mb_in_epoch = 0, 0, 0, 0
    if resume:
        # map to CPU: load_state_dict copies onto the GPU params, and RNG
        # states must stay ByteTensors on CPU
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        assert ck["variant"] == variant, \
            f"checkpoint is for {ck['variant']!r}, not {variant!r}"
        raw_model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        step, tokens_seen = ck["step"], ck["tokens_seen"]
        epoch, mb_in_epoch = ck["epoch"], ck["mb_in_epoch"]
        torch.set_rng_state(ck["rng"]["torch"])
        torch.cuda.set_rng_state_all(ck["rng"]["cuda"])
        print(f"resumed {resume}: step {step}, {tokens_seen:,} tokens, "
              f"epoch {epoch} (+{mb_in_epoch} micro-batches)")

    def save_ckpt() -> None:
        torch.save({
            "model": raw_model.state_dict(), "opt": opt.state_dict(),
            "step": step, "tokens_seen": tokens_seen,
            "epoch": epoch, "mb_in_epoch": mb_in_epoch,
            "rng": {"torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()},
            "variant": variant, "train_config": vars(tc),
            "model_config": dataclasses.asdict(cfg), "meta": meta,
        }, out / f"ckpt_{tokens_seen // 1_000_000:05d}M.pt")

    model.train()
    ds.set_epoch(epoch)
    dl_iter = iter(dl)
    for _ in range(mb_in_epoch):  # resume: fast-forward within the epoch
        next(dl_iter)
    next_ckpt = (tokens_seen // tc.ckpt_every_tokens + 1) * tc.ckpt_every_tokens
    # on-GPU spike stats: .item() syncs, so touch them only at log time
    loss_ema = torch.zeros((), device=device)
    ema_ready = False  # seed from the first step (incl. after resume)
    spikes = torch.zeros((), device=device)
    t0 = last_log_t = time.time()
    tokens_at_start, last_log_tokens = tokens_seen, tokens_seen
    while step < total_steps:
        for g in opt.param_groups:
            g["lr"] = lr_at(step, total_steps, tc)
        opt.zero_grad(set_to_none=True)
        loss_acc = torch.zeros((), device=device)
        for _ in range(accum):
            try:
                x, y = next(dl_iter)
                mb_in_epoch += 1
            except StopIteration:  # new epoch, fresh seeded shuffle (same across variants)
                epoch += 1
                mb_in_epoch = 0
                ds.set_epoch(epoch)
                dl_iter = iter(dl)
                x, y = next(dl_iter)
                mb_in_epoch += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, targets=y)
            (loss / accum).backward()
            loss_acc += loss.detach() / accum
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        opt.step()
        if not ema_ready:
            loss_ema.copy_(loss_acc)
            ema_ready = True
        else:
            spikes += (loss_acc > tc.spike_factor * loss_ema).float()
            loss_ema.mul_(tc.spike_ema_decay).add_((1 - tc.spike_ema_decay) * loss_acc)
        step += 1
        tokens_seen += tokens_per_step

        if step % tc.log_every == 0:
            now = time.time()
            tps_avg = (tokens_seen - tokens_at_start) / (now - t0)
            tps_inst = (tokens_seen - last_log_tokens) / (now - last_log_t)
            step_ms = (now - last_log_t) * 1000 / tc.log_every
            last_log_t, last_log_tokens = now, tokens_seen
            loss_v, spikes_v = loss_acc.item(), int(spikes.item())
            peak_gib = torch.cuda.max_memory_allocated() / 2**30
            print(f"step {step}/{total_steps} epoch {epoch} loss {loss_v:.4f} "
                  f"gnorm {grad_norm:.2f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"tok/s {tps_inst:,.0f} (avg {tps_avg:,.0f}) "
                  f"step_ms {step_ms:,.0f} mem {peak_gib:.1f}GiB spikes {spikes_v}")
            if wandb_run:
                wandb_run.log({
                    "train/loss": loss_v, "train/grad_norm": grad_norm,
                    "train/lr": opt.param_groups[0]["lr"], "train/epoch": epoch,
                    "train/tok_s": tps_inst, "train/tok_s_avg": tps_avg,
                    "train/step_time_ms": step_ms, "train/peak_mem_gib": peak_gib,
                    "train/loss_spikes": spikes_v, "tokens": tokens_seen,
                }, step=step)
        if val_data is not None and (step % tc.val_every == 0 or step == total_steps):
            vl = eval_val_loss(model, val_data, tc, device)
            # grads from this step are still populated (zeroed next iteration)
            gnb = grad_norm_breakdown(raw_model)
            arms = activation_rms(raw_model, probe_batch)
            print(f"step {step} VAL loss {vl:.4f} (ppl {math.exp(vl):.2f}, "
                  f"{vl / LN2:.3f} bits/tok)")
            print(f"  grad_norms {{{', '.join(f'{k}: {v:.3f}' for k, v in gnb.items())}}}")
            print(f"  act_rms {{{', '.join(f'{k}: {v:.2f}' for k, v in arms.items())}}}")
            if wandb_run:
                wandb_run.log({
                    "val/loss": vl, "val/ppl": math.exp(vl),
                    "val/bits_per_token": vl / LN2, "tokens": tokens_seen,
                    **{f"grad_norm/{k}": v for k, v in gnb.items()},
                    **{f"act_rms/{k}": v for k, v in arms.items()},
                }, step=step)
        if tokens_seen >= next_ckpt or step == total_steps:
            save_ckpt()
            next_ckpt += tc.ckpt_every_tokens
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", required=True, choices=list(VARIANTS))
    ap.add_argument("--data", default="data/babylm_strict/train.bin")
    ap.add_argument("--val-data", default="data/babylm_strict/val.bin")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="", help="suffix for run name / out dir")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the model (evaluate on the H100 first)")
    ap.add_argument("--resume", default=None,
                    help="checkpoint path to resume from (same variant)")
    tc0 = TrainConfig()
    ap.add_argument("--lr", type=float, default=tc0.lr)
    ap.add_argument("--seq-len", type=int, default=tc0.seq_len)
    ap.add_argument("--batch-size", type=int, default=tc0.batch_size)
    ap.add_argument("--micro-batch-size", type=int, default=tc0.micro_batch_size)
    ap.add_argument("--token-budget", type=int, default=tc0.token_budget)
    ap.add_argument("--warmup-steps", type=int, default=tc0.warmup_steps)
    args = ap.parse_args()
    tc = TrainConfig(lr=args.lr, seq_len=args.seq_len, batch_size=args.batch_size,
                     micro_batch_size=args.micro_batch_size,
                     token_budget=args.token_budget, warmup_steps=args.warmup_steps)
    run = f"{args.variant}{'_' + args.tag if args.tag else ''}"
    train(args.variant, args.data, args.out or f"checkpoints/{run}", tc,
          val_path=args.val_data, use_wandb=args.wandb, tag=args.tag,
          compile_model=args.compile, resume=args.resume)
