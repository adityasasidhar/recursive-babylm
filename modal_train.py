"""Modal launcher for the shared training loop (src/common/train.py) on an H100.

One-time setup:
    modal setup                                   # authenticate
    modal run modal_train.py::prep_data           # download + tokenize the official
                                                  # BabyLM 2026 Strict corpus onto the
                                                  # babylm-data volume (train/val.bin)

Smoke check the remote environment (no training):
    modal run modal_train.py::check_env

Launch everything for the paper (all 5 variants, one H100 each, in parallel;
preps the data first if needed):
    modal run --detach modal_train.py::train_all

Or a single variant (detached so it survives your laptop closing):
    modal run --detach modal_train.py::main --variant recursive_2to1 --use-wandb
    modal run --detach modal_train.py::main --variant baseline --lr 3e-4 --tag lr3e-4

Checkpoints land in the `babylm-checkpoints` Volume under
/<variant>[_<tag>]/ckpt_*.pt; pull them with
    modal volume get babylm-checkpoints /<run_name> ./checkpoints/

wandb: put WANDB_API_KEY in the repo-root .env (gitignored) or export it —
the key is forwarded into the container as a Modal Secret at launch time.

DO NOT launch real training without explicit project-owner sign-off
(same hard rule as src/common/train.py).
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


def _load_dotenv() -> None:
    """Load repo-root .env into the local environment (no python-dotenv dep);
    real env vars win over .env values."""
    p = Path(__file__).parent / ".env"
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv()

APP_NAME = "babylm-2026"

# Official BabyLM 2026 Strict corpus (100M words, 6 *.train.txt files) and the
# held-out dev split (6 *.dev files) — data.py routes .txt -> train, .dev -> val.
CORPUS_REPOS = (
    "BabyLM-community/BabyLM-2026-Strict",
    "BabyLM-community/BabyLM-dev",
)

# FA3 wheel is Hopper-only and needs the cu13x torch build (see pyproject.toml).
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
FA3_WHEEL = (
    "https://download.pytorch.org/whl/cu130/"
    "flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    # host compiler for nvcc — tilelang JIT-compiles its kernels at runtime
    .apt_install("g++")
    # pinned so image rebuilds can't drift the env between paper runs; the
    # baseline trained on 2.12.1 before this pin (documented in the paper),
    # the other four variants on 2.13.0
    .uv_pip_install("torch==2.13.0", extra_index_url=TORCH_INDEX)
    .uv_pip_install(
        "transformers>=4.40",
        "tokenizers>=0.19",
        "einops>=0.7",
        "numpy>=1.26",
        "wandb>=0.17",
        "tqdm>=4.66",
        "flash-linear-attention>=0.5",
        # fla's gated chunk_bwd kernel is wrong under Triton>=3.4 on Hopper
        # (fla #640); fla raises unless its tilelang backend is installed
        "tilelang",
        # tilelang locates CUDA via CUDA_HOME -> nvcc on PATH -> the
        # `nvidia-cuda-nvcc` pip pkg (>=13 only). debian_slim has no toolkit;
        # torch's cu13 wheels fill the unified nvidia/cu13 prefix with headers
        # + libs but no compiler — these wheels complete the toolchain there,
        # exercised at the first GDN backward when tilelang JIT-compiles fla's
        # kernel. EVERY member must sit at the same 13.0.* as torch's cu130
        # runtime: nvcc's wheel ships ptxas but declares its cicc/PTX-frontend
        # dep (nvidia-nvvm) UNPINNED, and a newer nvvm emits PTX that the
        # older ptxas rejects ("Unsupported .version 9.3").
        "nvidia-cuda-nvcc==13.0.*",   # nvcc driver + ptxas/nvlink/fatbinary
        "nvidia-nvvm==13.0.*",        # cicc PTX frontend (nvcc dep, unpinned upstream)
        "nvidia-cuda-crt==13.0.*",    # crt/ host headers (nvcc dep, unpinned upstream)
        "nvidia-cuda-cccl==13.0.*",   # libcu++ (<nv/target> etc.)
    )
    # GQA auto-falls back to SDPA if this import is absent, but on H100 we
    # want FA3, so a failed install should fail the build loudly.
    .uv_pip_install(FA3_WHEEL)
    .env({"TOKENIZERS_PARALLELISM": "false",
          "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
    .add_local_python_source("src")
)

app = modal.App(APP_NAME, image=image)

data_vol = modal.Volume.from_name("babylm-data", create_if_missing=True)
ckpt_vol = modal.Volume.from_name("babylm-checkpoints", create_if_missing=True)

DATA_DIR = "/data"
CKPT_DIR = "/checkpoints"

# The five runs the paper reports. VARIANTS also holds the revision arms added
# for the BabyLM review response, so the fan-out entrypoints default to this
# tuple rather than the whole registry — otherwise train_all would spawn an
# H100 per registered variant and blimp_eval_all would demand a checkpoint for
# every one of them.
PAPER_VARIANTS = ("baseline", "gdn_2to1", "gdn_3to1",
                  "recursive_2to1", "recursive_3to1")


def _resolve_variant(run_name: str, variants) -> str | None:
    """Recover the variant from a run-directory name.

    train_remote writes to `{variant}` or `{variant}_{tag}`, so the directory
    name alone is ambiguous once variant names share a prefix (`recursive_2to1`
    vs `recursive_2to1_uniqinit`). Longest match wins, which resolves both that
    case and tagged runs (`recursive_2to1_seed43` -> `recursive_2to1`).
    """
    hits = [v for v in variants if run_name == v or run_name.startswith(v + "_")]
    return max(hits, key=len) if hits else None


def _discover_runs(runs: str = ""):
    """[(run_name, variant, run_dir)] for every run directory under CKPT_DIR
    that holds checkpoints, newest-tagged and untagged alike.

    ``runs`` is an optional comma-separated filter of run names or variant
    names; empty means the paper five (untagged runs only).
    """
    from pathlib import Path

    from src.common.variants import VARIANTS

    wanted = [r.strip() for r in runs.split(",") if r.strip()] or list(PAPER_VARIANTS)
    found = []
    for name in wanted:
        run_dir = Path(CKPT_DIR, name)
        if not run_dir.is_dir() or not any(run_dir.glob("ckpt_*.pt")):
            print(f"skipping {name}: no checkpoints under {run_dir}")
            continue
        variant = _resolve_variant(name, VARIANTS)
        if variant is None:
            print(f"skipping {name}: no registered variant matches this run name")
            continue
        found.append((name, variant, run_dir))
    return found


# Baked from the local environment at `modal run` time; empty is fine when
# not using --use-wandb.
wandb_secret = modal.Secret.from_dict(
    {"WANDB_API_KEY": os.environ.get("WANDB_API_KEY", "")}
)


@app.function(timeout=60 * 60, cpu=8, memory=32 * 1024, volumes={DATA_DIR: data_vol})
def prep_data(force: bool = False) -> None:
    """Download the official corpus from the HF Hub and tokenize it into
    /data/train.bin + /data/val.bin through src.common.data.prepare — the
    exact same code path as local prep, so the suffix routing that keeps
    *.dev files out of train.bin applies here too. CPU-only, one-time."""
    from pathlib import Path

    from huggingface_hub import snapshot_download

    train_bin, val_bin = Path(DATA_DIR, "train.bin"), Path(DATA_DIR, "val.bin")
    if not force and train_bin.exists() and val_bin.exists():
        print("train.bin and val.bin already on the volume; use --force to rebuild")
        return
    raw = "/tmp/babylm_raw"
    for repo in CORPUS_REPOS:
        snapshot_download(repo, repo_type="dataset", local_dir=raw,
                          allow_patterns=["*.train.txt", "*.dev"])
    from src.common.data import prepare

    prepare(raw, train_bin, split="train")
    prepare(raw, val_bin, split="val")
    data_vol.commit()
    print(f"done: {train_bin} ({train_bin.stat().st_size / 1e6:.0f} MB), "
          f"{val_bin} ({val_bin.stat().st_size / 1e6:.0f} MB)")


@app.function(timeout=60 * 60, cpu=8, memory=32 * 1024, volumes={DATA_DIR: data_vol})
def val_domains(force: bool = False) -> dict:
    """Recover per-domain token ranges for the val.bin already on the volume.

    val.bin predates the domain manifest, and rebuilding it would risk silently
    swapping the evaluation set out from under every published number if the
    upstream dataset or the tokenizer has moved since 2026-07-09. So this
    re-tokenizes the dev files to a scratch path, asserts the result is
    byte-identical to the val.bin in use, and then writes only the sidecar
    manifest. It never touches val.bin itself."""
    import hashlib
    import json
    from pathlib import Path

    from huggingface_hub import snapshot_download

    from src.common.data import (
        domain_manifest_path,
        load_domain_manifest,
        prepare,
        write_domain_manifest,
    )

    val_bin = Path(DATA_DIR, "val.bin")
    manifest = domain_manifest_path(val_bin)
    if manifest.exists() and not force:
        print(f"{manifest} already present; use --force to rebuild")
        return {"domains": load_domain_manifest(val_bin)}

    raw = "/tmp/babylm_raw"
    for repo in CORPUS_REPOS:
        snapshot_download(repo, repo_type="dataset", local_dir=raw,
                          allow_patterns=["*.dev"])

    probe = Path("/tmp/val_probe.bin")
    prepare(raw, probe, split="val")

    live_digest = hashlib.sha256(val_bin.read_bytes()).hexdigest()
    probe_digest = hashlib.sha256(probe.read_bytes()).hexdigest()
    if live_digest != probe_digest:
        raise RuntimeError(
            "re-tokenized dev set does not match the val.bin in use "
            f"({val_bin.stat().st_size:,} vs {probe.stat().st_size:,} bytes; "
            f"sha256 {live_digest[:12]} vs {probe_digest[:12]}). The corpus or "
            "tokenizer has moved. Every published validation number was "
            "computed on the live file — do not overwrite it, investigate.")
    print(f"verified: re-tokenized dev set is byte-identical (sha256 "
          f"{live_digest[:12]}...)")

    rebuilt = json.loads(domain_manifest_path(probe).read_text())
    write_domain_manifest(val_bin, rebuilt["tokens"], rebuilt["domains"])
    data_vol.commit()
    for d in rebuilt["domains"]:
        print(f"  {d['name']:<18} [{d['start']:>10,}, {d['end']:>10,})  "
              f"{d['end'] - d['start']:>10,} tokens")
    return rebuilt


@app.function(gpu="H100", timeout=10 * 60, volumes={DATA_DIR: data_vol})
def check_env() -> None:
    """Verify GPU, kernels, data files, and every registered variant before spending on a run."""
    from pathlib import Path

    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")

    from src.common.attention import _HAS_FA3

    print(f"FA3 available: {_HAS_FA3}")

    import fla  # noqa: F401  # Triton GDN kernels import

    print("fla import OK")

    for f in ("train.bin", "val.bin"):
        p = Path(DATA_DIR) / f
        print(f"{p}: {'%.1f MB' % (p.stat().st_size / 1e6) if p.exists() else 'MISSING'}")

    from src.common.variants import VARIANTS, load_variant, non_embedding_params

    for name in VARIANTS:
        cfg, Model = load_variant(name)
        m = Model(cfg)
        print(f"{name}: {non_embedding_params(m):,} non-embedding params")

    # A real GDN forward+backward: on Hopper fla routes chunk_bwd_dqkwg to its
    # tilelang backend (fla #640), which JIT-compiles with nvcc on first use —
    # an import check alone misses a broken CUDA toolchain in the image.
    cfg, Model = load_variant("gdn_2to1")
    m = Model(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (2, 256), device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, loss = m(x, targets=torch.roll(x, -1, dims=1))
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
    print(f"GDN fwd+bwd OK (tilelang kernel path exercised), loss {loss.item():.3f}")


@app.function(gpu="H100", timeout=30 * 60)
def smoke_remote(steps: int = 30, variants: str = "") -> None:
    """Run src.common.smoke_test on an H100.

    The local 4GB card cannot hold the optimizer state for the deep revision
    arms (untied_3to1_deep is ~190M params: weights + grads + AdamW moments is
    ~3.0GB before activations), so the gate has to run here for those. Still a
    smoke test on dummy batches, not a training run.
    """
    import sys

    sys.argv = ["smoke_test", "--steps", str(steps)]
    if variants:
        sys.argv += ["--variants", variants]
    from src.common.smoke_test import main

    rc = main()
    if rc != 0:
        raise RuntimeError("smoke test failed — see output above")


@app.function(gpu="H100", timeout=8 * 60 * 60,
              volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol})
def ckpt_eval(val_tokens: int = 2_000_000, runs: str = "") -> dict:
    """Uniformly-weighted validation loss for every variant x every retained
    checkpoint, on the exact chunk grid eval_val_loss uses (stride seq_len from
    the start of val.bin). Inference only — not a training run.

    Motivation: eval_val_loss returns a mean of per-BATCH means with batch
    size = micro_batch_size, so 488 chunks split 30x16+8 for mb=16 runs
    (trailing 8 chunks double-weighted) but 61x8 exactly for the mb=8 run —
    the logged val losses are not comparable across micro-batch settings.
    This recomputes per-chunk losses so any weighting can be applied, plus
    per-position sums on the final checkpoints for long-range analysis."""
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    from src.common.variants import load_variant

    seq_len = 4096
    val = np.memmap(f"{DATA_DIR}/val.bin", dtype=np.uint16, mode="r")
    n_chunks = min(val_tokens, len(val) - 1) // seq_len
    dest_path = Path(CKPT_DIR, "analysis", "ckpt_eval.json")
    # Extend rather than replace: a --runs-filtered call must not drop the runs
    # it was not asked to evaluate.
    out: dict[str, dict] = (
        json.loads(dest_path.read_text()) if dest_path.exists() else {})
    for run_name, variant, run_dir in _discover_runs(runs):
        cfg, Model = load_variant(variant)
        model = Model(cfg).cuda().eval()
        out[run_name] = {}
        ck_paths = sorted(run_dir.glob("ckpt_*.pt"))
        for ck_path in ck_paths:
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            assert ck["variant"] == variant, \
                f"{ck_path} holds {ck['variant']!r}, not {variant!r}"
            model.load_state_dict(ck["model"])
            chunk_losses = []
            pos_sum = torch.zeros(seq_len, dtype=torch.float64, device="cuda")
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                for i in range(n_chunks):
                    s = i * seq_len  # eval_val_loss's grid: stride = seq_len
                    chunk = torch.from_numpy(
                        val[s : s + seq_len + 1].astype(np.int64)).cuda()
                    logits, _ = model(chunk[:-1][None])
                    tok_loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        chunk[1:].view(-1), reduction="none").double()
                    chunk_losses.append(round(tok_loss.mean().item(), 5))
                    pos_sum += tok_loss
            entry = {"variant": variant,
                     "tokens_M": int(ck_path.stem.split("_")[1].rstrip("M")),
                     "chunk_losses": chunk_losses}
            if ck_path == ck_paths[-1]:  # names zero-pad to 5 digits, so this is max tokens
                entry["per_pos"] = [round(v, 5)
                                    for v in (pos_sum / n_chunks).tolist()]
            out[run_name][ck_path.stem] = entry
            print(run_name, ck_path.stem,
                  f"uniform {sum(chunk_losses)/len(chunk_losses):.4f}")
        del model
        torch.cuda.empty_cache()
    # too big for stdout (Modal truncates long lines) — park it on the volume
    dest_path.parent.mkdir(exist_ok=True)
    dest_path.write_text(json.dumps(out))
    ckpt_vol.commit()
    print(f"wrote {dest_path}")
    return out


@app.function(gpu="H100", timeout=8 * 60 * 60,
              volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol})
def domain_eval(runs: str = "", max_windows: int = 0) -> dict:
    """Per-domain validation loss on each run's largest-budget checkpoint.

    ckpt_eval scores the first 2M tokens of val.bin, and because prepare()
    concatenates the dev files in sorted-filename order that prefix is almost
    entirely BNC Spoken plus a slice of CHILDES. Reviewer BRBm asked for a
    domain-balanced re-evaluation with per-domain deltas; this scores every dev
    domain separately, on windows that never straddle a domain boundary, and
    macro-averages them into a balanced number.

    ``max_windows`` caps windows per domain (0 = use all of every domain).
    Weighting is uniform per token within a domain, and uniform per domain in
    the ``balanced_macro`` figure."""
    import json
    from pathlib import Path

    import numpy as np
    import torch
    import torch.nn.functional as F

    from src.common.data import load_domain_manifest
    from src.common.variants import load_variant

    seq_len = 4096
    val_bin = Path(DATA_DIR, "val.bin")
    val = np.memmap(val_bin, dtype=np.uint16, mode="r")
    domains = load_domain_manifest(val_bin)

    # A window needs seq_len+1 tokens (inputs plus the shifted target), and must
    # sit wholly inside one domain or its loss is not attributable to either.
    grid = []
    for d in domains:
        n = (d["end"] - d["start"] - 1) // seq_len
        if max_windows:
            n = min(n, max_windows)
        if n == 0:
            print(f"skipping {d['name']}: shorter than one {seq_len}-token window")
            continue
        grid.append((d["name"], d["start"], n))
        print(f"  {d['name']:<18} {n:>5} windows ({n * seq_len:>10,} tokens)")

    dest = Path(CKPT_DIR, "analysis", "domain_eval.json")
    dest.parent.mkdir(exist_ok=True)
    # Extend rather than replace, so a --runs-filtered call adds to the file.
    out = json.loads(dest.read_text()) if dest.exists() else {}
    out.setdefault("protocol", {})
    out["protocol"] = {
        "seq_len": seq_len,
        "weighting": "uniform per token within a domain; uniform per domain "
                     "in balanced_macro",
        "windows_per_domain": {name: n for name, _, n in grid},
        "note": "windows are contained within one dev domain and never straddle "
                "a boundary; contrast with ckpt_eval's 2M-token prefix",
    }
    models = out.setdefault("models", {})

    for run_name, variant, run_dir in _discover_runs(runs):
        cfg, Model = load_variant(variant)
        model = Model(cfg).cuda().eval()
        ck_path = sorted(run_dir.glob("ckpt_*.pt"))[-1]
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        assert ck["variant"] == variant, \
            f"{ck_path} holds {ck['variant']!r}, not {variant!r}"
        model.load_state_dict(ck["model"])
        per_domain = {}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            for name, start, n in grid:
                window_losses = []
                for i in range(n):
                    s = start + i * seq_len
                    chunk = torch.from_numpy(
                        val[s : s + seq_len + 1].astype(np.int64)).cuda()
                    logits, _ = model(chunk[:-1][None])
                    tok_loss = F.cross_entropy(
                        logits.float().view(-1, logits.size(-1)),
                        chunk[1:].view(-1), reduction="none").double()
                    window_losses.append(round(tok_loss.mean().item(), 6))
                per_domain[name] = {
                    "n_windows": n,
                    "mean_loss": sum(window_losses) / len(window_losses),
                    "window_losses": window_losses,
                }
        balanced = sum(v["mean_loss"] for v in per_domain.values()) / len(per_domain)
        models[run_name] = {
            "variant": variant,
            "checkpoint": ck_path.name,
            "per_domain": per_domain,
            "balanced_macro": balanced,
        }
        dest.write_text(json.dumps(out))
        ckpt_vol.commit()
        print(f"{run_name} ({ck_path.name}) balanced {balanced:.4f}  " + "  ".join(
            f"{k} {v['mean_loss']:.4f}" for k, v in per_domain.items()))
        del model
        torch.cuda.empty_cache()

    print(f"wrote {dest}")
    return out


@app.function(gpu="H100", timeout=8 * 60 * 60,
              volumes={CKPT_DIR: ckpt_vol})
def blimp_eval_all(batch_size: int = 64, limit_per_file: int = 0,
                   runs: str = "") -> dict:
    """Evaluate each run's final checkpoint on the official 2026 full BLiMP
    and BLiMP Supplement sets. ``limit_per_file`` is a smoke-test knob; zero
    evaluates every item. Results are persisted on the checkpoint volume."""
    import json
    from pathlib import Path

    from src.common.blimp_eval import (
        download_data,
        evaluate_checkpoint,
        result_document,
    )

    data_root = download_data("/tmp/babylm_2026_evals")
    limit = limit_per_file or None
    dest = Path(CKPT_DIR, "analysis", "blimp_eval.json")
    dest.parent.mkdir(exist_ok=True)
    # Seed from what is already on the volume so a --runs-filtered invocation
    # extends the analysis file instead of replacing it with only its own runs.
    models = json.loads(dest.read_text())["models"] if dest.exists() else {}
    for run_name, variant, run_dir in _discover_runs(runs):
        # checkpoint names zero-pad the token count to 5 digits, so the last
        # sorted name is the largest budget — the 1.5B compute-matched runs end
        # at ckpt_01499M.pt, not the 500M grid's ckpt_00499M.pt
        checkpoint = sorted(run_dir.glob("ckpt_*.pt"))[-1]
        models[run_name] = evaluate_checkpoint(
            variant,
            checkpoint,
            data_root,
            batch_size=batch_size,
            limit_per_file=limit,
        )
        models[run_name]["run_name"] = run_name
        dest.write_text(json.dumps(result_document(models), indent=2))
        ckpt_vol.commit()
    result = result_document(models)
    print("summary:")
    for run_name, entry in models.items():
        suites = entry["suites"]
        print(
            f"  {run_name:<24} {Path(entry['checkpoint']).name:<16} "
            f"BLiMP {suites['blimp']['macro_accuracy']:.2f}  "
            f"supplement {suites['blimp_supplement']['macro_accuracy']:.2f}"
        )
    print(f"wrote {dest}")
    return result


@app.function(gpu="H100", timeout=8 * 60 * 60,
              volumes={CKPT_DIR: ckpt_vol})
def zeroshot_eval_all(batch_size: int = 128, limit_per_file: int = 0,
                      runs: str = "") -> dict:
    """Official COMPS + entity-tracking accuracy for each run's largest-budget
    checkpoint, via src.common.zeroshot_eval. ``limit_per_file`` is a smoke-test
    knob; zero evaluates every item. Extends the analysis file rather than
    replacing it, as blimp_eval_all does."""
    import json
    from pathlib import Path

    from src.common.zeroshot_eval import (
        download_data,
        evaluate_checkpoint,
        result_document,
    )

    data_root = download_data("/tmp/babylm_2026_zeroshot")
    limit = limit_per_file or None
    dest = Path(CKPT_DIR, "analysis", "zeroshot_eval.json")
    dest.parent.mkdir(exist_ok=True)
    models = json.loads(dest.read_text())["models"] if dest.exists() else {}
    # Entries evaluated under a different limit are not comparable, so a real
    # run must not silently inherit leftover smoke-test rows.
    stale = [k for k, v in models.items() if v.get("limit_per_file") != limit]
    for key in stale:
        print(f"dropping {key}: evaluated at limit_per_file="
              f"{models[key].get('limit_per_file')!r}, not {limit!r}")
        del models[key]

    for run_name, variant, run_dir in _discover_runs(runs):
        checkpoint = sorted(run_dir.glob("ckpt_*.pt"))[-1]
        print(f"{run_name} ({checkpoint.name})")
        models[run_name] = evaluate_checkpoint(
            variant, checkpoint, data_root,
            batch_size=batch_size, limit_per_file=limit,
        )
        models[run_name]["run_name"] = run_name
        models[run_name]["limit_per_file"] = limit
        dest.write_text(json.dumps(result_document(models), indent=2))
        ckpt_vol.commit()

    print("summary:")
    for run_name, entry in models.items():
        tasks = entry["tasks"]
        print(f"  {run_name:<24} COMPS {tasks['comps']['macro_accuracy']:.2f}  "
              f"entity {tasks['entity_tracking']['macro_accuracy']:.2f}")
    print(f"wrote {dest}")
    return result_document(models)


@app.function(
    gpu="H100",
    timeout=24 * 60 * 60,
    volumes={DATA_DIR: data_vol, CKPT_DIR: ckpt_vol},
    secrets=[wandb_secret],
)
def train_remote(
    variant: str,
    lr: float,
    seq_len: int,
    batch_size: int,
    micro_batch_size: int,
    token_budget: int,
    warmup_steps: int,
    tag: str,
    use_wandb: bool,
    compile_model: bool,
    resume: str | None,
    extra_meta: dict | None,
    seed: int | None = None,
    ckpt_every_tokens: int | None = None,
) -> None:
    from src.common.train import TrainConfig, train

    defaults = TrainConfig()
    tc = TrainConfig(
        lr=lr,
        seq_len=seq_len,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        token_budget=token_budget,
        warmup_steps=warmup_steps,
        seed=defaults.seed if seed is None else seed,
        ckpt_every_tokens=(defaults.ckpt_every_tokens
                           if ckpt_every_tokens is None else ckpt_every_tokens),
    )
    run_name = f"{variant}{'_' + tag if tag else ''}"
    try:
        train(
            variant,
            f"{DATA_DIR}/train.bin",
            f"{CKPT_DIR}/{run_name}",
            tc,
            val_path=f"{DATA_DIR}/val.bin",
            use_wandb=use_wandb,
            tag=tag,
            compile_model=compile_model,
            resume=resume,
            extra_meta=extra_meta,
        )
    finally:
        ckpt_vol.commit()  # background commits also run, but be explicit


@app.function(timeout=24 * 60 * 60)  # CPU-only shepherd; must outlast the children
def orchestrate_all(lr: float, token_budget: int, tag: str, use_wandb: bool,
                    compile_model: bool, extra_meta: dict,
                    variants: tuple[str, ...] = PAPER_VARIANTS,
                    seed: int | None = None,
                    ckpt_every_tokens: int | None = None) -> None:
    """Preps data, fans out the requested variants to parallel H100s, then
    blocks until every run finishes. Runs remotely so the whole grid survives
    the local client disconnecting (`modal run --detach` only guarantees the
    LAST-triggered call outlives the client — so that call must be this one).
    A child failure is reported but does not kill the siblings."""
    from src.common.train import TrainConfig

    # Effective depth 24 OOMs an 80GB H100 at micro-batch 16 (activations scale
    # with effective depth); accum rises to keep the same 32-sequence optimizer
    # step, so the math is unchanged. The untied deep arms carry the same
    # activation footprint as their recursive counterparts plus 3x the
    # optimizer state, so they need the smaller micro-batch too.
    micro_batch = {"recursive_3to1": 8,
                   "untied_2to1_deep": 8, "untied_3to1_deep": 8,
                   "recursive_3to1_uniqinit": 8}

    tc0 = TrainConfig()
    prep_data.remote()  # returns immediately if the bins already exist
    calls = {
        variant: train_remote.spawn(
            variant=variant,
            lr=lr,
            seq_len=tc0.seq_len,
            batch_size=tc0.batch_size,
            micro_batch_size=micro_batch.get(variant, tc0.micro_batch_size),
            token_budget=token_budget,
            warmup_steps=tc0.warmup_steps,
            tag=tag,
            use_wandb=use_wandb,
            compile_model=compile_model,
            resume=None,
            extra_meta=extra_meta,
            seed=seed,
            ckpt_every_tokens=ckpt_every_tokens,
        )
        for variant in variants
    }
    for variant, call in calls.items():
        print(f"spawned {variant}: {call.object_id}")
    failed = []
    for variant, call in calls.items():
        try:
            call.get()
            print(f"{variant}: finished")
        except Exception as e:  # keep waiting on the siblings
            failed.append(variant)
            print(f"{variant}: FAILED — {e}")
    if failed:
        raise RuntimeError(f"failed variants: {failed}")
    print(f"all {len(calls)} variants finished")


@app.local_entrypoint()
def train_all(
    lr: float = 6e-4,
    token_budget: int = 500_000_000,
    tag: str = "",
    use_wandb: bool = True,
    compile_model: bool = False,
    variants: str = "",
    seed: int = 0,
    ckpt_every_tokens: int = 0,
) -> None:
    """The whole paper grid in one command — all 5 paper variants in parallel,
    one H100 each, identical recipe:

        modal run --detach modal_train.py::train_all

    ``--variants`` is a comma-separated subset (default: the paper five, NOT the
    whole registry — the revision arms are registered but are not part of the
    reported grid). ``--seed`` drives both weight init and data order, so a seed
    group stays internally controlled across variants; 0 keeps the pinned 42.

        modal run --detach modal_train.py::train_all \\
            --variants recursive_2to1,gdn_2to1,recursive_3to1,gdn_3to1 \\
            --seed 43 --tag seed43 --ckpt-every-tokens 25000000

    Builds train/val.bin first if the data volume is empty (no-op otherwise).
    --detach is required: without it the runs die with the client.
    """
    from src.common.train import git_info
    from src.common.variants import VARIANTS

    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY not set (repo-root .env or shell env)")
    chosen = tuple(v.strip() for v in variants.split(",") if v.strip()) or PAPER_VARIANTS
    unknown = [v for v in chosen if v not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    extra_meta = {**git_info(), "launcher": "modal"}
    call = orchestrate_all.spawn(lr=lr, token_budget=token_budget, tag=tag,
                                 use_wandb=use_wandb, compile_model=compile_model,
                                 extra_meta=extra_meta, variants=chosen,
                                 seed=seed or None,
                                 ckpt_every_tokens=ckpt_every_tokens or None)
    print(f"orchestrator launched: {call.object_id}\n"
          f"it preps data, runs {len(chosen)} variants in parallel "
          f"({', '.join(chosen)}), and reports back — "
          "track progress in the Modal dashboard and wandb project babylm-2026")


@app.local_entrypoint()
def main(
    variant: str,
    lr: float = 6e-4,
    seq_len: int = 4096,
    batch_size: int = 32,
    micro_batch_size: int = 16,  # 32 OOMs the 80GB H100 (fp32 logits for CE)
    token_budget: int = 500_000_000,
    warmup_steps: int = 250,
    tag: str = "",
    use_wandb: bool = False,
    compile_model: bool = False,
    resume: str = "",
    seed: int = 0,  # 0 keeps the pinned 42; sets weight init AND data order
    ckpt_every_tokens: int = 0,  # 0 keeps the 100M-token grid
) -> None:
    from src.common.train import git_info
    from src.common.variants import VARIANTS

    if variant not in VARIANTS:
        raise SystemExit(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")
    if use_wandb and not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("--use-wandb requires WANDB_API_KEY in your local environment")
    # the container has no .git — capture provenance here and forward it
    extra_meta = {**git_info(), "launcher": "modal"}
    train_remote.remote(
        variant=variant,
        lr=lr,
        seq_len=seq_len,
        batch_size=batch_size,
        micro_batch_size=micro_batch_size,
        token_budget=token_budget,
        warmup_steps=warmup_steps,
        tag=tag,
        use_wandb=use_wandb,
        compile_model=compile_model,
        resume=resume or None,  # e.g. /checkpoints/<run_name>/ckpt_00300M.pt
        extra_meta=extra_meta,
        seed=seed or None,
        ckpt_every_tokens=ckpt_every_tokens or None,
    )
