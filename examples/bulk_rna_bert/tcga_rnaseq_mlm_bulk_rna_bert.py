"""Pre-train BulkRNABert on a bulk RNA-seq expression CSV (e.g. TCGA).

Usage example (discrete MLM, paper config):

    python examples/tcga_rnaseq_mlm_bulk_rna_bert.py \\
        --csv-path path/to/tcga_preprocessed.csv \\
        --output-dir output/bulk_rna_bert_pretrain_tcga_discrete \\
        --mode discrete --micro-batch-size 2 --accumulation-steps 80 \\
        --max-steps 600 --learning-rate 1e-4 --save-every 50 --log-every 10 --seed 42

The training recipe (micro_batch=2, accumulation=80, Adam, lr=1e-4) matches
the configuration reported in the paper, with bf16 autocast and
FlashAttention-2 forced via SDPA for speed on modern GPUs (Blackwell /
Ampere).

Checkpoints are written to ``<output-dir>/step_{N}/`` as ``params.pt`` +
``config.json`` every ``--save-every`` effective steps. Receiving SIGTERM (for
example from the companion GPU temperature watchdog) triggers a graceful stop
that writes one last checkpoint before exiting.

Next step: ``tcga_rnaseq_extract_embeddings_bulk_rna_bert.py``.

Author: Yohei Shibata (NetID: yoheis2)
Paper: BulkRNABert: Cancer prognosis from bulk RNA-seq based language models
       (Gelard et al., PMLR 259, 2025)
Paper link: https://proceedings.mlr.press/v259/gelard25a.html
Description: CLI for BulkRNABert MLM pre-training on a bulk RNA-seq
    expression CSV with gradient accumulation, step-based checkpointing,
    and SIGTERM-safe graceful stop.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

# Make PyHealth importable when running from a source checkout.
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyhealth.models import (  # noqa: E402
    BulkRNABert,
    BulkRNABertConfig,
    load_expression_csv,
)


# ---------------------------------------------------------------------------
# Graceful-shutdown flag (flipped by SIGTERM / SIGINT)
# ---------------------------------------------------------------------------

_STOP_REQUESTED = False


def _install_signal_handlers() -> None:
    def handler(signum, _frame):
        global _STOP_REQUESTED
        _STOP_REQUESTED = True
        print(
            f"[signal] received signal {signum}; will stop after current step",
            flush=True,
        )

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    p.add_argument("--csv-path", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument(
        "--mode", choices=["discrete", "continuous"], default="discrete"
    )
    p.add_argument("--micro-batch-size", type=int, default=2)
    p.add_argument("--accumulation-steps", type=int, default=80)
    p.add_argument("--max-steps", type=int, default=600,
                   help="Stop after this many effective (post-accumulation) steps.")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--schedule-total-steps", type=int, default=10000,
                   help="Total-step budget for the LR schedule shape. Training "
                        "stops at --max-steps but the schedule is shaped as if "
                        "training were to run this long.")
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-every", type=int, default=50,
                   help="Effective-step interval for checkpoint saving.")
    p.add_argument("--log-every", type=int, default=10,
                   help="Effective-step interval for log lines.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bins", type=int, default=64)
    p.add_argument("--normalization-factor", type=float, default=None)
    p.add_argument(
        "--already-log-normalized", action="store_true",
        help="Set if the CSV values are already log10(TPM+1) rather than raw TPM."
    )
    p.add_argument(
        "--autocast-dtype", choices=["bfloat16", "float16", "none"], default="bfloat16",
        help="Mixed-precision dtype for autocast inside forward. 'none' disables autocast."
    )
    p.add_argument(
        "--no-force-flash-attention", action="store_true",
        help="Disable the explicit FLASH_ATTENTION SDPA backend (default: on).",
    )
    p.add_argument(
        "--init-gene-embedding-from", type=Path, default=None,
        help="Path to a PyTorch .pt state_dict containing "
             "gene_embedding.embed.weight, gene_embedding.proj.weight, and "
             "gene_embedding.proj.bias. Only those three tensors are copied "
             "into model.gene_embedding before training starts; attention "
             "and LM-head weights stay at fresh initialization. Extra keys "
             "are ignored so a full model checkpoint can be passed directly.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_data(args: argparse.Namespace) -> Tuple[torch.Tensor, List[str]]:
    print(f"[data] loading {args.csv_path} (mode={args.mode}) ...", flush=True)
    t0 = time.time()
    load_kwargs = dict(
        mode=args.mode,
        n_bins=args.n_bins,
        already_log_normalized=args.already_log_normalized,
    )
    if args.normalization_factor is not None:
        load_kwargs["normalization_factor"] = args.normalization_factor
    data, gene_names = load_expression_csv(args.csv_path, **load_kwargs)
    print(
        f"[data] loaded tensor {tuple(data.shape)} dtype={data.dtype} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return data, gene_names


def _build_model(
    args: argparse.Namespace, n_genes: int
) -> Tuple[BulkRNABert, BulkRNABertConfig]:
    cfg = BulkRNABertConfig(
        n_genes=n_genes,
        n_bins=args.n_bins,
        expression_mode=args.mode,
        continuous_hidden_dim=None,
        autocast_dtype=None if args.autocast_dtype == "none" else args.autocast_dtype,
    )
    model = BulkRNABert(dataset=None, config=cfg, feature_key="expression")
    return model, cfg


GENE_EMBEDDING_PT_KEYS = (
    "gene_embedding.embed.weight",
    "gene_embedding.proj.weight",
    "gene_embedding.proj.bias",
)


def _load_gene_embedding_from_pt(model: BulkRNABert, path: Path) -> None:
    """Copy gene_embedding tensors from a PyTorch ``.pt`` state_dict.

    The file must be a dict (loadable with ``torch.load(..., weights_only=True)``)
    containing the three keys listed in :data:`GENE_EMBEDDING_PT_KEYS`. Extra
    keys are ignored so a full model state_dict can be passed directly. Only
    these three tensors are copied into ``model.gene_embedding``; attention
    and LM-head weights keep their fresh initialization.
    """
    gm = model.gene_embedding
    if gm is None:
        raise ValueError(
            "model has no gene_embedding module (config.use_gene_embedding=False)"
        )
    if gm.proj is None:
        raise ValueError(
            "cannot load external gene-embedding weights: model.gene_embedding "
            "has no proj layer (init_gene_embed_dim == embed_dim). Set "
            "init_gene_embed_dim=200 and embed_dim=256 to match the published "
            "checkpoint shape."
        )

    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(
            f"expected dict state_dict from {path}, got {type(state).__name__}"
        )
    missing = [k for k in GENE_EMBEDDING_PT_KEYS if k not in state]
    if missing:
        raise KeyError(
            f"{path} is missing required gene_embedding keys: {missing}. "
            f"Expected all of {list(GENE_EMBEDDING_PT_KEYS)}."
        )

    ge = state["gene_embedding.embed.weight"]
    pw = state["gene_embedding.proj.weight"]
    pb = state["gene_embedding.proj.bias"]

    if gm.embed.weight.shape != ge.shape:
        raise ValueError(
            f"gene_embedding.embed.weight shape mismatch: model "
            f"{tuple(gm.embed.weight.shape)} vs file {tuple(ge.shape)}"
        )
    if gm.proj.weight.shape != pw.shape:
        raise ValueError(
            f"gene_embedding.proj.weight shape mismatch: model "
            f"{tuple(gm.proj.weight.shape)} vs file {tuple(pw.shape)}"
        )
    if gm.proj.bias.shape != pb.shape:
        raise ValueError(
            f"gene_embedding.proj.bias shape mismatch: model "
            f"{tuple(gm.proj.bias.shape)} vs file {tuple(pb.shape)}"
        )

    with torch.no_grad():
        gm.embed.weight.copy_(ge)
        gm.proj.weight.copy_(pw)
        gm.proj.bias.copy_(pb)
    print(f"[init] loaded gene_embedding from {path}", flush=True)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def _save_checkpoint(
    model: BulkRNABert,
    cfg: BulkRNABertConfig,
    step: int,
    output_dir: Path,
    normalization_factor: Optional[float] = None,
) -> Path:
    ckpt_dir = output_dir / f"step_{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ckpt_dir / "params.pt")
    cfg_dict = asdict(cfg)
    if normalization_factor is not None:
        cfg_dict["normalization_factor"] = normalization_factor
    with open(ckpt_dir / "config.json", "w") as f:
        json.dump(cfg_dict, f, indent=2)
    return ckpt_dir


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _iter_micro_batches(
    data: torch.Tensor, micro_batch_size: int, generator: torch.Generator
):
    """Yield micro-batch tensors, reshuffling every full pass."""
    n = data.shape[0]
    while True:
        perm = torch.randperm(n, generator=generator)
        for i in range(0, n - micro_batch_size + 1, micro_batch_size):
            idx = perm[i : i + micro_batch_size]
            yield data[idx]


def _train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Save a run manifest so downstream analysis can recover the command line.
    with open(args.output_dir / "run_args.json", "w") as f:
        json.dump(
            {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            f,
            indent=2,
        )

    data, gene_names = _load_data(args)
    n_samples, n_genes = data.shape
    if args.micro_batch_size < 1:
        raise ValueError(
            f"--micro-batch-size must be >= 1 (got {args.micro_batch_size})"
        )
    if args.accumulation_steps < 1:
        raise ValueError(
            f"--accumulation-steps must be >= 1 "
            f"(got {args.accumulation_steps})"
        )
    if args.micro_batch_size > n_samples:
        raise ValueError(
            f"--micro-batch-size ({args.micro_batch_size}) exceeds "
            f"n_samples ({n_samples}); reduce --micro-batch-size so the "
            f"iterator can yield at least one batch."
        )
    effective_batch = args.micro_batch_size * args.accumulation_steps
    if effective_batch > n_samples:
        print(
            f"[warn] effective batch ({effective_batch}) exceeds n_samples "
            f"({n_samples}); samples will be reused within a single "
            f"accumulation window.",
            flush=True,
        )
    with open(args.output_dir / "gene_names.json", "w") as f:
        json.dump(gene_names, f)

    model, cfg = _build_model(args, n_genes=n_genes)
    if args.init_gene_embedding_from is not None:
        _load_gene_embedding_from_pt(model, args.init_gene_embedding_from)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(
        f"[model] n_genes={cfg.n_genes} mode={cfg.expression_mode} "
        f"params={sum(p.numel() for p in model.parameters()):,} device={device} "
        f"autocast={cfg.autocast_dtype}",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    def _lr_lambda(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        remaining = args.schedule_total_steps - args.warmup_steps
        return max(0.0, (args.schedule_total_steps - step) / max(1, remaining))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)
    model.train()

    generator = torch.Generator().manual_seed(args.seed)
    micro_iter = _iter_micro_batches(data, args.micro_batch_size, generator)

    from contextlib import nullcontext
    # Attention-backend context: FlashAttention only (fastest path on modern
    # GPUs when autocast is bf16/fp16).
    if args.no_force_flash_attention or not torch.cuda.is_available():
        attn_ctx = nullcontext()
    else:
        attn_ctx = sdpa_kernel([SDPBackend.FLASH_ATTENTION])

    steps_per_epoch = n_samples // (args.micro_batch_size * args.accumulation_steps)
    print(
        f"[train] micro_batch={args.micro_batch_size} accum={args.accumulation_steps} "
        f"effective_batch={args.micro_batch_size * args.accumulation_steps} "
        f"eff_steps_per_epoch~{steps_per_epoch} max_steps={args.max_steps}",
        flush=True,
    )

    eff_step = 0
    micro_accum_loss = 0.0
    t_start = time.time()
    t_last_log = t_start

    with attn_ctx:
        while eff_step < args.max_steps and not _STOP_REQUESTED:
            optimizer.zero_grad(set_to_none=True)
            total_loss_value = 0.0
            for _ in range(args.accumulation_steps):
                batch = next(micro_iter)
                out = model(expression=batch)
                loss = out["loss"] / args.accumulation_steps
                loss.backward()
                total_loss_value += float(loss.detach()) * args.accumulation_steps
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=args.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            eff_step += 1
            avg_micro_loss = total_loss_value / args.accumulation_steps
            micro_accum_loss += avg_micro_loss

            if eff_step % args.log_every == 0:
                elapsed = time.time() - t_start
                since_last = time.time() - t_last_log
                t_last_log = time.time()
                mean_loss = micro_accum_loss / args.log_every
                micro_accum_loss = 0.0
                peak_mem = (
                    torch.cuda.max_memory_allocated() / (1024 ** 3)
                    if torch.cuda.is_available()
                    else 0.0
                )
                current_lr = optimizer.param_groups[0]["lr"]
                print(
                    f"[step {eff_step:4d}/{args.max_steps}] "
                    f"loss={mean_loss:.4f} lr={current_lr:.2e} "
                    f"elapsed={elapsed:.1f}s (+{since_last:.1f}s) "
                    f"peak_mem={peak_mem:.2f}GB",
                    flush=True,
                )

            if eff_step % args.save_every == 0:
                path = _save_checkpoint(
                    model, cfg, eff_step, args.output_dir,
                    normalization_factor=args.normalization_factor,
                )
                print(f"[ckpt] saved {path}", flush=True)

        # Final safety save if a graceful stop happened off-boundary.
        if _STOP_REQUESTED and eff_step % args.save_every != 0:
            path = _save_checkpoint(
                model, cfg, eff_step, args.output_dir,
                normalization_factor=args.normalization_factor,
            )
            print(f"[ckpt] (graceful stop) saved {path}", flush=True)

    elapsed = time.time() - t_start
    print(
        f"[done] stopped at step {eff_step} after {elapsed:.1f}s "
        f"(stop_requested={_STOP_REQUESTED})",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    _install_signal_handlers()
    _train(args)


if __name__ == "__main__":
    main()
