"""ECG delineation on LUDB with UNet 3+ (reproduction of Park et al., CHIL 2025).

Paper:
    Park, J., Park, T., Kwon, J. & Jo, Y.. (2025). Benchmarking ECG
    Delineation using Deep Neural Network-based Semantic Segmentation
    Models. Proceedings of the sixth Conference on Health, Inference, and
    Learning, in Proceedings of Machine Learning Research 287:63-88
    Available from https://proceedings.mlr.press/v287/park25a.html.

Authors:
    Mateo Ignacio (Mateo.Ignacio@gatech.edu), Minwoo Jo (mjo38@gatech.edu)

Description:
    Full-pipeline example for the LUDB ECG delineation contribution: it
    reproduces the paper's UNet 3+ result on LUDB and runs a preprocessing
    filter ablation. ``--demo`` runs the same pipeline on a few synthetic
    records, so it works without downloading LUDB.

Pipeline (data loading -> task -> training -> evaluation):
    1. ``LUDBDataset`` indexes the 200 LUDB records.
    2. ``ECGDelineationLUDB`` turns each lead into a (1, 5000) filtered,
       z-scored signal and a (3, 5000) P / QRS / T mask (2,400 samples).
    3. The records are split with the authors' released 160 / 20 / 20 split
       (``LUDBDataset.default_subject_split``), so test scores compare
       directly with Table 1 of the paper.
    4. ``UNet3Plus`` is trained with the recipe of the paper's final stage
       (Stage 2, model fine-tuning): SGD with momentum 0.9, weight decay, a
       polynomial learning-rate scheduler and batch 64. The paper samples
       the learning rate (1e-4 to 1e-1), weight decay (5e-6 to 5e-4) and
       scheduler power (0 to 1, step 0.25) over 256 trials. Our defaults are
       the best of 6 trials sampled from the same ranges and trained for 30
       epochs each (validation mIoU 0.856). The paper trains for up to 500
       epochs; we default to 60, because the best validation epoch of every
       ablation run below fell between epochs 10 and 38. The epoch with the
       best validation mIoU is kept, as in the paper.
    5. Per-wave IoU and mIoU on the test records are compared with the
       paper's UNet 3+ result (P 0.814, QRS 0.881, T 0.868, mIoU 0.854).
       LUDB leaves the first and last beats of each lead unannotated, so
       scores (and validation model selection) use the annotated span of
       each lead; whole-lead scores are reported as well.

Ablation (preprocessing filter):
    The paper filters every lead with a 0.05-150 Hz Butterworth band-pass
    filter but never compares it with other settings, and the demo inputs
    the authors released are not filtered at all. The ablation retrains the
    same model with each ``filter_mode`` of the task, keeping everything
    else fixed, and reports the mean and standard deviation over seeds:

    - ``bandpass``: 0.05 Hz high-pass + 150 Hz low-pass (the paper).
    - ``highpass``: 0.05 Hz only; removes baseline wander.
    - ``lowpass``: 150 Hz only; removes high-frequency noise.
    - ``none``: the signal as stored in LUDB.

    The ``bandpass`` rows double as the reproduction of Table 1.

Run:
    # Quick check on 12 synthetic records: no download, about a minute on a
    # CPU. The synthetic beats are easy, so the scores only show that every
    # stage works; they say nothing about LUDB.
    python examples/ludb_ecg_delineation_unet3plus.py --demo

    # Single run with the paper's filter on LUDB v1.0.1
    # (https://physionet.org/content/ludb/1.0.1/).
    python examples/ludb_ecg_delineation_unet3plus.py \
        --root /path/to/lobachevsky-university-electrocardiography-database-1.0.1

    # Filter ablation: 4 filter modes x 3 seeds.
    python examples/ludb_ecg_delineation_unet3plus.py --root ... \
        --filter-modes bandpass highpass lowpass none --seeds 0 1 2

Results:
    Filter ablation on LUDB v1.0.1 with the authors' split: 60 epochs and
    seeds 0, 1, 2 for each filter, run one after another on a single
    RTX 5060 Ti (8 GB), 33-57 minutes per run. Test IoU on the annotated
    span of each lead, mean ± std over the 3 seeds; the last column scores
    the whole lead instead:

    filter     P              QRS            T              mIoU           whole lead
    bandpass   0.800 ± 0.005  0.882 ± 0.013  0.842 ± 0.009  0.841 ± 0.007  0.716 ± 0.009
    highpass   0.807 ± 0.018  0.880 ± 0.005  0.845 ± 0.004  0.844 ± 0.008  0.708 ± 0.004
    lowpass    0.806 ± 0.022  0.889 ± 0.001  0.856 ± 0.005  0.850 ± 0.007  0.708 ± 0.003
    none       0.813 ± 0.001  0.879 ± 0.005  0.852 ± 0.008  0.848 ± 0.004  0.697 ± 0.010
    paper      0.814          0.881          0.868          0.854          -

    Reproduction (``bandpass``, the paper's setting): test mIoU 0.841 ±
    0.007 against the paper's 0.854 (0.850-0.860 over its 5 seeds), so
    the result comes close but stays slightly below. QRS matches the paper
    (0.882 vs 0.881); most of the gap is in T (-0.026) and P (-0.014).
    Possible causes, not tested here: we train with BCE + Dice only and
    leave out the paper's MS-SSIM loss term (in the paper's loss ablation,
    Table 11, BCE + Dice alone reaches at most 0.850); our hyperparameters
    come from 6 trials instead of 256; and we train 60 epochs instead of up
    to 500.

    Filter ablation: on the annotated span all four filters land within
    0.009 of each other (0.841-0.850), about the size of the seed-to-seed
    spread (std 0.004-0.008), so with 3 seeds no filter is clearly better.
    The two settings without the 0.05 Hz high-pass (``lowpass``, ``none``)
    score slightly higher, mostly on T. On the whole lead the order
    changes: ``bandpass`` scores highest (0.716) and ``none`` lowest
    (0.697). Whole-lead scores are lower for every filter because LUDB
    leaves the first and last beats unlabelled, so waves the model marks
    there count as errors.

    The best validation epoch of every run fell between epochs 10 and 38.
    Every run is also written to ``--output`` as JSON (per-run scores,
    training history and the mean / std summary).
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import LUDBDataset, SubjectSplit, get_dataloader
from pyhealth.datasets.ludb import LUDB_LEADS, LUDB_SAMPLING_RATE
from pyhealth.models import UNet3Plus, segmentation_iou
from pyhealth.tasks import ECGDelineationLUDB

PAPER_UNET3PLUS = {"iou_P": 0.814, "iou_QRS": 0.881, "iou_T": 0.868, "miou": 0.854}
METRICS = ("iou_P", "iou_QRS", "iou_T", "miou")
FILTER_MODES = ("bandpass", "highpass", "lowpass", "none")


def write_demo_ludb(root: Path, n_records: int = 12, seed: int = 0) -> list[str]:
    """Write synthetic records in the LUDB v1.0.1 layout under ``root``.

    Each 10-second, 12-lead record is a train of beats built from Gaussian
    P, QRS and T bumps plus noise and slow baseline wander, with the same
    ``( peak )`` WFDB annotations LUDB uses. Every fourth record has no P
    waves and an irregular rhythm, like atrial fibrillation.

    Args:
        root: Folder to create; receives ``ludb.csv`` and ``data/``.
        n_records: Number of records to write.
        seed: Seed for the beat timing, amplitudes and noise.

    Returns:
        The record ids, as strings.
    """
    import wfdb

    data_dir = root / "data"
    data_dir.mkdir(parents=True)
    fs = LUDB_SAMPLING_RATE
    t = np.arange(10 * fs) / fs
    rng = np.random.default_rng(seed)

    def bump(center: float, width: float) -> np.ndarray:
        return np.exp(-0.5 * ((t - center) / width) ** 2)

    rows = []
    for rid in range(1, n_records + 1):
        has_p = rid % 4 != 0
        beats, b = [], rng.uniform(0.4, 0.8)
        while b + 0.45 < t[-1]:
            beats.append(b)
            b += rng.uniform(0.7, 1.0) if has_p else rng.uniform(0.5, 1.0)

        wave = np.zeros_like(t)
        events: list[tuple[int, str]] = []
        for b in beats:
            parts = [("p", b - 0.16, 0.02, 0.15)] if has_p else []
            parts += [("N", b, 0.008, 1.0), ("t", b + 0.28, 0.045, 0.3)]
            for symbol, center, width, amplitude in parts:
                wave += amplitude * bump(center, width)
                half = 2.5 * width
                for mark, at in (
                    ("(", center - half),
                    (symbol, center),
                    (")", center + half),
                ):
                    events.append((int(round(at * fs)), mark))
        events.sort()

        gains = rng.uniform(0.5, 1.5, len(LUDB_LEADS))
        gains[LUDB_LEADS.index("avr")] *= -1
        drift = 0.1 * np.sin(2 * np.pi * 0.2 * t + rng.uniform(0, 2 * np.pi))
        signal = np.stack(
            [g * wave + drift + 0.02 * rng.normal(size=t.size) for g in gains],
            axis=1,
        )
        wfdb.wrsamp(
            str(rid),
            fs=fs,
            units=["mV"] * len(LUDB_LEADS),
            sig_name=list(LUDB_LEADS),
            p_signal=signal,
            fmt=["16"] * len(LUDB_LEADS),
            write_dir=str(data_dir),
        )
        for lead in LUDB_LEADS:
            # wfdb only writes letter extensions; rename to ``.v1`` etc.
            wfdb.wrann(
                str(rid),
                "tmpann",
                sample=np.array([at for at, _ in events]),
                symbol=[mark for _, mark in events],
                write_dir=str(data_dir),
            )
            (data_dir / f"{rid}.tmpann").replace(data_dir / f"{rid}.{lead}")
        rows.append(
            {
                "ID": rid,
                "Sex": "F" if rid % 2 else "M",
                "Age": str(int(rng.integers(20, 80))),
                "Rhythms": "Sinus rhythm" if has_p else "Atrial fibrillation",
                "Electric axis of the heart": "Electric axis of the heart: normal",
            }
        )
    pd.DataFrame(rows).to_csv(root / "ludb.csv", index=False)
    return [str(row["ID"]) for row in rows]


def evaluate(
    model: UNet3Plus, loader: Any, annotated_only: bool = False
) -> dict[str, float]:
    """Return per-wave IoU and mIoU of ``model`` on ``loader``.

    Args:
        model: Trained segmentation model.
        loader: PyHealth data loader yielding ``signal`` and ``mask``.
        annotated_only: Score only the annotated span of each lead.

    Returns:
        ``{"iou_P", "iou_QRS", "iou_T", "miou"}``.
    """
    model.eval()
    trues, probs = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(**batch)
            trues.append(out["y_true"].cpu().numpy())
            probs.append(out["y_prob"].cpu().numpy())
    return segmentation_iou(
        np.concatenate(trues), np.concatenate(probs), annotated_only=annotated_only
    )


def train_one(
    args: argparse.Namespace, splits: tuple, filter_mode: str, seed: int
) -> dict[str, Any]:
    """Train one model on one filter mode with one seed, then test it.

    Args:
        args: Parsed command-line options (optimiser and model settings).
        splits: ``(train, val, test)`` sample datasets for ``filter_mode``.
        filter_mode: Task filter used to build ``splits``; only recorded.
        seed: Seed for weight initialisation and batch order.

    Returns:
        A dict with the run settings, the best validation scores, the test
        scores (annotated span and whole lead) and the training history.
    """
    train_ds, val_ds, test_ds = splits
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_loader = get_dataloader(
        train_ds, batch_size=args.batch_size, shuffle=True, seed=seed
    )
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size)
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size)

    model = UNet3Plus(
        train_ds,
        base_channels=args.base_channels,
        depth=5,
        kernel_size=args.kernel_size,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=args.epochs, power=args.poly_power
    )

    # Train, keeping the weights of the epoch with the best validation mIoU.
    best: dict[str, Any] = {"miou": -1.0}
    best_state = None
    history = []
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            loss = model(**batch)["loss"]
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        val = evaluate(model, val_loader, annotated_only=True)
        val_full = evaluate(model, val_loader)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                **val,
                "full_lead_miou": val_full["miou"],
            }
        )
        if val["miou"] > best["miou"]:
            best = {"epoch": epoch, **val}
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(
            f"[{filter_mode} seed {seed}] epoch {epoch:3d}  "
            f"loss {np.mean(losses):.4f}  val mIoU {val['miou']:.4f}  "
            f"(best {best['miou']:.4f} @ {best['epoch']})"
        )

    # Test the best-validation weights.
    model.load_state_dict(best_state)
    if args.checkpoint_dir:
        Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        torch.save(best_state, Path(args.checkpoint_dir) / f"{filter_mode}_{seed}.pt")
    return {
        "filter_mode": filter_mode,
        "seed": seed,
        "n_params": sum(p.numel() for p in model.parameters()),
        "device": device,
        "train_minutes": round((time.time() - start) / 60, 1),
        "best_val": best,
        "test_annotated_span": evaluate(model, test_loader, annotated_only=True),
        "test_full_lead": evaluate(model, test_loader),
        "history": history,
    }


def summarise(runs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mean and sample standard deviation of the test scores per filter mode.

    Args:
        runs: Outputs of :func:`train_one`.

    Returns:
        ``{filter_mode: {"n_seeds", "<metric>": {"mean", "std"}, ...}}`` for
        the annotated-span metrics, plus ``full_lead_miou``.
    """
    summary: dict[str, dict[str, Any]] = {}
    for mode in dict.fromkeys(run["filter_mode"] for run in runs):
        mode_runs = [run for run in runs if run["filter_mode"] == mode]
        columns = {m: [r["test_annotated_span"][m] for r in mode_runs] for m in METRICS}
        columns["full_lead_miou"] = [r["test_full_lead"]["miou"] for r in mode_runs]
        summary[mode] = {"n_seeds": len(mode_runs)}
        for name, values in columns.items():
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            summary[mode][name] = {"mean": float(np.mean(values)), "std": std}
    return summary


def print_summary(summary: dict[str, dict[str, Any]]) -> None:
    """Print the ablation table (test set, annotated span, mean ± std)."""
    header = f"{'filter':10s}{'seeds':>6s}" + "".join(
        f"{name:>16s}" for name in ("P", "QRS", "T", "mIoU", "mIoU whole lead")
    )
    print("\n" + header)
    for mode, row in summary.items():
        cells = [row[m] for m in (*METRICS, "full_lead_miou")]
        print(
            f"{mode:10s}{row['n_seeds']:>6d}"
            + "".join(f"{c['mean']:>9.3f} ± {c['std']:.3f}" for c in cells)
        )
    print(
        f"{'paper':10s}{'':>6s}"
        + "".join(f"{PAPER_UNET3PLUS[m]:>9.3f}{'':8s}" for m in METRICS)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", default=None, help="LUDB v1.0.1 folder")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="run on 12 synthetic records instead of LUDB (quick check)",
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument(
        "--filter-modes",
        nargs="+",
        default=["bandpass"],
        choices=FILTER_MODES,
        help="task filters to compare (the ablation); bandpass is the paper's",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument(
        "--epochs", type=int, default=None, help="default 60 (10 with --demo)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="default 64 (16 with --demo)"
    )
    parser.add_argument("--lr", type=float, default=5.47e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=8.17e-5)
    parser.add_argument(
        "--poly-power", type=float, default=1.0, help="0 keeps the lr constant"
    )
    parser.add_argument(
        "--base-channels", type=int, default=None, help="default 32 (8 with --demo)"
    )
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--output", default="ludb_unet3plus_results.json")
    parser.add_argument(
        "--checkpoint-dir", default=None, help="save best weights of every run here"
    )
    args = parser.parse_args()
    if not args.demo and args.root is None:
        parser.error("give --root (LUDB v1.0.1 folder) or --demo")
    if args.epochs is None:
        args.epochs = 10 if args.demo else 60
    if args.batch_size is None:
        args.batch_size = 16 if args.demo else 64
    if args.base_channels is None:
        args.base_channels = 8 if args.demo else 32

    demo_dir = None
    if args.demo:
        # Synthetic records: 8 train / 2 validation / 2 test.
        # A fixed folder, cleared first: on Windows the memory-mapped sample
        # cache stays locked until the process exits, so the previous run's
        # files can only be removed now.
        demo_dir = Path(tempfile.gettempdir()) / "pyhealth_ludb_demo"
        shutil.rmtree(demo_dir, ignore_errors=True)
        ids = write_demo_ludb(demo_dir / "ludb")
        root, cache_dir = str(demo_dir / "ludb"), str(demo_dir / "cache")
        split = SubjectSplit((ids[:-4], ids[-4:-2], ids[-2:]))
    else:
        # The authors' released LUDB split, so test scores match Table 1.
        root, cache_dir = args.root, args.cache_dir
        split = LUDBDataset.default_subject_split()
    try:
        run_all(args, root, cache_dir, split)
    finally:
        if demo_dir is not None:
            shutil.rmtree(demo_dir, ignore_errors=True)


def run_all(
    args: argparse.Namespace, root: str, cache_dir: str | None, split: SubjectSplit
) -> None:
    """Train every (filter mode, seed) pair and print the ablation table.

    Args:
        args: Parsed command-line options.
        root: LUDB folder (real or synthetic).
        cache_dir: PyHealth cache folder, or None for the default.
        split: Record-level train / validation / test split.
    """
    # 1. Dataset (shared by every run).
    dataset = LUDBDataset(root=root, cache_dir=cache_dir)

    runs = []
    for mode in args.filter_modes:
        # 2-3. Task samples for this filter, split by the authors' records.
        splits = dataset.set_task(ECGDelineationLUDB(filter_mode=mode), split=split)
        print(f"[{mode}] samples: " + " / ".join(str(len(s)) for s in splits))
        for seed in args.seeds:
            # 4-5. Train and test one model.
            runs.append(train_one(args, splits, mode, seed))
            result = {
                "args": vars(args),
                "paper_unet3plus": PAPER_UNET3PLUS,
                "summary": summarise(runs),
                "runs": runs,
            }
            # Rewritten after every run so a long ablation keeps partial results.
            Path(args.output).write_text(json.dumps(result, indent=1))

    print_summary(summarise(runs))
    print(f"results written to {args.output}")


if __name__ == "__main__":
    main()
