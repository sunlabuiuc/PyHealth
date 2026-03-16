"""Covariate-Shift Adaptive Conformal Prediction (CovariateLabel) on TUEV EEG Events using ContraWR.

This script:
1) Loads the TUEV dataset and applies the EEGEventsTUEV task (once, shared across all seeds).
2) Extracts the fixed test set (TUH eval partition — never changes across seeds).
3) For each seed: splits the TUH train partition into train/val/cal, trains ContraWR,
   extracts cal and test embeddings, calibrates a CovariateLabel predictor, and
   evaluates on the fixed test set.
4) Reports per-run results and mean ± std summary across all seeds.

Single-seed usage (from repo root):
  python examples/conformal_eeg/tuev_covariate_shift_conformal.py --root downloads/tuev/v2.0.1/edf

Multi-seed usage (recommended for papers):
  python examples/conformal_eeg/tuev_covariate_shift_conformal.py \\
      --root downloads/tuev/v2.0.1/edf --n-seeds 5 --seed 42 --alpha 0.1 \\
      --log-file tuev_covariate_alpha0.1_5seeds.log

Notes:
- CovariateLabel requires access to test embeddings to estimate density ratios.
- Test embeddings are recomputed each seed since the model changes.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


class _Tee:
    """Writes to both a stream and a file simultaneously."""

    def __init__(self, stream, file):
        self._stream = stream
        self._file = file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()

    def flush(self):
        self._stream.flush()
        self._file.flush()


from pyhealth.calib.predictionset.covariate import CovariateLabel
from pyhealth.calib.utils import extract_embeddings
from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_sample_conformal_tuh, split_by_sample_conformal
from pyhealth.models import ContraWR
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer, get_metrics_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Covariate-shift adaptive conformal prediction (CovariateLabel) on TUEV EEG events using ContraWR."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf",
        help="Path to TUEV edf/ folder.",
    )
    parser.add_argument("--subset", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base seed. With --n-seeds N, runs seeds seed, seed+1, ..., seed+N-1.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--alpha", type=float, default=0.1,
        help="Miscoverage rate (e.g., 0.1 => 90% target coverage).",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.6, 0.2, 0.2),
        metavar=("TRAIN", "VAL", "CAL"),
        help="Ratios for splitting the TUH train partition into train/val/cal. "
             "Must sum to 1.0. Test is fixed as the TUH eval partition.",
    )
    parser.add_argument("--n-fft", type=int, default=128, help="STFT FFT size used by ContraWR.")
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device string, e.g. 'cuda:0' or 'cpu'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of seeds to run sequentially for mean±std reporting. "
             "Seeds are seed, seed+1, ..., seed+n_seeds-1.",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Explicit comma-separated seeds, e.g. '42,43,44,45,46'. "
             "Overrides --seed and --n-seeds.",
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to log file. Stdout and stderr are teed to this file.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Smoke test: dev=True, max 2000 samples, 2 epochs.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_one_seed(
    args,
    sample_dataset,
    test_ds,
    test_loader,
    device: str,
    epochs: int,
    run_seed: int,
) -> dict:
    """Train ContraWR + calibrate CovariateLabel predictor for one seed.

    The test set is passed in pre-built (fixed TUH eval partition).
    Only train/val/cal (and model weights) vary per seed.
    Test embeddings are recomputed each seed because the model changes.

    Returns a dict with keys:
        accuracy, f1_weighted, coverage, miscoverage, avg_set_size
    """
    set_seed(run_seed)

    train_ds, val_ds, cal_ds, _ = split_by_sample_conformal_tuh(
        dataset=sample_dataset, ratios=list(args.ratios), seed=run_seed
    )
    print(f"  Split — Train: {len(train_ds)}, Val: {len(val_ds)}, "
          f"Cal: {len(cal_ds)}, Test: {len(test_ds)} (fixed)")

    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = (
        get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False)
        if len(val_ds) else None
    )

    print("  Training ContraWR...")
    model = ContraWR(dataset=sample_dataset, n_fft=args.n_fft).to(device)
    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="accuracy" if val_loader is not None else None,
    )

    # Base model metrics on fixed test set
    y_true_base, y_prob_base, _ = trainer.inference(test_loader)
    base_metrics = get_metrics_fn("multiclass")(
        y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"]
    )

    # Extract embeddings — both depend on the current model so must redo each seed
    print("  Extracting embeddings for calibration and test splits...")
    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=args.batch_size, device=device)
    test_embeddings = extract_embeddings(model, test_ds, batch_size=args.batch_size, device=device)

    # Conformal calibration + evaluation
    print("  Calibrating CovariateLabel predictor...")
    cov_predictor = CovariateLabel(model=model, alpha=float(args.alpha))
    cov_predictor.calibrate(
        cal_dataset=cal_ds,
        cal_embeddings=cal_embeddings,
        test_embeddings=test_embeddings,
    )

    print("  Evaluating CovariateLabel predictor on test set...")
    y_true, y_prob, _, extra = Trainer(model=cov_predictor).inference(
        test_loader, additional_outputs=["y_predset"]
    )
    conf_metrics = get_metrics_fn("multiclass")(
        y_true, y_prob,
        metrics=["accuracy", "miscoverage_ps"],
        y_predset=extra["y_predset"],
    )

    predset = extra["y_predset"]
    predset_t = torch.tensor(predset) if isinstance(predset, np.ndarray) else predset
    avg_set_size = predset_t.float().sum(dim=1).mean().item()

    miscoverage = conf_metrics["miscoverage_ps"]
    if isinstance(miscoverage, np.ndarray):
        miscoverage = float(miscoverage.item() if miscoverage.size == 1 else miscoverage.mean())
    else:
        miscoverage = float(miscoverage)

    return {
        "accuracy":    float(base_metrics["accuracy"]),
        "f1_weighted": float(base_metrics["f1_weighted"]),
        "coverage":    1.0 - miscoverage,
        "miscoverage": miscoverage,
        "avg_set_size": avg_set_size,
    }


def _print_single_run_results(metrics: dict, alpha: float) -> None:
    print("\nCovariateLabel Results:")
    print(f"  Accuracy:              {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted):         {metrics['f1_weighted']:.4f}")
    print(f"  Empirical coverage:    {metrics['coverage']:.4f}")
    print(f"  Empirical miscoverage: {metrics['miscoverage']:.4f}")
    print(f"  Average set size:      {metrics['avg_set_size']:.2f}")
    print(f"  Target coverage:       {1 - alpha:.0%} (alpha={alpha})")


def _print_multi_seed_summary(
    all_metrics: list, run_seeds: list, alpha: float, n_test: int
) -> None:
    accs       = np.array([m["accuracy"]     for m in all_metrics])
    f1s        = np.array([m["f1_weighted"]  for m in all_metrics])
    coverages  = np.array([m["coverage"]     for m in all_metrics])
    miscovs    = np.array([m["miscoverage"]  for m in all_metrics])
    set_sizes  = np.array([m["avg_set_size"] for m in all_metrics])
    n_runs = len(all_metrics)

    print("\n" + "=" * 80)
    print("Per-run CovariateLabel results (fixed test set = TUH eval partition)")
    print("=" * 80)
    print(f"  {'Run':<4} {'Seed':<6} {'Accuracy':<10} {'F1-Wt':<10} "
          f"{'Coverage':<10} {'Miscoverage':<12} {'Avg set size':<12}")
    print("  " + "-" * 68)
    for i in range(n_runs):
        m = all_metrics[i]
        print(f"  {i+1:<4} {run_seeds[i]:<6} {m['accuracy']:<10.4f} "
              f"{m['f1_weighted']:<10.4f} {m['coverage']:<10.4f} "
              f"{m['miscoverage']:<12.4f} {m['avg_set_size']:<12.2f}")

    print("\n" + "=" * 80)
    print(f"CovariateLabel summary (mean \u00b1 std over {n_runs} runs, fixed test set)")
    print("=" * 80)
    print(f"  Accuracy:              {accs.mean():.4f} \u00b1 {accs.std():.4f}")
    print(f"  F1 (weighted):         {f1s.mean():.4f} \u00b1 {f1s.std():.4f}")
    print(f"  Empirical coverage:    {coverages.mean():.4f} \u00b1 {coverages.std():.4f}")
    print(f"  Empirical miscoverage: {miscovs.mean():.4f} \u00b1 {miscovs.std():.4f}")
    print(f"  Average set size:      {set_sizes.mean():.2f} \u00b1 {set_sizes.std():.2f}")
    print(f"  Target coverage:       {1 - alpha:.0%} (alpha={alpha})")
    print(f"  Test set size:         {n_test} (fixed across runs)")
    print(f"  Run seeds:             {run_seeds}")
    print("\n--- Min / Max (across runs) ---")
    print(f"  Coverage:    [{coverages.min():.4f}, {coverages.max():.4f}]")
    print(f"  Set size:    [{set_sizes.min():.2f}, {set_sizes.max():.2f}]")
    print(f"  Accuracy:    [{accs.min():.4f}, {accs.max():.4f}]")


def _main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"TUEV root not found: {root}. "
            "Pass --root to point to your downloaded TUEV edf/ directory."
        )

    epochs = 2 if args.quick_test else args.epochs
    quick_test_max_samples = 2000
    if args.quick_test:
        print("*** QUICK TEST MODE (dev=True, 2 epochs, max 2000 samples) ***")

    # -------------------------------------------------------------------------
    # STEP 1: Load dataset ONCE — shared across all seeds
    # -------------------------------------------------------------------------
    print("=" * 80)
    print("STEP 1: Load TUEV + build task dataset (shared across all seeds)")
    print("=" * 80)
    dataset = TUEVDataset(root=str(root), subset=args.subset, dev=args.quick_test)
    sample_dataset = dataset.set_task(EEGEventsTUEV())
    if args.quick_test and len(sample_dataset) > quick_test_max_samples:
        sample_dataset = sample_dataset.subset(range(quick_test_max_samples))
        print(f"Capped to {quick_test_max_samples} samples for quick-test.")
    print(f"Task samples:  {len(sample_dataset)}")
    print(f"Input schema:  {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")
    if len(sample_dataset) == 0:
        raise RuntimeError("No samples produced. Verify TUEV root/subset/task.")

    # -------------------------------------------------------------------------
    # STEP 2: Extract the fixed test set ONCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Extract fixed test set (TUH eval partition — same for all seeds)")
    print("=" * 80)
    _, _, _, test_ds = split_by_sample_conformal_tuh(
        dataset=sample_dataset, ratios=list(args.ratios), seed=args.seed
    )
    if len(test_ds) == 0 and args.quick_test:
        print("  [quick-test] TUH eval partition empty in dev mode — using random 20% as test set.")
        _, _, _, test_ds = split_by_sample_conformal(
            dataset=sample_dataset, ratios=[0.6, 0.1, 0.1, 0.2], seed=args.seed
        )
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test: {len(test_ds)} (fixed)")

    # -------------------------------------------------------------------------
    # Determine run seeds
    # -------------------------------------------------------------------------
    if args.seeds is not None:
        run_seeds = [int(s.strip()) for s in args.seeds.split(",")]
    else:
        run_seeds = [args.seed + i for i in range(args.n_seeds)]

    use_multi_seed = len(run_seeds) > 1
    print(f"\nRun config: {'multi-seed (' + str(len(run_seeds)) + ' runs)' if use_multi_seed else 'single run'}")
    print(f"Seeds: {run_seeds}, alpha={args.alpha}, target coverage={1 - args.alpha:.0%}")

    # -------------------------------------------------------------------------
    # STEP 3+: Train + conformal (once per seed)
    # -------------------------------------------------------------------------
    all_metrics = []
    for run_i, run_seed in enumerate(run_seeds):
        print("\n" + "=" * 80)
        if use_multi_seed:
            print(f"Run {run_i + 1} / {len(run_seeds)}  (seed={run_seed})")
        else:
            print(f"STEP 3–4: Train ContraWR + Conformal Calibration  (seed={run_seed})")
        print("=" * 80)

        metrics = _run_one_seed(
            args, sample_dataset, test_ds, test_loader, device, epochs, run_seed
        )
        all_metrics.append(metrics)

        if use_multi_seed:
            print(f"  [Run {run_i + 1} result] "
                  f"acc={metrics['accuracy']:.4f}, f1={metrics['f1_weighted']:.4f}, "
                  f"cov={metrics['coverage']:.4f}, set_size={metrics['avg_set_size']:.2f}")

    if not use_multi_seed:
        _print_single_run_results(all_metrics[0], args.alpha)
    else:
        _print_multi_seed_summary(all_metrics, run_seeds, args.alpha, len(test_ds))


def main() -> None:
    args = parse_args()

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(orig_stdout, log_file)
        sys.stderr = _Tee(orig_stderr, log_file)

    try:
        _main(args)
    finally:
        if log_file is not None:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


if __name__ == "__main__":
    main()
