"""Neighborhood Conformal Prediction (NCP) on TUAB Abnormal EEG Detection using ContraWR.

This script:
1) Loads the TUAB dataset and applies the EEGAbnormalTUAB task.
2) Splits into train/val/cal/test using the TUH-aware split conformal protocol.
3) Trains a ContraWR model.
4) Extracts calibration embeddings and calibrates a NeighborhoodLabel (NCP) predictor.
5) Evaluates prediction-set coverage/miscoverage and efficiency on the test split.

With --n-seeds > 1: fixes the test set (TUH eval partition), runs multiple training runs
with different seeds (different train/val/cal splits and model init), reports
coverage / set size / accuracy as mean ± std (error bars).

Example (from repo root):
  python examples/conformal_eeg/tuab_ncp_conformal.py --root /srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf
  python examples/conformal_eeg/tuab_ncp_conformal.py --quick-test --log-file quicktest_ncp.log
  python examples/conformal_eeg/tuab_ncp_conformal.py --alpha 0.1 --n-seeds 5 --split-seed 0 --log-file ncp_seeds5.log
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch


class _Tee:
    """Writes to both a stream and a file."""

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


from pyhealth.calib.predictionset.cluster import NeighborhoodLabel
from pyhealth.calib.utils import extract_embeddings
from pyhealth.datasets import TUABDataset, get_dataloader, split_by_sample_conformal_tuh, split_by_sample_conformal
from pyhealth.models import ContraWR, TFMTokenizer
from pyhealth.tasks import EEGAbnormalTUAB
from pyhealth.trainer import Trainer, get_metrics_fn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neighborhood conformal prediction (NCP) on TUAB abnormal EEG detection using ContraWR."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf",
        help="Path to TUAB edf/ folder.",
    )
    parser.add_argument("--subset", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--seed", type=int, default=42, help="Run seed (or first of run seeds when n-seeds > 1).")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of runs for mean±std. Test set fixed (TUH eval partition); train/val/cal vary by seed.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Seed used to obtain the fixed test set in multi-seed mode. Since test = TUH eval partition, this only affects the initial train-pool shuffle (not which samples are in test).",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Comma-separated run seeds, e.g. 42,43,44,45,46. Overrides --seed and --n-seeds.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (e.g., 0.1 => 90% target coverage).")
    parser.add_argument(
        "--ratios",
        type=float,
        nargs=3,
        default=(0.6, 0.2, 0.2),
        metavar=("TRAIN", "VAL", "CAL"),
        help="Ratios for splitting the TUH train partition into train/val/cal. Must sum to 1.0. Test is fixed as the TUH eval partition.",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        default=50,
        help="Number of nearest calibration neighbors for NCP.",
    )
    parser.add_argument(
        "--lambda-L",
        type=float,
        default=100.0,
        help="Temperature for NCP exponential weights; smaller => more localization.",
    )
    parser.add_argument("--n-fft", type=int, default=128, help="STFT FFT size used by ContraWR.")
    parser.add_argument(
        "--model", type=str, default="contrawr", choices=["contrawr", "tfm"],
        help="Backbone model: 'contrawr' (default) or 'tfm' (TFMTokenizer).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string, e.g. 'cuda:0' or 'cpu'. Defaults to auto-detect.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file. Stdout and stderr are teed to this file.",
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Smoke test: dev=True, max 2000 samples, 2 epochs, ~5-10 min.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_pool_for_run(sample_dataset, ratios, run_seed):
    """Re-split the TUH train partition into train/val/cal for one run seed.

    The test set (TUH eval partition) is always fixed regardless of seed, so
    only train/val/cal change across runs in multi-seed mode.
    """
    train_ds, val_ds, cal_ds, _ = split_by_sample_conformal_tuh(
        sample_dataset, ratios=ratios, seed=run_seed
    )
    return train_ds, val_ds, cal_ds


def _run_one_ncp(
    sample_dataset,
    train_ds,
    val_ds,
    cal_ds,
    test_loader,
    args,
    device,
    epochs,
    return_metrics=False,
):
    """Train ContraWR, calibrate NCP, evaluate on test. Optionally return metrics dict for aggregation."""
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None

    print("\n" + "=" * 80)
    model_name = "TFMTokenizer" if args.model == "tfm" else "ContraWR"
    print(f"STEP 3: Train {model_name}")
    print("=" * 80)
    if args.model == "tfm":
        model = TFMTokenizer(dataset=sample_dataset).to(device)
    else:
        model = ContraWR(dataset=sample_dataset, n_fft=args.n_fft).to(device)
    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="roc_auc_weighted_ovr" if val_loader is not None else None,
    )

    print("\nBase model performance on test set:")
    y_true_base, y_prob_base, _loss_base = trainer.inference(test_loader)
    base_metrics = get_metrics_fn("multiclass")(
        y_true_base, y_prob_base, metrics=["accuracy", "roc_auc_weighted_ovr", "f1_weighted"]
    )
    for metric, value in base_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("STEP 4: Neighborhood Conformal Prediction (NCP / NeighborhoodLabel)")
    print("=" * 80)
    print(f"Target miscoverage alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")
    print(f"k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")

    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=args.batch_size, device=device)
    if not return_metrics:
        print(f"  cal_embeddings shape: {cal_embeddings.shape}")

    ncp_predictor = NeighborhoodLabel(
        model=model,
        alpha=float(args.alpha),
        k_neighbors=args.k_neighbors,
        lambda_L=args.lambda_L,
    )
    ncp_predictor.calibrate(cal_dataset=cal_ds, cal_embeddings=cal_embeddings)

    y_true, y_prob, _loss, extra = Trainer(model=ncp_predictor).inference(
        test_loader, additional_outputs=["y_predset"]
    )
    ncp_metrics = get_metrics_fn("multiclass")(
        y_true, y_prob, metrics=["accuracy", "miscoverage_ps"], y_predset=extra["y_predset"]
    )
    predset = extra["y_predset"]
    if isinstance(predset, np.ndarray):
        predset_t = torch.tensor(predset)
    else:
        predset_t = predset
    avg_set_size = predset_t.float().sum(dim=1).mean().item()
    miscoverage = ncp_metrics["miscoverage_ps"]
    if isinstance(miscoverage, np.ndarray):
        miscoverage = float(miscoverage.item() if miscoverage.size == 1 else miscoverage.mean())
    else:
        miscoverage = float(miscoverage)
    coverage = 1.0 - miscoverage

    if return_metrics:
        return {
            "accuracy":    float(base_metrics["accuracy"]),
            "roc_auc_weighted_ovr":     float(base_metrics["roc_auc_weighted_ovr"]),
            "f1_weighted":          float(base_metrics["f1_weighted"]),
            "coverage":    coverage,
            "miscoverage": miscoverage,
            "avg_set_size": avg_set_size,
        }

    print("\nNCP (NeighborhoodLabel) Results:")
    print(f"  Accuracy:              {base_metrics['accuracy']:.4f}")
    print(f"  ROC-AUC:               {base_metrics['roc_auc_weighted_ovr']:.4f}")
    print(f"  F1:                    {base_metrics['f1_weighted']:.4f}")
    print(f"  Empirical miscoverage: {miscoverage:.4f}")
    print(f"  Empirical coverage:    {coverage:.4f}")
    print(f"  Average set size:      {avg_set_size:.2f}")
    print(f"  k_neighbors: {args.k_neighbors}")


def main() -> None:
    args = parse_args()
    if args.n_seeds <= 1 and args.seeds is None:
        set_seed(args.seed)

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_file = None
    if args.log_file:
        log_file = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(orig_stdout, log_file)
        sys.stderr = _Tee(orig_stderr, log_file)

    try:
        _run(args)
    finally:
        if log_file is not None:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            log_file.close()


def _run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    root = Path(args.root)
    if not root.exists():
        raise FileNotFoundError(
            f"TUAB root not found: {root}. "
            "Pass --root to point to your downloaded TUAB edf/ directory."
        )

    epochs = 2 if args.quick_test else args.epochs
    quick_test_max_samples = 2000
    if args.quick_test:
        print("*** QUICK TEST MODE (dev=True, 2 epochs, max 2000 samples) ***")

    print("=" * 80)
    print("STEP 1: Load TUAB + build task dataset")
    print("=" * 80)
    dataset = TUABDataset(root=str(root), subset=args.subset, dev=args.quick_test)
    sample_dataset = dataset.set_task(EEGAbnormalTUAB())
    if args.quick_test and len(sample_dataset) > quick_test_max_samples:
        sample_dataset = sample_dataset.subset(range(quick_test_max_samples))
        print(f"Capped to {quick_test_max_samples} samples for quick-test.")

    print(f"Task samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    print("\n--- Experiment configuration ---")
    print(f"  dataset_root: {root}")
    print(f"  subset: {args.subset}, ratios: train/val/cal = {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f} (test = TUH eval partition)")
    print(f"  alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")
    print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
    print(f"  epochs: {epochs}, batch_size: {args.batch_size}, device: {device}, seed: {args.seed}")

    if len(sample_dataset) == 0:
        raise RuntimeError("No samples produced. Verify TUAB root/subset/task.")

    ratios = list(args.ratios)
    use_multi_seed = args.n_seeds > 1 or args.seeds is not None
    if use_multi_seed:
        run_seeds = (
            [int(s.strip()) for s in args.seeds.split(",")]
            if args.seeds
            else [args.seed + i for i in range(args.n_seeds)]
        )
        n_runs = len(run_seeds)
        print(f"  multi_seed: n_runs={n_runs}, run_seeds={run_seeds}, split_seed={args.split_seed} (fixed test set)")
        print(f"Multi-seed mode: {n_runs} runs (fixed test set), run seeds: {run_seeds}")

    if not use_multi_seed:
        print("\n" + "=" * 80)
        print("STEP 2: Split train/val/cal/test")
        print("=" * 80)
        train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal_tuh(
            dataset=sample_dataset, ratios=ratios, seed=args.seed
        )
        print(f"Train: {len(train_ds)}")
        print(f"Val:   {len(val_ds)}")
        print(f"Cal:   {len(cal_ds)}")
        print(f"Test:  {len(test_ds)}")

        test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
        _run_one_ncp(
            sample_dataset=sample_dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            cal_ds=cal_ds,
            test_loader=test_loader,
            args=args,
            device=device,
            epochs=epochs,
        )
        print("\n--- Split sizes and seed (for reporting) ---")
        print(f"  train={len(train_ds)}, val={len(val_ds)}, cal={len(cal_ds)}, test={len(test_ds)}, seed={args.seed}")
        return

    # Multi-seed: test set is always fixed as the TUH eval partition (no split-seed needed).
    # Each run uses a different seed to re-shuffle the train pool into train/val/cal.
    print("\n" + "=" * 80)
    print("STEP 2: Fix test set (TUH eval partition), then run multiple train/cal splits")
    print("=" * 80)
    _, _, _, test_ds = split_by_sample_conformal_tuh(
        dataset=sample_dataset, ratios=ratios, seed=args.split_seed
    )
    if len(test_ds) == 0 and args.quick_test:
        print("  [quick-test] TUH eval partition empty in dev mode — using random 20% as test set.")
        _, _, _, test_ds = split_by_sample_conformal(
            dataset=sample_dataset, ratios=[0.6, 0.1, 0.1, 0.2], seed=args.split_seed
        )
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_test = len(test_ds)
    print(f"Fixed test set size: {n_test} (TUH eval partition)")

    accs, roc_aucs, f1s, coverages, miscoverages, set_sizes = [], [], [], [], [], []
    for run_i, run_seed in enumerate(run_seeds):
        print("\n" + "=" * 80)
        print(f"Run {run_i + 1} / {n_runs} (seed={run_seed})")
        print("=" * 80)
        set_seed(run_seed)
        train_ds, val_ds, cal_ds = _split_train_pool_for_run(
            sample_dataset, ratios, run_seed
        )
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Cal: {len(cal_ds)}")

        metrics = _run_one_ncp(
            sample_dataset=sample_dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            cal_ds=cal_ds,
            test_loader=test_loader,
            args=args,
            device=device,
            epochs=epochs,
            return_metrics=True,
        )
        accs.append(metrics["accuracy"])
        roc_aucs.append(metrics["roc_auc_weighted_ovr"])
        f1s.append(metrics["f1_weighted"])
        coverages.append(metrics["coverage"])
        miscoverages.append(metrics["miscoverage"])
        set_sizes.append(metrics["avg_set_size"])

    accs          = np.array(accs)
    roc_aucs      = np.array(roc_aucs)
    f1s           = np.array(f1s)
    coverages     = np.array(coverages)
    miscoverages_arr = np.array(miscoverages)
    set_sizes     = np.array(set_sizes)

    print("\n" + "=" * 80)
    print("Per-run NCP results (fixed test set = TUH eval partition)")
    print("=" * 80)
    print(f"  {'Run':<4} {'Seed':<6} {'Accuracy':<10} {'ROC-AUC':<10} {'F1':<8} "
          f"{'Coverage':<10} {'Miscoverage':<12} {'Avg set size':<12}")
    print("  " + "-" * 76)
    for i in range(n_runs):
        print(f"  {i+1:<4} {run_seeds[i]:<6} {accs[i]:<10.4f} {roc_aucs[i]:<10.4f} "
              f"{f1s[i]:<8.4f} {coverages[i]:<10.4f} {miscoverages_arr[i]:<12.4f} {set_sizes[i]:<12.2f}")

    print("\n" + "=" * 80)
    print("NCP summary (mean \u00b1 std over {} runs, fixed test set)".format(n_runs))
    print("=" * 80)
    print(f"  Accuracy:              {accs.mean():.4f} \u00b1 {accs.std():.4f}")
    print(f"  ROC-AUC:               {roc_aucs.mean():.4f} \u00b1 {roc_aucs.std():.4f}")
    print(f"  F1:                    {f1s.mean():.4f} \u00b1 {f1s.std():.4f}")
    print(f"  Empirical coverage:    {coverages.mean():.4f} \u00b1 {coverages.std():.4f}")
    print(f"  Empirical miscoverage: {miscoverages_arr.mean():.4f} \u00b1 {miscoverages_arr.std():.4f}")
    print(f"  Average set size:      {set_sizes.mean():.2f} \u00b1 {set_sizes.std():.2f}")
    print(f"  Target coverage:       {1 - args.alpha:.0%} (alpha={args.alpha})")
    print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
    print(f"  Test set size:         {n_test} (fixed across runs)")
    print(f"  Run seeds:             {run_seeds}")
    print("\n--- Min / Max (across runs) ---")
    print(f"  Coverage:    [{coverages.min():.4f}, {coverages.max():.4f}]")
    print(f"  Set size:    [{set_sizes.min():.2f}, {set_sizes.max():.2f}]")
    print(f"  Accuracy:    [{accs.min():.4f}, {accs.max():.4f}]")
    print(f"  ROC-AUC:     [{roc_aucs.min():.4f}, {roc_aucs.max():.4f}]")


if __name__ == "__main__":
    main()
