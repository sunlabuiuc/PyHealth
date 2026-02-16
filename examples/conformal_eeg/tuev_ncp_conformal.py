"""Neighborhood Conformal Prediction (NCP) on TUEV EEG Events using ContraWR.

This script:
1) Loads the TUEV dataset and applies the EEGEventsTUEV task.
2) Splits into train/val/cal/test using split conformal protocol.
3) Trains a ContraWR model.
4) Extracts calibration embeddings and calibrates a NeighborhoodLabel (NCP) predictor.
5) Evaluates prediction-set coverage/miscoverage and efficiency on the test split.

With --n-seeds > 1: fixes the test set (--split-seed), runs multiple training runs
with different seeds (different train/val/cal splits and model init), reports
coverage / set size / accuracy as mean ± std (error bars).

Example (from repo root):
  python examples/conformal_eeg/tuev_ncp_conformal.py --root /srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf
  python examples/conformal_eeg/tuev_ncp_conformal.py --quick-test --log-file quicktest_ncp.log
  python examples/conformal_eeg/tuev_ncp_conformal.py --alpha 0.1 --n-seeds 5 --split-seed 0 --log-file ncp_seeds5.log
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

_script_dir = Path(__file__).resolve().parent
if str(_script_dir) not in sys.path:
    sys.path.insert(0, str(_script_dir))

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
from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_sample_conformal
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer, get_metrics_fn

from model_utils import AddSTFTDataset, get_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Neighborhood conformal prediction (NCP) on TUEV EEG events using ContraWR."
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf",
        help="Path to TUEV edf/ folder.",
    )
    parser.add_argument("--subset", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--seed", type=int, default=42, help="Run seed (or first of run seeds when n-seeds > 1).")
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=1,
        help="Number of runs for mean±std. Test set fixed; train/val/cal vary by seed.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=0,
        help="Fixed seed for initial split (fixes test set when n-seeds > 1).",
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
        nargs=4,
        default=(0.6, 0.1, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "CAL", "TEST"),
        help="Split ratios for train/val/cal/test. Must sum to 1.0.",
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
    parser.add_argument("--n-fft", type=int, default=128, help="STFT FFT size (ContraWR and TFM-Tokenizer).")
    parser.add_argument("--model", type=str, default="contrawr", choices=["contrawr", "tfm"], help="Backbone: contrawr or tfm (TFM-Tokenizer).")
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


def _split_remainder_into_train_val_cal(sample_dataset, remainder_indices, ratios, run_seed):
    """Split remainder indices into train/val/cal by renormalized ratios. Uses run_seed for shuffle."""
    r0, r1, r2, r3 = ratios
    remainder_frac = 1.0 - r3
    if remainder_frac <= 0:
        raise ValueError("Test ratio must be < 1 so remainder (train+val+cal) is non-empty.")
    # Renormalize so train/val/cal ratios sum to 1 on the remainder
    r_train = r0 / remainder_frac
    r_val = r1 / remainder_frac
    remainder = np.asarray(remainder_indices, dtype=np.int64)
    np.random.seed(run_seed)
    shuffled = np.random.permutation(remainder)
    M = len(shuffled)
    train_end = int(M * r_train)
    val_end = int(M * (r_train + r_val))
    train_index = shuffled[:train_end]
    val_index = shuffled[train_end:val_end]
    cal_index = shuffled[val_end:]
    train_ds = sample_dataset.subset(train_index.tolist())
    val_ds = sample_dataset.subset(val_index.tolist())
    cal_ds = sample_dataset.subset(cal_index.tolist())
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
    model_name = "TFM-Tokenizer" if args.model.lower() == "tfm" else "ContraWR"
    print(f"STEP 3: Train {model_name}")
    print("=" * 80)
    model = get_model(args, sample_dataset, device)
    trainer = Trainer(model=model, device=device, enable_logging=False)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=epochs,
        monitor="accuracy" if val_loader is not None else None,
    )

    if not return_metrics:
        print("\nBase model performance on test set:")
        y_true_base, y_prob_base, _loss_base = trainer.inference(test_loader)
        base_metrics = get_metrics_fn("multiclass")(
            y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"]
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
            "accuracy": float(ncp_metrics["accuracy"]),
            "coverage": coverage,
            "miscoverage": miscoverage,
            "avg_set_size": avg_set_size,
        }

    print("\nNCP (NeighborhoodLabel) Results:")
    print(f"  Accuracy: {ncp_metrics['accuracy']:.4f}")
    print(f"  Empirical miscoverage: {miscoverage:.4f}")
    print(f"  Empirical coverage: {coverage:.4f}")
    print(f"  Average set size: {avg_set_size:.2f}")
    print(f"  k_neighbors: {args.k_neighbors}")
    print("\n--- Single-run summary (for reporting) ---")
    print(f"  alpha={args.alpha}, target_coverage={1 - args.alpha:.2f}, empirical_coverage={coverage:.4f}, miscoverage={miscoverage:.4f}, accuracy={ncp_metrics['accuracy']:.4f}, avg_set_size={avg_set_size:.2f}")


def main() -> None:
    args = parse_args()
    # Seed set per run in multi-seed mode; for single run set once here
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
            f"TUEV root not found: {root}. "
            "Pass --root to point to your downloaded TUEV edf/ directory."
        )

    epochs = 2 if args.quick_test else args.epochs
    quick_test_max_samples = 2000  # cap samples so quick-test finishes in ~5-10 min
    if args.quick_test:
        print("*** QUICK TEST MODE (dev=True, 2 epochs, max 2000 samples) ***")

    print("=" * 80)
    print("STEP 1: Load TUEV + build task dataset")
    print("=" * 80)
    dataset = TUEVDataset(root=str(root), subset=args.subset, dev=args.quick_test)
    sample_dataset = dataset.set_task(EEGEventsTUEV(), cache_dir="examples/conformal_eeg/cache")
    if args.quick_test and len(sample_dataset) > quick_test_max_samples:
        sample_dataset = sample_dataset.subset(range(quick_test_max_samples))
        print(f"Capped to {quick_test_max_samples} samples for quick-test.")
    if args.model.lower() == "tfm":
        # TFM tokenizer needs n_fft=200, hop_length=100 so STFT time = temporal patches
        sample_dataset = AddSTFTDataset(
            sample_dataset, n_fft=200, hop_length=100
        )
        print("Wrapped dataset with STFT for TFM-Tokenizer.")

    print(f"Task samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    # Experiment configuration (for PI / reporting)
    print("\n--- Experiment configuration ---")
    print(f"  dataset_root: {root}")
    print(f"  subset: {args.subset}, ratios: train/val/cal/test = {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f}/{args.ratios[3]:.2f}")
    print(f"  alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")
    print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
    print(f"  epochs: {epochs}, batch_size: {args.batch_size}, device: {device}, seed: {args.seed}")

    if len(sample_dataset) == 0:
        raise RuntimeError("No samples produced. Verify TUEV root/subset/task.")

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
        # Single run: original behavior
        print("\n" + "=" * 80)
        print("STEP 2: Split train/val/cal/test")
        print("=" * 80)
        train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(
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

    # Multi-seed: fix test set, vary train/val/cal per run
    print("\n" + "=" * 80)
    print("STEP 2: Fix test set (split-seed), then run multiple train/cal splits")
    print("=" * 80)
    train_idx, val_idx, cal_idx, test_idx = split_by_sample_conformal(
        dataset=sample_dataset, ratios=ratios, seed=args.split_seed, get_index=True
    )
    # Convert to numpy for indexing
    train_index = train_idx.numpy() if hasattr(train_idx, "numpy") else np.array(train_idx)
    val_index = val_idx.numpy() if hasattr(val_idx, "numpy") else np.array(val_idx)
    cal_index = cal_idx.numpy() if hasattr(cal_idx, "numpy") else np.array(cal_idx)
    test_index = test_idx.numpy() if hasattr(test_idx, "numpy") else np.array(test_idx)
    remainder_indices = np.concatenate([train_index, val_index, cal_index])
    test_ds = sample_dataset.subset(test_index.tolist())
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_test = len(test_ds)
    print(f"Fixed test set size: {n_test}")

    accs, coverages, miscoverages, set_sizes = [], [], [], []
    for run_i, run_seed in enumerate(run_seeds):
        print("\n" + "=" * 80)
        print(f"Run {run_i + 1} / {n_runs} (seed={run_seed})")
        print("=" * 80)
        set_seed(run_seed)
        train_ds, val_ds, cal_ds = _split_remainder_into_train_val_cal(
            sample_dataset, remainder_indices, ratios, run_seed
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
        coverages.append(metrics["coverage"])
        miscoverages.append(metrics["miscoverage"])
        set_sizes.append(metrics["avg_set_size"])

    accs = np.array(accs)
    coverages = np.array(coverages)
    miscoverages_arr = np.array(miscoverages)
    set_sizes = np.array(set_sizes)

    # Per-run table (for PI / reporting)
    print("\n" + "=" * 80)
    print("Per-run NCP results (fixed test set)")
    print("=" * 80)
    print(f"  {'Run':<4} {'Seed':<6} {'Accuracy':<10} {'Coverage':<10} {'Miscoverage':<12} {'Avg set size':<12}")
    print("  " + "-" * 54)
    for i in range(n_runs):
        print(f"  {i+1:<4} {run_seeds[i]:<6} {accs[i]:<10.4f} {coverages[i]:<10.4f} {miscoverages_arr[i]:<12.4f} {set_sizes[i]:<12.2f}")

    print("\n" + "=" * 80)
    print("NCP summary (mean ± std over {} runs, fixed test set)".format(n_runs))
    print("=" * 80)
    print(f"  Accuracy:           {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Empirical coverage: {coverages.mean():.4f} ± {coverages.std():.4f}")
    print(f"  Empirical miscoverage: {miscoverages_arr.mean():.4f} ± {miscoverages_arr.std():.4f}")
    print(f"  Average set size:  {set_sizes.mean():.2f} ± {set_sizes.std():.2f}")
    print(f"  Target coverage:  {1 - args.alpha:.0%} (alpha={args.alpha})")
    print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
    print(f"  Test set size: {n_test} (fixed across runs)")
    print(f"  Run seeds: {run_seeds}")
    print("\n--- Min / Max (across runs) ---")
    print(f"  Coverage:    [{coverages.min():.4f}, {coverages.max():.4f}]")
    print(f"  Set size:    [{set_sizes.min():.2f}, {set_sizes.max():.2f}]")
    print(f"  Accuracy:   [{accs.min():.4f}, {accs.max():.4f}]")


if __name__ == "__main__":
    main()
