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
import os
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
from pyhealth.datasets import TUEVDataset, get_dataloader, split_by_patient_conformal_tuh, split_by_sample_conformal_tuh, split_by_sample_conformal
from pyhealth.models import ContraWR, TFMTokenizer
from pyhealth.tasks import EEGEventsTUEV
from pyhealth.trainer import Trainer, get_metrics_fn


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
        "--alphas", type=str, default=None,
        help="Comma-separated miscoverage rates, e.g. '0.2,0.1,0.05,0.01'. Overrides --alpha.",
    )
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
    parser.add_argument(
        "--weights-dir",
        type=str,
        default="weightfiles/TFM_Tokenizer_multiple_finetuned_on_TUEV",
        help="Root folder of fine-tuned TFM classifier checkpoints (only with --model tfm).",
    )
    parser.add_argument(
        "--tokenizer-weights",
        type=str,
        default="weightfiles/tfm_tokenizer_last.pth",
        help="Path to the pre-trained TFM tokenizer weights (only with --model tfm).",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default="patient",
        choices=["patient", "sample"],
        help="Split strategy: 'patient' (default, patient-level, no leakage) or "
             "'sample' (original sample-level, for comparison).",
    )
    return parser.parse_args()


def _do_split(dataset, ratios, seed, split_type):
    """Dispatch to the correct TUH split function based on split_type."""
    if split_type == "patient":
        return split_by_patient_conformal_tuh(dataset=dataset, ratios=list(ratios), seed=seed)
    else:
        return split_by_sample_conformal_tuh(dataset=dataset, ratios=list(ratios), seed=seed)


def _load_tfm_weights(model, args, run_idx: int) -> None:
    """Load pre-trained tokenizer + fine-tuned classifier for run_idx (0-based)."""
    base = os.path.basename(args.weights_dir)
    classifier_path = os.path.join(args.weights_dir, f"{base}_{run_idx + 1}", "best_model.pth")
    print(f"  Loading TFM weights (run {run_idx + 1}): {classifier_path}")
    model.load_pretrained_weights(
        tokenizer_checkpoint_path=args.tokenizer_weights,
        classifier_checkpoint_path=classifier_path,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_train_pool_for_run(sample_dataset, ratios, run_seed, split_type="patient"):
    """Re-split the TUH train partition into train/val/cal for one run seed.

    The test set (TUH eval partition) is always fixed regardless of seed, so
    only train/val/cal change across runs in multi-seed mode.
    """
    train_ds, val_ds, cal_ds, _ = _do_split(sample_dataset, ratios, run_seed, split_type)
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
    alphas: list,
    run_idx: int = 0,
):
    """Train model + calibrate NCP for one seed across all alphas.

    Training, embedding extraction, and base inference are done once; calibration
    loops over alphas (fast — only threshold recomputed per alpha).

    Returns {alpha: metrics_dict} where metrics_dict has keys:
        accuracy, f1_weighted, coverage, miscoverage, avg_set_size
    """
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None

    if args.model == "tfm":
        model = TFMTokenizer(dataset=sample_dataset).to(device)
        _load_tfm_weights(model, args, run_idx)
    else:
        model = ContraWR(dataset=sample_dataset, n_fft=args.n_fft).to(device)
        print("  Training ContraWR...")
        trainer_tmp = Trainer(model=model, device=device, enable_logging=False)
        trainer_tmp.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="accuracy" if val_loader is not None else None,
        )
    trainer = Trainer(model=model, device=device, enable_logging=False)

    # Base model metrics — computed once, shared across all alphas
    y_true_base, y_prob_base, _ = trainer.inference(test_loader)
    base_metrics = get_metrics_fn("multiclass")(
        y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"]
    )

    # Extract calibration embeddings once — reused for every alpha
    print("  Extracting calibration embeddings...")
    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=args.batch_size, device=device)

    # Calibration + evaluation — fast; loop over every alpha
    results = {}
    for alpha in alphas:
        print(f"  Calibrating NCP predictor (alpha={alpha})...")
        ncp_predictor = NeighborhoodLabel(
            model=model,
            alpha=float(alpha),
            k_neighbors=args.k_neighbors,
            lambda_L=args.lambda_L,
        )
        ncp_predictor.calibrate(cal_dataset=cal_ds, cal_embeddings=cal_embeddings)

        y_true, y_prob, _, extra = Trainer(model=ncp_predictor).inference(
            test_loader, additional_outputs=["y_predset"]
        )
        ncp_metrics = get_metrics_fn("multiclass")(
            y_true, y_prob, metrics=["accuracy", "miscoverage_ps"], y_predset=extra["y_predset"]
        )
        predset = extra["y_predset"]
        predset_t = torch.tensor(predset) if isinstance(predset, np.ndarray) else predset
        avg_set_size = predset_t.float().sum(dim=1).mean().item()

        miscoverage = ncp_metrics["miscoverage_ps"]
        if isinstance(miscoverage, np.ndarray):
            miscoverage = float(miscoverage.item() if miscoverage.size == 1 else miscoverage.mean())
        else:
            miscoverage = float(miscoverage)

        results[alpha] = {
            "accuracy":    float(base_metrics["accuracy"]),
            "f1_weighted": float(base_metrics["f1_weighted"]),
            "coverage":    1.0 - miscoverage,
            "miscoverage": miscoverage,
            "avg_set_size": avg_set_size,
        }
    return results


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
    sample_dataset = dataset.set_task(EEGEventsTUEV())
    if args.quick_test and len(sample_dataset) > quick_test_max_samples:
        sample_dataset = sample_dataset.subset(range(quick_test_max_samples))
        print(f"Capped to {quick_test_max_samples} samples for quick-test.")

    print(f"Task samples: {len(sample_dataset)}")
    print(f"Input schema: {sample_dataset.input_schema}")
    print(f"Output schema: {sample_dataset.output_schema}")

    if len(sample_dataset) == 0:
        raise RuntimeError("No samples produced. Verify TUEV root/subset/task.")

    # Parse alphas and run seeds
    alphas = [float(a.strip()) for a in args.alphas.split(",")] if args.alphas else [args.alpha]
    ratios = list(args.ratios)
    use_multi_seed = args.n_seeds > 1 or args.seeds is not None
    run_seeds = (
        [int(s.strip()) for s in args.seeds.split(",")]
        if args.seeds
        else [args.seed + i for i in range(args.n_seeds)]
    )
    n_runs = len(run_seeds)

    # -------------------------------------------------------------------------
    # STEP 2: Extract the fixed test set ONCE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2: Extract fixed test set (TUH eval partition — same for all seeds)")
    print("=" * 80)
    _, _, _, test_ds = _do_split(
        sample_dataset, ratios, args.split_seed, args.split_type
    )
    if len(test_ds) == 0 and args.quick_test:
        print("  [quick-test] TUH eval partition empty in dev mode — using random 20% as test set.")
        _, _, _, test_ds = split_by_sample_conformal(
            dataset=sample_dataset, ratios=[0.6, 0.1, 0.1, 0.2], seed=args.split_seed
        )
    test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
    n_test = len(test_ds)
    print(f"Test: {n_test} (fixed)")

    print(f"\nRun config: {'multi-seed (' + str(n_runs) + ' runs)' if use_multi_seed else 'single run'}")
    print(f"Seeds: {run_seeds}, alphas={alphas}, k_neighbors={args.k_neighbors}")

    # -------------------------------------------------------------------------
    # STEP 3+: Train once per seed; calibrate for every alpha (fast)
    # -------------------------------------------------------------------------
    all_metrics = {alpha: [] for alpha in alphas}
    for run_i, run_seed in enumerate(run_seeds):
        print("\n" + "=" * 80)
        if use_multi_seed:
            print(f"Run {run_i + 1} / {n_runs}  (seed={run_seed})")
        else:
            print(f"STEP 3–4: Train + NCP Calibration  (seed={run_seed})")
        print("=" * 80)
        set_seed(run_seed)
        train_ds, val_ds, cal_ds = _split_train_pool_for_run(sample_dataset, ratios, run_seed, args.split_type)
        print(f"  Split — Train: {len(train_ds)}, Val: {len(val_ds)}, "
              f"Cal: {len(cal_ds)}, Test: {n_test} (fixed)")

        seed_results = _run_one_ncp(
            sample_dataset=sample_dataset,
            train_ds=train_ds,
            val_ds=val_ds,
            cal_ds=cal_ds,
            test_loader=test_loader,
            args=args,
            device=device,
            epochs=epochs,
            alphas=alphas,
            run_idx=run_i,
        )
        for alpha in alphas:
            all_metrics[alpha].append(seed_results[alpha])

        if use_multi_seed:
            m = seed_results[alphas[0]]
            print(f"  [Run {run_i + 1} result (alpha={alphas[0]})] "
                  f"acc={m['accuracy']:.4f}, f1={m['f1_weighted']:.4f}, "
                  f"cov={m['coverage']:.4f}, set_size={m['avg_set_size']:.2f}")

    for alpha in alphas:
        mlist = all_metrics[alpha]
        accs       = np.array([m["accuracy"]     for m in mlist])
        f1s        = np.array([m["f1_weighted"]  for m in mlist])
        coverages  = np.array([m["coverage"]     for m in mlist])
        miscovs    = np.array([m["miscoverage"]  for m in mlist])
        set_sizes  = np.array([m["avg_set_size"] for m in mlist])

        if not use_multi_seed:
            print(f"\nNCP Results (alpha={alpha}):")
            print(f"  Accuracy:              {accs[0]:.4f}")
            print(f"  F1 (weighted):         {f1s[0]:.4f}")
            print(f"  Empirical coverage:    {coverages[0]:.4f}")
            print(f"  Empirical miscoverage: {miscovs[0]:.4f}")
            print(f"  Average set size:      {set_sizes[0]:.2f}")
            print(f"  Target coverage:       {1 - alpha:.0%} (alpha={alpha})")
            print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
        else:
            print("\n" + "=" * 80)
            print(f"Per-run NCP results — alpha={alpha} (target coverage={1-alpha:.0%})")
            print("=" * 80)
            print(f"  {'Run':<4} {'Seed':<6} {'Accuracy':<10} {'F1-Wt':<10} "
                  f"{'Coverage':<10} {'Miscoverage':<12} {'Avg set size':<12}")
            print("  " + "-" * 68)
            for i in range(n_runs):
                print(f"  {i+1:<4} {run_seeds[i]:<6} {accs[i]:<10.4f} {f1s[i]:<10.4f} "
                      f"{coverages[i]:<10.4f} {miscovs[i]:<12.4f} {set_sizes[i]:<12.2f}")

            print("\n" + "=" * 80)
            print(f"NCP summary — alpha={alpha} (mean ± std over {n_runs} runs, fixed test set)")
            print("=" * 80)
            print(f"  Accuracy:              {accs.mean():.4f} ± {accs.std():.4f}")
            print(f"  F1 (weighted):         {f1s.mean():.4f} ± {f1s.std():.4f}")
            print(f"  Empirical coverage:    {coverages.mean():.4f} ± {coverages.std():.4f}")
            print(f"  Empirical miscoverage: {miscovs.mean():.4f} ± {miscovs.std():.4f}")
            print(f"  Average set size:      {set_sizes.mean():.2f} ± {set_sizes.std():.2f}")
            print(f"  Target coverage:       {1 - alpha:.0%} (alpha={alpha})")
            print(f"  k_neighbors: {args.k_neighbors}, lambda_L: {args.lambda_L}")
            print(f"  Test set size:         {n_test} (fixed across runs)")
            print(f"  Run seeds:             {run_seeds}")
            print("\n--- Min / Max (across runs) ---")
            print(f"  Coverage:    [{coverages.min():.4f}, {coverages.max():.4f}]")
            print(f"  Set size:    [{set_sizes.min():.2f}, {set_sizes.max():.2f}]")
            print(f"  Accuracy:    [{accs.min():.4f}, {accs.max():.4f}]")


if __name__ == "__main__":
    main()
