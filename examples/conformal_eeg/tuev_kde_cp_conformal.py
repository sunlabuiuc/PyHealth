"""CP with covariate shift correction (KDE) on TUEV EEG Events using ContraWR.

Baseline: CovariateLabel with KDE-based likelihood ratio weighting (CoDrug-style).
Requires cal and test embeddings to fit KDEs and compute weights.

With --n-seeds > 1: fixes the test set (--split-seed), runs multiple training runs
with different seeds, reports coverage / set size / accuracy as mean ± std.

Example (from repo root):
  python examples/conformal_eeg/tuev_kde_cp_conformal.py --alpha 0.1 --n-seeds 5 --split-seed 0 --log-file kde_cp_alpha01_seeds5.log

Run for PI baselines (alpha=0.2, 0.1, 0.05 with error bars):
  for a in 0.2 0.1 0.05; do python examples/conformal_eeg/tuev_kde_cp_conformal.py --alpha $a --n-seeds 5 --split-seed 0 --log-file kde_cp_alpha${a}_seeds5.log; done
  Or in parallel on 3 GPUs: same with CUDA_VISIBLE_DEVICES=0/1/2 and & wait.
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
from pyhealth.datasets import TUEVDataset, TUABDataset, get_dataloader, split_by_sample_conformal
from pyhealth.tasks import EEGEventsTUEV, EEGAbnormalTUAB
from pyhealth.trainer import Trainer, get_metrics_fn

from model_utils import AddSTFTDataset, get_model

DEFAULT_ROOT = {"tuev": "/srv/local/data/TUH/tuh_eeg_events/v2.0.0/edf", "tuab": "/srv/local/data/TUH/tuh_eeg_abnormal/v3.0.0/edf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CP with covariate shift correction (KDE / CovariateLabel) on TUEV/TUAB EEG using ContraWR or TFM."
    )
    parser.add_argument("--dataset", type=str, default="tuev", choices=["tuev", "tuab"], help="EEG dataset: tuev or tuab.")
    parser.add_argument("--root", type=str, default=None, help="Path to dataset edf/ folder. Default per --dataset.")
    parser.add_argument("--subset", type=str, default="both", choices=["train", "eval", "both"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-seeds", type=int, default=1, help="Number of runs for mean±std. Test set fixed when > 1.")
    parser.add_argument("--split-seed", type=int, default=0, help="Fixed seed for initial split (fixes test set when n-seeds > 1).")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated run seeds. Overrides --seed and --n-seeds.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate (e.g. 0.1 => 90%% target coverage).")
    parser.add_argument("--ratios", type=float, nargs=4, default=(0.6, 0.1, 0.15, 0.15), metavar=("TRAIN", "VAL", "CAL", "TEST"))
    parser.add_argument("--n-fft", type=int, default=128)
    parser.add_argument("--model", type=str, default="contrawr", choices=["contrawr", "tfm"], help="Backbone: contrawr or tfm (TFM-Tokenizer).")
    parser.add_argument("--tfm-checkpoint", type=str, default=None, help="Path to TFM checkpoint (full model or tokenizer). Use {seed} for per-seed paths.")
    parser.add_argument("--tfm-tokenizer-checkpoint", type=str, default=None, help="Path to pretrained TFM tokenizer (shared). Use with --tfm-classifier-checkpoint for inference.")
    parser.add_argument("--tfm-classifier-checkpoint", type=str, default=None, help="Path to finetuned classifier. Use {seed} for per-seed paths.")
    parser.add_argument("--tfm-skip-train", action="store_true", help="Skip training; load checkpoint(s) and run calibration + inference only.")
    parser.add_argument("--tfm-freeze-tokenizer", action="store_true", help="Freeze tokenizer when fine-tuning; only train classifier.")
    parser.add_argument("--tfm-epochs", type=int, default=5, help="Epochs when fine-tuning TFM. Ignored if --tfm-skip-train.")
    parser.add_argument("--tfm-lr", type=float, default=1e-4, help="Learning rate when fine-tuning TFM.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--cache-dir", type=str, default=None, help="Per-job cache dir to avoid races when running 8 in parallel.")
    parser.add_argument("--quick-test", action="store_true", help="dev=True, max 2000 samples, 2 epochs.")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _split_remainder_into_train_val_cal(sample_dataset, remainder_indices, ratios, run_seed):
    r0, r1, r2, r3 = ratios
    remainder_frac = 1.0 - r3
    if remainder_frac <= 0:
        raise ValueError("Test ratio must be < 1.")
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


def _run_one_kde_cp(
    sample_dataset,
    train_ds,
    val_ds,
    cal_ds,
    test_ds,
    test_loader,
    args,
    device,
    epochs,
    task_mode="multiclass",
    return_metrics=False,
):
    """Train ContraWR, extract cal + test embeddings, calibrate CovariateLabel (KDE), evaluate."""
    train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = get_dataloader(val_ds, batch_size=args.batch_size, shuffle=False) if len(val_ds) else None

    model_name = "TFM-Tokenizer" if args.model.lower() == "tfm" else "ContraWR"
    print("\n" + "=" * 80)
    print(f"STEP 3: Train {model_name}" if not getattr(args, "tfm_skip_train", False) else f"STEP 3: Load {model_name} (skip train)")
    print("=" * 80)
    model = get_model(args, sample_dataset, device)
    trainer = Trainer(model=model, device=device, enable_logging=False)
    if not getattr(args, "tfm_skip_train", False):
        optimizer_params = None
        if args.model.lower() == "tfm" and (
            getattr(args, "tfm_checkpoint", None)
            or (getattr(args, "tfm_tokenizer_checkpoint", None) and getattr(args, "tfm_classifier_checkpoint", None))
        ):
            optimizer_params = {"lr": getattr(args, "tfm_lr", 1e-4)}
        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=epochs,
            monitor="accuracy" if val_loader is not None else None,
            optimizer_params=optimizer_params,
        )

    if not return_metrics:
        print("\nBase model performance on test set:")
        y_true_base, y_prob_base, _ = trainer.inference(test_loader)
        base_metrics = get_metrics_fn(task_mode)(y_true_base, y_prob_base, metrics=["accuracy", "f1_weighted"])
        for k, v in base_metrics.items():
            print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 80)
    print("STEP 4: CP with covariate shift correction (CovariateLabel / KDE)")
    print("=" * 80)
    print(f"Target miscoverage alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")

    print("Extracting calibration embeddings...")
    cal_embeddings = extract_embeddings(model, cal_ds, batch_size=args.batch_size, device=device)
    print("Extracting test embeddings...")
    test_embeddings = extract_embeddings(model, test_ds, batch_size=args.batch_size, device=device)
    if not return_metrics:
        print(f"  cal_embeddings shape: {cal_embeddings.shape}, test_embeddings shape: {test_embeddings.shape}")

    predictor = CovariateLabel(model=model, alpha=float(args.alpha))
    predictor.calibrate(
        cal_dataset=cal_ds,
        cal_embeddings=cal_embeddings,
        test_embeddings=test_embeddings,
    )

    y_true, y_prob, _loss, extra = Trainer(model=predictor, enable_logging=False).inference(test_loader, additional_outputs=["y_predset"])
    metrics = get_metrics_fn(task_mode)(y_true, y_prob, metrics=["accuracy", "miscoverage_ps"], y_predset=extra["y_predset"])
    predset = extra["y_predset"]
    predset_t = torch.tensor(predset) if isinstance(predset, np.ndarray) else predset
    avg_set_size = predset_t.float().sum(dim=1).mean().item()
    miscoverage = metrics["miscoverage_ps"]
    if isinstance(miscoverage, np.ndarray):
        miscoverage = float(miscoverage.item() if miscoverage.size == 1 else miscoverage.mean())
    else:
        miscoverage = float(miscoverage)
    coverage = 1.0 - miscoverage

    if return_metrics:
        return {"accuracy": float(metrics["accuracy"]), "coverage": coverage, "miscoverage": miscoverage, "avg_set_size": avg_set_size}

    print("\nCovariateLabel (KDE) Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Empirical miscoverage: {miscoverage:.4f}")
    print(f"  Empirical coverage: {coverage:.4f}")
    print(f"  Average set size: {avg_set_size:.2f}")
    print("\n--- Single-run summary (for reporting) ---")
    print(f"  alpha={args.alpha}, target_coverage={1 - args.alpha:.2f}, empirical_coverage={coverage:.4f}, miscoverage={miscoverage:.4f}, accuracy={metrics['accuracy']:.4f}, avg_set_size={avg_set_size:.2f}")


def main() -> None:
    args = parse_args()
    if args.n_seeds <= 1 and args.seeds is None:
        set_seed(args.seed)

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    log_file = None
    if args.log_file:
        p = Path(args.log_file)
        if "/" not in args.log_file and not p.is_absolute():
            Path("logs").mkdir(parents=True, exist_ok=True)
            args.log_file = str(Path("logs") / args.log_file)
        log_file = open(args.log_file, "w", encoding="utf-8")
        sys.stdout = _Tee(orig_stdout, log_file)
        sys.stderr = _Tee(orig_stderr, log_file)
    try:
        _run(args)
    finally:
        if log_file is not None:
            sys.stdout, sys.stderr = orig_stdout, orig_stderr
            log_file.close()


def _run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset_name = getattr(args, "dataset", "tuev")
    root = Path(args.root or DEFAULT_ROOT[dataset_name])
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {root}. Set --root for {dataset_name}.")

    if args.model.lower() == "tfm":
        if not getattr(args, "tfm_tokenizer_checkpoint", None):
            args.tfm_tokenizer_checkpoint = "/srv/local/data/arjunc4/tfm_tokenizer_last.pth"
        if not getattr(args, "tfm_classifier_checkpoint", None):
            args.tfm_classifier_checkpoint = (
                "/srv/local/data/arjunc4/TFM_Tokenizer_multiple_finetuned_on_TUAB/TFM_Tokenizer_multiple_finetuned_on_TUAB_{seed}/best_model.pth"
                if dataset_name == "tuab"
                else "/srv/local/data/arjunc4/TFM_Tokenizer_multiple_finetuned_on_TUEV/TFM_Tokenizer_multiple_finetuned_on_TUEV_{seed}/best_model.pth"
            )

    if args.quick_test:
        epochs = 2
    elif args.model.lower() == "tfm" and (
        getattr(args, "tfm_checkpoint", None)
        or (getattr(args, "tfm_tokenizer_checkpoint", None) and getattr(args, "tfm_classifier_checkpoint", None))
    ):
        epochs = getattr(args, "tfm_epochs", 5)
    else:
        epochs = args.epochs
    quick_test_max = 2000
    if args.quick_test:
        print("*** QUICK TEST MODE ***")

    task_mode = "binary" if dataset_name == "tuab" else "multiclass"
    print("=" * 80)
    print(f"STEP 1: Load {dataset_name.upper()} + build task dataset")
    print("=" * 80)
    cache_base = getattr(args, "cache_dir", None)
    if dataset_name == "tuab":
        cache_dir = (cache_base.rstrip("/") + "_tuab") if cache_base else "examples/conformal_eeg/cache_tuab"
        dataset = TUABDataset(root=str(root), subset=args.subset, dev=args.quick_test)
        sample_dataset = dataset.set_task(EEGAbnormalTUAB(), cache_dir=cache_dir)
    else:
        cache_dir = cache_base or "examples/conformal_eeg/cache"
        dataset = TUEVDataset(root=str(root), subset=args.subset, dev=args.quick_test)
        sample_dataset = dataset.set_task(
            EEGEventsTUEV(resample_rate=200), cache_dir=cache_dir
        )
    if args.quick_test and len(sample_dataset) > quick_test_max:
        sample_dataset = sample_dataset.subset(range(quick_test_max))
        print(f"Capped to {quick_test_max} samples.")
    if args.model.lower() == "tfm":
        sample_dataset = AddSTFTDataset(sample_dataset, n_fft=200, hop_length=100)
        print("Wrapped dataset with STFT for TFM-Tokenizer.")
    print(f"Task samples: {len(sample_dataset)} (task_mode={task_mode})")

    print("\n--- Experiment configuration ---")
    print(f"  dataset: {dataset_name}, dataset_root: {root}, subset: {args.subset}")
    print(f"  ratios: train/val/cal/test = {args.ratios[0]:.2f}/{args.ratios[1]:.2f}/{args.ratios[2]:.2f}/{args.ratios[3]:.2f}")
    print(f"  alpha: {args.alpha} (target coverage {1 - args.alpha:.0%})")
    print(f"  epochs: {epochs}, batch_size: {args.batch_size}, device: {device}, seed: {args.seed}")

    if len(sample_dataset) == 0:
        raise RuntimeError("No samples.")

    ratios = list(args.ratios)
    use_multi_seed = args.n_seeds > 1 or args.seeds is not None
    if use_multi_seed:
        run_seeds = [int(s.strip()) for s in args.seeds.split(",")] if args.seeds else [args.seed + i for i in range(args.n_seeds)]
        n_runs = len(run_seeds)
        print(f"  multi_seed: n_runs={n_runs}, run_seeds={run_seeds}, split_seed={args.split_seed} (fixed test set)")
        print(f"Multi-seed mode: {n_runs} runs (fixed test set), run seeds: {run_seeds}")

    if not use_multi_seed:
        print("\n" + "=" * 80)
        print("STEP 2: Split train/val/cal/test")
        print("=" * 80)
        train_ds, val_ds, cal_ds, test_ds = split_by_sample_conformal(dataset=sample_dataset, ratios=ratios, seed=args.seed)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Cal: {len(cal_ds)}, Test: {len(test_ds)}")
        test_loader = get_dataloader(test_ds, batch_size=args.batch_size, shuffle=False)
        _run_one_kde_cp(sample_dataset, train_ds, val_ds, cal_ds, test_ds, test_loader, args, device, epochs, task_mode=task_mode)
        print("\n--- Split sizes and seed (for reporting) ---")
        print(f"  train={len(train_ds)}, val={len(val_ds)}, cal={len(cal_ds)}, test={len(test_ds)}, seed={args.seed}")
        return

    print("\n" + "=" * 80)
    print("STEP 2: Fix test set (split-seed), then run multiple train/cal splits")
    print("=" * 80)
    train_idx, val_idx, cal_idx, test_idx = split_by_sample_conformal(dataset=sample_dataset, ratios=ratios, seed=args.split_seed, get_index=True)
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
    tfm_ckpt_original = getattr(args, "tfm_checkpoint", None)
    tfm_classifier_ckpt_original = getattr(args, "tfm_classifier_checkpoint", None)
    for run_i, run_seed in enumerate(run_seeds):
        print("\n" + "=" * 80)
        print(f"Run {run_i + 1} / {n_runs} (seed={run_seed})")
        print("=" * 80)
        if tfm_ckpt_original and "{seed}" in tfm_ckpt_original:
            args.tfm_checkpoint = tfm_ckpt_original.replace("{seed}", str(run_seed))
        if tfm_classifier_ckpt_original and "{seed}" in tfm_classifier_ckpt_original:
            args.tfm_classifier_checkpoint = tfm_classifier_ckpt_original.replace("{seed}", str(run_seed))
        set_seed(run_seed)
        train_ds, val_ds, cal_ds = _split_remainder_into_train_val_cal(sample_dataset, remainder_indices, ratios, run_seed)
        print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Cal: {len(cal_ds)}")
        m = _run_one_kde_cp(sample_dataset, train_ds, val_ds, cal_ds, test_ds, test_loader, args, device, epochs, task_mode=task_mode, return_metrics=True)
        if tfm_ckpt_original and "{seed}" in tfm_ckpt_original:
            args.tfm_checkpoint = tfm_ckpt_original
        if tfm_classifier_ckpt_original and "{seed}" in tfm_classifier_ckpt_original:
            args.tfm_classifier_checkpoint = tfm_classifier_ckpt_original
        accs.append(m["accuracy"])
        coverages.append(m["coverage"])
        miscoverages.append(m["miscoverage"])
        set_sizes.append(m["avg_set_size"])

    accs = np.array(accs)
    coverages = np.array(coverages)
    miscoverages_arr = np.array(miscoverages)
    set_sizes = np.array(set_sizes)

    print("\n" + "=" * 80)
    print("Per-run KDE CP results (fixed test set)")
    print("=" * 80)
    print(f"  {'Run':<4} {'Seed':<6} {'Accuracy':<10} {'Coverage':<10} {'Miscoverage':<12} {'Avg set size':<12}")
    print("  " + "-" * 54)
    for i in range(n_runs):
        print(f"  {i+1:<4} {run_seeds[i]:<6} {accs[i]:<10.4f} {coverages[i]:<10.4f} {miscoverages_arr[i]:<12.4f} {set_sizes[i]:<12.2f}")

    print("\n" + "=" * 80)
    print("KDE CP summary (mean ± std over {} runs, fixed test set)".format(n_runs))
    print("=" * 80)
    print(f"  Accuracy:             {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"  Empirical coverage:   {coverages.mean():.4f} ± {coverages.std():.4f}")
    print(f"  Empirical miscoverage: {miscoverages_arr.mean():.4f} ± {miscoverages_arr.std():.4f}")
    print(f"  Average set size:     {set_sizes.mean():.2f} ± {set_sizes.std():.2f}")
    print(f"  Target coverage: {1 - args.alpha:.0%} (alpha={args.alpha})")
    print(f"  Test set size: {n_test} (fixed across runs)")
    print(f"  Run seeds: {run_seeds}")
    print("\n--- Min / Max (across runs) ---")
    print(f"  Coverage:  [{coverages.min():.4f}, {coverages.max():.4f}]")
    print(f"  Set size:  [{set_sizes.min():.2f}, {set_sizes.max():.2f}]")
    print(f"  Accuracy:  [{accs.min():.4f}, {accs.max():.4f}]")


if __name__ == "__main__":
    main()
