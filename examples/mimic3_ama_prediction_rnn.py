"""Ablation study for MIMIC-III Against-Medical-Advice (AMA) discharge prediction (RNN).

This script demonstrates ``AMAPredictionMIMIC3`` with three ``feature_keys``
ablations (baseline, +race, +substance), training PyHealth's ``RNN`` so the
mapping from task tensors to logits is non-linear.  One visit
per sequence yields a compact recurrent block over the selected features.
Labels and task schemas match ``AMAPredictionMIMIC3`` (AMA from discharge
location; ``input_schema`` / ``output_schema`` unchanged).

Paper:
    Boag, W.; Suresh, H.; Celi, L. A.; Szolovits, P.; and Ghassemi, M.
    "Racial Disparities and Mistrust in End-of-Life Care."
    Machine Learning for Healthcare Conference, PMLR, 2018.

Ablation configurations tested:
    1. BASELINE: demographics + age + LOS.
    2. BASELINE+RACE: adds ``race`` (normalized ethnicity).
    3. BASELINE+RACE+SUBSTANCE: adds ``has_substance_use`` from diagnosis text.

Results:
    For each baseline configuration we report:
    - Overall AUROC over N random 60/40 patient-level splits.
    - Subgroup AUROC by race, age group, and insurance.

Synthetic data:
    ``generate_synthetic_mimic3`` is imported from
    ``examples/mimic3_ama_prediction_logistic_regression.py`` (single shared
    implementation of the CSV writer).

Usage:
    # Synthetic exhaustive grid (default when no ``--root``)
    cd /path/to/PyHealth && python examples/mimic3_ama_prediction_rnn.py \\
        --data-source synthetic

    # Real MIMIC-III 1.4 (set ``--root`` to your extract path)
    cd /path/to/PyHealth && python examples/mimic3_ama_prediction_rnn.py \\
        --data-source real --root /path/to/mimic-iii/1.4 --splits 100 --epochs 10

"""

import argparse
import importlib.util
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RNN
from pyhealth.tasks import AMAPredictionMIMIC3
from pyhealth.trainer import Trainer

_LR_EXAMPLE = Path(__file__).resolve().parent / (
    "mimic3_ama_prediction_logistic_regression.py"
)
_spec = importlib.util.spec_from_file_location(
    "mimic3_ama_lr_example",
    _LR_EXAMPLE,
)
_lr_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_lr_mod)
generate_synthetic_mimic3 = _lr_mod.generate_synthetic_mimic3

BASELINES = {
    "BASELINE": ["demographics", "age", "los"],
    "BASELINE+RACE": ["demographics", "age", "los", "race"],
    "BASELINE+RACE+SUBSTANCE": [
        "demographics",
        "age",
        "los",
        "race",
        "has_substance_use",
    ],
}


# ------------------------------------------------------------------
# Helpers -- demographics lookup
# ------------------------------------------------------------------


def _build_demographics_lookup(
    dataset: Any,
    task: AMAPredictionMIMIC3,
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Map each visit to race, age, and insurance for subgroup metrics.

    Args:
        dataset: ``MIMIC3Dataset`` with ``iter_patients()``.
        task: Same ``AMAPredictionMIMIC3`` used with ``set_task``.

    Returns:
        ``(patient_id, visit_id)`` strings -> ``{"race","age","insurance"}``.
    """
    lookup: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for patient in dataset.iter_patients():
        for sample in task(patient):
            pid = str(sample["patient_id"])
            vid = str(sample["visit_id"])
            race = sample["race"][0].split(":", 1)[1]
            age = sample["age"][0]
            insurance = "Other"
            for t in sample["demographics"]:
                if t.startswith("insurance:"):
                    insurance = t.split(":", 1)[1]
                    break
            lookup[(pid, vid)] = {
                "race": race,
                "age": age,
                "insurance": insurance,
            }
    return lookup


def _age_group(age: float) -> str:
    """Map age in years to Young / Middle / Senior subgroup labels."""
    if age < 45:
        return "Young (18-44)"
    if age < 65:
        return "Middle (45-64)"
    return "Senior (65+)"


# ------------------------------------------------------------------
# Helpers -- inference with demographic labels
# ------------------------------------------------------------------


def _get_predictions(
    model: Any,
    dataloader: Any,
    lookup: Dict[Tuple[str, str], Dict[str, Any]],
) -> Tuple[Any, Any, Dict[str, Any]]:
    """Evaluate ``model`` on ``dataloader``; attach subgroup labels.

    Args:
        model: Trained ``RNN`` accepting batch kwargs.
        dataloader: Test loader with ``patient_id`` / ``visit_id``.
        lookup: Demographics map from :func:`_build_demographics_lookup`.

    Returns:
        ``(y_prob, y_true, groups)`` numpy arrays / group dict.
    """
    model.eval()
    all_probs, all_labels = [], []
    all_races, all_ages, all_ins = [], [], []

    with torch.no_grad():
        for batch in dataloader:
            output = model(**batch)
            all_probs.append(output["y_prob"].detach().cpu())
            all_labels.append(output["y_true"].detach().cpu())

            pids = batch["patient_id"]
            vids = batch["visit_id"]
            if isinstance(vids, torch.Tensor):
                vids = vids.tolist()
            if isinstance(pids, torch.Tensor):
                pids = pids.tolist()

            for pid, vid in zip(pids, vids):
                info = lookup.get((str(pid), str(vid)), {})
                all_races.append(info.get("race", "Other"))
                all_ages.append(_age_group(info.get("age", 0.0)))
                all_ins.append(info.get("insurance", "Other"))

    y_prob = torch.cat(all_probs).numpy().ravel()
    y_true = torch.cat(all_labels).numpy().ravel()
    groups = {
        "Race": np.array(all_races),
        "Age Group": np.array(all_ages),
        "Insurance": np.array(all_ins),
    }
    return y_prob, y_true, groups


# ------------------------------------------------------------------
# Helpers -- safe metrics
# ------------------------------------------------------------------


def _safe_auroc(y: Any, p: Any) -> float:
    """ROC-AUC with guards for single-class slices."""
    if len(np.unique(y)) < 2:
        return float("nan")
    try:
        return roc_auc_score(y, p)
    except ValueError:
        return float("nan")


# ------------------------------------------------------------------
# Single split
# ------------------------------------------------------------------


def _create_model(
    sample_dataset: Any,
    feature_keys: List[str],
    embedding_dim: int = 128,
    hidden_dim: int = 64,
) -> RNN:
    """Build ``RNN`` over ``feature_keys`` (ablation-controlled inputs).

    Args:
        sample_dataset: ``SampleDataset`` after ``set_task``.
        feature_keys: Keys from ``AMAPredictionMIMIC3.input_schema``.
        embedding_dim: Token embedding size.
        hidden_dim: RNN hidden size (``fc`` input is ``len(keys)*hidden_dim``).

    Returns:
        Configured ``RNN`` instance.
    """
    model = RNN(
        dataset=sample_dataset,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    )
    model.feature_keys = list(feature_keys)
    output_size = model.get_output_size()
    model.fc = torch.nn.Linear(
        len(feature_keys) * hidden_dim,
        output_size,
    )
    return model


def _run_single_split(
    sample_dataset: Any,
    feature_keys: List[str],
    lookup: Dict[Tuple[str, str], Dict[str, Any]],
    seed: int,
    epochs: int,
    batch_size: int = 32,
) -> Optional[Dict[str, Any]]:
    """One patient-level split: train RNN, then test + subgroup AUROC.

    Args:
        sample_dataset: AMA task samples.
        feature_keys: Subset of input schema keys for this ablation.
        lookup: Demographics for post-hoc slicing.
        seed: Split RNG seed.
        epochs: Training epochs.
        batch_size: Loader batch size.

    Returns:
        Metric dict or ``None`` on training failure.
    """
    train_ds, _, test_ds = split_by_patient(
        sample_dataset,
        [0.6, 0.0, 0.4],
        seed=seed,
    )
    train_dl = get_dataloader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = get_dataloader(test_ds, batch_size=batch_size, shuffle=False)

    # ``feature_keys`` selects which task outputs the RNN reads for this run.
    model = _create_model(sample_dataset, feature_keys)
    trainer = Trainer(model=model)
    try:
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=None,
            epochs=epochs,
            monitor=None,
        )
    except Exception as exc:
        print(f"    train failed: {exc}")
        return None

    y_prob, y_true, groups = _get_predictions(model, test_dl, lookup)

    overall_auroc = _safe_auroc(y_true, y_prob)

    subgroup = {}
    for attr_name, attr_vals in groups.items():
        subgroup[attr_name] = {}
        for grp in sorted(set(attr_vals)):
            mask = attr_vals == grp
            n = int(mask.sum())
            if n < 2:
                continue
            yt, yp = y_true[mask], y_prob[mask]
            subgroup[attr_name][grp] = {
                "auroc": _safe_auroc(yt, yp),
                "n": n,
            }

    return {
        "auroc": overall_auroc,
        "subgroups": subgroup,
    }


# ------------------------------------------------------------------
# Aggregation
# ------------------------------------------------------------------


def _nanmean(lst: List[float]) -> float:
    """Mean ignoring NaNs."""
    v = [x for x in lst if not np.isnan(x)]
    return float(np.mean(v)) if v else float("nan")


def _nanstd(lst: List[float]) -> float:
    """Std ignoring NaNs."""
    v = [x for x in lst if not np.isnan(x)]
    return float(np.std(v)) if v else float("nan")


def _aggregate(
    results: List[Optional[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    """Mean/std of overall AUROC and per-subgroup metrics across splits."""
    valid = [r for r in results if r is not None]
    if not valid:
        return None

    agg = {
        "n": len(valid),
        "auroc_mean": _nanmean([r["auroc"] for r in valid]),
        "auroc_std": _nanstd([r["auroc"] for r in valid]),
    }

    all_attrs = set()
    for r in valid:
        all_attrs.update(r["subgroups"].keys())

    agg["subgroups"] = {}
    for attr in sorted(all_attrs):
        agg["subgroups"][attr] = {}
        all_grps = set()
        for r in valid:
            if attr in r["subgroups"]:
                all_grps.update(r["subgroups"][attr].keys())

        for grp in sorted(all_grps):
            aurocs, ns = [], []
            for r in valid:
                m = r["subgroups"].get(attr, {}).get(grp)
                if m is None:
                    continue
                aurocs.append(m["auroc"])
                ns.append(m["n"])

            agg["subgroups"][attr][grp] = {
                "auroc_mean": _nanmean(aurocs),
                "auroc_std": _nanstd(aurocs),
                "n_avg": int(np.mean(ns)) if ns else 0,
            }
    return agg


# ------------------------------------------------------------------
# Pretty-printing
# ------------------------------------------------------------------


def _fmt(val: float, digits: int = 4) -> str:
    """Human-readable float or ``N/A`` for NaN."""
    return "N/A" if np.isnan(val) else f"{val:.{digits}f}"


def _print_results(
    name: str,
    feature_keys: List[str],
    agg: Optional[Dict[str, Any]],
) -> None:
    """Print ablation summary for RNN runs."""
    w = 70
    print(f"\n{'=' * w}")
    print(f"  {name}  (RNN  hidden_dim=64)")
    print(f"  Features: {feature_keys}")
    print(f"{'=' * w}")

    if agg is None:
        print("  No valid results.\n")
        return

    ci_lo = agg["auroc_mean"] - 1.96 * agg["auroc_std"]
    ci_hi = agg["auroc_mean"] + 1.96 * agg["auroc_std"]
    print(f"\n  1. Overall Performance ({agg['n']} splits)")
    print(
        f"     AUROC:  {_fmt(agg['auroc_mean'])} +/- {_fmt(agg['auroc_std'])}"
        f"  95% CI ({_fmt(ci_lo)}, {_fmt(ci_hi)})"
    )

    print("\n  2. Subgroup Performance")
    for attr, grps in agg["subgroups"].items():
        print(f"     {attr}:")
        print(f"       {'Group':<20} {'AUROC':>15} {'n_avg':>7}")
        print(f"       {'-' * 42}")
        for grp, m in grps.items():
            a_str = f"{_fmt(m['auroc_mean'])}+/-{_fmt(m['auroc_std'])}"
            print(f"       {grp:<20} {a_str:>15} {m['n_avg']:>7}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    """CLI entry: load MIMIC-III (or synthetic CSVs), apply AMA task, train RNN.

    Resolves ``--data-source`` / ``--root`` / ``--synthetic-mode``, builds a
    ``MIMIC3Dataset`` and ``SampleDataset`` via ``set_task(AMAPredictionMIMIC3)``,
    builds a demographics lookup for subgroup metrics, then for each entry in
    ``BASELINES`` runs ``--splits`` train/eval loops.  Each loop uses the
    listed ``feature_keys`` so the ``RNN`` reads only that subset of task
    outputs; the task ``input_schema`` / ``output_schema`` are unchanged.

    Side effects: prints progress; creates temporary data and cache dirs.
    """
    parser = argparse.ArgumentParser(
        description="AMA prediction ablation -- RNN",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "synthetic", "real"),
        default="auto",
        help="auto: use --root if set, else synthetic. synthetic: local CSVs. "
        "real: require --root.",
    )
    parser.add_argument(
        "--synthetic-mode",
        choices=("exhaustive", "random"),
        default="exhaustive",
        help="Synthetic grid: exhaustive cross-product or random; "
        "see generate_synthetic_mimic3 docstring in the loaded module.",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="MIMIC-III root (required for --data-source real).",
    )
    parser.add_argument(
        "--patients", type=int, default=100, help="Random synthetic mode only."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="RNG seed for random synthetic mode."
    )
    parser.add_argument(
        "--splits", type=int, default=5, help="Number of random 60/40 splits"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Training epochs per split"
    )
    parser.add_argument(
        "--dev", action="store_true", help="Random synthetic: 10 patients only."
    )
    args = parser.parse_args()

    if args.data_source == "real" and not args.root:
        parser.error("--data-source real requires --root /path/to/mimic-iii/1.4")

    use_synthetic = args.data_source == "synthetic" or (
        args.data_source == "auto" and args.root is None
    )
    if args.data_source == "synthetic" and args.root:
        print(
            "Note: --root ignored with --data-source synthetic.\n",
        )

    cache_dir = tempfile.mkdtemp(prefix="ama_rnn_")

    if use_synthetic:
        print("[Setup] Generating synthetic MIMIC-III dataset...")
        data_dir = tempfile.mkdtemp(prefix="synthetic_mimic3_")
        n_patients = (
            10 if args.dev and args.synthetic_mode == "random" else args.patients
        )
        generate_synthetic_mimic3(
            data_dir,
            n_patients=n_patients,
            avg_admissions_per_patient=2,
            seed=args.seed,
            mode=args.synthetic_mode,
        )
        args.root = str(data_dir)
        print(f"        Synthetic: {data_dir}  mode={args.synthetic_mode}\n")
    else:
        print(f"Using real MIMIC-III from: {args.root}\n")

    print(f"Cache: {cache_dir}")
    print(f"Root:  {args.root}")
    print(f"Splits: {args.splits}  |  Epochs: {args.epochs}")

    print("\n[1/4] Loading dataset...")
    t0 = time.time()
    dataset = MIMIC3Dataset(
        root=args.root,
        tables=[],
        cache_dir=cache_dir,
        dev=args.dev,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")
    dataset.stats()

    print("\n[2/4] Applying AMA task...")
    task = AMAPredictionMIMIC3()
    try:
        sample_dataset = dataset.set_task(task)
    except ValueError as exc:
        if "unique labels" not in str(exc).lower():
            raise
        print(f"\n  {exc}")
        print("  The dataset contains no AMA-positive cases.")
        print("  AMA prevalence is ~2% so small/synthetic data")
        print("  often lacks positives.  Demonstrating the task")
        print("  on raw patients instead:\n")
        total = 0
        for patient in dataset.iter_patients():
            samples = task(patient)
            total += len(samples)
        print(f"  Task produced {total} samples (all label=0)")
        print("\n  Re-run with real MIMIC-III:")
        print("    python examples/mimic3_ama_prediction_rnn.py \\")
        print("        --data-source real --root /path/to/mimic-iii/1.4")
        print("\nDone.")
        return
    print(f"  Samples: {len(sample_dataset)}")

    print("\n[3/4] Building demographics lookup...")
    t0 = time.time()
    lookup = _build_demographics_lookup(dataset, task)
    print(f"  {len(lookup)} entries in {time.time() - t0:.1f}s")

    print(
        f"\n[4/4] Running ablation ({len(BASELINES)} baselines "
        f"x {args.splits} splits)...\n"
    )

    t_total = time.time()
    for name, feature_keys in BASELINES.items():
        split_results = []
        for i in range(args.splits):
            t0 = time.time()
            res = _run_single_split(
                sample_dataset,
                feature_keys,
                lookup,
                seed=i,
                epochs=args.epochs,
            )
            elapsed = time.time() - t0
            if res is not None:
                print(
                    f"  [{name}] split {i + 1:3d}/{args.splits}: "
                    f"AUROC={_fmt(res['auroc'])}  ({elapsed:.1f}s)"
                )
            else:
                print(f"  [{name}] split {i + 1:3d}/{args.splits}: FAILED")
            split_results.append(res)

        agg = _aggregate(split_results)
        _print_results(name, feature_keys, agg)

    print(f"\nTotal time: {time.time() - t_total:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
