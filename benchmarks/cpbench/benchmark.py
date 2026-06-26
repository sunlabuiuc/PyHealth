import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROTOCOL = {
    "version": "v1",
    "split_ratios": [0.60, 0.15, 0.10, 0.15],
    "seeds": [42, 43, 44, 45, 46],
    "batch_size": 32,
    "standard_alphas": [0.01, 0.05, 0.10, 0.20],
}

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sleep_staging_isruc": {
        "description": "5-class sleep staging on ISRUC-I (polysomnography)",
        "dataset_class": "ISRUCDataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "sleep_staging_isruc_fn",
        "task_module": "pyhealth.tasks",
        "model_class": "SparcNet",
        "model_module": "pyhealth.models",
        "feature_keys": ["signal"],
        "label_key": "label",
        "mode": "multiclass",
        "split_strategy": "by_patient",
        "monitor": "accuracy",
        "monitor_criterion": "max",
        "default_epochs": 30,
    },
    "sleep_staging_sleepedf": {
        "description": "5-class sleep staging on Sleep-EDF",
        "dataset_class": "SleepEDFDataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "sleep_staging_sleepedf_fn",
        "task_module": "pyhealth.tasks",
        "model_class": "SparcNet",
        "model_module": "pyhealth.models",
        "feature_keys": ["signal"],
        "label_key": "label",
        "mode": "multiclass",
        "split_strategy": "by_patient",
        "monitor": "accuracy",
        "monitor_criterion": "max",
        "default_epochs": 30,
    },
    "sleep_staging_shhs": {
        "description": "5-class sleep staging on SHHS",
        "dataset_class": "SHHSDataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "sleep_staging_shhs_fn",
        "task_module": "pyhealth.tasks",
        "model_class": "SparcNet",
        "model_module": "pyhealth.models",
        "feature_keys": ["signal"],
        "label_key": "label",
        "mode": "multiclass",
        "split_strategy": "by_patient",
        "monitor": "accuracy",
        "monitor_criterion": "max",
        "default_epochs": 30,
    },
    "eeg_tuev": {
        "description": "6-class EEG event detection on TUEV",
        "dataset_class": "TUEVDataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "EEGEventsTUEV",
        "task_module": "pyhealth.tasks",
        "model_class": "ContraWR",
        "model_module": "pyhealth.models",
        "feature_keys": ["signal"],
        "label_key": "label",
        "mode": "multiclass",
        "split_strategy": "by_patient",
        "monitor": "accuracy",
        "monitor_criterion": "max",
        "default_epochs": 30,
    },
}

METHOD_REGISTRY: Dict[str, Dict[str, Any]] = {
    "base": {
        "description": "Split conformal prediction with APS score (Vovk et al. 2005)",
        "class": "BaseConformal",
        "module": "pyhealth.calib.predictionset",
        "needs_embeddings": False,
        "extra_kwargs": {"score_type": "aps"},
        "paper": "Vovk, Gammerman, Shafer. Algorithmic Learning in a Random World (2005)",
    },
    "label": {
        "description": "LABEL: Least Ambiguous set-valued classifiers with bounded error (Sadinle et al. 2019)",
        "class": "LABEL",
        "module": "pyhealth.calib.predictionset",
        "needs_embeddings": False,
        "extra_kwargs": {},
        "paper": "Sadinle, Lei, Wasserman. JASA (2019)",
    },
    "cluster": {
        "description": "Cluster-based conformal prediction with per-cluster thresholds",
        "class": "ClusterLabel",
        "module": "pyhealth.calib.predictionset",
        "needs_embeddings": True,
        "extra_kwargs": {"n_clusters": 5},
        "paper": "Cluster-based CP baseline",
    },
    "neighborhood": {
        "description": "Neighborhood-based conformal prediction using patient similarity",
        "class": "NeighborhoodLabel",
        "module": "pyhealth.calib.predictionset",
        "needs_embeddings": True,
        "extra_kwargs": {},
        "paper": "Neighborhood-based CP baseline",
    },
    "covariate": {
        "description": "Covariate shift-corrected conformal prediction via KDE (Tibshirani et al. 2019)",
        "class": "CovariateLabel",
        "module": "pyhealth.calib.predictionset",
        "needs_embeddings": True,
        "extra_kwargs": {},
        "paper": "Tibshirani, Barber, Candes, Ramdas. NeurIPS (2019)",
    },
}

AGGREGATE_METRICS = [
    "empirical_coverage",
    "coverage_gap",
    "avg_set_size",
    "rejection_rate",
    "accuracy",
]


def _import(module: str, name: str) -> Any:
    mod = importlib.import_module(module)
    return getattr(mod, name)


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_header(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 68 - len(title)))


def load_dataset(task_cfg: Dict[str, Any], data_path: str, dev: bool) -> Any:
    DatasetClass = _import(task_cfg["dataset_module"], task_cfg["dataset_class"])
    dataset = DatasetClass(root=data_path, dev=dev)
    try:
        task_fn = _import(task_cfg["task_module"], task_cfg["task_fn"])
    except AttributeError:
        raise ValueError(
            f"Task function '{task_cfg['task_fn']}' not found in "
            f"'{task_cfg['task_module']}'. Check TASK_REGISTRY."
        )
    dataset = dataset.set_task(task_fn)
    return dataset


def split_dataset(
    dataset: Any,
    task_cfg: Dict[str, Any],
    seed: int,
    ratios: List[float],
) -> Tuple[Any, Any, Any, Any]:
    from pyhealth.datasets import split_by_patient_conformal, split_by_sample_conformal

    strategy = task_cfg.get("split_strategy", "by_patient")
    if strategy == "by_patient":
        return split_by_patient_conformal(dataset, ratios=ratios, seed=seed)
    elif strategy == "by_sample":
        return split_by_sample_conformal(dataset, ratios=ratios, seed=seed)
    else:
        raise ValueError(f"Unknown split strategy: {strategy!r}")


def build_model(task_cfg: Dict[str, Any], dataset: Any) -> Any:
    ModelClass = _import(task_cfg["model_module"], task_cfg["model_class"])
    return ModelClass(
        dataset=dataset,
        feature_keys=task_cfg["feature_keys"],
        label_key=task_cfg["label_key"],
        mode=task_cfg["mode"],
    )


def train_model(
    model: Any,
    train_data: Any,
    val_data: Any,
    task_cfg: Dict[str, Any],
    epochs: int,
    batch_size: int,
    output_dir: str,
    seed: int,
) -> Any:
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer

    train_dl = get_dataloader(train_data, batch_size=batch_size, shuffle=True)
    val_dl = get_dataloader(val_data, batch_size=batch_size, shuffle=False)

    trainer = Trainer(
        model=model,
        device=_device(),
        enable_logging=True,
        output_path=output_dir,
        exp_name=f"seed_{seed}",
    )
    trainer.train(
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        epochs=epochs,
        monitor=task_cfg.get("monitor"),
        monitor_criterion=task_cfg.get("monitor_criterion", "max"),
        load_best_model_at_last=True,
    )
    return model


def build_cp_model(
    method_cfg: Dict[str, Any],
    base_model: Any,
    alpha: float,
    cal_data: Any,
    test_data: Any,
    dev: bool,
) -> Any:
    CPClass = _import(method_cfg["module"], method_cfg["class"])
    extra = method_cfg.get("extra_kwargs", {})
    cp_model = CPClass(model=base_model, alpha=alpha, debug=dev, **extra)

    if method_cfg["needs_embeddings"]:
        cp_model.calibrate(cal_dataset=cal_data, test_dataset=test_data)
    else:
        cp_model.calibrate(cal_dataset=cal_data)

    return cp_model


def evaluate(
    cp_model: Any,
    test_data: Any,
    alpha: float,
    batch_size: int,
    mode: str,
) -> Dict[str, Any]:
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer, get_metrics_fn
    import pyhealth.metrics.prediction_set as pset_metrics

    test_dl = get_dataloader(test_data, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model=cp_model, device=_device(), enable_logging=False)
    y_true_all, y_prob_all, _, extra = trainer.inference(
        test_dl, additional_outputs=["y_predset"]
    )

    y_predset = extra["y_predset"]
    miscoverage = pset_metrics.miscoverage_overall_ps(y_predset, y_true_all)

    base_metrics = get_metrics_fn(mode)(y_true_all, y_prob_all, metrics=["accuracy"])

    return {
        "alpha": alpha,
        "target_coverage": round(1.0 - alpha, 4),
        "empirical_coverage": round(float(1.0 - miscoverage), 4),
        "coverage_gap": round(float(abs(alpha - miscoverage)), 4),
        "avg_set_size": round(float(pset_metrics.size(y_predset)), 4),
        "rejection_rate": round(float(pset_metrics.rejection_rate(y_predset)), 4),
        "accuracy": round(float(base_metrics.get("accuracy", float("nan"))), 4),
        "n_test": int(len(y_true_all)),
        "n_cal": None,
    }


def aggregate_results(
    per_seed_results: List[List[Dict[str, Any]]],
    alphas: List[float],
) -> List[Dict[str, Any]]:
    aggregated = []
    for i, alpha in enumerate(alphas):
        alpha_rows = [seed_rows[i] for seed_rows in per_seed_results]
        row: Dict[str, Any] = {
            "alpha": alpha,
            "target_coverage": round(1.0 - alpha, 4),
            "n_seeds": len(alpha_rows),
            "n_test": alpha_rows[0]["n_test"],
            "n_cal": alpha_rows[0]["n_cal"],
        }
        for metric in AGGREGATE_METRICS:
            vals = [r[metric] for r in alpha_rows]
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals)), 4)
        aggregated.append(row)
    return aggregated


def print_aggregated_table(
    aggregated: List[Dict[str, Any]],
    task: str,
    method: str,
    seeds: List[int],
) -> None:
    _print_section("Results")
    print(f"\n  Task: {task}   Method: {method}")
    print(f"  Protocol: splits={PROTOCOL['split_ratios']}  seeds={seeds}")
    print(f"  Reporting mean ± std across {len(seeds)} seeds\n")

    col = 16
    header = (
        f"  {'alpha':>6}  {'target_cov':>10}  "
        f"{'emp_cov':<{col}}  {'cov_gap':<{col}}  "
        f"{'set_size':<{col}}  {'reject%':<{col}}  {'accuracy':<{col}}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in aggregated:
        def fmt(m: str) -> str:
            return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"

        gap_mean = r["coverage_gap_mean"]
        flag = " ✓" if gap_mean <= 0.01 else (" ✗" if gap_mean > 0.05 else "  ")

        print(
            f"  {r['alpha']:>6.2f}  {r['target_coverage']:>10.4f}  "
            f"{fmt('empirical_coverage'):<{col}}  {fmt('coverage_gap'):<{col}}  "
            f"{fmt('avg_set_size'):<{col}}  {fmt('rejection_rate'):<{col}}  "
            f"{fmt('accuracy'):<{col}}{flag}"
        )

    print()
    print("  ✓ = mean coverage_gap ≤ 0.01   ✗ = mean coverage_gap > 0.05")
    print(f"  n_test ≈ {aggregated[0]['n_test']}   n_cal ≈ {aggregated[0]['n_cal']}")


def save_results(
    aggregated: List[Dict[str, Any]],
    per_seed_results: List[List[Dict[str, Any]]],
    seeds: List[int],
    task: str,
    method: str,
    output_path: str,
) -> None:
    record = {
        "cpbench_protocol": PROTOCOL,
        "task": task,
        "method": method,
        "method_paper": METHOD_REGISTRY[method]["paper"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seeds": seeds,
        "aggregated": aggregated,
        "per_seed": {
            str(seed): per_seed_results[i] for i, seed in enumerate(seeds)
        },
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Results saved → {output_path}")


def run_benchmark(
    task: str,
    method: str,
    data_path: str,
    alphas: List[float],
    seeds: List[int],
    epochs: int,
    output_path: Optional[str],
    dev: bool,
    output_dir: str,
) -> List[Dict[str, Any]]:
    task_cfg = TASK_REGISTRY[task]
    method_cfg = METHOD_REGISTRY[method]
    batch_size = PROTOCOL["batch_size"]
    ratios = PROTOCOL["split_ratios"]

    _print_header(f"CPBench  |  task: {task}  |  method: {method}")
    print(f"  Seeds: {seeds}   Alphas: {alphas}")

    _print_section("Loading dataset")
    print(f"  {task_cfg['description']}")
    print(f"  data_path = {data_path}")
    dataset = load_dataset(task_cfg, data_path, dev)
    print(f"  Dataset size: {len(dataset)} samples")

    per_seed_results: List[List[Dict[str, Any]]] = []

    for seed in seeds:
        _print_header(f"Seed {seed}  ({seeds.index(seed) + 1}/{len(seeds)})")

        _print_section("Splitting")
        print(f"  ratios={ratios}  seed={seed}")
        train_data, val_data, cal_data, test_data = split_dataset(dataset, task_cfg, seed, ratios)
        print(
            f"  train={len(train_data)}  val={len(val_data)}  "
            f"cal={len(cal_data)}  test={len(test_data)}"
        )
        n_cal = len(cal_data)

        _print_section(f"Training {task_cfg['model_class']} ({epochs} epochs)")
        model = build_model(task_cfg, dataset)
        t0 = time.time()
        model = train_model(model, train_data, val_data, task_cfg, epochs, batch_size, output_dir, seed)
        print(f"  Training time: {time.time() - t0:.1f}s")
        model.to(_device())
        model.eval()

        seed_rows: List[Dict[str, Any]] = []

        for alpha in alphas:
            _print_section(f"α={alpha}  method={method}")
            print(f"  Calibrating on {n_cal} samples...")

            t0 = time.time()
            try:
                cp_model = build_cp_model(method_cfg, model, alpha, cal_data, test_data, dev)
            except TypeError:
                cp_model = _import(method_cfg["module"], method_cfg["class"])(
                    model=model, alpha=alpha, debug=dev, **method_cfg.get("extra_kwargs", {})
                )
                cp_model.calibrate(cal_dataset=cal_data)

            cp_model.to(_device())
            cal_time = time.time() - t0
            print(f"  Calibration: {cal_time:.1f}s  |  Evaluating on {len(test_data)} samples...")

            result = evaluate(cp_model, test_data, alpha, batch_size, task_cfg["mode"])
            result["n_cal"] = n_cal
            result["cal_time_s"] = round(cal_time, 2)
            result["seed"] = seed
            seed_rows.append(result)

        per_seed_results.append(seed_rows)

    aggregated = aggregate_results(per_seed_results, alphas)
    print_aggregated_table(aggregated, task, method, seeds)

    if output_path:
        save_results(aggregated, per_seed_results, seeds, task, method, output_path)

    return aggregated


def list_registry() -> None:
    print("\nRegistered Tasks:")
    print("-" * 60)
    for name, cfg in TASK_REGISTRY.items():
        print(f"  {name:<30}  {cfg['description']}")

    print("\nRegistered CP Methods:")
    print("-" * 60)
    for name, cfg in METHOD_REGISTRY.items():
        print(f"  {name:<14}  {cfg['description']}")
        print(f"  {'':14}  Paper: {cfg['paper']}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPBench: Standardized conformal prediction benchmarking for PyHealth",
    )

    parser.add_argument("--list", action="store_true", help="List all registered tasks and methods, then exit.")
    parser.add_argument("--task", type=str, choices=list(TASK_REGISTRY.keys()), help="Task name (see --list).")
    parser.add_argument("--method", type=str, choices=list(METHOD_REGISTRY.keys()), help="CP method name (see --list).")
    parser.add_argument("--data-path", type=str, help="Path to the raw dataset on disk.")
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=None,
        help=f"Miscoverage rate(s). Defaults to standard set {PROTOCOL['standard_alphas']}.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=f"Seeds for splitting. Defaults to {PROTOCOL['seeds']}.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Defaults to per-task default.",
    )
    parser.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    parser.add_argument("--output-dir", type=str, default="./cpbench_runs", help="Directory for Trainer logs.")
    parser.add_argument("--dev", action="store_true", help="Dev mode: tiny subset, single seed (42), single alpha (0.10).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_registry()
        sys.exit(0)

    missing = [f for f in ("task", "method", "data_path") if not getattr(args, f, None)]
    if missing:
        print(f"Error: --{', --'.join(m.replace('_', '-') for m in missing)} required.")
        sys.exit(1)

    if args.dev:
        alphas = [0.10]
        seeds = [42]
    else:
        alphas = args.alpha if args.alpha is not None else PROTOCOL["standard_alphas"]
        seeds = args.seeds if args.seeds is not None else PROTOCOL["seeds"]

    task_cfg = TASK_REGISTRY[args.task]
    epochs = args.epochs if args.epochs is not None else task_cfg.get("default_epochs", 20)

    os.makedirs(args.output_dir, exist_ok=True)

    run_benchmark(
        task=args.task,
        method=args.method,
        data_path=args.data_path,
        alphas=alphas,
        seeds=seeds,
        epochs=epochs,
        output_path=args.output,
        dev=args.dev,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
