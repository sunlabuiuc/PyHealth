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
    "split_seed": 42,
    "batch_size": 32,
    "standard_alphas": [0.05, 0.10, 0.20],
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
    "mortality_mimic4": {
        "description": "In-hospital mortality prediction on MIMIC-IV (binary)",
        "dataset_class": "MIMIC4Dataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "mortality_prediction_mimic4_fn",
        "task_module": "pyhealth.tasks",
        "model_class": "Transformer",
        "model_module": "pyhealth.models",
        "feature_keys": ["conditions", "procedures", "drugs"],
        "label_key": "label",
        "mode": "binary",
        "split_strategy": "by_patient",
        "monitor": "roc_auc",
        "monitor_criterion": "max",
        "default_epochs": 20,
    },
    "readmission_mimic4": {
        "description": "30-day readmission prediction on MIMIC-IV (binary)",
        "dataset_class": "MIMIC4Dataset",
        "dataset_module": "pyhealth.datasets",
        "task_fn": "readmission_prediction_mimic4_fn",
        "task_module": "pyhealth.tasks",
        "model_class": "Transformer",
        "model_module": "pyhealth.models",
        "feature_keys": ["conditions", "procedures", "drugs"],
        "label_key": "label",
        "mode": "binary",
        "split_strategy": "by_patient",
        "monitor": "roc_auc",
        "monitor_criterion": "max",
        "default_epochs": 20,
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


def build_model(task_cfg: Dict[str, Any], dataset: Any, checkpoint: Optional[str]) -> Any:
    ModelClass = _import(task_cfg["model_module"], task_cfg["model_class"])
    model = ModelClass(
        dataset=dataset,
        feature_keys=task_cfg["feature_keys"],
        label_key=task_cfg["label_key"],
        mode=task_cfg["mode"],
    )
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        print(f"  Loaded checkpoint: {checkpoint}")
    return model


def train_model(
    model: Any,
    train_data: Any,
    val_data: Any,
    task_cfg: Dict[str, Any],
    epochs: int,
    batch_size: int,
    output_dir: str,
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
        exp_name="base_model",
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
    batch_size: int,
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
) -> Dict[str, float]:
    from pyhealth.datasets import get_dataloader
    from pyhealth.trainer import Trainer, get_metrics_fn
    import pyhealth.metrics.prediction_set as pset_metrics

    test_dl = get_dataloader(test_data, batch_size=batch_size, shuffle=False)
    trainer = Trainer(model=cp_model, device=_device(), enable_logging=False)
    y_true_all, y_prob_all, loss, extra = trainer.inference(
        test_dl, additional_outputs=["y_predset"]
    )

    y_predset = extra["y_predset"]
    miscoverage = pset_metrics.miscoverage_overall_ps(y_predset, y_true_all)
    empirical_coverage = 1.0 - miscoverage
    coverage_gap = abs(alpha - miscoverage)
    avg_set_size = float(pset_metrics.size(y_predset))
    rejection_rate = float(pset_metrics.rejection_rate(y_predset))

    base_metrics = get_metrics_fn(mode)(y_true_all, y_prob_all, metrics=["accuracy"])
    accuracy = base_metrics.get("accuracy", float("nan"))

    return {
        "alpha": alpha,
        "empirical_coverage": round(float(empirical_coverage), 4),
        "target_coverage": round(1.0 - alpha, 4),
        "coverage_gap": round(float(coverage_gap), 4),
        "avg_set_size": round(avg_set_size, 4),
        "rejection_rate": round(rejection_rate, 4),
        "accuracy": round(float(accuracy), 4),
        "n_test": int(len(y_true_all)),
        "n_cal": None,
    }


def print_results_table(all_results: List[Dict[str, Any]], task: str, method: str) -> None:
    _print_section("Results")
    header = (
        f"  {'alpha':>6}  {'target_cov':>10}  {'emp_cov':>8}  "
        f"{'cov_gap':>8}  {'set_size':>9}  {'reject%':>8}  {'accuracy':>9}"
    )
    print(f"\n  Task: {task}   Method: {method}")
    print(f"  Protocol: splits={PROTOCOL['split_ratios']}  seed={PROTOCOL['split_seed']}")
    print()
    print(header)
    print("  " + "-" * (len(header) - 2))
    for r in all_results:
        flag = " ✓" if r["coverage_gap"] <= 0.01 else (" ✗" if r["coverage_gap"] > 0.05 else "  ")
        print(
            f"  {r['alpha']:>6.2f}  {r['target_coverage']:>10.4f}  "
            f"{r['empirical_coverage']:>8.4f}  {r['coverage_gap']:>8.4f}  "
            f"{r['avg_set_size']:>9.4f}  {r['rejection_rate']:>8.4f}  "
            f"{r['accuracy']:>9.4f}{flag}"
        )
    print()
    print("  ✓ = coverage_gap ≤ 0.01   ✗ = coverage_gap > 0.05")
    print(f"  n_test = {all_results[0]['n_test']}   n_cal = {all_results[0]['n_cal']}")


def save_results(all_results: List[Dict[str, Any]], task: str, method: str, output_path: str) -> None:
    record = {
        "cpbench_protocol": PROTOCOL,
        "task": task,
        "method": method,
        "method_paper": METHOD_REGISTRY[method]["paper"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "results": all_results,
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Results saved → {output_path}")


def run_benchmark(
    task: str,
    method: str,
    data_path: str,
    alphas: List[float],
    seed: int,
    epochs: int,
    checkpoint: Optional[str],
    output_path: Optional[str],
    dev: bool,
    output_dir: str,
) -> List[Dict[str, Any]]:
    task_cfg = TASK_REGISTRY[task]
    method_cfg = METHOD_REGISTRY[method]
    batch_size = PROTOCOL["batch_size"]
    ratios = PROTOCOL["split_ratios"]

    _print_header(f"CPBench  |  task: {task}  |  method: {method}")

    _print_section("Loading dataset")
    print(f"  {task_cfg['description']}")
    print(f"  data_path = {data_path}")
    dataset = load_dataset(task_cfg, data_path, dev)
    print(f"  Dataset size: {len(dataset)} samples")

    _print_section("Splitting dataset (fixed protocol)")
    print(f"  ratios={ratios}  seed={seed}")
    train_data, val_data, cal_data, test_data = split_dataset(dataset, task_cfg, seed, ratios)
    print(
        f"  train={len(train_data)}  val={len(val_data)}  "
        f"cal={len(cal_data)}  test={len(test_data)}"
    )
    n_cal = len(cal_data)

    _print_section("Base model")
    print(f"  {task_cfg['model_class']} (mode={task_cfg['mode']})")
    model = build_model(task_cfg, dataset, checkpoint)

    if checkpoint is None:
        _print_section(f"Training base model ({epochs} epochs)")
        t0 = time.time()
        model = train_model(model, train_data, val_data, task_cfg, epochs, batch_size, output_dir)
        print(f"  Training time: {time.time() - t0:.1f}s")
    else:
        print(f"  Skipping training (checkpoint provided)")

    model.to(_device())
    model.eval()

    all_results: List[Dict[str, Any]] = []

    for alpha in alphas:
        _print_section(f"Conformal method: {method}  (α={alpha})")
        print(f"  {method_cfg['description']}")
        print(f"  Calibrating on {n_cal} samples...")

        t0 = time.time()
        try:
            cp_model = build_cp_model(
                method_cfg, model, alpha, cal_data, test_data, batch_size, dev
            )
        except TypeError:
            cp_model = _import(method_cfg["module"], method_cfg["class"])(
                model=model,
                alpha=alpha,
                debug=dev,
                **method_cfg.get("extra_kwargs", {}),
            )
            cp_model.calibrate(cal_dataset=cal_data)

        cp_model.to(_device())
        cal_time = time.time() - t0
        print(f"  Calibration time: {cal_time:.1f}s")

        print(f"  Evaluating on {len(test_data)} test samples...")
        result = evaluate(cp_model, test_data, alpha, batch_size, task_cfg["mode"])
        result["n_cal"] = n_cal
        result["cal_time_s"] = round(cal_time, 2)
        all_results.append(result)

    print_results_table(all_results, task, method)

    if output_path:
        save_results(all_results, task, method, output_path)

    return all_results


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
        "--seed",
        type=int,
        default=PROTOCOL["split_seed"],
        help=f"Split seed. Default: {PROTOCOL['split_seed']}. Override only for sensitivity analysis.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs. Defaults to per-task default. Ignored when --checkpoint is given.",
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Pre-trained model checkpoint (.pth). Skips training.")
    parser.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    parser.add_argument("--output-dir", type=str, default="./cpbench_runs", help="Directory for Trainer logs.")
    parser.add_argument("--dev", action="store_true", help="Dev mode: tiny dataset subset, single alpha (0.10).")

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

    if args.alpha is not None:
        alphas = args.alpha
        if alphas != PROTOCOL["standard_alphas"]:
            print(
                f"  [cpbench] Note: using non-standard alpha={alphas}. "
                f"Published results must include all of {PROTOCOL['standard_alphas']}."
            )
    elif args.dev:
        alphas = [0.10]
    else:
        alphas = PROTOCOL["standard_alphas"]

    task_cfg = TASK_REGISTRY[args.task]
    epochs = args.epochs if args.epochs is not None else task_cfg.get("default_epochs", 20)

    os.makedirs(args.output_dir, exist_ok=True)

    run_benchmark(
        task=args.task,
        method=args.method,
        data_path=args.data_path,
        alphas=alphas,
        seed=args.seed,
        epochs=epochs,
        checkpoint=args.checkpoint,
        output_path=args.output,
        dev=args.dev,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
