"""
CPBench: a generic conformal-prediction benchmark harness for PyHealth.

Rather than a curated, hand-maintained list of pre-wired (dataset, task,
model, CP method) combinations, CPBench lets you name ANY BaseDataset
subclass, ANY BaseTask subclass, ANY BaseModel subclass, and ANY
SetPredictor (conformal prediction-set method) subclass already available
in pyhealth, and it trains + calibrates + evaluates that exact combination.
Adding a new dataset, task, model, or CP method to pyhealth automatically
makes it available here too -- nothing in this file needs to change, as
long as the new piece follows pyhealth's existing BaseDataset/BaseTask/
BaseModel/SetPredictor conventions.

Usage:
    List everything available:
        python benchmark.py --list

    Full benchmark run:
        python benchmark.py \\
            --dataset TUEVDataset --dataset-root /path/to/tuev \\
            --task EEGEventsTUEV \\
            --model ContraWR \\
            --cp-method LABEL \\
            --alpha 0.05 0.1 --seeds 42 43 --epochs 30

    Quick test (tiny data subset, single seed/alpha, 2 training epochs):
        python benchmark.py \\
            --dataset TUEVDataset --dataset-root /path/to/tuev \\
            --task EEGEventsTUEV --model ContraWR --cp-method LABEL --quick-test

    Dry run (resolve and validate the combination without touching data):
        python benchmark.py \\
            --dataset TUEVDataset --task EEGEventsTUEV \\
            --model ContraWR --cp-method LABEL --dry-run

    A dataset/task/model needing extra constructor args (beyond `root=`,
    no-args, or `dataset=`) can be given them via JSON passthrough, e.g.:
        --dataset-kwargs '{"subset": "cassette"}'
        --task-kwargs '{"chunk_duration": 30.0}'
        --model-kwargs '{"embedding_dim": 256}'
        --cp-kwargs '{"n_clusters": 5}'
"""

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

PROTOCOL = {
    "version": "v2",
    "split_ratios": [0.60, 0.15, 0.10, 0.15],
    "seeds": [42, 43, 44, 45, 46],
    "batch_size": 32,
    "standard_alphas": [0.01, 0.05, 0.10, 0.20],
}

# --quick-test overrides, matching the convention already established in
# examples/conformal_eeg/tuev_conventional_conformal.py.
QUICK_TEST_EPOCHS = 2
QUICK_TEST_MAX_SAMPLES = 2000


# --------------------------------------------------------------------------
# Discovery: enumerate everything CPBench can plug together, straight from
# pyhealth's own public API. Nothing here is a hand-maintained registry --
# if a class is exported from the relevant pyhealth package and subclasses
# the right base class, it is automatically available.
# --------------------------------------------------------------------------

def _discover(package_name: str, base_class_path: str) -> Dict[str, type]:
    """Return {class_name: class} for every public class in `package_name`
    that subclasses the class at `base_class_path` ("module.ClassName"),
    excluding the base class itself."""
    module = importlib.import_module(package_name)
    base_module_name, base_class_name = base_class_path.rsplit(".", 1)
    base_class = getattr(importlib.import_module(base_module_name), base_class_name)

    found = {}
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        if (
            inspect.isclass(obj)
            and issubclass(obj, base_class)
            and obj is not base_class
        ):
            found[name] = obj
    return found


def discover_datasets() -> Dict[str, type]:
    return _discover("pyhealth.datasets", "pyhealth.datasets.BaseDataset")


def discover_tasks() -> Dict[str, type]:
    return _discover("pyhealth.tasks", "pyhealth.tasks.BaseTask")


def discover_models() -> Dict[str, type]:
    return _discover("pyhealth.models", "pyhealth.models.BaseModel")


def discover_cp_methods() -> Dict[str, type]:
    return _discover(
        "pyhealth.calib.predictionset", "pyhealth.calib.base_classes.SetPredictor"
    )


def _resolve(registry: Dict[str, type], name: str, kind: str) -> type:
    if name not in registry:
        choices = ", ".join(sorted(registry.keys())) or "(none found)"
        raise ValueError(
            f"Unknown {kind} '{name}'.\nAvailable {kind}s: {choices}\n"
            "(Run with --list to see this with descriptions.)"
        )
    return registry[name]


# --------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _print_header(title: str) -> None:
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_section(title: str) -> None:
    print(f"\n── {title} " + "─" * max(0, 68 - len(title)))


def _parse_kwargs_json(raw: Optional[str], flag_name: str) -> Dict[str, Any]:
    if raw is None:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{flag_name} must be a valid JSON object: {e}") from e
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must be a JSON object, got {type(parsed).__name__}")
    return parsed


# --------------------------------------------------------------------------
# Generic construction helpers -- each one introspects the target class's
# constructor rather than assuming a fixed argument set, and raises a clear,
# actionable error (pointing at the relevant --*-kwargs flag) if the given
# combination genuinely doesn't fit together, instead of guessing.
# --------------------------------------------------------------------------

def build_dataset(
    dataset_cls: type,
    root: Optional[str],
    quick_test: bool,
    dataset_kwargs: Dict[str, Any],
) -> Any:
    ctor_params = inspect.signature(dataset_cls.__init__).parameters
    accepts_catchall_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in ctor_params.values()
    )

    kwargs = dict(dataset_kwargs)
    if root is not None:
        kwargs.setdefault("root", root)
    # Datasets that support a tiny-subset mode do so via a constructor kwarg
    # literally named `dev` (a pyhealth-wide convention) -- unrelated to
    # CPBench's own --quick-test flag name.
    if "dev" not in kwargs:
        if "dev" in ctor_params or accepts_catchall_kwargs:
            kwargs["dev"] = quick_test
        elif quick_test:
            print(
                f"  Note: {dataset_cls.__name__} does not support a 'dev' "
                "flag; loading the full dataset (still capped to "
                f"{QUICK_TEST_MAX_SAMPLES} samples after task processing)."
            )

    try:
        return dataset_cls(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {dataset_cls.__name__}(**{kwargs}): {e}\n"
            "Pass extra/renamed constructor args via --dataset-kwargs "
            "(a JSON object)."
        ) from e


def build_task(task_cls: type, task_kwargs: Dict[str, Any]) -> Any:
    try:
        return task_cls(**task_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {task_cls.__name__}(**{task_kwargs}): {e}\n"
            "Pass its constructor args via --task-kwargs (a JSON object)."
        ) from e


def build_model(model_cls: type, dataset: Any, model_kwargs: Dict[str, Any]) -> Any:
    try:
        return model_cls(dataset=dataset, **model_kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {model_cls.__name__}(dataset=..., "
            f"**{model_kwargs}): {e}\n"
            "Pass extra constructor args via --model-kwargs (a JSON object)."
        ) from e


def split_dataset(
    dataset: Any, split_strategy: str, seed: int, ratios: List[float]
) -> Tuple[Any, Any, Any, Any]:
    from pyhealth.datasets import split_by_patient_conformal, split_by_sample_conformal

    if split_strategy == "by_patient":
        return split_by_patient_conformal(dataset, ratios=ratios, seed=seed)
    elif split_strategy == "by_sample":
        return split_by_sample_conformal(dataset, ratios=ratios, seed=seed)
    else:
        raise ValueError(
            f"Unknown split strategy: {split_strategy!r}. Use 'by_patient' "
            "or 'by_sample' (--split-strategy)."
        )


def split_dataset_with_quick_test_fallback(
    dataset: Any, split_strategy: str, seed: int, ratios: List[float], quick_test: bool
) -> Tuple[Any, Any, Any, Any]:
    """Splits the dataset, falling back to a by-sample split under
    --quick-test if the requested strategy emptied any partition.

    A patient-level split (the default) can leave a partition empty once the
    dataset has been capped down to QUICK_TEST_MAX_SAMPLES, since patients
    aren't evenly represented in a small, arbitrary sample slice. A
    by-sample split doesn't depend on patient-level structure, so it's a
    reasonable fallback to keep --quick-test runnable on any dataset,
    mirroring the same fallback already used in
    examples/conformal_eeg/tuev_conventional_conformal.py.
    """
    train_data, val_data, cal_data, test_data = split_dataset(
        dataset, split_strategy, seed, ratios
    )
    if quick_test and split_strategy == "by_patient" and any(
        len(part) == 0 for part in (train_data, val_data, cal_data, test_data)
    ):
        print(
            "  [quick-test] A partition was empty after 'by_patient' "
            "splitting on the capped subset -- falling back to "
            "'by_sample' splitting for this run."
        )
        train_data, val_data, cal_data, test_data = split_dataset(
            dataset, "by_sample", seed, ratios
        )
    return train_data, val_data, cal_data, test_data


def train_model(
    model: Any,
    train_data: Any,
    val_data: Any,
    epochs: int,
    batch_size: int,
    output_dir: str,
    seed: int,
    monitor: Optional[str],
    monitor_criterion: str,
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
        monitor=monitor,
        monitor_criterion=monitor_criterion,
        load_best_model_at_last=(monitor is not None),
    )
    return model


def construct_cp_predictor(
    cp_cls: type,
    base_model: Any,
    alpha: Optional[float],
    cp_kwargs: Dict[str, Any],
    quick_test: bool,
) -> Tuple[Any, bool]:
    """Construct (but do not calibrate) any SetPredictor generically, with
    clear error wrapping.

    Construction only needs `base_model.mode` (set synchronously in
    BaseModel.__init__ from the dataset's schema) -- it does not require the
    model to be trained. This lets callers validate a (model, CP method)
    pairing cheaply, before paying for training.

    Not every CP method is parameterized by a scalar/array `alpha` (e.g.
    SCRIB uses `risk`, FavMac uses `target_cost`/`delta`). Returns
    (cp_model, used_alpha): used_alpha is False when the class has no
    `alpha` constructor parameter, in which case its real parameters must be
    supplied via --cp-kwargs.
    """
    ctor_params = inspect.signature(cp_cls.__init__).parameters
    kwargs = dict(cp_kwargs)
    used_alpha = "alpha" in ctor_params
    if used_alpha and "alpha" not in kwargs:
        kwargs["alpha"] = alpha
    # CP methods that support a verbose/debug mode do so via a constructor
    # kwarg literally named `debug` -- unrelated to CPBench's --quick-test.
    if "debug" in ctor_params and "debug" not in kwargs:
        kwargs["debug"] = quick_test

    try:
        cp_model = cp_cls(model=base_model, **kwargs)
    except NotImplementedError as e:
        raise NotImplementedError(
            f"{cp_cls.__name__}(model=..., **{kwargs}) failed to construct "
            f"for a model with mode='{base_model.mode}': {e}\n"
            "This usually means the CP method's required classification "
            "mode (see its docstring) doesn't match the task's output mode."
        ) from e
    except TypeError as e:
        raise TypeError(
            f"Failed to construct {cp_cls.__name__}(model=..., **{kwargs}): "
            f"{e}\nCheck its constructor signature (see its docstring) and "
            "pass required args via --cp-kwargs (a JSON object)."
        ) from e

    return cp_model, used_alpha


def build_cp_model(
    cp_cls: type,
    base_model: Any,
    alpha: Optional[float],
    cp_kwargs: Dict[str, Any],
    train_data: Any,
    cal_data: Any,
    test_data: Any,
    quick_test: bool,
) -> Tuple[Any, bool]:
    """Construct and calibrate any SetPredictor generically.

    Every SetPredictor in pyhealth.calib.predictionset is expected to accept
    calibrate(cal_dataset, train_dataset=None, test_dataset=None) -- see
    pyhealth.calib.base_classes.SetPredictor -- so this function never needs
    to know which extra data a particular CP method needs; each method
    extracts whatever it needs internally.
    """
    cp_model, used_alpha = construct_cp_predictor(
        cp_cls, base_model, alpha, cp_kwargs, quick_test
    )
    cp_model.calibrate(
        cal_dataset=cal_data, train_dataset=train_data, test_dataset=test_data
    )
    return cp_model, used_alpha


def evaluate(
    cp_model: Any,
    test_data: Any,
    alpha: Optional[float],
    used_alpha: bool,
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

    # These are computed purely from the output prediction sets, so they're
    # meaningful for every SetPredictor regardless of its calibration target.
    base_metrics = get_metrics_fn(mode)(y_true_all, y_prob_all, metrics=["accuracy"])
    result: Dict[str, Any] = {
        "avg_set_size": round(float(pset_metrics.size(y_predset)), 4),
        "rejection_rate": round(float(pset_metrics.rejection_rate(y_predset)), 4),
        "accuracy": round(float(base_metrics.get("accuracy", float("nan"))), 4),
        "n_test": int(len(y_true_all)),
        "n_cal": None,
    }

    # Coverage-vs-target reporting only makes sense for methods parameterized
    # by a scalar miscoverage rate alpha.
    if used_alpha and alpha is not None:
        miscoverage = pset_metrics.miscoverage_overall_ps(y_predset, y_true_all)
        result["alpha"] = alpha
        result["target_coverage"] = round(1.0 - alpha, 4)
        result["empirical_coverage"] = round(float(1.0 - miscoverage), 4)
        result["coverage_gap"] = round(float(abs(alpha - miscoverage)), 4)

    return result


def aggregate_results(per_seed_results: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Aggregate mean/std across seeds for each alpha (or the single
    no-alpha run, for methods without a scalar alpha)."""
    n_rows = len(per_seed_results[0])
    aggregated = []
    for i in range(n_rows):
        rows = [seed_rows[i] for seed_rows in per_seed_results]
        row: Dict[str, Any] = {
            "n_seeds": len(rows),
            "n_test": rows[0]["n_test"],
            "n_cal": rows[0]["n_cal"],
        }
        if "alpha" in rows[0]:
            row["alpha"] = rows[0]["alpha"]
            row["target_coverage"] = rows[0]["target_coverage"]

        metric_names = [
            m for m in (
                "empirical_coverage", "coverage_gap",
                "avg_set_size", "rejection_rate", "accuracy",
            )
            if m in rows[0]
        ]
        for metric in metric_names:
            vals = [r[metric] for r in rows]
            row[f"{metric}_mean"] = round(float(np.mean(vals)), 4)
            row[f"{metric}_std"] = round(float(np.std(vals)), 4)
        aggregated.append(row)
    return aggregated


def print_aggregated_table(
    aggregated: List[Dict[str, Any]],
    dataset_name: str,
    task_name: str,
    model_name: str,
    cp_method_name: str,
    seeds: List[int],
) -> None:
    _print_section("Results")
    print(f"\n  Dataset={dataset_name}  Task={task_name}  Model={model_name}  CP method={cp_method_name}")
    print(f"  Protocol: splits={PROTOCOL['split_ratios']}  seeds={seeds}")
    print(f"  Reporting mean ± std across {len(seeds)} seeds\n")

    has_alpha = "alpha" in aggregated[0]
    col = 16

    if has_alpha:
        header = (
            f"  {'alpha':>6}  {'target_cov':>10}  "
            f"{'emp_cov':<{col}}  {'cov_gap':<{col}}  "
            f"{'set_size':<{col}}  {'reject%':<{col}}  {'accuracy':<{col}}"
        )
    else:
        header = (
            f"  {'set_size':<{col}}  {'reject%':<{col}}  {'accuracy':<{col}}"
        )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in aggregated:
        def fmt(m: str) -> str:
            return f"{r[f'{m}_mean']:.4f} ± {r[f'{m}_std']:.4f}"

        if has_alpha:
            gap_mean = r["coverage_gap_mean"]
            flag = " ✓" if gap_mean <= 0.01 else (" ✗" if gap_mean > 0.05 else "  ")
            print(
                f"  {r['alpha']:>6.2f}  {r['target_coverage']:>10.4f}  "
                f"{fmt('empirical_coverage'):<{col}}  {fmt('coverage_gap'):<{col}}  "
                f"{fmt('avg_set_size'):<{col}}  {fmt('rejection_rate'):<{col}}  "
                f"{fmt('accuracy'):<{col}}{flag}"
            )
        else:
            print(
                f"  {fmt('avg_set_size'):<{col}}  {fmt('rejection_rate'):<{col}}  "
                f"{fmt('accuracy'):<{col}}"
            )

    print()
    if has_alpha:
        print("  ✓ = mean coverage_gap ≤ 0.01   ✗ = mean coverage_gap > 0.05")
    print(f"  n_test ≈ {aggregated[0]['n_test']}   n_cal ≈ {aggregated[0]['n_cal']}")


def save_results(
    aggregated: List[Dict[str, Any]],
    per_seed_results: List[List[Dict[str, Any]]],
    seeds: List[int],
    dataset_name: str,
    task_name: str,
    model_name: str,
    cp_method_name: str,
    output_path: str,
) -> None:
    record = {
        "cpbench_protocol": PROTOCOL,
        "dataset": dataset_name,
        "task": task_name,
        "model": model_name,
        "cp_method": cp_method_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "seeds": seeds,
        "aggregated": aggregated,
        "per_seed": {str(seed): per_seed_results[i] for i, seed in enumerate(seeds)},
    }
    with open(output_path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\n  Results saved → {output_path}")


def run_benchmark(args: argparse.Namespace) -> List[Dict[str, Any]]:
    datasets = discover_datasets()
    tasks = discover_tasks()
    models = discover_models()
    cp_methods = discover_cp_methods()

    DatasetClass = _resolve(datasets, args.dataset, "dataset")
    TaskClass = _resolve(tasks, args.task, "task")
    ModelClass = _resolve(models, args.model, "model")
    CPClass = _resolve(cp_methods, args.cp_method, "CP method")

    dataset_kwargs = _parse_kwargs_json(args.dataset_kwargs, "--dataset-kwargs")
    task_kwargs = _parse_kwargs_json(args.task_kwargs, "--task-kwargs")
    model_kwargs = _parse_kwargs_json(args.model_kwargs, "--model-kwargs")
    cp_kwargs = _parse_kwargs_json(args.cp_kwargs, "--cp-kwargs")

    if args.dry_run:
        _print_header(
            f"CPBench dry-run  |  {args.dataset} + {args.task} + {args.model} "
            f"+ {args.cp_method}"
        )
        print(f"  dataset kwargs: {dataset_kwargs}")
        print(f"  task kwargs:    {task_kwargs}")
        print(f"  model kwargs:   {model_kwargs}")
        print(f"  cp kwargs:      {cp_kwargs}")
        print("\n  All four classes resolved successfully. Re-run without "
              "--dry-run to actually train/calibrate/evaluate.")
        return []

    batch_size = PROTOCOL["batch_size"]
    ratios = PROTOCOL["split_ratios"]

    _print_header(
        f"CPBench  |  dataset: {args.dataset}  |  task: {args.task}  |  "
        f"model: {args.model}  |  cp-method: {args.cp_method}"
    )
    print(f"  Seeds: {args.seeds}   Alphas: {args.alpha}")

    _print_section("Loading dataset + task")
    print(f"  data_path = {args.dataset_root}")
    dataset = build_dataset(
        DatasetClass, args.dataset_root, args.quick_test, dataset_kwargs
    )
    task = build_task(TaskClass, task_kwargs)
    dataset = dataset.set_task(task)
    print(f"  Dataset size: {len(dataset)} samples")

    if args.quick_test and len(dataset) > QUICK_TEST_MAX_SAMPLES:
        dataset = dataset.subset(range(QUICK_TEST_MAX_SAMPLES))
        print(f"  [quick-test] Capped to {QUICK_TEST_MAX_SAMPLES} samples.")

    _print_section("Validating model / cp-method compatibility")
    # Construction only needs model.mode (set from the dataset's schema, not
    # from trained weights) -- so an incompatible (model, cp-method) pairing
    # (e.g. a multilabel-only CP method against a multiclass task) is caught
    # here, before spending time training, rather than only surfacing after
    # every seed has already finished training.
    probe_model = build_model(ModelClass, dataset, model_kwargs)
    probe_alpha = args.alpha[0] if args.alpha else 0.1
    construct_cp_predictor(CPClass, probe_model, probe_alpha, cp_kwargs, args.quick_test)
    print(f"  model.mode='{probe_model.mode}' is compatible with cp-method '{args.cp_method}'")
    del probe_model

    epochs = QUICK_TEST_EPOCHS if args.quick_test else args.epochs

    per_seed_results: List[List[Dict[str, Any]]] = []

    for seed in args.seeds:
        _print_header(f"Seed {seed}  ({args.seeds.index(seed) + 1}/{len(args.seeds)})")

        _print_section("Splitting")
        print(f"  ratios={ratios}  seed={seed}  strategy={args.split_strategy}")
        train_data, val_data, cal_data, test_data = split_dataset_with_quick_test_fallback(
            dataset, args.split_strategy, seed, ratios, args.quick_test
        )
        print(
            f"  train={len(train_data)}  val={len(val_data)}  "
            f"cal={len(cal_data)}  test={len(test_data)}"
        )
        n_cal = len(cal_data)

        _print_section(f"Training {args.model} ({epochs} epochs)")
        model = build_model(ModelClass, dataset, model_kwargs)
        t0 = time.time()
        model = train_model(
            model, train_data, val_data, epochs, batch_size, args.output_dir,
            seed, args.monitor, args.monitor_criterion,
        )
        print(f"  Training time: {time.time() - t0:.1f}s")
        model.to(_device())
        model.eval()

        seed_rows: List[Dict[str, Any]] = []
        alphas = args.alpha

        for alpha in alphas:
            _print_section(f"α={alpha}  cp-method={args.cp_method}")
            print(f"  Calibrating on {n_cal} samples...")

            t0 = time.time()
            cp_model, used_alpha = build_cp_model(
                CPClass, model, alpha, cp_kwargs, train_data, cal_data, test_data,
                args.quick_test,
            )
            cp_model.to(_device())
            cal_time = time.time() - t0
            print(f"  Calibration: {cal_time:.1f}s  |  Evaluating on {len(test_data)} samples...")

            result = evaluate(cp_model, test_data, alpha, used_alpha, batch_size, model.mode)
            result["n_cal"] = n_cal
            result["cal_time_s"] = round(cal_time, 2)
            result["seed"] = seed
            seed_rows.append(result)

            if not used_alpha:
                # This CP method has no scalar alpha to sweep over -- one run
                # per seed is all there is; skip the rest of the alpha list.
                if len(alphas) > 1:
                    print(
                        f"  Note: {args.cp_method} has no 'alpha' constructor "
                        "parameter; ignoring the remaining --alpha values "
                        "for this method."
                    )
                break

        per_seed_results.append(seed_rows)

    aggregated = aggregate_results(per_seed_results)
    print_aggregated_table(
        aggregated, args.dataset, args.task, args.model, args.cp_method, args.seeds
    )

    if args.output:
        save_results(
            aggregated, per_seed_results, args.seeds,
            args.dataset, args.task, args.model, args.cp_method, args.output,
        )

    return aggregated


def list_registry() -> None:
    datasets = discover_datasets()
    tasks = discover_tasks()
    models = discover_models()
    cp_methods = discover_cp_methods()

    def _first_line(cls: type) -> str:
        doc = (cls.__doc__ or "").strip()
        return doc.splitlines()[0].strip() if doc else ""

    for title, registry in (
        ("Datasets (pyhealth.datasets, BaseDataset subclasses)", datasets),
        ("Tasks (pyhealth.tasks, BaseTask subclasses)", tasks),
        ("Models (pyhealth.models, BaseModel subclasses)", models),
        ("CP methods (pyhealth.calib.predictionset, SetPredictor subclasses)", cp_methods),
    ):
        print(f"\n{title}:")
        print("-" * 70)
        for name in sorted(registry):
            print(f"  {name:<30}  {_first_line(registry[name])}")
    print(
        "\nNote: only classes that already subclass BaseDataset/BaseTask/"
        "BaseModel/SetPredictor show up here. Legacy pre-2.0 datasets/tasks "
        "not yet migrated to this API (e.g. classes still built on the "
        "deprecated BaseSignalDataset/BaseEHRDataset stubs) are intentionally "
        "excluded -- they are not usable with the current set_task()/model "
        "API regardless of CPBench.\n"
        "Not every dataset/task pairing is guaranteed to be compatible "
        "(pyhealth has no dataset<->task compatibility registry yet); an "
        "incompatible pairing will fail with an error when you run it.\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CPBench: generic conformal-prediction benchmarking for PyHealth. "
        "Name any dataset/task/model/CP-method already in pyhealth.",
    )

    parser.add_argument("--list", action="store_true", help="List all available datasets, tasks, models, and CP methods, then exit.")
    parser.add_argument("--dataset", type=str, help="Dataset class name (see --list), e.g. TUEVDataset.")
    parser.add_argument("--dataset-root", type=str, default=None, help="Path to the raw dataset on disk (passed as root=).")
    parser.add_argument("--dataset-kwargs", type=str, default=None, help="JSON object of extra dataset constructor kwargs.")
    parser.add_argument("--task", type=str, help="Task class name (see --list), e.g. EEGEventsTUEV.")
    parser.add_argument("--task-kwargs", type=str, default=None, help="JSON object of task constructor kwargs.")
    parser.add_argument("--model", type=str, help="Model class name (see --list), e.g. ContraWR.")
    parser.add_argument("--model-kwargs", type=str, default=None, help="JSON object of extra model constructor kwargs.")
    parser.add_argument("--cp-method", type=str, help="CP method class name (see --list), e.g. LABEL.")
    parser.add_argument("--cp-kwargs", type=str, default=None, help="JSON object of extra CP-method constructor kwargs (required for methods with no 'alpha' param, e.g. SCRIB's risk=, FavMac's target_cost=/delta=).")
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=None,
        help=f"Miscoverage rate(s), for CP methods parameterized by alpha. Defaults to standard set {PROTOCOL['standard_alphas']}.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help=f"Seeds for splitting. Defaults to {PROTOCOL['seeds']}.",
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs. Default 20.")
    parser.add_argument("--split-strategy", type=str, default="by_patient", choices=["by_patient", "by_sample"], help="Split strategy. Default by_patient.")
    parser.add_argument("--monitor", type=str, default="accuracy", help="Validation metric to monitor for checkpoint selection. Default accuracy.")
    parser.add_argument("--monitor-criterion", type=str, default="max", choices=["max", "min"], help="Whether higher or lower --monitor is better. Default max.")
    parser.add_argument("--output", type=str, default=None, help="Path to write results JSON.")
    parser.add_argument("--output-dir", type=str, default="./cpbench_runs", help="Directory for Trainer logs.")
    parser.add_argument(
        "--quick-test", dest="quick_test", action="store_true",
        help=(
            "Quick test: single seed (42), single alpha (0.10), "
            f"{QUICK_TEST_EPOCHS} training epochs, dataset's own tiny-subset "
            f"mode if supported, and the resulting sample-processed dataset "
            f"further capped to {QUICK_TEST_MAX_SAMPLES} samples. Falls back "
            "to a by-sample split if by-patient splitting empties a "
            "partition on the capped data."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Resolve and validate the dataset/task/model/cp-method combination without loading data or training.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        list_registry()
        sys.exit(0)

    missing = [f for f in ("dataset", "task", "model", "cp_method") if not getattr(args, f, None)]
    if missing:
        print(f"Error: --{', --'.join(m.replace('_', '-') for m in missing)} required. Run --list to see options.")
        sys.exit(1)

    if args.quick_test:
        args.alpha = [0.10]
        args.seeds = [42]
    else:
        args.alpha = args.alpha if args.alpha is not None else PROTOCOL["standard_alphas"]
        args.seeds = args.seeds if args.seeds is not None else PROTOCOL["seeds"]

    os.makedirs(args.output_dir, exist_ok=True)

    run_benchmark(args)


if __name__ == "__main__":
    main()
