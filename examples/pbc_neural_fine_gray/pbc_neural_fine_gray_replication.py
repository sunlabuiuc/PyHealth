"""Run PBC Neural Fine-Gray with paired missingness and architecture ablations.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
Install optional requirements-neural-fine-gray.txt for comparators/reference
metrics. See README.md for data access, commands, and the study protocol.
"""

import argparse
import json
import math
import platform
import time
from collections.abc import Sequence
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pbc_evaluation as evaluation
import pbc_experiment as experiment
import torch
from pbc_experiment import (
    COX_PENALTIES,
    DROPOUTS,
    HORIZONS,
    MODES,
    PAPER_HORIZONS,
    RATES,
    WIDTHS,
)
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from pyhealth.tasks import CompetingRisksPreprocessor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Read experiment options and reject invalid budgets before data loading.

    Args:
        argv: Command-line arguments, or None to read the process arguments.

    Returns:
        Parsed data source, output directory, device, seed, and training budget.

    Raises:
        SystemExit: Required options are absent or a budget/device is invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--synthetic", action="store_true")
    data.add_argument("--data-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--historical", action="store_true")
    parser.add_argument(
        "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--sensitivity-seeds",
        type=int,
        default=0,
        help="Number of fresh matched training seeds; 0 disables sensitivity.",
    )
    parser.add_argument(
        "--paper-time",
        action="store_true",
        help="Also score 2.19, 3.95 and 6.45 unshifted years.",
    )
    parser.add_argument(
        "--architecture-ablation",
        action="store_true",
        help="Vary widths and dropout with the other selected settings frozen.",
    )
    parser.add_argument(
        "--cox-penalties",
        type=float,
        nargs="+",
        default=list(COX_PENALTIES),
        help="Shared L2 candidates selected on inner-validation likelihood.",
    )
    args = parser.parse_args(argv)
    if args.folds < 2 or min(args.trials, args.epochs, args.patience) < 1:
        parser.error("Require folds >= 2 and positive trials, epochs, and patience")
    if not 0 <= args.seed < 2**32 or args.sensitivity_seeds < 0:
        parser.error("Require seed in [0, 2**32) and sensitivity-seeds >= 0")
    if len(set(args.cox_penalties)) != len(args.cox_penalties) or any(
        not math.isfinite(penalty) or penalty <= 0 for penalty in args.cox_penalties
    ):
        parser.error("Require distinct positive finite Cox penalties")
    if args.device != "cpu":
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            parser.error("Requested device is unavailable; use cpu or available cuda")
        if device.index is not None and device.index >= torch.cuda.device_count():
            parser.error("Requested CUDA index is unavailable")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the PBC pipeline with patient-level folds and paired ablations.

    The original arm selects its configuration on inner-validation likelihood.
    The other arms reuse it. Test patients shape neither preprocessing nor
    model selection. A completion record is written only after the full metric
    grid has been checked.

    Args:
        argv: Command-line arguments, or None for the process arguments.

    Raises:
        ValueError: Cohort size, saved provenance, or metric rows are invalid.
        ImportError: Optional experiment dependencies are missing.
    """
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    samples, audit = experiment.load_samples(args)
    saved_manifest = args.output_dir / "manifest.json"
    previous_manifest = (
        json.loads(saved_manifest.read_text()) if saved_manifest.exists() else None
    )
    if previous_manifest is not None and previous_manifest["audit"] != audit:
        raise ValueError("Changed source data: use a new output directory")
    if args.audit_only:
        experiment.write_json(args.output_dir / "cohort_audit.json", audit)
        print(json.dumps(audit, indent=2), flush=True)
        return
    try:
        from lifelines import AalenJohansenFitter

        if args.historical:
            import sksurv.metrics  # noqa: F401
    except ImportError as error:
        raise ImportError(
            "Install examples/pbc_neural_fine_gray/requirements-neural-fine-gray.txt"
        ) from error
    root = Path(__file__).resolve().parents[2]
    manifest = {
        "dataset": "PBC",
        "audit": audit,
        "seed": args.seed,
        "folds": args.folds,
        "trials": args.trials,
        "epochs": args.epochs,
        "patience": args.patience,
        "device": args.device,
        "historical": args.historical,
        "float": "float64",
        "horizons_years": list(HORIZONS),
        "rates": list(RATES),
        "primary": "corrected five-year death Brier",
        "cox_penalties": args.cox_penalties,
        "cox_selection": "inner-validation partial log-likelihood; no refit",
        "sensitivity_training_seeds": [
            args.seed + offset + 1 for offset in range(args.sensitivity_seeds)
        ],
        "paper_horizons_years": list(PAPER_HORIZONS) if args.paper_time else [],
        "architecture_ablation": args.architecture_ablation,
        "architecture_widths": list(WIDTHS),
        "architecture_dropouts": list(DROPOUTS),
        "paper_commit": "8fcd60bd2c8a9306504f2fe674bf60703c4b2abf",
        "data_commit": "e4b07b3f497f2266eaa71d0e182195e95663d367",
        "python": platform.python_version(),
        "versions": {
            name: version(name)
            for name in (
                "torch",
                "pyhealth",
                "numpy",
                "pandas",
                "dask",
                "polars",
                "pyarrow",
                "PyYAML",
                "scikit-learn",
                "lifelines",
                *(("scikit-survival",) if args.historical else ()),
            )
        },
        "source_hashes": experiment.pipeline_source_hashes(root),
    }
    if previous_manifest is not None and previous_manifest != manifest:
        raise ValueError(
            "Changed source, data, or protocol: use a new output directory"
        )
    experiment.write_json(saved_manifest, manifest)
    experiment.write_json(args.output_dir / "cohort_audit.json", audit)
    print(json.dumps(audit, indent=2), flush=True)
    labels = np.asarray([s["event"] for s in samples])
    if np.bincount(labels, minlength=3).min() < args.folds:
        raise ValueError("Insufficient patients per event class for requested folds")
    grid = {
        "layers": [[width] * depth for width in (25, 50) for depth in (1, 2, 3, 4)],
        "layers_surv": [
            [width] * depth for width in (25, 50) for depth in (1, 2, 3, 4)
        ],
        "dropout": [0.0, 0.25, 0.5, 0.75],
        "learning_rate": [1e-3, 1e-4],
        "batch": [100, 250],
    }
    configs = list(ParameterSampler(grid, n_iter=args.trials, random_state=args.seed))
    rows, paper_rows, sensitivity_rows, architecture_rows = [], [], [], []
    architecture_fits = 0
    splitter = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for fold, (development, testing) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        directory = args.output_dir / f"fold-{fold}"
        directory.mkdir(exist_ok=True)
        training, validating = train_test_split(
            development,
            test_size=0.1,
            stratify=labels[development],
            random_state=args.seed + fold,
        )
        train, validation, test = [
            [samples[index] for index in indices]
            for indices in (training, validating, testing)
        ]
        experiment.write_json(
            directory / "private_splits.json",
            {
                name: [s["patient_id"] for s in group]
                for name, group in (
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                )
            },
        )
        processor = CompetingRisksPreprocessor().fit(train)
        experiment.write_json(directory / "preprocessing.json", processor.state)
        processed_train, processed_validation = (
            processor.transform(train),
            processor.transform(validation),
        )
        started = time.perf_counter()
        best_info = None
        for trial, config in enumerate(configs):
            _, info = experiment.fit_model(
                processed_train,
                processed_validation,
                config,
                "original",
                directory / f"trial-{trial}",
                args,
            )
            if (
                best_info is None
                or info["validation_nll_scaled"] < best_info["validation_nll_scaled"]
            ):
                best_info = info
            print(
                f"fold={fold} trial={trial} validation={info['validation_nll_scaled']:.6f}",
                flush=True,
            )
        if best_info is None:
            raise RuntimeError("No validated model selected")
        fold_rows = []
        for mode in MODES:
            if mode == "original":
                best_trial = next(
                    i
                    for i in range(len(configs))
                    if json.loads((directory / f"trial-{i}" / "fit.json").read_text())
                    == best_info
                )
                model, info = experiment.fit_model(
                    processed_train,
                    processed_validation,
                    best_info["identity"]["config"],
                    mode,
                    directory / f"trial-{best_trial}",
                    args,
                )
            else:
                model, info = experiment.fit_model(
                    processed_train,
                    processed_validation,
                    best_info["identity"]["config"],
                    mode,
                    directory / mode,
                    args,
                )
            destination = directory / mode
            destination.mkdir(exist_ok=True)
            if mode == "original":
                # Record the selected trial without creating another training fit.
                experiment.write_json(
                    destination / "selection.json",
                    {
                        "trial": best_trial,
                        "checkpoint_sha256": info["checkpoint_sha256"],
                    },
                )
            measured, supplementary = evaluation.score_neural(
                model,
                processor,
                train,
                test,
                fold,
                mode,
                destination,
                args.seed + 1000 + fold,
                paper_time=args.paper_time,
            )
            fold_rows.extend(measured)
            paper_rows.extend(supplementary)
            if mode == "original" and args.historical:
                with np.load(destination / "private_predictions_0.0.npz") as saved:
                    fold_rows.extend(
                        evaluation.historical_metrics(saved["cif"], train, test, fold)
                    )
            print(
                f"fold={fold} arm={mode} parameters={info['parameters']} complete",
                flush=True,
            )
        horizons = np.array(
            sorted({*HORIZONS, *(PAPER_HORIZONS if args.paper_time else ())})
        )
        primary_indices = np.searchsorted(horizons, HORIZONS)
        paper_indices = np.searchsorted(horizons, PAPER_HORIZONS)
        marginal = np.zeros((len(test), len(horizons), 2))
        for risk in (1, 2):
            estimator = AalenJohansenFitter(calculate_variance=False, seed=args.seed)
            estimator.fit(
                [s["time"] for s in train],
                [s["event"] for s in train],
                event_of_interest=risk,
            )
            marginal[:, :, risk - 1] = (
                estimator.cumulative_density_.reindex(
                    estimator.cumulative_density_.index.union(horizons)
                )
                .sort_index()
                .ffill()
                .reindex(horizons)
                .fillna(0)
                .to_numpy()
                .T
            )
        cox_models, columns, cox_selection = experiment.select_cox(
            processed_train, processed_validation, args.cox_penalties
        )
        experiment.write_json(directory / "cox_selection.json", cox_selection)
        for rate in RATES:
            fold_rows.extend(
                evaluation.evaluate(
                    marginal[:, primary_indices],
                    train,
                    test,
                    fold,
                    "aalen_johansen",
                    rate,
                )
            )
            processed = processor.transform(
                experiment.corrupt(test, rate, args.seed + 1000 + fold)
            )
            prediction = experiment.predict_cox(
                cox_models,
                columns,
                processed,
                horizons / processor.state["time_scale"],
            )
            fold_rows.extend(
                evaluation.evaluate(
                    prediction[:, primary_indices],
                    train,
                    test,
                    fold,
                    "cause_specific_cox",
                    rate,
                )
            )
            if args.paper_time:
                for label, curves in (
                    ("aalen_johansen", marginal),
                    ("cause_specific_cox", prediction),
                ):
                    paper_rows.extend(
                        evaluation.evaluate(
                            curves[:, paper_indices],
                            train,
                            test,
                            fold,
                            label,
                            rate,
                            horizons=PAPER_HORIZONS,
                        )
                    )
        selected_config = best_info["identity"]["config"]
        for seed in manifest["sensitivity_training_seeds"]:
            for mode in MODES:
                destination = directory / "seed_sensitivity" / f"seed-{seed}" / mode
                model, _ = experiment.fit_model(
                    processed_train,
                    processed_validation,
                    selected_config,
                    mode,
                    destination,
                    args,
                    training_seed=seed,
                )
                measured, _ = evaluation.score_neural(
                    model,
                    processor,
                    train,
                    test,
                    fold,
                    mode,
                    destination,
                    args.seed + 1000 + fold,
                )
                sensitivity_rows.extend(
                    {**row, "training_seed": seed} for row in measured
                )
        if args.architecture_ablation:
            cases = experiment.architecture_configurations(selected_config)
            architecture_fits += (
                len({json.dumps(config, sort_keys=True) for config in cases.values()})
                - 1
            )
            for label, config in cases.items():
                destination = directory / "architecture" / label
                checkpoint_dir = (
                    directory / f"trial-{best_trial}"
                    if config == selected_config
                    else destination
                )
                model, info = experiment.fit_model(
                    processed_train,
                    processed_validation,
                    config,
                    "original",
                    checkpoint_dir,
                    args,
                )
                experiment.write_json(
                    destination / "selection.json",
                    {
                        "config": config,
                        "checkpoint": checkpoint_dir.relative_to(directory).as_posix(),
                        "checkpoint_sha256": info["checkpoint_sha256"],
                        "parameters": info["parameters"],
                        "selected_on_test": False,
                    },
                )
                measured, _ = evaluation.score_neural(
                    model,
                    processor,
                    train,
                    test,
                    fold,
                    label,
                    destination,
                    args.seed + 1000 + fold,
                )
                architecture_rows.extend(measured)
        evaluation.validate_rows(fold_rows, [fold], args.historical)
        experiment.write_json(directory / "metrics.json", fold_rows)
        timing = directory / "timing.json"
        if not timing.exists():
            experiment.write_json(timing, {"seconds": time.perf_counter() - started})
        rows.extend(fold_rows)
        evaluation.report(args.output_dir, rows)
        print(f"Completed PBC fold {fold + 1}/{args.folds}", flush=True)
    evaluation.validate_rows(rows, list(range(args.folds)), args.historical)
    folds = list(range(args.folds))
    if args.paper_time:
        evaluation.validate_rows(paper_rows, folds, False, horizons=PAPER_HORIZONS)
        evaluation.report(args.output_dir / "paper_time", paper_rows, h2=False)
    if args.sensitivity_seeds:
        evaluation.report_sensitivity(
            args.output_dir / "seed_sensitivity",
            sensitivity_rows,
            folds,
            manifest["sensitivity_training_seeds"],
        )
    if args.architecture_ablation:
        cases = experiment.architecture_configurations(configs[0])
        evaluation.validate_rows(architecture_rows, folds, False, modes=tuple(cases))
        evaluation.report(args.output_dir / "architecture", architecture_rows, h2=False)
    primary_fits = args.folds * (len(configs) + len(MODES) - 1)
    sensitivity_fits = args.folds * args.sensitivity_seeds * len(MODES)
    experiment.write_json(
        args.output_dir / "completion.json",
        {
            "status": "complete",
            "folds": args.folds,
            "metric_rows": len(rows),
            "training_fits": primary_fits + sensitivity_fits + architecture_fits,
            "primary_training_fits": primary_fits,
            "sensitivity_training_fits": sensitivity_fits,
            "architecture_training_fits": architecture_fits,
            "cox_fits": args.folds * len(args.cox_penalties) * 2,
            "paper_time_metric_rows": len(paper_rows),
            "sensitivity_metric_rows": len(sensitivity_rows),
            "architecture_metric_rows": len(architecture_rows),
            "seconds": sum(
                json.loads(
                    (args.output_dir / f"fold-{fold}" / "timing.json").read_text()
                )["seconds"]
                for fold in range(args.folds)
            ),
        },
    )
    print(f"Complete: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
