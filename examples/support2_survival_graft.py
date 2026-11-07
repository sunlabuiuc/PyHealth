"""GRAFT loading, fitting, configuration comparisons and survival evaluation.

The default creates 96 fabricated SUPPORT2 rows and uses the existing dataset
loader and event/time task. It downloads nothing and runs at most eight epochs.
Use --variant full/no_stg/linear and --noise-multiplier 3 --rho 0/0.5/0.9 to
exercise study configurations. Demo metrics do not estimate benchmark accuracy.

    python examples/support2_survival_graft.py --output /tmp/graft-demo
    python examples/support2_survival_graft.py --variant no_stg \
        --noise-multiplier 3 --rho 0.9 --output /tmp/graft-no-stg

--root selects an existing directory containing support2.csv with an ID column
named sno. Some UCI copies call it id; rename that column before loading.
The raw-data route uses a fixed 65/15/20 split and onehot22 by default.
--encoding benchmark14 changes its encoding, not its splitting protocol.

--benchmark-file explicitly selects the verified processed SUPPORT feather
file (8,873 rows and 14 predictors) and reproduces one GRAFT study cell using
the fixed three-fold/inner-validation protocol. It defaults to 1,000 maximum
epochs with validation stopping, never launches a full matrix and downloads
no data. --fold and --seed select the cell. An existing result is not replaced.

    python examples/support2_survival_graft.py --benchmark-file support.feather \
        --fold 0 --seed 42 --variant full --output /tmp/graft-cell

See support2_survival_graft_results.md for installation, data provenance,
complete study commands, measured results and interpretation limits. Fitting
requires the optional graft dependencies. CPU fitting uses two torch threads.
"""

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from pyhealth.datasets import Support2Dataset, get_dataloader
from pyhealth.models import GRAFT
from pyhealth.tasks import SurvivalTimeSupport2


def make_synthetic_support2(root: Path, n: int = 96, seed: int = 598) -> None:
    """Write fabricated clinical columns to exercise the real dataset loader.

    Examples:
        >>> # make_synthetic_support2(Path('/tmp/synthetic-support2'))
    """
    import pyhealth.datasets.support2 as support_module

    if n < 64:
        raise ValueError(
            "Use at least 64 synthetic rows for default local neighborhoods."
        )
    root.mkdir(parents=True, exist_ok=True)
    config = Path(support_module.__file__).parent / "configs/support2.yaml"
    columns = yaml.safe_load(config.read_text())["tables"]["support2"]["attributes"]
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        age = rng.uniform(25, 80)
        event_time = np.exp(3.2 + 0.025 * (age - 50) + rng.normal(0, 0.6))
        censor_time = np.exp(3.8 + rng.normal(0, 0.7))
        row = dict.fromkeys(columns)
        row.update(
            {
                "sno": f"synthetic_{i:04d}",
                "age": age,
                "num.co": i % 4,
                "diabetes": i % 2,
                "dementia": int(i % 7 == 0),
                "meanbp": rng.uniform(65, 110),
                "hrt": rng.uniform(55, 110),
                "resp": rng.uniform(12, 30),
                "temp": rng.normal(37, 0.5),
                "wblc": rng.uniform(4, 14),
                "sod": rng.uniform(130, 145),
                "crea": rng.uniform(0.5, 2),
                "sex": ("female", "male")[i % 2],
                "race": None
                if i % 6 == 0
                else ("asian", "black", "hispanic", "other", "white")[i % 5],
                "ca": ("metastatic", "no", "yes")[i % 3],
                "d.time": min(event_time, censor_time),
                "death": int(event_time <= censor_time),
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(root / "support2.csv", index=False)


def make_nuisance_features(
    subjects: int, predictors: int, seed: int, rho: float
) -> np.ndarray:
    """Generate exactly 3p Gaussian nuisance columns in blocks of five.

    Args:
        subjects: Number of rows in stable cohort order.
        predictors: Original feature count p.
        seed: Nonnegative training seed; the same base draws serve every rho.
        rho: Within-block correlation in [0, 1].

    Returns:
        Array of shape (subjects, 3 * predictors), independent of outcomes
        by construction. The final block is truncated to exactly 3p columns.

    Examples:
        >>> make_nuisance_features(2, 1, 42, 0.5).shape
        (2, 3)
    """
    if subjects < 1 or predictors < 1 or seed < 0 or not 0 <= rho <= 1:
        raise ValueError("Positive counts, nonnegative seed and rho in [0,1] required.")
    generator = np.random.Generator(
        np.random.PCG64(np.random.SeedSequence([5982026, seed, 1]))
    )
    epsilon = generator.standard_normal((subjects, 3 * predictors))
    common = generator.standard_normal((subjects, (3 * predictors + 4) // 5))
    repeated = np.repeat(common, 5, axis=1)[:, : 3 * predictors]
    return np.sqrt(rho) * repeated + np.sqrt(1 - rho) * epsilon


def _arrays(cohort: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(cohort, Mapping):
        return (
            np.asarray(cohort["features"], dtype=float),
            np.asarray(cohort["duration"], dtype=float),
            np.asarray(cohort["event"], dtype=bool),
        )
    return (
        np.stack([np.asarray(sample["features"], dtype=float) for sample in cohort]),
        np.asarray([float(sample["duration"]) for sample in cohort]),
        np.asarray([bool(sample["event"]) for sample in cohort]),
    )


def _output_directory(output: Path) -> None:
    if any((output / name).exists() for name in ["results.json", "graft.pt"]):
        raise FileExistsError(
            "Choose a new output directory; existing results are kept."
        )
    output.mkdir(parents=True, exist_ok=True)


def _condition(
    features: np.ndarray, seed: int, noise_multiplier: int, rho: float
) -> np.ndarray:
    if noise_multiplier not in (0, 3) or (noise_multiplier == 0 and rho != 0):
        raise ValueError("Use noise multiplier 0 or 3; nonzero rho requires 3.")
    if noise_multiplier == 0:
        return features
    noise = make_nuisance_features(len(features), features.shape[1], seed, rho)
    return np.column_stack([features, noise])


def _model(width: int, variant: str, dataset: Any = None) -> GRAFT:
    if variant not in ("full", "no_stg", "linear"):
        raise ValueError("Variant must be full, no_stg or linear.")
    return GRAFT(
        dataset=dataset,
        n_features=width,
        use_gates=variant == "full",
        use_residual=variant != "linear",
    )


def _fit_evaluate(
    model: GRAFT,
    train: Any,
    validation: Any,
    test: Any,
    output: Path,
    epochs: int,
    batch_size: int,
    seed: int,
    fold: int,
    train_rows: np.ndarray,
) -> dict[str, Any]:
    from sksurv.nonparametric import CensoringDistributionEstimator
    from sksurv.util import Surv

    torch.set_num_threads(2)
    model.fit(
        train,
        validation,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
        fold=fold,
        row_ids=train_rows,
    )
    model.fit_calibration(train)
    _, train_times, train_events = _arrays(train)
    censor = CensoringDistributionEstimator().fit(
        Surv.from_arrays(train_events, train_times)
    )
    supported = censor.unique_time_[censor.prob_ >= 0.05]
    supported = supported[np.isfinite(supported)]
    horizon = min(float(np.quantile(train_times, 0.9)), float(supported.max()))
    grid = np.linspace(
        float(np.quantile(train_times, 0.05)), np.nextafter(horizon, -np.inf), 50
    )
    metrics = model.evaluate_survival(test, grid, horizon)
    features, times, events = _arrays(test)
    forward = model(features=features, duration=times, event=events)
    if "loss" in forward:
        raise AssertionError(
            "Held-out labels must not trigger training/imputation loss."
        )
    scores = model.predict_score(features)
    curves = model.predict_survival(features, grid)
    checkpoint = output / "graft.pt"
    model.save(checkpoint)
    restored = GRAFT.load(checkpoint)
    np.testing.assert_array_equal(scores, restored.predict_score(features))
    np.testing.assert_array_equal(curves, restored.predict_survival(features, grid))
    assert restored.dataset is None
    np.savez(
        output / "reload_inputs.npz",
        features=features,
        grid=grid,
        scores=scores,
        curves=curves,
    )
    results = {
        "status": "PASS",
        "feature_width": model.n_features,
        "training": len(train_times),
        "validation": len(_arrays(validation)[1]),
        "evaluation": len(times),
        "mode": model.mode,
        "fit_outside_trainer": model.fit_outside_trainer,
        "held_out_forward_has_no_loss": True,
        "checkpoint_reload_exact": True,
        "reload_needs_training_records": False,
        "metrics": metrics,
        "training_info": model.training_info_,
        "calibration_coefficient": model.calibration_["coefficient"],
        "calibration_nonnegative": model.calibration_["nonnegative_slope"],
        "training_seed": seed,
        "fold": fold,
    }
    return results


def run_pipeline(
    root: Path,
    output: Path,
    epochs: int = 8,
    encoding: str = "onehot22",
    *,
    variant: str = "full",
    noise_multiplier: int = 0,
    rho: float = 0.0,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the existing SUPPORT2 loader/task route and a single configuration.

    Args:
        root: Existing local directory containing support2.csv with sno IDs.
        output: New result directory; existing results are never replaced.
        epochs: Maximum epochs, with validation stopping.
        encoding: Fixed onehot22 or numeric-coded benchmark14 task encoding.
        variant: full, no_stg or linear GRAFT architecture.
        noise_multiplier: Zero or three times the task's original feature width.
        rho: Within-block nuisance correlation; nonzero requires added noise.
        seed: Training and nuisance seed. The 65/15/20 membership stays fixed.

    Returns:
        Result metadata and the actual task subsets/model/dataloader batch.

    Examples:
        >>> # results, artifacts = run_pipeline(data_root, new_output_directory)
    """
    if not (root / "support2.csv").is_file():
        raise FileNotFoundError("Supply an existing support2.csv; no download is made.")
    _output_directory(output)
    task = SurvivalTimeSupport2(encoding=encoding)
    base = Support2Dataset(
        root=str(root),
        tables=["support2"],
        num_workers=1,
        cache_dir=str(output / "cache"),
    )
    dataset = base.set_task(task, num_workers=1)
    if len(dataset) < 64:
        raise ValueError("Example needs at least 64 retained subjects.")
    order = np.random.default_rng(42).permutation(len(dataset))
    nt, nv = int(0.65 * len(order)), int(0.15 * len(order))
    indices = [
        np.sort(order[:nt]),
        np.sort(order[nt : nt + nv]),
        np.sort(order[nt + nv :]),
    ]
    subsets = [dataset.subset(rows.tolist()) for rows in indices]
    features, times, events = _arrays(dataset)
    conditioned = _condition(features, seed, noise_multiplier, rho)
    if noise_multiplier:
        cohorts = [
            {
                "features": conditioned[rows],
                "duration": times[rows],
                "event": events[rows],
            }
            for rows in indices
        ]
    else:
        cohorts = subsets
    model = _model(conditioned.shape[1], variant, dataset)
    results = _fit_evaluate(model, *cohorts, output, epochs, 16, seed, 0, indices[0])
    batch = next(
        iter(get_dataloader(subsets[2], batch_size=len(subsets[2]), shuffle=False))
    )
    if noise_multiplier:
        batch["features"] = torch.as_tensor(conditioned[indices[2]])
    if "loss" in model(**batch):
        raise AssertionError("Held-out task/dataloader labels must not create a loss.")
    results.update(
        {
            "encoding": encoding,
            "subjects": len(dataset),
            "censored_subjects": int((~events).sum()),
            "variant": variant,
            "noise_multiplier": noise_multiplier,
            "rho": rho,
            "original_feature_width": features.shape[1],
            "scope": (
                "Loader/task execution and configuration example; fixed "
                "65/15/20 split, not the benchmark CV protocol."
            ),
        }
    )
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    artifacts = {
        "model": model,
        "train": cohorts[0],
        "validation": cohorts[1],
        "test": cohorts[2],
        "dataset": dataset,
        "batch": batch,
    }
    return results, artifacts


def run_benchmark(
    file: Path,
    output: Path,
    *,
    epochs: int = 1000,
    variant: str = "full",
    noise_multiplier: int = 0,
    rho: float = 0.0,
    fold: int = 0,
    seed: int = 42,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reproduce one verified processed-SUPPORT GRAFT study cell, without I/O downloads.

    Args:
        file: Local pycox SUPPORT feather file with verified content/order.
        output: New result directory.
        epochs: Maximum epochs; the reported study uses 1000.
        variant: full, no_stg or linear.
        noise_multiplier: Zero or three, giving 14 or 56 total features.
        rho: Nuisance block correlation, used only with added columns.
        fold: Outer fold 0, 1 or 2. The saved study uses all three.
        seed: Training/nuisance seed; the saved study uses 42, 43 and 44.

    Returns:
        Results, model and array cohorts. Original labels remain uncensored
        for fitting/C-index; IBS applies the training-defined horizon only.

    Examples:
        >>> # results, artifacts = run_benchmark(local_feather, new_output_directory)
    """
    from sklearn.model_selection import KFold, train_test_split

    if fold not in (0, 1, 2):
        raise ValueError("The benchmark has outer folds 0, 1 and 2.")
    frame = pd.read_feather(file)
    checksum = hashlib.sha256(frame.to_csv().encode()).hexdigest()
    if checksum != "b07a9d216bf04501e832084e5b7955cb84dfef834810037c548dee82ea251f8d":
        raise ValueError(
            "Input is not the verified 8,873-row processed SUPPORT cohort/order."
        )
    _output_directory(output)
    original = frame[[f"x{column}" for column in range(14)]].to_numpy(dtype=float)
    features = _condition(original, seed, noise_multiplier, rho)
    times, events = (
        frame.duration.to_numpy(dtype=float),
        frame.event.to_numpy(dtype=bool),
    )
    outer_train, test = list(KFold(3, shuffle=True, random_state=42).split(frame))[fold]
    train, validation = train_test_split(
        outer_train,
        test_size=0.2,
        random_state=4242 + fold,
        stratify=events[outer_train],
    )
    indices = [np.sort(train), np.sort(validation), np.sort(test)]
    cohorts = [
        {"features": features[rows], "duration": times[rows], "event": events[rows]}
        for rows in indices
    ]
    model = _model(features.shape[1], variant)
    results = _fit_evaluate(model, *cohorts, output, epochs, 64, seed, fold, indices[0])
    results.update(
        {
            "data_mode": "verified_processed_benchmark",
            "encoding": "benchmark14",
            "subjects": len(frame),
            "variant": variant,
            "noise_multiplier": noise_multiplier,
            "rho": rho,
            "original_feature_width": 14,
            "dataframe_sha256": checksum,
            "split_index_sha256": {
                name: hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()
                for name, rows in zip(["train", "validation", "test"], indices)
            },
            "scope": (
                "One processed-SUPPORT GRAFT cell; fixed project protocol, "
                "not the author's executable protocol."
            ),
        }
    )
    (output / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n"
    )
    return results, {
        "model": model,
        "train": cohorts[0],
        "validation": cohorts[1],
        "test": cohorts[2],
    }


def main() -> None:
    """Run the downloadable-data-free synthetic demonstration by default.

    Examples:
        >>> # python examples/support2_survival_graft.py --epochs 8
    """
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group()
    data.add_argument(
        "--root", type=Path, help="Existing SUPPORT2 directory; no downloads"
    )
    data.add_argument(
        "--benchmark-file",
        type=Path,
        help="Verified local processed SUPPORT feather; one CV cell",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--epochs", type=int, help="Maximum epochs: 8 demo/raw, 1000 benchmark"
    )
    parser.add_argument(
        "--variant", choices=["full", "no_stg", "linear"], default="full"
    )
    parser.add_argument("--noise-multiplier", type=int, choices=[0, 3], default=0)
    parser.add_argument("--rho", type=float, choices=[0.0, 0.5, 0.9], default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fold", type=int, choices=[0, 1, 2], default=0)
    parser.add_argument(
        "--encoding", choices=["onehot22", "benchmark14"], default="onehot22"
    )
    args = parser.parse_args()
    if args.seed < 0 or (args.epochs is not None and args.epochs < 1):
        parser.error("Use a nonnegative seed and a positive epoch maximum.")
    if args.noise_multiplier == 0 and args.rho:
        parser.error("Nonzero --rho requires --noise-multiplier 3.")
    if args.fold and args.benchmark_file is None:
        parser.error("--fold applies to --benchmark-file only.")
    with tempfile.TemporaryDirectory(prefix="pyhealth-graft-") as temporary:
        output = args.output or Path(temporary) / "output"
        settings = {
            "variant": args.variant,
            "noise_multiplier": args.noise_multiplier,
            "rho": args.rho,
            "seed": args.seed,
        }
        if args.benchmark_file is not None:
            result, _ = run_benchmark(
                args.benchmark_file,
                output,
                epochs=args.epochs or 1000,
                fold=args.fold,
                **settings,
            )
        else:
            root = args.root or Path(temporary) / "data"
            if args.root is None:
                make_synthetic_support2(root)
            result, _ = run_pipeline(
                root, output, args.epochs or 8, args.encoding, **settings
            )
        print(
            json.dumps(
                {k: v for k, v in result.items() if k != "training_info"}, indent=2
            )
        )


if __name__ == "__main__":
    main()
