"""Authors: Aritra Saha and Ugochukwu Njoku.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Run optional Cox/DeepSurv benchmarks using native PyHealth survival tasks.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, get_dataloader
from pyhealth.metrics import d_calibration, survival_curve_metrics_fn

from .baselines import CoxAdapter, DeepSurvAdapter
from .run_survival_mdn import (
    BENCHMARKS,
    artifact_hashes,
    git_commit,
    implementation_hashes,
    input_hashes,
    prepare_output,
    reference_compatible_widest,
    resolve_run,
    split_parts,
)


def package_version(name: str) -> str | None:
    """Read optional distribution versions without requiring every backend.

    Args:
        name: Installed package distribution name.

    Returns:
        Version string, or None when this optional dependency is absent.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def fingerprint(path: Path) -> str:
    """Hash a local input file.

    Args:
        path: File whose bytes identify an input or implementation.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arrays(part: SampleDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract a full patient-ordered batch from a native survival task.

    Args:
        part: One training, validation or test SampleDataset.

    Returns:
        Features [N,D], packed outcomes [N,2] and patient IDs [N].
    """
    batch = next(iter(get_dataloader(part, batch_size=len(part), shuffle=False)))
    if not {"duration", "event"}.issubset(batch):
        raise ValueError("Survival tasks must emit separate duration and event tensors")
    return (
        np.asarray(batch["features"], dtype=np.float32),
        np.asarray(
            torch.cat([batch["duration"], batch["event"]], dim=1), dtype=np.float32
        ),
        np.asarray(batch["patient_id"]),
    )


def score(
    adapter: CoxAdapter | DeepSurvAdapter, features: np.ndarray, labels: np.ndarray
) -> tuple[dict[str, float], dict[str, float], dict[str, Any], np.ndarray]:
    """Score exact curves at evaluator-selected times and each patient's time.

    Args:
        adapter: Fitted Cox or DeepSurv adapter.
        features: Patient-ordered feature rows.
        labels: Matching positive times and binary event flags, shape [N,2].

    Returns:
        Aligned metrics, named reference-compatible metrics, D-calibration,
        and exact observed-time survival probabilities in patient order.
    """
    times, events = labels.T
    cache = {}

    def predict(grid):
        key = np.asarray(grid, dtype=float).tobytes()
        if key not in cache:
            cache[key] = adapter.predict_survival(features, grid)
        return cache[key]

    metrics = survival_curve_metrics_fn(times, events, predict)
    reference = reference_compatible_widest(times, events, predict)
    unique, indices = np.unique(times, return_inverse=True)
    observed = predict(unique)[np.arange(len(times)), indices]
    calibration = d_calibration(observed, events)
    return metrics, reference, calibration, observed


def run_fit(
    model_name: str,
    config: dict[str, Any],
    parts: tuple[SampleDataset, ...],
    run_dir: Path,
) -> dict[str, Any]:
    """Fit, restore and evaluate one optional baseline.

    Args:
        model_name: cox or deepsurv.
        config: Adapter constructor options.
        parts: Separate training, validation and test SampleDatasets.
        run_dir: Destination for the model, predictions and split IDs.

    Returns:
        Fit metadata, validation/test scores, calibration and reload error.

    Raises:
        ValueError: If source IDs overlap or the requested model is unknown.
    """
    if model_name not in ("cox", "deepsurv"):
        raise ValueError("model_name must be cox or deepsurv")
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    x, y, train_ids = arrays(parts[0])
    xv, yv, valid_ids = arrays(parts[1])
    xt, yt, test_ids = arrays(parts[2])
    for a, b in ((train_ids, valid_ids), (train_ids, test_ids), (valid_ids, test_ids)):
        if set(a) & set(b):
            raise ValueError("source patient IDs overlap between splits")
    adapter = CoxAdapter(**config) if model_name == "cox" else DeepSurvAdapter(**config)
    start = time.perf_counter()
    adapter.fit(x, y, validation=(xv, yv), checkpoint_dir=run_dir / "checkpoint")
    train_seconds = time.perf_counter() - start
    adapter.save(run_dir / "checkpoint")
    restored = type(adapter).load(run_dir / "checkpoint")
    grid = np.linspace(0.0, float(y[:, 0].max()), 200)
    predictions = adapter.predict_survival(xt, grid)
    reload_predictions = restored.predict_survival(xt, grid)
    np.testing.assert_allclose(predictions, reload_predictions, rtol=0, atol=0)
    if (np.diff(predictions, axis=1) > 1e-6).any():
        raise ValueError("predicted curves increase over time")
    start = time.perf_counter()
    validation_metrics, _, _, _ = score(adapter, xv, yv)
    metrics, reference, calibration, observed = score(adapter, xt, yt)
    evaluation_seconds = time.perf_counter() - start
    np.savez_compressed(
        run_dir / "predictions.npz",
        patient_id=test_ids,
        time=yt[:, 0],
        event=yt[:, 1],
        grid=grid,
        survival=predictions,
        survival_at_observed_time=observed,
    )
    (run_dir / "split_ids.json").write_text(
        json.dumps(
            {
                "train": train_ids.tolist(),
                "valid": valid_ids.tolist(),
                "test": test_ids.tolist(),
            },
            indent=2,
        )
        + "\n"
    )
    return {
        "model": model_name,
        "config": adapter.config,
        "counts": {
            name: len(part) for name, part in zip(("train", "valid", "test"), parts)
        },
        "fit": adapter.fit_info,
        "adapter_preprocessing": {
            "standardize_input": adapter.config["standardize_input"],
            "feature_mean": adapter.mean.tolist(),
            "feature_scale": adapter.scale.tolist(),
        },
        "train_seconds": train_seconds,
        "evaluation_seconds": evaluation_seconds,
        "validation_metrics": validation_metrics,
        "selection_metric": "validation_ibs",
        "selection_value": validation_metrics["ibs"],
        "test_metrics": metrics,
        "test_metrics_reference_compatible": reference,
        "d_calibration": calibration,
        "checkpoint_prediction_max_error": float(
            np.abs(predictions - reload_predictions).max()
        ),
        "curve_tail_policy": (
            "right-continuous training-hazard steps; 1 before first knot; "
            "constant beyond last knot"
        ),
        "test_times_after_last_training_time": int((yt[:, 0] > y[:, 0].max()).sum()),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Run baseline fits from the command line.

    Args:
        argv: CLI arguments, or None to read the process command line.
    """
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=sorted(BENCHMARKS), required=True)
    p.add_argument("--root", type=Path)
    p.add_argument("--released-dir", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--models", choices=["cox", "deepsurv"], nargs="+", default=["cox", "deepsurv"]
    )
    p.add_argument("--splits", type=int, nargs="+", default=[1])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--config",
        type=Path,
        help="JSON with optional cox/deepsurv parameter dictionaries",
    )
    a = p.parse_args(argv)
    if a.dataset != "synthetic" and (a.root is None or a.released_dir is None):
        p.error("Real datasets require --root and --released-dir")
    if any(v < 1 for v in a.splits) or a.seed < 0 or a.epochs < 1 or a.patience < 1:
        p.error(
            "Split/epoch/patience values must be positive; seed must be nonnegative"
        )
    prepare_output(a.out, a.dataset, "baselines")
    overrides = json.loads(a.config.read_text()) if a.config else {}
    if not isinstance(overrides, dict) or set(overrides) - {"cox", "deepsurv"}:
        p.error("Config must contain only cox/deepsurv parameter objects")
    for name, parameters in overrides.items():
        allowed = {"alpha"} if name == "cox" else {"hidden", "lr"}
        if not isinstance(parameters, dict) or set(parameters) - allowed:
            p.error(f"Config {name} accepts only: {sorted(allowed)}")
    environment = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(a.device)
        if a.device.startswith("cuda") and "deepsurv" in a.models
        else None,
        "platform": platform.platform(),
        "device": a.device,
        "feature_commit": git_commit(Path(__import__("pyhealth").__file__).parent),
        "example_commit": git_commit(Path(__file__).parent),
        "packages": {
            name: package_version(name)
            for name in (
                "numpy",
                "scipy",
                "scikit-learn",
                "scikit-survival",
                "pycox",
                "torchtuples",
            )
        },
    }
    benchmark = None
    records = []
    for split in a.splits:
        inputs = input_hashes(a.root, a.released_dir, a.dataset, split)
        prepared = None
        for kind in a.models:
            config = (
                {"alpha": 0.1, "standardize_input": a.dataset != "support2"}
                if kind == "cox"
                else {
                    "hidden": [32, 32],
                    "lr": 0.001,
                    "epochs": a.epochs,
                    "patience": a.patience,
                    "seed": a.seed,
                    "device": a.device,
                    "standardize_input": a.dataset != "support2",
                }
            )
            config.update(overrides.get(kind, {}))
            identity = {
                "dataset": a.dataset,
                "split": split,
                "seed": a.seed,
                "model": kind,
                "config": config,
                "inputs": inputs,
                "implementation": implementation_hashes(),
                "runtime": {
                    "python": environment["python"],
                    "torch": environment["torch"],
                    "device": config.get("device", "cpu"),
                    "cuda": environment["cuda"] if kind == "deepsurv" else None,
                    "gpu": environment["gpu"] if kind == "deepsurv" else None,
                    "packages": environment["packages"],
                },
            }
            parent = a.out / kind / f"split_{split}" / f"seed_{a.seed}"
            run_dir, saved = resolve_run(parent, identity)
            existing = run_dir / "record.json"
            if saved is not None:
                records.append(saved)
                print(f"Verified existing record: {existing}")
                continue
            if benchmark is None:
                benchmark = BENCHMARKS[a.dataset](
                    a.root, a.released_dir, a.out / "cache"
                )
            start = time.perf_counter()
            if prepared is None:
                prepared = benchmark.split(split)
            ids, samples, preprocessing = prepared
            parts = split_parts(samples, ids, a.seed)
            preparation_seconds = time.perf_counter() - start
            record = run_fit(kind, config, parts, run_dir)
            record.update(
                request=identity,
                dataset=a.dataset,
                split=split,
                seed=a.seed,
                preprocessing=preprocessing,
                environment=environment,
                preparation_seconds=preparation_seconds,
                run_directory=str(run_dir.relative_to(a.out)),
            )
            record["artifacts"] = artifact_hashes(run_dir)
            existing.write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(f"Saved run: {run_dir}")
            print(
                f"{a.dataset} {kind} split {split}: "
                f"C^td={record['test_metrics']['c_td']:.4f}, "
                f"IBS={record['test_metrics']['ibs']:.4f}, "
                f"reload error={record['checkpoint_prediction_max_error']:.2g}"
            )
    (a.out / "pilot_summary.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
