"""Authors: Aritra Saha and Ugochukwu Njoku.
Paper: Survival Mixture Density Networks, Han et al., MLHC 2022.
https://proceedings.mlr.press/v182/han22a.html

Run reproducible MDN benchmarks with the standard PyHealth Trainer.

From the repository root, without downloads:
    python -m examples.survival_benchmark.run_survival_mdn --dataset synthetic \
        --splits 1 --epochs 3 --out /tmp/survival-mdn-demo

Real benchmarks require local CSVs and the authors' released split arrays.
See README.md in this directory. The example is self-contained in this checkout.
"""

import argparse
import hashlib
import itertools
import json
import logging
import platform
import random
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

import pyhealth
from pyhealth.datasets import (
    GBSGDataset,
    SampleDataset,
    Support2Dataset,
    create_sample_dataset,
    gbsg_paper_time_transform,
    gbsg_released_split_ids,
    get_dataloader,
    support2_paper_time_transform,
    support2_released_split_ids,
)
from pyhealth.metrics import d_calibration, survival_curve_metrics_fn
from pyhealth.metrics.survival_curve import (
    integrated_binomial_loglik_td,
    integrated_brier_score_td,
    integration_grid,
)
from pyhealth.models import SurvivalMDN
from pyhealth.tasks import SurvivalTimeToEventGBSG, SurvivalTimeToEventSupport2
from pyhealth.trainer import Trainer
from pyhealth.trainer import logger as trainer_logger

SPLITS = ("train", "valid", "test")

DEFAULT_MODEL = {
    "num_components": 10,
    "time_transform": "softplus",
    "hidden_dim": 32,
    "num_hidden_layers": 3,
    "min_scale": 1e-8,
    "include_jacobian": True,
    "reference_initialization": True,
}

# The authors' main.py defaults: 100 epochs, patience 10 on validation loss,
# RMSprop, gradient clipping at 100 (configs/train.json).
DEFAULT_TRAIN = {
    "epochs": 100,
    "patience": 10,
    "optimizer": "RMSprop",
    "lr": 1e-3,
    "weight_decay": 0.0,
    "batch_size": 512,
    "grad_clip": 100.0,
}
PREDICTION_GRID_POINTS = 200


def input_hashes(
    root: Path | None, released_dir: Path | None, dataset: str, split: int
) -> dict[str, str]:
    """Identify the cohort and released files used by one fit.

    Args:
        root: CSV directory, required for real datasets.
        released_dir: Local released NPZ directory for real datasets.
        dataset: One of gbsg, support2 or synthetic.
        split: Positive released split number or synthetic partition seed.

    Returns:
        Input file fingerprints, or the synthetic generation recipe identity.
    """
    if dataset == "synthetic":
        return {"synthetic": "six-patient-survival-features7-generation-seed0-v2"}
    paths = {
        "cohort": Path(root) / ("gbsg.csv" if dataset == "gbsg" else "support2.csv")
    }
    paths.update({name: Path(released_dir) / f"{name}_{split}.npz" for name in SPLITS})
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def validate_resume(
    record: dict[str, Any], request: dict[str, Any], run_dir: Path | None = None
) -> None:
    """Reject changed requests or missing/altered completed outputs.

    Args:
        record: Previously saved fit record.
        request: Settings, input and source identity of the requested fit.
        run_dir: Optional fit directory whose recorded files must match.

    Raises:
        ValueError: If identities differ or recorded output files are invalid.
    """
    if not isinstance(record, dict) or record.get("request") != request:
        raise ValueError(
            "Saved run identity does not match the requested fit; preserve "
            "the existing files and inspect the record before rerunning"
        )

    if run_dir is not None:
        artifacts = record.get("artifacts", {})
        if not isinstance(artifacts, dict) or "predictions.npz" not in artifacts:
            raise ValueError(
                f"Incomplete run at {run_dir}: no verified predictions; "
                "preserve this directory before explicitly restarting the fit"
            )
        for name, expected in artifacts.items():
            path = run_dir / name
            if (
                not path.is_file()
                or hashlib.sha256(path.read_bytes()).hexdigest() != expected
            ):
                raise ValueError(
                    f"Missing or changed result artifact: {path}. Preserve this "
                    "run and inspect the files before explicitly restarting; "
                    "automatic repair is disabled"
                )


def resolve_run(
    parent: Path, request: dict[str, Any]
) -> tuple[Path, dict[str, Any] | None]:
    """Create a versioned fit directory or verify an identical completed run.

    Args:
        parent: Readable model/split/seed directory below the output root.
        request: JSON settings, input, source and runtime identity of one fit.

    Returns:
        The fingerprint-named directory and its verified record, or None
        for a newly created run. Existing files are never overwritten.

    Raises:
        ValueError: If an existing run is incomplete, unreadable or altered.
    """
    encoded = json.dumps(
        request, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    run_id = hashlib.sha256(encoded).hexdigest()[:16]
    run_dir = parent / f"run_{run_id}"
    parent.mkdir(parents=True, exist_ok=True)
    try:
        run_dir.mkdir()
    except FileExistsError:
        record_path = run_dir / "record.json"
        if not record_path.is_file():
            raise ValueError(
                f"Incomplete run at {run_dir}: no completed record. Another "
                "process may still be running. Preserve this directory before "
                "explicitly restarting the fit; automatic repair is disabled"
            ) from None
        try:
            record = json.loads(record_path.read_text())
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Unreadable completed record at {record_path}; preserve the "
                "run and inspect it before explicitly restarting"
            ) from error
        validate_resume(record, request, run_dir)
        return run_dir, record
    (run_dir / "request.json").write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n"
    )
    return run_dir, None


def artifact_hashes(run_dir: Path) -> dict[str, str]:
    """Hash completed outputs before recording a resumable fit.

    Args:
        run_dir: Completed fit directory, excluding its record.json.

    Returns:
        Relative file paths mapped to SHA-256 fingerprints.
    """
    paths = [run_dir / "predictions.npz"]
    paths += [
        run_dir / name
        for name in ("request.json", "split_ids.json")
        if (run_dir / name).is_file()
    ]
    paths += sorted((run_dir / "checkpoint").glob("*"))
    paths += sorted((run_dir / "trainer").glob("*.ckpt"))
    if not paths[0].is_file():
        raise ValueError("A completed fit requires predictions.npz")
    return {
        str(p.relative_to(run_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def implementation_hashes() -> dict[str, str]:
    """Identify example and native model, metric, task and loader source.

    Returns:
        Source identifiers mapped to SHA-256 fingerprints. Installed source
        is read directly, so an editable checkout and a source wheel work.
    """
    paths = {p.name: p for p in Path(__file__).parent.glob("*.py")}
    package = Path(pyhealth.__file__).parent
    for name in [
        "models/survival_mdn.py",
        "metrics/survival_curve.py",
        "tasks/survival_time_to_event.py",
        "datasets/gbsg.py",
        "datasets/support2.py",
    ]:
        paths["pyhealth/" + name] = package / name
    return {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in paths.items()
    }


def prepare_output(out: Path, dataset: str, runner: str) -> None:
    """Keep one dataset and runner type per output root.

    Args:
        out: Benchmark output root.
        dataset: Real dataset name or synthetic.
        runner: MDN or baseline runner identifier.

    Raises:
        ValueError: If another dataset or runner already owns this folder.
    """
    identity = {"dataset": dataset, "runner": runner}
    out.mkdir(parents=True, exist_ok=True)
    marker = out / "benchmark.json"
    if marker.exists() and json.loads(marker.read_text()) != identity:
        raise ValueError("Use separate output folders for each dataset and runner")
    marker.write_text(json.dumps(identity, indent=2) + "\n")


def git_commit(path: Path) -> str | None:
    """Read the checkout revision for provenance, without modifying Git.

    Args:
        path: Directory inside a Git checkout.

    Returns:
        HEAD, with a dirty suffix for modified tracked files, or None when Git
        metadata or the executable is unavailable.
    """
    try:
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return head + ("+dirty" if dirty else "")


def config_grid(
    base: dict[str, Any], args: argparse.Namespace
) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    """Expand component-count and transform choices into named fits.

    Args:
        base: Optional model/train dictionaries overriding documented defaults.
        args: Parsed CLI settings, with command-line values taking precedence.

    Returns:
        Tuples of fit name, model settings and training settings.
    """
    if not isinstance(base, dict) or set(base) - {"model", "train"}:
        raise ValueError("Config accepts only model and train parameter objects")
    for group, allowed in [("model", DEFAULT_MODEL), ("train", DEFAULT_TRAIN)]:
        values = base.get(group, {})
        if not isinstance(values, dict) or set(values) - set(allowed):
            raise ValueError(
                f"Unknown {group} settings; allowed keys: {sorted(allowed)}"
            )
    model = {**DEFAULT_MODEL, **base.get("model", {})}
    for key in ("num_components", "hidden_dim", "num_hidden_layers"):
        if (
            isinstance(model[key], bool)
            or not isinstance(model[key], int)
            or model[key] < 1
        ):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("include_jacobian", "reference_initialization"):
        if not isinstance(model[key], bool):
            raise TypeError(f"{key} must be boolean")
    if model["time_transform"] not in {"softplus", "exp"}:
        raise ValueError("time_transform must be softplus or exp")
    if not np.isfinite(model["min_scale"]) or model["min_scale"] <= 0:
        raise ValueError("min_scale must be finite and positive")
    train = {**DEFAULT_TRAIN, **base.get("train", {})}
    for key in ("epochs", "patience", "lr", "batch_size"):
        if getattr(args, key) is not None:
            train[key] = getattr(args, key)
    for key, minimum in [("epochs", 1), ("patience", 1), ("batch_size", 2)]:
        value = train[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{key} must be an integer of at least {minimum}")
    for key, minimum in [("lr", 0), ("grad_clip", 0), ("weight_decay", 0)]:
        value = train[key]
        if (
            not np.isfinite(value)
            or value < minimum
            or (key != "weight_decay" and value == 0)
        ):
            raise ValueError(f"Invalid finite training setting: {key}")
    if train["optimizer"] not in {"RMSprop", "Adam", "AdamW", "SGD"}:
        raise ValueError("optimizer must be RMSprop, Adam, AdamW or SGD")
    ks = args.num_components or [model["num_components"]]
    transforms = args.transform or [model["time_transform"]]
    configs = []
    for k, tf in itertools.product(ks, transforms):
        m = {**model, "num_components": k, "time_transform": tf}
        configs.append((f"{args.name}_K{k}_{tf}", m, dict(train)))
    return configs


class GBSGBenchmark:
    """GBSG released partitions with explicit feature and time conventions.

    Args:
        root: Directory containing gbsg.csv.
        released_dir: Directory containing the released train/valid/test NPZs.
        cache_dir: Destination for native dataset/task caches.
        standardize: Fit feature means/scales on each split's training patients.
            Default False preserves the raw-feature paper protocol.
    """

    def __init__(
        self,
        root: Path,
        released_dir: Path,
        cache_dir: Path,
        standardize: bool = False,
    ) -> None:
        self.root, self.released_dir = root, released_dir
        self.standardize = standardize
        self.dataset = GBSGDataset(root=str(root), cache_dir=cache_dir, num_workers=1)
        task = SurvivalTimeToEventGBSG(**gbsg_paper_time_transform(root))
        self.samples = self.dataset.set_task(task)
        self.preprocessing = {"features": "raw", **gbsg_paper_time_transform(root)}

    def split(
        self, split: int
    ) -> tuple[dict[str, list[str]], SampleDataset, dict[str, Any]]:
        """Load one released patient assignment without changing row order.

        Args:
            split: Released split number from 1 through 10.

        Returns:
            Train/validation/test IDs, native samples and time metadata.
        """
        ids = gbsg_released_split_ids(self.root, self.released_dir, split)
        if not self.standardize:
            return ids, self.samples, self.preprocessing
        raw = {
            sample["patient_id"]: sample["features"].numpy() for sample in self.samples
        }
        train = np.stack([raw[pid] for pid in ids["train"]]).astype(float)
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale == 0] = 1.0
        time = gbsg_paper_time_transform(self.root)
        task = SurvivalTimeToEventGBSG(mean.tolist(), scale.tolist(), **time)
        samples = self.dataset.set_task(task)
        preprocessing = {
            "features": "standardized on this split's training patients",
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            **time,
        }
        return ids, samples, preprocessing


class Support2Benchmark:
    """Authors' protocol: 27 features standardized on each split's training
    patients, days / 365.25 + 0.001.

    Args:
        root: Directory containing support2.csv with the sno patient header.
        released_dir: Directory containing released train/valid/test NPZs.
        cache_dir: Destination for native dataset/task caches.
    """

    def __init__(self, root: Path, released_dir: Path, cache_dir: Path) -> None:
        self.root, self.released_dir = root, released_dir
        self.dataset = Support2Dataset(
            root=str(root), tables=["support2"], cache_dir=cache_dir, num_workers=1
        )
        self.time = support2_paper_time_transform()
        raw = self.dataset.set_task(SurvivalTimeToEventSupport2(**self.time))
        self.raw_features = {
            raw[i]["patient_id"]: raw[i]["features"].numpy().astype(float)
            for i in range(len(raw))
        }

    def split(
        self, split: int
    ) -> tuple[dict[str, list[str]], SampleDataset, dict[str, Any]]:
        """Fit the feature transform on this released split's training rows.

        Args:
            split: Released split number from 1 through 10.

        Returns:
            Patient IDs, samples using the fitted transform and its metadata.
        """
        ids = support2_released_split_ids(self.root, self.released_dir, split)
        train = np.stack([self.raw_features[p] for p in ids["train"]])
        mean, scale = train.mean(axis=0), train.std(axis=0)
        scale[scale == 0] = 1.0  # as sklearn's StandardScaler
        task = SurvivalTimeToEventSupport2(mean.tolist(), scale.tolist(), **self.time)
        samples = self.dataset.set_task(task)
        preprocessing = {
            "features": "standardized on this split's training patients",
            "feature_mean": mean.tolist(),
            "feature_scale": scale.tolist(),
            **self.time,
        }
        return ids, samples, preprocessing


class SyntheticBenchmark:
    """Six-patient, download-free interface demonstration.

    The training pair includes one censored patient. Validation and test
    pairs have observed outcomes so tiny-sample curve scores are defined.
    These scores demonstrate execution, not statistical performance.

    Args:
        root: Unused; kept for the shared benchmark interface.
        released_dir: Unused; no released arrays are needed.
        cache_dir: Unused; the synthetic dataset stays in memory.
    """

    def __init__(self, root=None, released_dir=None, cache_dir=None) -> None:
        rng = np.random.default_rng(0)
        features = rng.normal(size=(6, 7))
        times = np.exp(0.8 + 0.2 * features[:, 0] + rng.normal(0, 0.3, 6))
        # Keep the censored training patient at risk when its partner has an
        # event, so the tiny Cox demonstration has a nonempty risk comparison.
        times[0] = times[1:].max() + 1.0
        samples = [
            {
                "patient_id": f"synthetic-{i}",
                "features": x.tolist(),
                "duration": [float(t)],
                "event": [float(i != 0)],
            }
            for i, (x, t) in enumerate(zip(features, times))
        ]
        self.samples = create_sample_dataset(
            samples,
            {"features": "tensor"},
            {"duration": "tensor", "event": "tensor"},
            dataset_name="survival-benchmark-demo",
        )

    def split(
        self, split: int
    ) -> tuple[dict[str, list[str]], SampleDataset, dict[str, Any]]:
        """Partition six patients into disjoint training/validation/test pairs.

        Args:
            split: Positive partition seed, independent of initialization.

        Returns:
            IDs in partition order, native samples and smoke-data metadata.
        """
        if split < 1:
            raise ValueError("split must be positive")
        observed = np.random.default_rng(split).permutation(np.arange(1, 6))
        groups = [[0, int(observed[0])], observed[1:3], observed[3:5]]
        ids = {
            name: [f"synthetic-{i}" for i in group]
            for name, group in zip(SPLITS, groups)
        }
        return ids, self.samples, {"features": "raw", "synthetic": True}


BENCHMARKS = {
    "gbsg": GBSGBenchmark,
    "support2": Support2Benchmark,
    "synthetic": SyntheticBenchmark,
}


def split_parts(
    samples: SampleDataset, ids: dict[str, list[str]], seed: int
) -> tuple[SampleDataset, ...]:
    """Fresh train/valid/test subsets for one fit.

    The training shuffle comes from get_dataloader's seed (PyHealth #1280),
    so results do not depend on run order. Fresh subsets are kept anyway,
    so no dataset state can carry over from an earlier fit.

    Args:
        samples: Native patient-indexed survival samples.
        ids: Source IDs for the train, valid and test groups in release order.
        seed: Fit seed retained in the shared caller interface. This function
            preserves source order; the training loader applies the seed.

    Returns:
        Fresh training, validation and test subsets in that order.
    """
    return tuple(
        samples.subset([i for pid in ids[name] for i in samples.patient_to_index[pid]])
        for name in SPLITS
    )


def safe_batch_size(n: int, requested: int) -> int:
    """Avoid singleton batches, including a three-patient training cohort.

    Args:
        n: Training patient count, at least two.
        requested: Requested batch size, at least two.

    Returns:
        A safe size, using the full cohort if every bounded choice is unsafe.

    Examples:
        >>> safe_batch_size(3, 2)
        3
        >>> safe_batch_size(513, 512)
        511
    """
    if n < 2 or requested < 2:
        raise ValueError("BatchNorm training needs at least two patients per batch")
    size = min(requested, n)
    while size > 2 and n % size == 1:
        size -= 1
    return n if n % size == 1 else size


def reference_compatible_widest(
    times: np.ndarray, events: np.ndarray, predict: Callable[[np.ndarray], np.ndarray]
) -> dict[str, float]:
    """Widest-horizon IBS/IBLL as the authors' evaluator computes them.

    Their model code evaluates curves on linspace(t_min, t_max, 1000) but
    scores them as if they were on linspace(t_min, tau, 1000), where tau is
    the widest horizon. When the last test time is censored, tau < t_max
    and the grids are misaligned. The
    paper's Table 1 widest-horizon IBS/IBLL include this effect, so these
    values are the ones to compare with Table 1. PyHealth's aligned values
    stay in test_metrics. Checked against the authors' metrics.py to 2e-15
    using independent synthetic probes.

    Args:
        times: Positive observed event/censoring times, shape [N].
        events: Matching binary event flags.
        predict: Callable returning [N,T] survival values for a requested grid.

    Returns:
        Separately named reference-compatible widest-horizon IBS and IBLL.
        These deliberately reproduce the grid mismatch, not aligned scoring.
    """
    grid = integration_grid(times, events, 1e-8)
    surv = predict(np.linspace(times.min(), times.max(), len(grid)))
    return {
        "ibs@1e-08": float(integrated_brier_score_td(times, events, surv, grid)),
        "ibll@1e-08": float(integrated_binomial_loglik_td(times, events, surv, grid)),
    }


def close_trainer_logs(handlers: Sequence[logging.Handler]) -> None:
    """Close only file handlers created by this benchmark fit.

    Args:
        handlers: Handlers owned by the fit, excluding preexisting handlers.
    """
    for handler in handlers:
        if isinstance(handler, logging.FileHandler):
            trainer_logger.removeHandler(handler)
            handler.close()


def epochs_from_log(log: Path) -> dict[str, Any]:
    """Read early-stopping metadata from the fit's own Trainer log.

    Args:
        log: Log path; absent logs return unknown epoch counts.

    Returns:
        Best epoch, completed epoch count and early-stopping flag.
    """
    text = log.read_text() if log.exists() else ""
    best = re.findall(r"New best loss score \([^)]*\) at epoch-(\d+)", text)
    trained = re.findall(r"--- Train epoch-(\d+)", text)
    stopped = re.search(r"Early stopping at epoch-(\d+)", text)
    return {
        "best_epoch": int(best[-1]) if best else None,
        "epochs_run": int(trained[-1]) + 1 if trained else None,
        "early_stopped": stopped is not None,
    }


def run_one(
    name: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split: int,
    seed: int,
    parts: tuple[SampleDataset, ...],
    run_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train one fit and release only its own Trainer file handlers.

    Args:
        name: Configuration name.
        model_cfg: SurvivalMDN constructor settings.
        train_cfg: Training and validation early-stopping settings.
        split: Released split number or synthetic partition seed.
        seed: Model initialization and minibatch seed.
        parts: Separate training, validation and test datasets.
        run_dir: Existing fit output directory.
        device: Torch CPU/CUDA device string.

    Returns:
        Recorded validation loss, test metrics, calibration and fit metadata.
    """
    previous = set(trainer_logger.handlers)
    try:
        return _run_one(name, model_cfg, train_cfg, split, seed, parts, run_dir, device)
    finally:
        close_trainer_logs([h for h in trainer_logger.handlers if h not in previous])


def _run_one(
    name: str,
    model_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    split: int,
    seed: int,
    parts: tuple[SampleDataset, ...],
    run_dir: Path,
    device: str,
) -> dict[str, Any]:
    """Train and score one MDN fit, restoring its best validation weights.

    Args:
        name: Human-readable configuration name.
        model_cfg: SurvivalMDN constructor options.
        train_cfg: Epoch, optimizer, batch and early-stopping settings.
        split: Released split number or synthetic partition seed.
        seed: Random initialization and training-shuffle seed.
        parts: Separate training, validation and test SampleDatasets.
        run_dir: Fit output directory, already created.
        device: Torch device string.

    Returns:
        Configuration, validation loss, test metrics, calibration and timing.
        Test scores do not participate in checkpoint or configuration selection.
    """
    train, valid, test = parts
    # Seed every RNG the data pipeline may draw from, so a fit gives the
    # same result whether or not other fits ran earlier in this process.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SurvivalMDN(train, **model_cfg)
    # Logging must stay on: without an output path the Trainer saves no
    # best.ckpt, and load_best_model_at_last would silently keep the last
    # epoch instead of the best validation epoch.
    trainer = Trainer(
        model, device=device, output_path=str(run_dir), exp_name="trainer"
    )
    batch = safe_batch_size(len(train), train_cfg["batch_size"])
    val_loader = get_dataloader(valid, batch_size=len(valid))
    start = time.perf_counter()
    trainer.train(
        get_dataloader(train, batch_size=batch, shuffle=True, seed=seed),
        val_loader,
        epochs=train_cfg["epochs"],
        optimizer_class=getattr(torch.optim, train_cfg["optimizer"]),
        optimizer_params={"lr": train_cfg["lr"]},
        weight_decay=train_cfg["weight_decay"],
        max_grad_norm=train_cfg["grad_clip"],
        monitor="loss",
        monitor_criterion="min",
        patience=train_cfg["patience"],
        load_best_model_at_last=True,
    )
    train_seconds = time.perf_counter() - start
    val_nll = float(trainer.evaluate(val_loader)["loss"])

    heldout = next(iter(get_dataloader(test, batch_size=len(test))))
    features = heldout["features"].to(device)
    times = heldout["duration"][:, 0].numpy()
    events = heldout["event"][:, 0].numpy()

    def predict(grid):
        g = torch.as_tensor(np.asarray(grid), dtype=torch.float32, device=device)
        with torch.no_grad():
            return model.predict_survival(features, g).cpu().numpy()

    metrics = survival_curve_metrics_fn(times, events, predict)
    observed_grid, observed_columns = np.unique(times, return_inverse=True)
    observed_survival = predict(observed_grid)[np.arange(len(times)), observed_columns]
    train_max = max(float(s["duration"][0]) for s in train)
    grid = np.linspace(
        min(float(s["duration"][0]) for s in train), train_max, PREDICTION_GRID_POINTS
    )
    np.savez_compressed(
        run_dir / "predictions.npz",
        patient_id=np.asarray(heldout["patient_id"]),
        time=times,
        event=events,
        grid=grid,
        survival=predict(grid),
        survival_at_observed_time=observed_survival,
    )
    order = np.argsort(times, kind="stable")
    return {
        "config": name,
        "split": split,
        "seed": seed,
        "model": model_cfg,
        "train": {**train_cfg, "batch_size_used": batch},
        "counts": {s: len(p) for s, p in zip(SPLITS, parts)},
        "val_nll": val_nll,
        "test_metrics": {k: float(v) for k, v in metrics.items()},
        "d_calibration": d_calibration(observed_survival, events),
        "test_metrics_reference_compatible": reference_compatible_widest(
            times, events, predict
        ),
        "last_test_time_censored": bool(events[order[-1]] == 0),
        "train_seconds": train_seconds,
        "epochs_requested": train_cfg["epochs"],
        **epochs_from_log(run_dir / "trainer" / "log.txt"),
    }


def summarize(out: Path, records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean and SD across splits, per config.

    Seeds are averaged within each split first, so the SD is across the
    released splits, as in the paper's tables.

    Args:
        out: Destination for the current command's aggregate output.
        records: Only the completed or verified fits requested by this command.

    Returns:
        Aggregate metric/validation rows, also saved to summary.json and CSV.
        Sorting uses validation loss; test scores do not select settings.
    """
    rows = {}
    for rec in records:
        rows.setdefault(rec["config"], {}).setdefault(rec["split"], []).append(rec)
    summary = []
    for name, by_split in rows.items():
        recs = [r for split_recs in by_split.values() for r in split_recs]

        split_records = tuple(by_split.values())

        def per_split(get, split_records=split_records):
            return [float(np.mean([get(r) for r in rs])) for rs in split_records]

        entry = {
            "config": name,
            "fits": len(recs),
            "splits": sorted(by_split),
            "seeds_per_split": sorted({len(rs) for rs in by_split.values()}),
            "val_nll_mean": float(np.mean(per_split(lambda r: r["val_nll"]))),
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in recs])),
        }
        for metric in recs[0]["test_metrics"]:
            vals = per_split(lambda r, m=metric: r["test_metrics"][m])
            entry[f"{metric}_mean"] = float(np.mean(vals))
            entry[f"{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        for metric in recs[0]["test_metrics_reference_compatible"]:
            vals = per_split(
                lambda r, m=metric: r["test_metrics_reference_compatible"][m]
            )
            entry[f"ref_{metric}_mean"] = float(np.mean(vals))
            entry[f"ref_{metric}_sd"] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            )
        summary.append(entry)
    summary.sort(key=lambda e: e["val_nll_mean"])
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if summary:
        cols = list(summary[0])
        lines = [",".join(cols)] + [
            ",".join(
                " ".join(map(str, e[c]))
                if isinstance(e[c], list)
                else (f"{e[c]:.6g}" if isinstance(e[c], float) else str(e[c]))
                for c in cols
            )
            for e in summary
        ]
        (out / "summary.csv").write_text("\n".join(lines) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    """Run the module's command-line benchmark.

    Args:
        argv: CLI arguments, or None to read the process command line.
    """
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--dataset", choices=sorted(BENCHMARKS), default="gbsg")
    p.add_argument(
        "--root",
        type=Path,
        help="Folder with gbsg.csv, or support2.csv for --dataset support2",
    )
    p.add_argument(
        "--released-dir",
        type=Path,
        help="Authors' data/gbsg or data/support folder with the .npz splits",
    )
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--name", default="mdn", help="Config name prefix")
    p.add_argument("--config", type=Path, help="JSON with 'model' and 'train'")
    p.add_argument("--splits", type=int, nargs="+", default=[1])
    p.add_argument("--seeds", type=int, nargs="+", default=[1])
    p.add_argument("--num-components", type=int, nargs="+")
    p.add_argument("--transform", nargs="+", choices=["softplus", "exp"])
    p.add_argument("--epochs", type=int)
    p.add_argument("--patience", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument(
        "--standardize-gbsg",
        action="store_true",
        help="Fit GBSG feature standardization on training patients only",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args(argv)
    if args.standardize_gbsg and args.dataset != "gbsg":
        p.error("--standardize-gbsg requires --dataset gbsg")
    if args.dataset != "synthetic" and (args.root is None or args.released_dir is None):
        p.error("Real datasets require --root and --released-dir")
    if any(v < 1 for v in args.splits) or any(v < 0 for v in args.seeds):
        p.error("Split numbers must be positive and seeds nonnegative")
    for field in ("epochs", "patience", "batch_size"):
        value = getattr(args, field)
        if value is not None and value < (2 if field == "batch_size" else 1):
            p.error(f"--{field.replace('_', '-')} is too small")

    base = json.loads(args.config.read_text()) if args.config else {}
    configs = config_grid(base, args)
    prepare_output(args.out, args.dataset, "mdn")
    env = {
        "pyhealth_commit": git_commit(Path(pyhealth.__file__).parent),
        "runner_commit": git_commit(Path(__file__).parent),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0)
        if args.device.startswith("cuda")
        else None,
        "python": platform.python_version(),
    }
    benchmark = None  # built on first use, so a finished run resumes instantly
    implementation = implementation_hashes()
    records = []
    total = len(configs) * len(args.splits) * len(args.seeds)
    done = 0
    for split in args.splits:
        prepared = None
        inputs = input_hashes(args.root, args.released_dir, args.dataset, split)
        for (name, model_cfg, train_cfg), seed in itertools.product(
            configs, args.seeds
        ):
            done += 1
            request = {
                "dataset": args.dataset,
                "split": split,
                "seed": seed,
                "model": model_cfg,
                "train": train_cfg,
                "inputs": inputs,
                "implementation": implementation,
                "runtime": {key: env[key] for key in ("python", "torch", "cuda", "gpu")}
                | {"device": args.device},
            }
            if args.dataset == "gbsg":
                request["gbsg_standardize"] = args.standardize_gbsg
            parent = args.out / name / f"split_{split}" / f"seed_{seed}"
            run_dir, saved = resolve_run(parent, request)
            if saved is not None:
                records.append(saved)
                print(f"[{done}/{total}] {run_dir} done, skipping")
                continue
            if benchmark is None:
                benchmark = BENCHMARKS[args.dataset](
                    args.root,
                    args.released_dir,
                    args.out / "cache",
                    **(
                        {"standardize": args.standardize_gbsg}
                        if args.dataset == "gbsg"
                        else {}
                    ),
                )
            if prepared is None:
                prepared = benchmark.split(split)
            ids, samples, preprocessing = prepared
            run_dir.mkdir(parents=True, exist_ok=True)
            parts = split_parts(samples, ids, seed)
            record = run_one(
                name, model_cfg, train_cfg, split, seed, parts, run_dir, args.device
            )
            record["dataset"] = args.dataset
            record["preprocessing"] = preprocessing
            record["environment"] = {**env, "device": args.device}
            record["request"] = request
            record["run_directory"] = str(run_dir.relative_to(args.out))
            record["artifacts"] = artifact_hashes(run_dir)
            (run_dir / "record.json").write_text(json.dumps(record, indent=2) + "\n")
            records.append(record)
            print(f"Saved run: {run_dir}")
            m = record["test_metrics"]
            print(
                f"[{done}/{total}] {name} split {split} seed {seed}: "
                f"val NLL {record['val_nll']:.4f}  C^td {m['c_td']:.3f}  "
                f"IBS {m['ibs']:.3f}  IBLL {m['ibll']:.3f}  "
                f"({record['train_seconds']:.1f}s)"
            )
    for entry in summarize(args.out, records):
        print(
            f"{entry['config']}: {entry['fits']} fits, val NLL "
            f"{entry['val_nll_mean']:.4f}, C^td {entry['c_td_mean']:.3f} "
            f"± {entry['c_td_sd']:.3f}, IBS {entry['ibs_mean']:.3f} "
            f"± {entry['ibs_sd']:.3f}, IBLL {entry['ibll_mean']:.3f} "
            f"± {entry['ibll_sd']:.3f} | reference-compatible IBS "
            f"{entry['ref_ibs@1e-08_mean']:.3f}, IBLL "
            f"{entry['ref_ibll@1e-08_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
