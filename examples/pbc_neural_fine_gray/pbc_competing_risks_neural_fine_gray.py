"""Run PBC Neural Fine-Gray with paired missingness and architecture ablations.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
Install optional requirements-neural-fine-gray.txt for comparators/reference
metrics. See README.md for data access, commands, and the study protocol.
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import time
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import ParameterSampler, StratifiedKFold, train_test_split

from pyhealth.datasets import PBCDataset, create_sample_dataset, get_dataloader
from pyhealth.metrics.competing_risks import (
    CensoringDistribution,
    MetricSupportError,
    competing_risk_brier,
    competing_risk_concordance,
)
from pyhealth.models import NeuralFineGray
from pyhealth.models.neural_fine_gray import MaskMode
from pyhealth.tasks import CompetingRisksPBC, CompetingRisksPreprocessor
from pyhealth.tasks.competing_risks_pbc import (
    PBC_CATEGORIES,
    PBC_FEATURES,
    PBC_MASK_INDICES,
    PBC_NUMERIC,
)
from pyhealth.trainer import Trainer

Sample = dict[str, Any]
HORIZONS = (3.19, 4.95, 5.0, 7.45)
PAPER_HORIZONS = (2.19, 3.95, 6.45)
RATES = (0.0, 0.1, 0.3)
MODES: tuple[MaskMode, ...] = ("original", "constant", "observed", "augmented")
COMPARATORS = ("aalen_johansen", "cause_specific_cox")
COX_PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)
WIDTHS = (25, 50)
DROPOUTS = (0.0, 0.25, 0.5, 0.75)
LOGGER = logging.getLogger(__name__)


def write_json(path: Path, payload: Any) -> None:
    """Replace a JSON file only after its new contents have been written.

    Args:
        path: Destination in the experiment's controlled output directory.
        payload: JSON-serializable metadata. NaN and infinity are not accepted.

    Raises:
        ValueError: The payload contains a nonfinite number.
        OSError: Writing or replacing the file fails.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(path: Path) -> str:
    """Compute a file's SHA-256 for source, data, and checkpoint checks.

    Args:
        path: Existing local file to hash.

    Returns:
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_source_hashes(root: Path) -> dict[str, str]:
    """Bind resume to library code/configuration and both pipeline entry points.

    Args:
        root: PyHealth checkout containing pyhealth and examples.

    Returns:
        Relative source paths mapped to content hashes, including local edits
        and newly added Python/configuration files but excluding bytecode/caches.
    """
    sources = [
        path
        for path in (root / "pyhealth").rglob("*")
        if path.is_file() and path.suffix in {".py", ".yaml", ".yml"}
    ]
    example = root / "examples" / "pbc_neural_fine_gray"
    sources.extend(example.rglob("*.py"))
    sources.extend(
        [root / "pyproject.toml", example / "requirements-neural-fine-gray.txt"]
    )
    return {
        path.relative_to(root).as_posix(): fingerprint(path) for path in sorted(sources)
    }


def synthetic_csv(path: Path, seed: int, size: int = 60) -> None:
    """Write invented PBC-shaped rows for an end-to-end smoke run.

    The records are not sampled from the medical dataset. Event classes are
    balanced, and some baseline numeric values are missing so the smoke run
    exercises imputation as well as dataset loading.

    Args:
        path: Local CSV destination.
        seed: Random seed for repeatable fabricated records.
        size: Number of rows. The runner uses 60; unit tests use tiny fixtures.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(size):
        row = {
            "id": f"synthetic-{index}",
            "year": 0.0,
            "years": rng.uniform(1.0, 12.0),
            "status": ("alive", "dead", "transplanted")[index % 3],
            **{name: rng.choice(levels) for name, levels in PBC_CATEGORIES.items()},
            **{name: rng.uniform(1.0, 10.0) for name in PBC_NUMERIC},
        }
        for name in PBC_NUMERIC:
            if rng.random() < 0.05:
                row[name] = ""
        rows.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_samples(args: argparse.Namespace) -> tuple[list[Sample], dict[str, Any]]:
    """Load baseline samples through PBCDataset and its public set_task method.

    The task's raw processors do not learn feature statistics. Imputation and
    scaling happen later, after patient-level splitting. The returned audit
    contains cohort counts and schema details, never individual records or IDs.

    Args:
        args: Parsed synthetic/data-root, output-directory, and seed options.

    Returns:
        Raw baseline samples and a JSON-serializable aggregate cohort audit.

    Raises:
        ValueError: The task returns no samples or duplicate baseline patients.
    """
    if args.synthetic:
        source = args.output_dir / "private-source" / "pbc2.csv"
        if not source.exists():
            synthetic_csv(source, args.seed)
    else:
        source = args.data_root / "pbc2.csv"
    cohort = PBCDataset(
        str(source.parent.resolve()),
        cache_dir=str((args.output_dir / "private-cache").resolve()),
        num_workers=1,
    )
    dataset = cohort.set_task(CompetingRisksPBC(), num_workers=1)
    samples = sorted(dataset, key=lambda sample: sample["patient_id"])
    if not samples or len({s["patient_id"] for s in samples}) != len(samples):
        raise ValueError("Expected exactly one nonempty baseline sample per patient")
    features = np.asarray([s["features"] for s in samples], float)
    with source.open(newline="", encoding="utf-8-sig") as stream:
        visit_rows = sum(1 for _ in csv.DictReader(stream))
    audit = {
        "source": "fabricated" if args.synthetic else "author-fork pbc2.csv",
        "sha256": fingerprint(source),
        "visit_rows": visit_rows,
        "patients": len(samples),
        "feature_count": len(PBC_FEATURES),
        "feature_names": list(PBC_FEATURES),
        "event_mapping": {"0": "censored", "1": "death", "2": "transplant"},
        "event_counts": {
            str(risk): sum(s["event"] == risk for s in samples) for risk in (0, 1, 2)
        },
        "missing_by_feature": dict(
            zip(PBC_FEATURES, np.isnan(features).sum(0).tolist())
        ),
        "baseline": "earliest whole row; no backfill; baseline age only",
        "time": "unshifted years, baseline entry at zero",
    }
    return samples, audit


def corrupt(samples: list[Sample], rate: float, seed: int) -> list[Sample]:
    """Mask three baseline biomarkers using shared, nested random draws.

    Reusing the seed makes the 10% mask a subset of the 30% mask. Every model
    sees the same held-out corruption. Existing missing values stay missing.

    Args:
        samples: Raw, un-imputed samples in a fixed patient order.
        rate: Additional missingness probability in [0, 1].
        seed: Seed shared across arms and corruption rates within a fold.

    Returns:
        Sample copies with selected bilirubin, albumin, and cholesterol values
        set to NaN. Outcomes and other features are unchanged.

    Raises:
        ValueError: The missingness probability is invalid.
    """
    if not np.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Missingness rate must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    result = []
    for sample in samples:
        features = np.asarray(sample["features"], float).copy()
        indices = np.asarray(PBC_MASK_INDICES)[rng.random(len(PBC_MASK_INDICES)) < rate]
        features[indices] = np.nan
        result.append({**sample, "features": features.tolist()})
    return result


def curve_checks(predictions: np.ndarray, origin: bool = False) -> dict[str, bool]:
    """Check whether a predicted grid can represent competing first events.

    Args:
        predictions: CIF array [patients, ordered horizons, two risks].
        origin: Whether the first horizon is zero and must have zero incidence.

    Returns:
        Shape, finiteness, monotonicity, probability-mass, and optional origin
        checks for the saved audit.

    Raises:
        FloatingPointError: A probability or curve invariant fails.
    """
    checks = {
        "shape": predictions.ndim == 3 and predictions.shape[2] == 2,
        "finite": bool(np.isfinite(predictions).all()),
        "monotonic": bool((np.diff(predictions, axis=1) >= -1e-10).all()),
        "mass_valid": bool(
            (predictions >= 0).all() and (predictions.sum(2) <= 1 + 1e-10).all()
        ),
    }
    if origin:
        checks["origin_zero"] = bool((predictions[:, 0] == 0).all())
    if not all(checks.values()):
        raise FloatingPointError(f"Invalid cumulative incidence: {checks}")
    return checks


def fit_model(
    train: list[Sample],
    validation: list[Sample],
    config: dict[str, Any],
    mode: MaskMode,
    directory: Path,
    args: argparse.Namespace,
    *,
    training_seed: int | None = None,
) -> tuple[NeuralFineGray, dict[str, Any]]:
    """Train one arm with PyHealth Trainer or restore its verified checkpoint.

    Early stopping uses inner-validation loss. The best weights, not the last
    epoch, are saved. On resume, both the checkpoint hash and its recomputed
    validation loss must agree with the saved metadata.

    Args:
        train: Samples transformed with inner-training statistics.
        validation: Inner-validation samples using those same statistics.
        config: Encoder/head layers, dropout, batch size, and learning rate.
        mode: Original, constant, observed, or augmented missingness arm.
        directory: This fit's private checkpoint, log, and metadata directory.
        args: Seed, device, epoch limit, and early-stopping patience.
        training_seed: Optional initialization/shuffle/dropout seed override.
            Patient splits and held-out corruption keep the study seed.

    Returns:
        The selected model in evaluation mode and its fit metadata.

    Raises:
        ValueError: Saved configuration, checkpoint hash, or validation loss
            does not match this fit.
        FloatingPointError: Validation produces a nonfinite likelihood.
    """
    task = CompetingRisksPBC()
    train_dataset = create_sample_dataset(train, task.input_schema, task.output_schema)
    seed = args.seed if training_seed is None else training_seed
    torch.manual_seed(seed)
    model = NeuralFineGray(
        train_dataset,
        len(PBC_FEATURES),
        hidden_dims=config["layers"],
        survival_dims=config["layers_surv"],
        dropout=config["dropout"],
        mask_mode=mode,
        mask_indices=PBC_MASK_INDICES,
    ).to(args.device)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint, metadata = directory / "best.ckpt", directory / "fit.json"
    identity = {
        "config": config,
        "mode": mode,
        "epochs": args.epochs,
        "patience": args.patience,
        "seed": seed,
    }
    batch = {
        name: [s[name] for s in validation]
        for name in ("features", "missing_mask", "time", "event")
    }
    if metadata.exists():
        info = json.loads(metadata.read_text())
        if info["identity"] != identity or info["checkpoint_sha256"] != fingerprint(
            checkpoint
        ):
            raise ValueError(f"Checkpoint identity/hash mismatch: {directory}")
        model.load_state_dict(
            torch.load(checkpoint, map_location=args.device, weights_only=True)
        )
        model.eval()
        with torch.no_grad():
            loss = float(model(**batch)["loss"])
        if not math.isclose(loss, info["validation_nll_scaled"], abs_tol=1e-10):
            raise ValueError("Restored checkpoint validation loss differs")
        return model, info
    train_loader = get_dataloader(
        train_dataset, batch_size=config["batch"], shuffle=True
    )
    val_dataset = create_sample_dataset(
        validation,
        task.input_schema,
        task.output_schema,
    )
    val_loader = get_dataloader(val_dataset, batch_size=config["batch"], shuffle=False)
    trainer = Trainer(model, device=args.device, enable_logging=False)
    steps = [0]

    def count_steps(module: NeuralFineGray, inputs: Any, output: Any) -> None:
        """Count training batches without counting validation forwards."""
        if module.training:
            steps[0] += 1

    hook = model.register_forward_hook(count_steps)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    trainer_logger = logging.getLogger("pyhealth.trainer")
    previous_propagation = trainer_logger.propagate
    try:
        with (
            (directory / "training.log").open("w") as stream,
            redirect_stdout(stream),
            redirect_stderr(stream),
        ):
            handler = logging.StreamHandler(stream)
            trainer_logger.addHandler(handler)
            trainer_logger.propagate = False
            try:
                trainer.train(
                    train_loader,
                    val_loader,
                    epochs=args.epochs,
                    patience=args.patience,
                    optimizer_params={"lr": config["learning_rate"]},
                    monitor="loss",
                    monitor_criterion="min",
                    max_grad_norm=10.0,
                )
            finally:
                trainer_logger.removeHandler(handler)
                handler.close()
                trainer_logger.propagate = previous_propagation
        model.eval()
        with torch.no_grad():
            loss = float(model(**batch)["loss"])
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation likelihood")
        trainer.save_ckpt(str(checkpoint))
        info = {
            "identity": identity,
            "validation_nll_scaled": loss,
            "seconds": time.perf_counter() - started,
            "epochs_completed": math.ceil(steps[0] / len(train_loader)),
            "training_steps": steps[0],
            "parameters": sum(p.numel() for p in model.parameters()),
            "checkpoint_sha256": fingerprint(checkpoint),
            "peak_gpu_allocated_mib": (
                torch.cuda.max_memory_allocated(args.device) / 2**20
                if args.device.startswith("cuda")
                else 0.0
            ),
        }
        write_json(metadata, info)
    finally:
        hook.remove()
    return model, info


def evaluate(
    predictions: np.ndarray,
    train: list[Sample],
    test: list[Sample],
    fold: int,
    mode: str,
    rate: float,
    *,
    horizons: Sequence[float] = HORIZONS,
) -> list[Sample]:
    """Score Brier and concordance separately for each cause and horizon.

    A fold without comparable transplant pairs can still have a valid
    transplant Brier score. Only expected support errors become unsupported
    rows; malformed data and other failures still stop the run.

    Args:
        predictions: Held-out CIFs [patients, horizons, two risks].
        train: Raw training labels used to estimate censoring.
        test: Raw held-out labels, in prediction order.
        fold: Outer-fold number.
        mode: Model or comparator name.
        rate: Held-out biomarker corruption probability.
        horizons: Ordered scoring times in unshifted years.

    Returns:
        Four rows per horizon, with finite values or unsupported reasons.
    """
    curve_checks(predictions)
    if predictions.shape != (len(test), len(horizons), 2):
        raise ValueError("Prediction grid does not match held-out patients/horizons")
    censoring = CensoringDistribution(
        np.array([s["time"] for s in train]),
        np.array([s["event"] for s in train]),
    )
    times, events = (
        np.array([s["time"] for s in test]),
        np.array([s["event"] for s in test]),
    )
    rows = []
    for position, horizon in enumerate(horizons):
        for risk in (1, 2):
            for metric, function in (
                ("brier", competing_risk_brier),
                ("concordance", competing_risk_concordance),
            ):
                row = {
                    "fold": fold,
                    "mode": mode,
                    "missing_rate": rate,
                    "horizon": horizon,
                    "risk": risk,
                    "metric": metric,
                    "lane": "corrected",
                }
                try:
                    value = function(
                        times,
                        events,
                        predictions[:, position, risk - 1],
                        horizon,
                        risk,
                        censoring,
                    )
                    if isinstance(value, tuple):
                        value, pairs = value
                        row["pairs"] = pairs
                    row.update(status="ok", value=float(value))
                except MetricSupportError as error:
                    row.update(status="unsupported", reason=str(error))
                rows.append(row)
    return rows


def historical_metrics(
    predictions: np.ndarray,
    train: list[Sample],
    test: list[Sample],
    fold: int,
) -> list[Sample]:
    """Apply the reference analysis's binary-cause metric convention.

    This lane is descriptive. It treats each cause as a binary survival label
    and therefore does not estimate the same quantity as the corrected
    competing-risk scores. It never selects a model or replaces a primary score.

    Args:
        predictions: Clean original-arm CIFs on HORIZONS.
        train: Training patients used for reference censoring weights.
        test: Held-out patients in prediction order.
        fold: Outer-fold number.

    Returns:
        Twelve rows at 3.19, 4.95, and 7.45 years, with support reasons retained.

    Raises:
        ValueError: The reference library reports an unexpected input error.
        FloatingPointError: A returned score is nonfinite.
    """
    from sksurv.metrics import brier_score, concordance_index_ipcw

    train_time, test_time = (
        np.array([s["time"] for s in group]) for group in (train, test)
    )
    train_event, test_event = (
        np.array([s["event"] for s in group]) for group in (train, test)
    )
    rows = []
    for risk in (1, 2):
        train_labels = np.array(
            list(zip(train_event == risk, train_time)),
            dtype=[("event", bool), ("time", float)],
        )
        selection = (test_time < train_time.max()) | (test_event != risk)
        test_labels = np.array(
            list(zip(test_event[selection] == risk, test_time[selection])),
            dtype=train_labels.dtype,
        )
        for horizon in (3.19, 4.95, 7.45):
            prediction = predictions[selection, HORIZONS.index(horizon), risk - 1]
            for metric in ("brier", "concordance"):
                row = {
                    "fold": fold,
                    "mode": "original",
                    "missing_rate": 0.0,
                    "horizon": horizon,
                    "risk": risk,
                    "metric": metric,
                    "lane": "paper_binary_cause",
                }
                # Valid arrays reach the reference library; its known support
                # exceptions are reported, while other errors remain failures.
                try:
                    if metric == "brier":
                        value = brier_score(
                            train_labels,
                            test_labels,
                            (1 - prediction)[:, None],
                            [horizon],
                        )[1][0]
                    else:
                        value = concordance_index_ipcw(
                            train_labels,
                            test_labels,
                            prediction,
                            tau=horizon,
                        )[0]
                    if not np.isfinite(value):
                        raise FloatingPointError("Nonfinite historical score")
                    row.update(status="ok", value=float(value))
                except ValueError as error:
                    support_messages = (
                        "time must be smaller",
                        "censoring survival function is zero",
                        "all samples are censored",
                        "all times must be within follow-up",
                        "No comparable pairs",
                    )
                    if not any(
                        text.lower() in str(error).lower() for text in support_messages
                    ):
                        raise
                    row.update(status="unsupported", reason=str(error))
                rows.append(row)
    return rows


def fit_cox(train: list[Sample], penalty: float = 0.1) -> tuple[list[Any], list[str]]:
    """Fit one cause-specific Cox model per first event.

    Args:
        train: Inner-training samples with imputed and standardized features.
        penalty: Positive finite L2 penalty shared by the two cause models.

    Returns:
        Two lifelines models and the retained feature names.
        Columns constant in training are excluded from both fits.
    """
    from lifelines import CoxPHFitter

    if not math.isfinite(penalty) or penalty <= 0:
        raise ValueError("Cox penalty must be positive and finite")
    frame = pd.DataFrame([s["features"] for s in train], columns=PBC_FEATURES)
    columns = frame.columns[frame.std() > 1e-10].tolist()
    frame = frame[columns].copy()
    frame["duration"] = [s["time"] for s in train]
    models = []
    for risk in (1, 2):
        frame["observed"] = [s["event"] == risk for s in train]
        model = CoxPHFitter(penalizer=penalty)
        model.fit(frame, duration_col="duration", event_col="observed")
        models.append(model)
    return models, columns


def select_cox(
    train: list[Sample],
    validation: list[Sample],
    penalties: Sequence[float],
) -> tuple[list[Any], list[str], dict[str, Any]]:
    """Select a shared Cox penalty using held-out partial log-likelihood.

    Args:
        train: Inner-training samples, also used to select retained columns.
        validation: Inner-validation samples in the same preprocessing units.
        penalties: Nonempty, distinct positive L2 penalties.

    Returns:
        Selected training-only models, columns, and all validation scores.
        Neither preprocessing nor the selected models are refit on validation.

    Raises:
        ValueError: The candidate grid or validation cohort is invalid.
        FloatingPointError: A validation likelihood is nonfinite.
    """
    if (
        not validation
        or not penalties
        or len(set(penalties)) != len(penalties)
        or any(not math.isfinite(penalty) or penalty <= 0 for penalty in penalties)
    ):
        raise ValueError("Require validation patients and distinct Cox penalties")
    candidates = []
    best: tuple[list[Any], list[str]] | None = None
    best_score = -math.inf
    selected_penalty = None
    for penalty in penalties:
        models, columns = fit_cox(train, penalty)
        frame = pd.DataFrame([s["features"] for s in validation], columns=PBC_FEATURES)
        frame = frame[columns].copy()
        frame["duration"] = [s["time"] for s in validation]
        scores = []
        for risk, model in enumerate(models, start=1):
            frame["observed"] = [s["event"] == risk for s in validation]
            score = float(model.score(frame, scoring_method="log_likelihood"))
            if not math.isfinite(score):
                raise FloatingPointError("Nonfinite Cox validation likelihood")
            scores.append(score)
        if len(scores) != 2:
            raise ValueError("Expected two fitted cause-specific Cox models")
        score = sum(scores)
        if not math.isfinite(score):
            raise FloatingPointError("Nonfinite combined Cox validation likelihood")
        candidates.append(
            {"penalty": penalty, "validation_partial_log_likelihood": score}
        )
        if score > best_score:
            best, best_score, selected_penalty = (models, columns), score, penalty
    if best is None:
        raise RuntimeError("No validated Cox model selected")
    return *best, {
        "criterion": "summed mean cause-specific validation partial log-likelihood",
        "selected_penalty": selected_penalty,
        "candidates": candidates,
        "refit_on_validation": False,
    }


def predict_cox(
    models: list[Any],
    columns: list[str],
    samples: list[Sample],
    horizons: np.ndarray,
) -> np.ndarray:
    """Combine fitted cause hazards without allocating more than unit mass.

    Each interval allocates its event probability in proportion to the cause
    hazard increments, then updates event-free survival. This piecewise-constant
    approximation is not the paper's R riskRegression estimator.

    Args:
        models: Two fitted cause-specific Cox models.
        columns: Feature names retained by fit_cox.
        samples: Transformed held-out samples in prediction order.
        horizons: Prediction times in the fitted preprocessing units.

    Returns:
        Competing-risk CIFs [patients, horizons, two risks].
    """
    frame = pd.DataFrame([s["features"] for s in samples], columns=PBC_FEATURES)
    grid = np.unique(
        np.concatenate(
            (
                [0.0],
                horizons,
                *[
                    model.baseline_cumulative_hazard_.index.to_numpy()
                    for model in models
                ],
            )
        )
    )
    hazards = np.stack(
        [
            model.predict_cumulative_hazard(frame[columns])
            .reindex(grid)
            .ffill()
            .fillna(0)
            .to_numpy()
            .T
            for model in models
        ],
        axis=2,
    )
    survival, incidence = np.ones(len(samples)), np.zeros((len(samples), 2))
    curves = np.zeros_like(hazards)
    for position in range(1, len(grid)):
        increment = np.maximum(hazards[:, position] - hazards[:, position - 1], 0)
        total = increment.sum(1)
        fraction = np.divide(
            increment,
            total[:, None],
            out=np.zeros_like(increment),
            where=total[:, None] > 0,
        )
        incidence += (survival * -np.expm1(-total))[:, None] * fraction
        survival *= np.exp(-total)
        curves[:, position] = incidence
    return curves[:, np.searchsorted(grid, horizons)]


def validate_rows(
    rows: list[Sample],
    folds: Sequence[int],
    historical: bool,
    *,
    modes: Sequence[str] = (*MODES, *COMPARATORS),
    horizons: Sequence[float] = HORIZONS,
) -> None:
    """Require every expected metric row, even when its score is unsupported.

    Args:
        rows: Metric records produced by evaluation.
        folds: Outer folds that must be present.
        historical: Whether clean-original binary-cause rows are also required.
        modes: Required model or architecture labels.
        horizons: Required corrected scoring times.

    Raises:
        ValueError: Rows are missing, duplicated, nonfinite, or lack a support
            reason. Unsupported values are never replaced with zero.
    """
    expected = {
        (fold, mode, rate, horizon, risk, metric, "corrected")
        for fold in folds
        for mode in modes
        for rate in RATES
        for horizon in horizons
        for risk in (1, 2)
        for metric in ("brier", "concordance")
    }
    if historical:
        expected |= {
            (fold, "original", 0.0, horizon, risk, metric, "paper_binary_cause")
            for fold in folds
            for horizon in (3.19, 4.95, 7.45)
            for risk in (1, 2)
            for metric in ("brier", "concordance")
        }
    fields = ("fold", "mode", "missing_rate", "horizon", "risk", "metric", "lane")
    keys = [tuple(row[field] for field in fields) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("Incomplete or duplicate metric grid")
    for row in rows:
        if row["status"] == "ok":
            if not math.isfinite(row["value"]):
                raise ValueError("Supported metric must be finite")
        elif row["status"] != "unsupported" or not row.get("reason"):
            raise ValueError("Unsupported metrics require explicit reasons")


def _metric_frame(rows: list[Sample]) -> pd.DataFrame:
    """Retain the score column when every metric is unsupported."""
    frame = pd.DataFrame(rows)
    if "value" not in frame:
        if not (frame.status == "unsupported").all():
            raise ValueError("Supported metrics must include a score")
        frame["value"] = np.nan
    return frame


def report(directory: Path, rows: list[Sample], *, h2: bool = True) -> None:
    """Write aggregate scores and paired-fold missingness effects.

    H2 compares augmentation with the constant-mask control at 10% and 30%
    corruption, and with the original arm on clean data. Negative Brier
    differences favor augmentation. These comparisons are descriptive, not
    significance tests, and negative findings remain in the report.

    Args:
        directory: Controlled output directory for CSV, JSON, and Markdown.
        rows: Metric rows from the completed folds available so far.
        h2: Whether these rows contain the primary four-arm H2 comparison.
    """
    directory.mkdir(parents=True, exist_ok=True)
    frame = _metric_frame(rows)
    frame.to_csv(directory / "metrics.csv", index=False)
    summary = (
        frame[frame.status == "ok"]
        .groupby(["mode", "missing_rate", "horizon", "risk", "metric", "lane"])
        .value.agg(mean="mean", sd="std", supported_folds="count")
        .reset_index()
    )
    summary.to_csv(directory / "summary.csv", index=False)
    if not h2:
        return
    primary = frame[
        (frame.lane == "corrected")
        & (frame.metric == "brier")
        & (frame.risk == 1)
        & (frame.horizon == 5.0)
        & (frame.status == "ok")
    ]
    pairs = []
    for rate, control in ((0.0, "original"), (0.1, "constant"), (0.3, "constant")):
        selected = primary[primary.missing_rate == rate]
        augmented = selected[selected["mode"] == "augmented"].set_index("fold").value
        comparison = selected[selected["mode"] == control].set_index("fold").value
        common = augmented.index.intersection(comparison.index)
        for fold in common:
            pairs.append(
                {
                    "fold": int(fold),
                    "rate": rate,
                    "control": control,
                    "augmented_minus_control": float(
                        augmented[fold] - comparison[fold]
                    ),
                }
            )
    pd.DataFrame(
        pairs, columns=["fold", "rate", "control", "augmented_minus_control"]
    ).to_csv(directory / "paired_effects.csv", index=False)
    lines = [
        "# PBC measured results",
        "",
        (
            "Negative Brier differences favor augmentation. These are descriptive "
            "paired-fold effects, not an exact-replication or superiority claim."
        ),
        "",
        "| Test missingness | Control | Paired folds | Mean Brier difference |",
        "|---:|---|---:|---:|",
    ]
    effects = {}
    for rate, control in ((0.0, "original"), (0.1, "constant"), (0.3, "constant")):
        values = [
            row["augmented_minus_control"] for row in pairs if row["rate"] == rate
        ]
        effect = float(np.mean(values)) if values else None
        effects[str(rate)] = {"paired_folds": len(values), "mean_difference": effect}
        display = "unsupported" if effect is None else f"{effect:.6f}"
        lines.append(f"| {rate:.0%} | {control} | {len(values)} | {display} |")
    complete = len(frame.fold.unique())
    full_support = all(value["paired_folds"] == complete for value in effects.values())
    h2 = full_support and (
        effects["0.0"]["mean_difference"] <= 0
        and effects["0.1"]["mean_difference"] < 0
        and effects["0.3"]["mean_difference"] < 0
    )
    write_json(
        directory / "h2.json",
        {
            "effects": effects,
            "all_available_folds_supported": full_support,
            "directional_criteria_met": bool(h2),
        },
    )
    lines.extend(
        [
            "",
            f"H2 directional criteria met on these folds: **{h2}**.",
            f"Unsupported metric rows: {int((frame.status != 'ok').sum())}.",
            "See metrics.csv for every score and explicit unsupported reason.",
        ]
    )
    (directory / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def architecture_configurations(selected: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Vary joint hidden width or encoder dropout, keeping other settings fixed.

    Args:
        selected: Configuration selected by original-arm validation likelihood.

    Returns:
        The selected baseline, two width cases, and four dropout cases.
        Encoder/head depths, learning rate, and batch size remain unchanged.
    """
    cases = {"selected": dict(selected)}
    for width in WIDTHS:
        cases[f"width-{width}"] = {
            **selected,
            "layers": [width] * len(selected["layers"]),
            "layers_surv": [width] * len(selected["layers_surv"]),
        }
    for dropout in DROPOUTS:
        cases[f"dropout-{dropout:g}"] = {**selected, "dropout": dropout}
    return cases


def score_neural(
    model: NeuralFineGray,
    processor: CompetingRisksPreprocessor,
    train: list[Sample],
    test: list[Sample],
    fold: int,
    label: str,
    directory: Path,
    corruption_seed: int,
    *,
    paper_time: bool = False,
) -> tuple[list[Sample], list[Sample]]:
    """Save and score neural CIFs using identical held-out masks across fits.

    Args:
        model: Selected model in evaluation mode.
        processor: Preprocessing fitted on inner-training patients.
        train: Raw inner-training censoring labels.
        test: Raw outer-test samples; outcomes are never changed by masking.
        fold: Outer fold index.
        label: Model or architecture label in the metric rows.
        directory: Controlled prediction and curve-check destination.
        corruption_seed: Shared seed for nested test corruption.
        paper_time: Also score the supplementary unshifted paper-time grid.

    Returns:
        Primary and supplementary paper-time rows, kept separate.
    """
    directory.mkdir(parents=True, exist_ok=True)
    processed = processor.transform(test)
    scale = processor.state["time_scale"]
    with torch.no_grad():
        curves = (
            model.predict_cif(
                [s["features"] for s in processed[:10]],
                np.linspace(0.0, max(s["time"] for s in train), 51) / scale,
                [s["missing_mask"] for s in processed[:10]],
            )
            .cpu()
            .numpy()
        )
    write_json(directory / "curve_checks.json", curve_checks(curves, origin=True))
    horizons = np.array(sorted({*HORIZONS, *(PAPER_HORIZONS if paper_time else ())}))
    primary_indices = np.searchsorted(horizons, HORIZONS)
    paper_indices = np.searchsorted(horizons, PAPER_HORIZONS)
    rows, paper_rows = [], []
    for rate in RATES:
        processed = processor.transform(corrupt(test, rate, corruption_seed))
        with torch.no_grad():
            prediction = (
                model.predict_cif(
                    [s["features"] for s in processed],
                    horizons / scale,
                    [s["missing_mask"] for s in processed],
                )
                .cpu()
                .numpy()
            )
        curve_checks(prediction)
        saved = {"cif": prediction[:, primary_indices]}
        rows.extend(evaluate(saved["cif"], train, test, fold, label, rate))
        if paper_time:
            saved["paper_cif"] = prediction[:, paper_indices]
            paper_rows.extend(
                evaluate(
                    saved["paper_cif"],
                    train,
                    test,
                    fold,
                    label,
                    rate,
                    horizons=PAPER_HORIZONS,
                )
            )
        np.savez_compressed(directory / f"private_predictions_{rate:.1f}.npz", **saved)
    return rows, paper_rows


def report_sensitivity(
    directory: Path, rows: list[Sample], folds: Sequence[int], seeds: Sequence[int]
) -> None:
    """Report each matched training seed without pooling it into primary H2.

    Args:
        directory: Supplementary aggregate-output directory.
        rows: Four-arm primary-horizon scores with a training_seed field.
        folds: Required outer folds for every seed.
        seeds: Fresh training seeds specified in the study manifest.

    Raises:
        ValueError: A seed, fold, model, or metric row is missing or duplicated.
    """
    if (
        not seeds
        or len(set(seeds)) != len(seeds)
        or {row["training_seed"] for row in rows} != set(seeds)
    ):
        raise ValueError("Incomplete sensitivity seed grid")
    directory.mkdir(parents=True, exist_ok=True)
    summaries, decisions = [], []
    for seed in seeds:
        selected = [row for row in rows if row["training_seed"] == seed]
        validate_rows(selected, folds, False, modes=MODES)
        destination = directory / f"seed-{seed}"
        report(destination, selected)
        decision = json.loads((destination / "h2.json").read_text())
        decisions.append({"training_seed": seed, **decision})
        for rate, effect in decision["effects"].items():
            summaries.append(
                {"training_seed": seed, "missing_rate": float(rate), **effect}
            )
    _metric_frame(rows).to_csv(directory / "metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(directory / "paired_summary.csv", index=False)
    write_json(
        directory / "h2.json",
        {
            "seeds": decisions,
            "seed_count": len(seeds),
            "directional_criteria_met_count": sum(
                decision["directional_criteria_met"] for decision in decisions
            ),
            "primary_analysis_unchanged": True,
        },
    )


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
    samples, audit = load_samples(args)
    saved_manifest = args.output_dir / "manifest.json"
    previous_manifest = (
        json.loads(saved_manifest.read_text()) if saved_manifest.exists() else None
    )
    if previous_manifest is not None and previous_manifest["audit"] != audit:
        raise ValueError("Changed source data: use a new output directory")
    if args.audit_only:
        write_json(args.output_dir / "cohort_audit.json", audit)
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
        "source_hashes": pipeline_source_hashes(root),
    }
    if previous_manifest is not None and previous_manifest != manifest:
        raise ValueError(
            "Changed source, data, or protocol: use a new output directory"
        )
    write_json(saved_manifest, manifest)
    write_json(args.output_dir / "cohort_audit.json", audit)
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
        write_json(
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
        write_json(directory / "preprocessing.json", processor.state)
        processed_train, processed_validation = (
            processor.transform(train),
            processor.transform(validation),
        )
        started = time.perf_counter()
        best_info = None
        for trial, config in enumerate(configs):
            _, info = fit_model(
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
                model, info = fit_model(
                    processed_train,
                    processed_validation,
                    best_info["identity"]["config"],
                    mode,
                    directory / f"trial-{best_trial}",
                    args,
                )
            else:
                model, info = fit_model(
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
                write_json(
                    destination / "selection.json",
                    {
                        "trial": best_trial,
                        "checkpoint_sha256": info["checkpoint_sha256"],
                    },
                )
            measured, supplementary = score_neural(
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
                        historical_metrics(saved["cif"], train, test, fold)
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
        cox_models, columns, cox_selection = select_cox(
            processed_train, processed_validation, args.cox_penalties
        )
        write_json(directory / "cox_selection.json", cox_selection)
        for rate in RATES:
            fold_rows.extend(
                evaluate(
                    marginal[:, primary_indices],
                    train,
                    test,
                    fold,
                    "aalen_johansen",
                    rate,
                )
            )
            processed = processor.transform(
                corrupt(test, rate, args.seed + 1000 + fold)
            )
            prediction = predict_cox(
                cox_models,
                columns,
                processed,
                horizons / processor.state["time_scale"],
            )
            fold_rows.extend(
                evaluate(
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
                        evaluate(
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
                model, _ = fit_model(
                    processed_train,
                    processed_validation,
                    selected_config,
                    mode,
                    destination,
                    args,
                    training_seed=seed,
                )
                measured, _ = score_neural(
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
            cases = architecture_configurations(selected_config)
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
                model, info = fit_model(
                    processed_train,
                    processed_validation,
                    config,
                    "original",
                    checkpoint_dir,
                    args,
                )
                write_json(
                    destination / "selection.json",
                    {
                        "config": config,
                        "checkpoint": checkpoint_dir.relative_to(directory).as_posix(),
                        "checkpoint_sha256": info["checkpoint_sha256"],
                        "parameters": info["parameters"],
                        "selected_on_test": False,
                    },
                )
                measured, _ = score_neural(
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
        validate_rows(fold_rows, [fold], args.historical)
        write_json(directory / "metrics.json", fold_rows)
        timing = directory / "timing.json"
        if not timing.exists():
            write_json(timing, {"seconds": time.perf_counter() - started})
        rows.extend(fold_rows)
        report(args.output_dir, rows)
        print(f"Completed PBC fold {fold + 1}/{args.folds}", flush=True)
    validate_rows(rows, list(range(args.folds)), args.historical)
    folds = list(range(args.folds))
    if args.paper_time:
        validate_rows(paper_rows, folds, False, horizons=PAPER_HORIZONS)
        report(args.output_dir / "paper_time", paper_rows, h2=False)
    if args.sensitivity_seeds:
        report_sensitivity(
            args.output_dir / "seed_sensitivity",
            sensitivity_rows,
            folds,
            manifest["sensitivity_training_seeds"],
        )
    if args.architecture_ablation:
        cases = architecture_configurations(configs[0])
        validate_rows(architecture_rows, folds, False, modes=tuple(cases))
        report(args.output_dir / "architecture", architecture_rows, h2=False)
    primary_fits = args.folds * (len(configs) + len(MODES) - 1)
    sensitivity_fits = args.folds * args.sensitivity_seeds * len(MODES)
    write_json(
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
