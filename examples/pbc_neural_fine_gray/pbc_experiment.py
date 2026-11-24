"""Data, training, comparators, and provenance for the PBC example.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import argparse
import csv
import hashlib
import json
import logging
import math
import time
from collections.abc import Sequence
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import PBCDataset, create_sample_dataset, get_dataloader
from pyhealth.models import NeuralFineGray
from pyhealth.models.neural_fine_gray import MaskMode
from pyhealth.tasks import CompetingRisksPBC
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
    """Bind resume to library code/configuration, the entry point, and example helpers.

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
