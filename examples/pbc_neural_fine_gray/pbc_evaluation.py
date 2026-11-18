"""PBC scoring, support validation, and aggregate study reports.

Contributors: Amey Deotale (ameysd2), Anushka Shinde (ashinde4),
Abid Hossain (ahossa5).
Paper: Neural Fine-Gray: Monotonic neural networks for competing risks.
https://proceedings.mlr.press/v209/jeanselme23a.html
"""

import json
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from pbc_experiment import (
    COMPARATORS,
    HORIZONS,
    MODES,
    PAPER_HORIZONS,
    RATES,
    Sample,
    corrupt,
    write_json,
)

from pyhealth.metrics.competing_risks import (
    CensoringDistribution,
    MetricSupportError,
    competing_risk_brier,
    competing_risk_concordance,
)
from pyhealth.models import NeuralFineGray
from pyhealth.tasks import CompetingRisksPreprocessor


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
