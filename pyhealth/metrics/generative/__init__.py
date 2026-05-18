"""Evaluation metrics for synthetic (generative) EHR data.

This subpackage provides metrics for assessing synthetic electronic health
record (EHR) data along three axes:

    - **Privacy** (:mod:`pyhealth.metrics.generative.privacy`): NNAAR,
      membership inference, and discriminator-based adversarial accuracy.
    - **Utility / fidelity** (:mod:`pyhealth.metrics.generative.utility`):
      machine learning efficacy (TRTR vs TSTR) and code-prevalence similarity.

The convenience function :func:`evaluate_synthetic_ehr` runs the full suite
and returns a single merged dictionary of ``{metric_name: (mean, std)}``.
"""

import logging
from typing import Dict, Optional, Tuple

import pandas as pd

from .privacy import (
    calc_membership_inference,
    calc_nnaar,
    compute_discriminator_privacy,
)
from .utility import compute_mle, compute_prevalence_metrics
from .utils import train_lstm_model, train_sklearn_model

logger = logging.getLogger(__name__)

__all__ = [
    "calc_nnaar",
    "calc_membership_inference",
    "compute_discriminator_privacy",
    "compute_mle",
    "compute_prevalence_metrics",
    "evaluate_synthetic_ehr",
]


def evaluate_synthetic_ehr(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    sample_size: int = 1000,
    mode: str = "lstm",
    metrics: str = "all",
    lstm_params: Optional[Dict] = None,
    sklearn_params: Optional[Dict] = None,
    n_bootstraps: int = 100,
    n_runs: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Runs the full synthetic-EHR evaluation suite.

    Computes privacy and/or utility metrics comparing synthetic EHR data
    against real train/test data, and returns a single merged dictionary.

    Args:
        train_ehr: Real training EHR dataframe.
        test_ehr: Real held-out test EHR dataframe.
        syn_ehr: Synthetic EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the label.
        sample_size: Number of patients sampled per dataset for the
            privacy metrics.
        mode: Predictive backbone for the utility metrics; ``"lstm"`` uses the
            built-in LSTM classifier, ``"rf"`` uses a random forest.
        metrics: Which metric group to compute: ``"all"``, ``"privacy"`` or
            ``"utility"``.
        lstm_params: Optional overrides for the LSTM (``embed_dim``,
            ``hidden_dim``, ``batch_size``, ``epochs``).
        sklearn_params: Optional overrides for the sklearn model (``model``).
        n_bootstraps: Number of bootstrap resamples for the utility metrics.
        n_runs: Number of sampling runs for the privacy metrics.

    Returns:
        Dictionary mapping each metric name to a ``(mean, std)`` tuple.

    Raises:
        ValueError: If ``metrics`` or ``mode`` is not a recognized value.
    """
    if metrics not in ("all", "privacy", "utility"):
        raise ValueError(
            f"Unknown metrics group: {metrics!r}. "
            "Expected 'all', 'privacy' or 'utility'."
        )
    if mode not in ("lstm", "rf"):
        raise ValueError(f"Unknown mode: {mode!r}. Expected 'lstm' or 'rf'.")

    lstm_params = lstm_params or {}
    sklearn_params = sklearn_params or {}
    final_output: Dict[str, Tuple[float, float]] = {}

    if metrics in ("all", "privacy"):
        final_output.update(
            calc_nnaar(
                train_ehr,
                test_ehr,
                syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                sample_size=sample_size,
                n_runs=n_runs,
            )
        )
        final_output.update(
            calc_membership_inference(
                train_ehr,
                test_ehr,
                syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                num_attack_samples=sample_size,
                n_runs=n_runs,
            )
        )

    if metrics in ("all", "utility"):
        if mode == "lstm":
            train_fn = train_lstm_model
            train_kwargs = {
                "embed_dim": lstm_params.get("embed_dim", 32),
                "hidden_dim": lstm_params.get("hidden_dim", 32),
                "batch_size": lstm_params.get("batch_size", 32),
                "epochs": lstm_params.get("epochs", 5),
                "verbose": False,
            }
        else:
            train_fn = train_sklearn_model
            train_kwargs = {"model": sklearn_params.get("model", "rf")}

        final_output.update(
            compute_mle(
                train_fn=train_fn,
                train_ehr=train_ehr,
                test_ehr=test_ehr,
                syn_ehr=syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                n_bootstraps=n_bootstraps,
                **train_kwargs,
            )
        )
        final_output.update(
            compute_discriminator_privacy(
                train_fn=train_fn,
                train_ehr=train_ehr,
                test_ehr=test_ehr,
                syn_ehr=syn_ehr,
                subject_col=subject_col,
                visit_col=visit_col,
                code_col=code_col,
                label_col=label_col,
                n_bootstraps=n_bootstraps,
                **train_kwargs,
            )
        )
        final_output.update(
            compute_prevalence_metrics(
                train_ehr,
                syn_ehr,
                subject_col=subject_col,
                code_col=code_col,
                n_bootstraps=n_bootstraps,
            )
        )

    return final_output
