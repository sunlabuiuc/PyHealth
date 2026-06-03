"""Utility and statistical-fidelity metrics for synthetic EHR data.

These metrics quantify how *useful* synthetic EHR data is as a stand-in for
real data:

    - Machine Learning Efficacy (MLE): compares a model trained on real data
      against one trained on synthetic data, both evaluated on real data.
    - Code-prevalence similarity: compares per-code patient-level prevalence
      between real and synthetic data (R-squared, Pearson correlation, RMSE).

All functions take flat / long-format EHR dataframes -- one row per
``(patient, visit, code)`` event, with default columns
``[id, time, visit_codes, labels]`` (see :mod:`pyhealth.metrics.generative`
for the full column contract) -- and return ``{metric_name: (mean, std)}``
summaries over bootstrap resamples. The real (``train_ehr``, ``test_ehr``) and
synthetic (``syn_ehr``) dataframes must share the same schema.
"""

import copy
import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics

from .utils import build_next_visit_prediction_dataset, summarize_metric_runs

logger = logging.getLogger(__name__)

__all__ = [
    "compute_mle",
    "compute_prevalence_metrics",
]


def compute_mle(
    train_fn: Callable,
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    n_bootstraps: int = 5,
    **kwargs,
) -> Dict[str, Tuple[float, float]]:
    """Computes Machine Learning Efficacy (utility) for synthetic data.

    Two classifiers are trained on a next-visit prediction task: one on real
    training data (Train-Real-Test-Real, TRTR) and one on synthetic data
    (Train-Synthetic-Test-Real, TSTR). Both are evaluated on the same real test
    set. Synthetic accuracy/F1 close to real accuracy/F1 indicates high utility.

    Note:
        The current implementation hard-codes the downstream task to
        next-visit prediction (built via
        :func:`build_next_visit_prediction_dataset`). This is degenerate for
        bag-of-codes generators such as MedGAN and CorGAN, which emit a
        single aggregate visit per patient and so always get label=0. A
        future revision will let callers plug in static-label tasks
        (mortality, readmission, "ever diagnosed with X", ...) so MLE is
        meaningful for both sequential (HALO, GPT2, PromptEHR) and
        bag-of-codes (MedGAN, CorGAN) generators.

    Args:
        train_fn: A training function such as
            :func:`pyhealth.metrics.generative.utils.train_lstm_model` or
            ``train_sklearn_model``, returning ``(model, y_true, y_pred)``.
        train_ehr: Real training EHR dataframe, flat
            ``[id, time, visit_codes, labels]`` format.
        test_ehr: Real held-out test EHR dataframe; same schema as ``train_ehr``.
        syn_ehr: Synthetic EHR dataframe; same schema as ``train_ehr``.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes (one code per row).
        label_col: Column name for the label (overwritten by the next-visit
            prediction label).
        n_bootstraps: Number of bootstrap resamples of the predictions.
        **kwargs: Extra keyword arguments forwarded to ``train_fn``.

    Returns:
        Dictionary mapping the MLE metrics (real/synthetic accuracy and F1,
            their difference and ratio) to their ``(mean, std)`` across
            bootstraps.

    Examples:
        >>> from pyhealth.metrics.generative.utility import compute_mle
        >>> from pyhealth.metrics.generative.utils import train_lstm_model
        >>> # train_ehr, test_ehr, syn_ehr are flat
        >>> # [id, time, visit_codes, labels] dataframes sharing one schema --
        >>> # see evaluate_synthetic_ehr for how to build them.
        >>> result = compute_mle(train_lstm_model, train_ehr, test_ehr, syn_ehr)
        >>> synth_acc_mean, synth_acc_std = result["MLE_Synth_Accuracy"]
    """
    logger.info("Computing MLE (utility)")

    train_task = build_next_visit_prediction_dataset(
        train_ehr, subject_col, visit_col, label_col
    )
    test_task = build_next_visit_prediction_dataset(
        test_ehr, subject_col, visit_col, label_col
    )
    syn_task = build_next_visit_prediction_dataset(
        syn_ehr, subject_col, visit_col, label_col
    )

    # Train on Real, test on Real (TRTR).
    _, real_y_true, real_y_pred = train_fn(
        copy.deepcopy(train_task),
        copy.deepcopy(test_task),
        subject_col=subject_col,
        visit_col=visit_col,
        code_col=code_col,
        label_col=label_col,
        **kwargs,
    )
    # Train on Synthetic, test on Real (TSTR).
    _, syn_y_true, syn_y_pred = train_fn(
        copy.deepcopy(syn_task),
        copy.deepcopy(test_task),
        subject_col=subject_col,
        visit_col=visit_col,
        code_col=code_col,
        label_col=label_col,
        **kwargs,
    )

    metrics_runs = []
    n_samples = len(real_y_true)
    for _ in range(n_bootstraps):
        if n_samples > 0:
            indices = np.random.choice(n_samples, n_samples, replace=True)
            r_true, r_pred = real_y_true[indices], real_y_pred[indices]
            s_true, s_pred = syn_y_true[indices], syn_y_pred[indices]
        else:
            r_true, r_pred = real_y_true, real_y_pred
            s_true, s_pred = syn_y_true, syn_y_pred

        real_acc = (
            sklearn_metrics.accuracy_score(r_true, r_pred)
            if len(r_true) > 0
            else 0.0
        )
        syn_acc = (
            sklearn_metrics.accuracy_score(s_true, s_pred)
            if len(s_true) > 0
            else 0.0
        )
        real_f1 = (
            sklearn_metrics.f1_score(r_true, r_pred, average="macro")
            if len(r_true) > 0
            else 0.0
        )
        syn_f1 = (
            sklearn_metrics.f1_score(s_true, s_pred, average="macro")
            if len(s_true) > 0
            else 0.0
        )

        metrics_runs.append(
            {
                "MLE_Real_Accuracy": real_acc,
                "MLE_Synth_Accuracy": syn_acc,
                "MLE_Difference": real_acc - syn_acc,
                "MLE_Ratio": syn_acc / real_acc if real_acc > 0 else 0.0,
                "MLE_Real_F1": real_f1,
                "MLE_Synth_F1": syn_f1,
            }
        )

    summary = summarize_metric_runs(metrics_runs)
    logger.info("MLE results: %s", summary)
    return summary


def compute_prevalence_metrics(
    train_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    code_col: str = "visit_codes",
    n_bootstraps: int = 5,
) -> Dict[str, Tuple[float, float]]:
    """Compares per-code patient-level prevalence of real vs synthetic data.

    For every code, prevalence is the fraction of unique patients who have that
    code at least once. The real and synthetic prevalence vectors are compared
    with R-squared, Pearson correlation and RMSE; bootstrap resampling is over
    codes.

    This metric only reads ``subject_col`` and ``code_col``, but ``train_ehr``
    and ``syn_ehr`` are expected to be the same flat
    ``[id, time, visit_codes, labels]`` frames used by the other metrics.

    Args:
        train_ehr: Real training EHR dataframe, flat
            ``[id, time, visit_codes, labels]`` format.
        syn_ehr: Synthetic EHR dataframe; same schema as ``train_ehr``.
        subject_col: Column name for patient/subject identifiers.
        code_col: Column name for the medical codes (one code per row).
        n_bootstraps: Number of bootstrap resamples over codes.

    Returns:
        Dictionary mapping ``"Prevalence_R2"``, ``"Prevalence_Pearson"`` and
            ``"Prevalence_RMSE"`` to their ``(mean, std)`` across bootstraps.

    Examples:
        >>> from pyhealth.metrics.generative.utility import (
        ...     compute_prevalence_metrics,
        ... )
        >>> # train_ehr and syn_ehr are flat [id, time, visit_codes, labels]
        >>> # dataframes sharing one schema -- see evaluate_synthetic_ehr for
        >>> # how to build them.
        >>> result = compute_prevalence_metrics(train_ehr, syn_ehr)
        >>> r2_mean, r2_std = result["Prevalence_R2"]
    """
    logger.info("Computing prevalence metrics")

    all_codes = set()
    all_codes.update(train_ehr[code_col].unique().tolist())
    all_codes.update(syn_ehr[code_col].unique().tolist())

    n_train = train_ehr[subject_col].nunique()
    n_syn = syn_ehr[subject_col].nunique()
    if n_train == 0 or n_syn == 0:
        return {
            "Prevalence_R2": (0.0, 0.0),
            "Prevalence_Pearson": (0.0, 0.0),
            "Prevalence_RMSE": (0.0, 0.0),
        }

    # Count unique patients per code.
    train_counts = train_ehr.groupby(code_col)[subject_col].nunique()
    syn_counts = syn_ehr.groupby(code_col)[subject_col].nunique()
    for code in all_codes:
        if code not in train_counts.index:
            train_counts.loc[code] = 0
        if code not in syn_counts.index:
            syn_counts.loc[code] = 0

    train_probs = train_counts / n_train
    syn_probs = syn_counts / n_syn
    df_compare = pd.DataFrame(
        {"real": train_probs, "syn": syn_probs}
    ).fillna(0)

    metrics_runs = []
    n_samples = len(df_compare)
    for _ in range(n_bootstraps):
        if n_samples > 0:
            df_sampled = df_compare.sample(n=n_samples, replace=True)
            real_vec = df_sampled["real"].values
            syn_vec = df_sampled["syn"].values
        else:
            real_vec = df_compare["real"].values
            syn_vec = df_compare["syn"].values

        r2 = (
            sklearn_metrics.r2_score(real_vec, syn_vec)
            if n_samples > 1
            else 0.0
        )
        # Pearson correlation via numpy (avoids a hard scipy dependency).
        if len(np.unique(real_vec)) > 1 and len(np.unique(syn_vec)) > 1:
            rho = float(np.corrcoef(real_vec, syn_vec)[0, 1])
        else:
            rho = 0.0
        rmse = (
            float(np.sqrt(sklearn_metrics.mean_squared_error(real_vec, syn_vec)))
            if n_samples > 0
            else 0.0
        )

        metrics_runs.append(
            {
                "Prevalence_R2": r2,
                "Prevalence_Pearson": rho,
                "Prevalence_RMSE": rmse,
            }
        )

    summary = summarize_metric_runs(metrics_runs)
    logger.info("Prevalence results: %s", summary)
    return summary
