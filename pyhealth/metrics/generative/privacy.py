"""Privacy metrics for synthetic EHR data.

These metrics quantify how much a synthetic EHR dataset leaks about the real
records it was trained on. They include:

    - Nearest Neighbor Adversarial Accuracy Risk (NNAAR)
    - Membership Inference Attack (MIA) metrics
    - A discriminator-based adversarial-accuracy privacy score

All functions take flat EHR dataframes (one row per patient/visit/code event)
and return ``{metric_name: (mean, std)}`` summaries computed over multiple runs
or bootstrap resamples.
"""

import copy
import logging
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn import metrics as sklearn_metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .utils import (
    convert_visits_to_sets,
    find_nearest_neighbor_dist,
    summarize_metric_runs,
)

logger = logging.getLogger(__name__)

__all__ = [
    "calc_nnaar",
    "calc_membership_inference",
    "compute_discriminator_privacy",
]


def calc_nnaar(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    sample_size: int = 1000,
    n_runs: int = 5,
    verbose: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Computes the Nearest Neighbor Adversarial Accuracy Risk (NNAAR).

    NNAAR measures whether the synthetic data sits closer to the real training
    data than to held-out test data, which would indicate memorization::

        NNAAR = AA_ES - AA_TS

    where ``AA_ES`` is the adversarial accuracy between test and synthetic data
    and ``AA_TS`` is the adversarial accuracy between train and synthetic data.
    Values near 0 indicate low privacy risk.

    Args:
        train_ehr: Real training EHR dataframe.
        test_ehr: Real held-out test EHR dataframe.
        syn_ehr: Synthetic EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the label (unused, kept for a uniform API).
        sample_size: Number of patients to sample per dataset per run.
        n_runs: Number of independent sampling runs.
        verbose: Whether to show per-run progress bars.

    Returns:
        Dictionary mapping ``"nnaar"``, ``"aa_es"`` and ``"aa_ts"`` to their
            ``(mean, std)`` across runs.
    """
    logger.info(
        "Calculating NNAAR (sample_size=%d, n_runs=%d)", sample_size, n_runs
    )
    train = convert_visits_to_sets(train_ehr, subject_col, visit_col, code_col)
    test = convert_visits_to_sets(test_ehr, subject_col, visit_col, code_col)
    synthetic = convert_visits_to_sets(syn_ehr, subject_col, visit_col, code_col)

    metrics_runs = []
    n = min(sample_size, len(train), len(test), len(synthetic))

    for _ in range(n_runs):
        if len(train) > n:
            inds = np.random.choice(len(train), n, replace=False)
            s_train = [train[i] for i in inds]
        else:
            s_train = list(train)
        if len(test) > n:
            inds = np.random.choice(len(test), n, replace=False)
            s_test = [test[i] for i in inds]
        else:
            s_test = list(test)
        if len(synthetic) > n:
            inds = np.random.choice(len(synthetic), n, replace=False)
            s_syn = [synthetic[i] for i in inds]
        else:
            s_syn = list(synthetic)

        # AA_ES (test vs synthetic).
        val1 = sum(
            1
            for p in tqdm(s_test, desc="Test vs Syn", disable=not verbose)
            if find_nearest_neighbor_dist(p, s_syn)
            > find_nearest_neighbor_dist(p, s_test)
        )
        val2 = sum(
            1
            for p in tqdm(s_syn, desc="Syn vs Test", disable=not verbose)
            if find_nearest_neighbor_dist(p, s_test)
            > find_nearest_neighbor_dist(p, s_syn)
        )
        # AA_TS (train vs synthetic).
        val3 = sum(
            1
            for p in tqdm(s_train, desc="Train vs Syn", disable=not verbose)
            if find_nearest_neighbor_dist(p, s_syn)
            > find_nearest_neighbor_dist(p, s_train)
        )
        val4 = sum(
            1
            for p in tqdm(s_syn, desc="Syn vs Train", disable=not verbose)
            if find_nearest_neighbor_dist(p, s_train)
            > find_nearest_neighbor_dist(p, s_syn)
        )

        aa_es = 0.5 * (val1 / n + val2 / n)
        aa_ts = 0.5 * (val3 / n + val4 / n)
        metrics_runs.append(
            {"nnaar": aa_es - aa_ts, "aa_es": aa_es, "aa_ts": aa_ts}
        )

    return summarize_metric_runs(metrics_runs)


def calc_membership_inference(
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    num_attack_samples: int = 1000,
    n_runs: int = 5,
    verbose: bool = False,
) -> Dict[str, Tuple[float, float]]:
    """Computes Membership Inference Attack (MIA) metrics.

    An attacker tries to tell members (training patients) from non-members
    (test patients) using proximity to the synthetic data: members are expected
    to be closer to synthetic records. Predictions are made by thresholding the
    nearest-neighbor distance at its median; F1, precision, recall and accuracy
    near 0.5 indicate low membership-inference risk.

    Args:
        train_ehr: Real training EHR dataframe (members).
        test_ehr: Real held-out test EHR dataframe (non-members).
        syn_ehr: Synthetic EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the label (unused, kept for a uniform API).
        num_attack_samples: Total attack-set size (half members, half not).
        n_runs: Number of independent sampling runs.
        verbose: Whether to show per-run progress bars.

    Returns:
        Dictionary mapping ``"MIA_F1"``, ``"MIA_Precision"``, ``"MIA_Recall"``
            and ``"MIA_Accuracy"`` to their ``(mean, std)`` across runs.
    """
    logger.info(
        "Calculating Membership Inference (attack_size=%d, n_runs=%d)",
        num_attack_samples,
        n_runs,
    )
    train = convert_visits_to_sets(train_ehr, subject_col, visit_col, code_col)
    test = convert_visits_to_sets(test_ehr, subject_col, visit_col, code_col)
    synthetic = convert_visits_to_sets(syn_ehr, subject_col, visit_col, code_col)

    metrics_runs = []
    for _ in range(n_runs):
        # Build a balanced attack set: 50% members, 50% non-members.
        n_half = min(len(train), len(test), num_attack_samples) // 2
        if n_half == 0:
            continue

        pos_inds = np.random.choice(len(train), n_half, replace=False)
        pos_samples = [train[i] for i in pos_inds]
        neg_inds = np.random.choice(len(test), n_half, replace=False)
        neg_samples = [test[i] for i in neg_inds]

        attack_data = pos_samples + neg_samples
        attack_labels = [1] * len(pos_samples) + [0] * len(neg_samples)

        distances = [
            find_nearest_neighbor_dist(record, synthetic)
            for record in tqdm(
                attack_data, desc="Calculating Distances", disable=not verbose
            )
        ]
        if len(distances) == 0:
            continue

        # Members are expected to be closer (smaller distance) to synthetic.
        median_dist = np.median(distances)
        predictions = [1 if d < median_dist else 0 for d in distances]

        metrics_runs.append(
            {
                "MIA_F1": sklearn_metrics.f1_score(attack_labels, predictions),
                "MIA_Precision": sklearn_metrics.precision_score(
                    attack_labels, predictions, zero_division=0
                ),
                "MIA_Recall": sklearn_metrics.recall_score(
                    attack_labels, predictions, zero_division=0
                ),
                "MIA_Accuracy": sklearn_metrics.accuracy_score(
                    attack_labels, predictions
                ),
            }
        )

    summary = summarize_metric_runs(metrics_runs)
    logger.info("MIA results: %s", summary)
    return summary


def compute_discriminator_privacy(
    train_fn: Callable,
    train_ehr: pd.DataFrame,
    test_ehr: pd.DataFrame,
    syn_ehr: pd.DataFrame,
    subject_col: str = "id",
    visit_col: str = "time",
    code_col: str = "visit_codes",
    label_col: str = "labels",
    n_bootstraps: int = 5,
    seed: int = 4,
    **kwargs,
) -> Dict[str, Tuple[float, float]]:
    """Computes a discriminator-based adversarial-accuracy privacy score.

    A classifier is trained to predict whether a record is real (1) or
    synthetic (0). An accuracy near 0.5 means real and synthetic data are
    indistinguishable (good privacy); accuracy well above 0.5 means the
    synthetic data is easy to tell apart (poor privacy). The ``Privacy_Score``
    rescales accuracy so 1.0 is perfect privacy and 0.0 is none.

    Args:
        train_fn: A training function such as
            :func:`pyhealth.metrics.generative.utils.train_lstm_model` or
            ``train_sklearn_model``. It must accept ``train_ehr``, ``test_ehr``,
            the four column-name arguments and return ``(model, y_true,
            y_pred)``.
        train_ehr: Real training EHR dataframe.
        test_ehr: Real held-out test EHR dataframe (unused; kept for a uniform
            API with the other metrics).
        syn_ehr: Synthetic EHR dataframe.
        subject_col: Column name for patient/subject identifiers.
        visit_col: Column name for visit/timestep identifiers.
        code_col: Column name for the medical codes.
        label_col: Column name for the original label (unused; the
            discriminator target replaces it).
        n_bootstraps: Number of bootstrap resamples of the predictions.
        seed: Random seed for the patient-level train/test split.
        **kwargs: Extra keyword arguments forwarded to ``train_fn``.

    Returns:
        Dictionary mapping ``"Privacy_Discriminator_Accuracy"`` and
            ``"Privacy_Score"`` to their ``(mean, std)`` across bootstraps.
    """
    logger.info("Computing discriminator privacy")

    # Label data: real = 1, synthetic = 0.
    real_df = copy.deepcopy(train_ehr)
    syn_df = copy.deepcopy(syn_ehr)
    disc_label = "is_real"
    real_df[disc_label] = 1
    syn_df[disc_label] = 0

    # Disambiguate subject IDs so real/synthetic patients never collide.
    real_df[subject_col] = real_df[subject_col].astype(str) + "_real"
    syn_df[subject_col] = syn_df[subject_col].astype(str) + "_syn"

    combined_df = pd.concat([real_df, syn_df])
    unique_patients = combined_df[subject_col].unique()
    train_ids, test_ids = train_test_split(
        unique_patients, test_size=0.2, random_state=seed
    )
    disc_train = combined_df[combined_df[subject_col].isin(train_ids)]
    disc_test = combined_df[combined_df[subject_col].isin(test_ids)]

    logger.info(
        "Discriminator train size=%d, test size=%d",
        len(disc_train),
        len(disc_test),
    )
    _, y_true, y_pred = train_fn(
        train_ehr=disc_train,
        test_ehr=disc_test,
        subject_col=subject_col,
        visit_col=visit_col,
        code_col=code_col,
        label_col=disc_label,
        **kwargs,
    )

    metrics_runs = []
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        if n_samples > 0:
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_t, y_p = y_true[indices], y_pred[indices]
        else:
            y_t, y_p = y_true, y_pred

        acc = sklearn_metrics.accuracy_score(y_t, y_p) if len(y_t) > 0 else 0.0
        metrics_runs.append(
            {
                "Privacy_Discriminator_Accuracy": acc,
                # 1.0 = perfect privacy (acc 0.5); 0.0 = no privacy.
                "Privacy_Score": 1.0 - 2 * abs(0.5 - acc),
            }
        )

    summary = summarize_metric_runs(metrics_runs)
    logger.info("Discriminator privacy results: %s", summary)
    return summary
