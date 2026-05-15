"""Temporal evaluation utilities for clinical prediction.

This module implements a lightweight temporal evaluation pipeline inspired by:

Zhou, H., Chen, Y., and Lipton, Z. C.
Evaluating Model Performance in Medical Datasets Over Time.

The main idea is to evaluate models in a deployment-like setting:
train on records up to a time cutoff and test on records after that cutoff.

This file is intentionally lightweight so it can be tested entirely with
synthetic data in milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


Record = Dict[str, Any]


@dataclass
class TemporalExperimentResult:
    """Container for experiment outputs.

    Attributes:
        experiment_type: Either "temporal" or "random".
        split_year: Temporal cutoff year, or None for random split.
        train_size: Number of training examples.
        test_size: Number of testing examples.
        accuracy: Accuracy on the test split.
        auroc: AUROC on the test split, or None if unavailable.
        auprc: AUPRC on the test split, or None if unavailable.
        brier: Brier score on the test split, or None if unavailable.
        f1: F1 score on the test split.
    """

    experiment_type: str
    split_year: int | None
    train_size: int
    test_size: int
    accuracy: float
    auroc: float | None
    auprc: float | None
    brier: float | None
    f1: float


def validate_dataset(dataset: Sequence[Record]) -> None:
    """Validates the dataset format.

    Expected keys for each record:
    - "year"
    - "label"
    - "features"

    Args:
        dataset: Sequence of record dictionaries.

    Raises:
        ValueError: If the dataset is empty or malformed.
    """
    if not dataset:
        raise ValueError("Dataset must not be empty.")

    required_keys = {"year", "label", "features"}
    for i, row in enumerate(dataset):
        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(
                f"Record at index {i} is missing required keys: {sorted(missing)}"
            )

        if not isinstance(row["features"], (list, tuple)):
            raise ValueError(
                f"Record at index {i} must have a list or tuple in 'features'."
            )

        if len(row["features"]) == 0:
            raise ValueError(
                f"Record at index {i} must contain at least one feature."
            )

        try:
            int(row["year"])
            int(row["label"])
            [float(value) for value in row["features"]]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Record at index {i} contains non-numeric year, label, or features."
            ) from exc


def prepare_data(dataset: Sequence[Record]) -> Tuple[List[List[float]], List[int]]:
    """Converts record dictionaries into feature matrix and label vector.

    Args:
        dataset: Sequence of records with "features" and "label".

    Returns:
        Tuple of (X, y).
    """
    validate_dataset(dataset)
    x = [[float(value) for value in row["features"]] for row in dataset]
    y = [int(row["label"]) for row in dataset]
    return x, y


def temporal_split(
    dataset: Sequence[Record], split_year: int
) -> Tuple[List[Record], List[Record]]:
    """Splits records into past-vs-future subsets.

    Train split contains years <= split_year.
    Test split contains years > split_year.

    Args:
        dataset: Sequence of records containing a "year" key.
        split_year: Temporal cutoff year.

    Returns:
        Tuple of (train_records, test_records).

    Raises:
        ValueError: If either split is empty.
    """
    validate_dataset(dataset)
    train = [row for row in dataset if int(row["year"]) <= split_year]
    test = [row for row in dataset if int(row["year"]) > split_year]

    if not train:
        raise ValueError("Temporal split produced an empty training set.")
    if not test:
        raise ValueError("Temporal split produced an empty testing set.")

    return train, test


def _check_binary_labels(y_train: Sequence[int], y_test: Sequence[int]) -> None:
    """Ensures both train and test splits are valid for binary classification."""
    if len(set(y_train)) < 2:
        raise ValueError("Training set must contain at least two classes.")
    if len(set(y_test)) < 2:
        raise ValueError("Testing set must contain at least two classes.")


def train_logistic_regression(
    x_train: Sequence[Sequence[float]],
    y_train: Sequence[int],
    max_iter: int = 1000,
) -> LogisticRegression:
    """Trains a logistic regression model.

    Args:
        x_train: Training features.
        y_train: Training labels.
        max_iter: Maximum number of optimizer iterations.

    Returns:
        Trained LogisticRegression model.

    Raises:
        ValueError: If training labels contain fewer than two classes.
    """
    if len(set(y_train)) < 2:
        raise ValueError("Training labels must contain at least two classes.")

    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_train, y_train)
    return model


def evaluate_model(
    model: LogisticRegression,
    x_test: Sequence[Sequence[float]],
    y_test: Sequence[int],
) -> Tuple[float, float | None, float | None, float | None, float]:
    """Evaluates a binary classifier on a test set.

    Args:
        model: Trained logistic regression model.
        x_test: Testing features.
        y_test: Testing labels.

    Returns:
        Tuple of (accuracy, auroc, auprc, brier, f1).
        AUROC/AUPRC/Brier may be None if unavailable.
    """
    predictions = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))

    auroc: float | None
    auprc: float | None
    brier: float | None

    try:
        probabilities = model.predict_proba(x_test)[:, 1]
        auroc = float(roc_auc_score(y_test, probabilities))
        auprc = float(average_precision_score(y_test, probabilities))
        brier = float(brier_score_loss(y_test, probabilities))
    except Exception:
        auroc = None
        auprc = None
        brier = None

    return accuracy, auroc, auprc, brier, f1


def run_temporal_experiment(
    dataset: Sequence[Record], split_year: int
) -> TemporalExperimentResult:
    """Runs one temporal train/test experiment.

    Args:
        dataset: Sequence of patient records.
        split_year: Train on years <= split_year, test on years > split_year.

    Returns:
        TemporalExperimentResult with summary metrics.
    """
    train_records, test_records = temporal_split(dataset, split_year)
    x_train, y_train = prepare_data(train_records)
    x_test, y_test = prepare_data(test_records)

    _check_binary_labels(y_train, y_test)

    model = train_logistic_regression(x_train, y_train)
    accuracy, auroc, auprc, brier, f1 = evaluate_model(model, x_test, y_test)

    return TemporalExperimentResult(
        experiment_type="temporal",
        split_year=split_year,
        train_size=len(train_records),
        test_size=len(test_records),
        accuracy=accuracy,
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        f1=f1,
    )


def run_random_experiment(
    dataset: Sequence[Record],
    test_size: float = 0.4,
    random_state: int = 42,
) -> TemporalExperimentResult:
    """Runs a random train/test baseline experiment.

    Args:
        dataset: Sequence of patient records.
        test_size: Fraction used for testing.
        random_state: Random seed.

    Returns:
        TemporalExperimentResult with summary metrics.

    Raises:
        ValueError: If the dataset does not contain both classes.
    """
    x, y = prepare_data(dataset)

    if len(set(y)) < 2:
        raise ValueError("Dataset must contain at least two classes.")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    _check_binary_labels(y_train, y_test)

    model = train_logistic_regression(x_train, y_train)
    accuracy, auroc, auprc, brier, f1 = evaluate_model(model, x_test, y_test)

    return TemporalExperimentResult(
        experiment_type="random",
        split_year=None,
        train_size=len(x_train),
        test_size=len(x_test),
        accuracy=accuracy,
        auroc=auroc,
        auprc=auprc,
        brier=brier,
        f1=f1,
    )


def run_ablation(
    dataset: Sequence[Record],
    split_years: Iterable[int],
) -> List[TemporalExperimentResult]:
    """Runs multiple temporal cutoffs for an ablation study.

    Args:
        dataset: Sequence of patient records.
        split_years: Iterable of temporal cutoff years.

    Returns:
        List of TemporalExperimentResult objects.
    """
    results: List[TemporalExperimentResult] = []
    for year in split_years:
        results.append(run_temporal_experiment(dataset, split_year=year))
    return results


def generate_synthetic_temporal_shift_data(
    n_patients_per_year: int = 40,
    start_year: int = 2010,
    end_year: int = 2021,
    seed: int = 42,
) -> List[Record]:
    """Generates realistic synthetic clinical data with temporal drift.

    Features:
        - age_like
        - comorbidity_like
        - lab_like
        - utilization_like

    The data distribution shifts over time, which makes temporal evaluation
    meaningfully different from random splits.

    Args:
        n_patients_per_year: Number of synthetic patients generated per year.
        start_year: First year.
        end_year: Last year.
        seed: Random seed.

    Returns:
        List of synthetic patient records.
    """
    rng = np.random.default_rng(seed)
    records: List[Record] = []
    patient_id = 1

    for year in range(start_year, end_year + 1):
        drift = (year - start_year) / max(1, (end_year - start_year))

        for _ in range(n_patients_per_year):
            age_like = np.clip(rng.normal(loc=55 + 8 * drift, scale=10), 18, 90)
            comorbidity_like = np.clip(
                rng.normal(loc=1.5 + 1.2 * drift, scale=0.9), 0, 6
            )
            lab_like = np.clip(
                rng.normal(loc=0.45 + 0.20 * drift, scale=0.15), 0.05, 1.20
            )
            utilization_like = np.clip(
                rng.normal(loc=1.2 + 0.8 * drift, scale=0.7), 0, 5
            )

            # The label relationship changes slightly over time.
            score = (
                -6.0
                + 0.035 * age_like
                + 0.55 * comorbidity_like
                + 3.2 * lab_like
                + 0.35 * utilization_like
                - 0.9 * drift
                + rng.normal(0, 0.35)
            )

            probability = 1.0 / (1.0 + np.exp(-score))
            label = int(rng.random() < probability)

            records.append(
                {
                    "patient_id": patient_id,
                    "year": year,
                    "features": [
                        round(float(age_like), 3),
                        round(float(comorbidity_like), 3),
                        round(float(lab_like), 3),
                        round(float(utilization_like), 3),
                    ],
                    "label": label,
                }
            )
            patient_id += 1

    return records

def _to_python_scalar(value: Any) -> Any:
    """Converts tensor-like objects to Python values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def sample_dataset_to_temporal_records(sample_dataset: Sequence[Any]) -> List[Record]:
    """Converts a PyHealth SampleDataset into temporal evaluation records.

    Args:
        sample_dataset: Sequence of PyHealth samples.

    Returns:
        List of records compatible with temporal evaluation utilities.
    """
    records: List[Record] = []

    for i, sample in enumerate(sample_dataset):
        features = _to_python_scalar(sample["features"])
        year = _to_python_scalar(sample["year"])
        label = _to_python_scalar(sample["label"])

        if isinstance(year, list):
            year = year[0]
        if isinstance(label, list):
            label = label[0]

        records.append(
            {
                "patient_id": i,
                "year": int(year),
                "features": [float(x) for x in features],
                "label": int(label),
            }
        )

    return records

if __name__ == "__main__":
    dataset = generate_synthetic_temporal_shift_data()
    results = run_ablation(dataset, [2013, 2015, 2017])

    print("=== Temporal ablation ===")
    for r in results:
        print(r)

    print("\n=== Random baseline ===")
    print(run_random_experiment(dataset))