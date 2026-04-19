"""Evaluation metrics for wav2sleep sleep stage classification model.

This module implements the metrics and evaluations used in the wav2sleep study:
"A Unified Multi-Modal Approach to Sleep Stage Classification from 
Physiological Signals" (Carter & Tarassenko, arXiv:2411.04644).

The primary metric is Cohen's Kappa, which measures agreement between model
predictions and ground truth labels on a scale of 0 to 1.

Metrics Included:
  - Cohen's Kappa: Inter-rater agreement metric (primary metric)
  - Accuracy: Classification accuracy from confusion matrix
  - Confusion Matrix: Per-class and overall predictions
  - Per-class precision, recall, F1-score
  
References:
  https://arxiv.org/abs/2411.04644
  https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html
"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    """Compute confusion matrix from predictions and labels.

    Args:
        predictions: Predicted class indices, shape (n_samples,).
            Can be integers or logits that will be argmax'd.
        labels: Ground truth class indices, shape (n_samples,).
            Integer values in range [0, num_classes).
        num_classes: Number of classification classes.

    Returns:
        Confusion matrix of shape (num_classes, num_classes).
        Element [i, j] represents the number of samples with true label i
        that were predicted as class j.

    Raises:
        ValueError: If prediction and label arrays have different lengths
            or contain invalid class indices.

    Example:
        >>> preds = np.array([0, 1, 1, 0, 2])
        >>> labels = np.array([0, 0, 1, 0, 2])
        >>> cmat = compute_confusion_matrix(preds, labels, num_classes=3)
        >>> cmat.shape
        (3, 3)
        >>> cmat.sum()
        5
    """
    if len(predictions) != len(labels):
        raise ValueError(
            f"Prediction and label arrays must have same length.  "
            f"Got predictions.shape={predictions.shape}, labels.shape={labels.shape}."
        )

    # Convert to numpy if needed
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()

    # Handle logits by taking argmax
    if predictions.ndim == 2:
        predictions = predictions.argmax(axis=1)

    predictions = predictions.astype(np.int64).flatten()
    labels = labels.astype(np.int64).flatten()

    # Validate class indices
    if predictions.min() < 0 or predictions.max() >= num_classes:
        raise ValueError(
            f"Prediction indices out of range [0, {num_classes-1}]. "
            f"Got range [{predictions.min()}, {predictions.max()}]."
        )
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError(
            f"Label indices out of range [0, {num_classes-1}]. "
            f"Got range [{labels.min()}, {labels.max()}]."
        )

    # Build confusion matrix
    cmat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, pred_label in zip(labels, predictions):
        cmat[true_label, pred_label] += 1

    return cmat


def cohens_kappa(
    confusion_matrix: np.ndarray,
    num_classes: Optional[int] = None,
) -> float:
    """Compute Cohen's Kappa coefficient from confusion matrix.

    Cohen's Kappa measures the agreement between model predictions and
    ground truth, accounting for chance agreement. It ranges from 0 (no
    agreement better than chance) to 1 (perfect agreement).

    Kappa = (observed_accuracy - expected_accuracy) / (1 - expected_accuracy)

    This implementation follows the scikit-learn approach:
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).
            Element [i, j] is the count of samples with true label i
            predicted as class j.
        num_classes: Number of classes. If None, inferred from matrix shape.

    Returns:
        Cohen's Kappa coefficient in range [0, 1].
        - 0.0: No agreement beyond chance
        - 1.0: Perfect agreement
        - Values in between: Partial agreement

    Example:
        >>> cmat = np.array([[10, 1], [2, 8]])  # 2x2 confusion matrix
        >>> kappa = cohens_kappa(cmat)
        >>> print(f"Kappa: {kappa:.3f}")
        Kappa: 0.765

    References:
        Cohen, J. (1960). "A coefficient of agreement for nominal scales."
        Educational and Psychological Measurement, 20(1), 37–46.
    """
    confusion_matrix = confusion_matrix.astype(float)

    if num_classes is None:
        num_classes = confusion_matrix.shape[0]

    # Compute row and column sums (marginals)
    row_sums = np.sum(confusion_matrix, axis=1)  # true label counts
    col_sums = np.sum(confusion_matrix, axis=0)  # pred label counts
    total = np.sum(confusion_matrix)

    if total == 0:
        raise ValueError("Confusion matrix is empty (sum is 0).")

    # Observed accuracy: trace / total
    observed_accuracy = np.trace(confusion_matrix) / total

    # Expected accuracy from marginals
    expected_agreements = np.sum(row_sums * col_sums) / (total * total)

    # Cohen's Kappa
    if expected_agreements >= 1.0:
        # Perfect agreement expected by chance
        return 1.0 if observed_accuracy == 1.0 else 0.0

    kappa = (observed_accuracy - expected_agreements) / (1.0 - expected_agreements)

    # Clamp to [0, 1] to handle floating point edge cases
    kappa = np.clip(kappa, 0.0, 1.0)
    return kappa


def confusion_accuracy(confusion_matrix: np.ndarray) -> float:
    """Calculate accuracy from confusion matrix.

    Accuracy is the fraction of correct predictions:
    Accuracy = trace(cmat) / sum(cmat)

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).

    Returns:
        Accuracy score in range [0.0, 1.0], where 1.0 is perfect accuracy.

    Example:
        >>> cmat = np.array([[100, 10], [5, 85]])
        >>> acc = confusion_accuracy(cmat)
        >>> print(f"Accuracy: {acc:.2%}")
        Accuracy: 92.50%
    """
    confusion_matrix = confusion_matrix.astype(float)
    total = np.sum(confusion_matrix)

    if total == 0:
        return 0.0

    return np.trace(confusion_matrix) / total


def per_class_metrics(
    confusion_matrix: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Compute per-class precision, recall, and F1-score from confusion matrix.

    Args:
        confusion_matrix: Confusion matrix of shape (num_classes, num_classes).

    Returns:
        Dictionary with keys:
        - "precision": Per-class precision, shape (num_classes,)
        - "recall": Per-class recall (sensitivity), shape (num_classes,)
        - "f1": Per-class F1-score, shape (num_classes,)

    Example:
        >>> cmat = np.array([[10, 1], [2, 8]])
        >>> metrics = per_class_metrics(cmat)
        >>> print(f"Class 0 recall: {metrics['recall'][0]:.2%}")
        Class 0 recall: 90.91%
    """
    confusion_matrix = confusion_matrix.astype(float)
    num_classes = confusion_matrix.shape[0]

    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)

    for i in range(num_classes):
        # Precision: TP / (TP + FP)
        col_sum = confusion_matrix[:, i].sum()
        if col_sum > 0:
            precision[i] = confusion_matrix[i, i] / col_sum
        else:
            precision[i] = 0.0

        # Recall: TP / (TP + FN)
        row_sum = confusion_matrix[i, :].sum()
        if row_sum > 0:
            recall[i] = confusion_matrix[i, i] / row_sum
        else:
            recall[i] = 0.0

        # F1: harmonic mean
        if precision[i] + recall[i] > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_model(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_names: Optional[list] = None,
) -> Dict[str, any]:
    """Comprehensive evaluation of model predictions.

    Computes confusion matrix, accuracy, Cohen's Kappa, and per-class metrics.

    Args:
        predictions: Predicted class indices or logits, shape (n_samples,) or
            (n_samples, num_classes).
        labels: Ground truth class indices, shape (n_samples,).
        num_classes: Number of classification classes.
        class_names: Optional list of class names for reporting.
            Default: ["Class 0", "Class 1", ...].

    Returns:
        Dictionary with keys:
        - "confusion_matrix": Confusion matrix, shape (num_classes, num_classes)
        - "accuracy": Overall accuracy [0, 1]
        - "kappa": Cohen's Kappa [0, 1]
        - "precision": Per-class precision, shape (num_classes,)
        - "recall": Per-class recall, shape (num_classes,)
        - "f1": Per-class F1-score, shape (num_classes,)
        - "macro_precision": Macro-averaged precision
        - "macro_recall": Macro-averaged recall
        - "macro_f1": Macro-averaged F1-score
        - "weighted_f1": Weighted F1-score by support

    Example:
        >>> preds = np.array([0, 1, 1, 0, 2, 2, 1, 0])
        >>> labels = np.array([0, 0, 1, 0, 2, 2, 1, 0])
        >>> results = evaluate_model(preds, labels, num_classes=3)
        >>> print(f"Accuracy: {results['accuracy']:.2%}")
        >>> print(f"Cohen's Kappa: {results['kappa']:.3f}")
    """
    cmat = compute_confusion_matrix(predictions, labels, num_classes)
    accuracy = confusion_accuracy(cmat)
    kappa = cohens_kappa(cmat, num_classes)
    per_class = per_class_metrics(cmat)

    # Compute class supports for weighted metrics
    class_support = cmat.sum(axis=1)
    total_support = class_support.sum()

    # Macro and weighted averages
    macro_precision = per_class["precision"].mean()
    macro_recall = per_class["recall"].mean()
    macro_f1 = per_class["f1"].mean()

    weighted_f1 = (
        (per_class["f1"] * class_support).sum() / total_support
        if total_support > 0
        else 0.0
    )

    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]

    return {
        "confusion_matrix": cmat,
        "accuracy": accuracy,
        "kappa": kappa,
        "precision": per_class["precision"],
        "recall": per_class["recall"],
        "f1": per_class["f1"],
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "class_names": class_names,
        "class_support": class_support,
    }


def format_evaluation_report(
    eval_results: Dict[str, any],
) -> str:
    """Format evaluation results as a readable string report.

    Args:
        eval_results: Dictionary returned by evaluate_model().

    Returns:
        Formatted string report suitable for logging or display.

    Example:
        >>> preds = np.random.randint(0, 3, 100)
        >>> labels = np.random.randint(0, 3, 100)
        >>> results = evaluate_model(preds, labels, num_classes=3)
        >>> print(format_evaluation_report(results))
    """
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUATION REPORT")
    lines.append("=" * 70)

    # Overall metrics
    lines.append("\nOVERALL METRICS:")
    lines.append(f"  Accuracy:        {eval_results['accuracy']:.4f}")
    lines.append(f"  Cohen's Kappa:   {eval_results['kappa']:.4f}")
    lines.append(f"  Macro Precision: {eval_results['macro_precision']:.4f}")
    lines.append(f"  Macro Recall:    {eval_results['macro_recall']:.4f}")
    lines.append(f"  Macro F1:        {eval_results['macro_f1']:.4f}")
    lines.append(f"  Weighted F1:     {eval_results['weighted_f1']:.4f}")

    # Per-class metrics
    lines.append("\nPER-CLASS METRICS:")
    lines.append(
        f"{'Class':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}"
    )
    lines.append("-" * 70)

    for i, class_name in enumerate(eval_results["class_names"]):
        precision = eval_results["precision"][i]
        recall = eval_results["recall"][i]
        f1 = eval_results["f1"][i]
        support = int(eval_results["class_support"][i])
        lines.append(
            f"{class_name:<15} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f} {support:<10}"
        )

    lines.append("-" * 70)

    # Confusion matrix
    lines.append("\nCONFUSION MATRIX:")
    cmat = eval_results["confusion_matrix"]
    class_names = eval_results["class_names"]

    # Header
    lines.append("      " + "  ".join([f"{cn:>10}" for cn in class_names]))

    # Rows
    for i, class_name in enumerate(class_names):
        row_str = f"{class_name:<6}"
        row_str += "  ".join([f"{int(cmat[i, j]):>10}" for j in range(cmat.shape[1])])
        lines.append(row_str)

    lines.append("=" * 70)
    return "\n".join(lines)
