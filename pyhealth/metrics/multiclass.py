from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset


def multiclass_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    y_predset: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes metrics for multiclass classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_macro_ovo: area under the receiver operating characteristic curve,
            macro averaged over one-vs-one multiclass classification
        - roc_auc_macro_ovr: area under the receiver operating characteristic curve,
            macro averaged over one-vs-rest multiclass classification
        - roc_auc_weighted_ovo: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-one multiclass classification
        - roc_auc_weighted_ovr: area under the receiver operating characteristic curve,
            weighted averaged over one-vs-rest multiclass classification
        - accuracy: accuracy score
        - balanced_accuracy: balanced accuracy score (usually used for imbalanced
            datasets)
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - cohen_kappa: Cohen's kappa score
        - brier_top1: brier score between the top prediction and the true label
        - ECE: Expected Calibration Error (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_multiclass`.
        - ECE_adapt: adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_confidence_multiclass`.
        - cwECEt: classwise ECE with threshold=min(0.01,1/K). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - cwECEt_adapt: classwise adaptive ECE with threshold=min(0.01,1/K). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - hits@K: fraction of samples whose true class is among the K
            highest-probability classes, for any positive integer K
            (e.g. "hits@3" returns the key "HITS@3")
        - hits@n: shorthand that returns HITS@1, HITS@5 and HITS@10 together
        - mean_rank: mean rank of the true class (1 is best); also returns
            mean_reciprocal_rank

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:
        - rejection_rate: Frequency of rejection, where rejection happens when the prediction set has cardinality other than 1. Check :func:`pyhealth.metrics.prediction_set.rejection_rate`.
        - set_size: Average size of the prediction sets. Check :func:`pyhealth.metrics.prediction_set.size`.
        - miscoverage_ps:  Prob(k not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_ps`.
        - miscoverage_mean_ps: The average (across different classes k) of miscoverage_ps.
        - miscoverage_overall_ps: Prob(Y not in prediction set). Check :func:`pyhealth.metrics.prediction_set.miscoverage_overall_ps`.
        - error_ps: Same as miscoverage_ps, but retricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_ps`.
        - error_mean_ps: The average (across different classes k) of error_ps.
        - error_overall_ps: Same as miscoverage_overall_ps, but restricted to un-rejected samples. Check :func:`pyhealth.metrics.prediction_set.error_overall_ps`.

    If no metrics are specified, accuracy, f1_macro, and f1_micro are computed
    by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples,).
        y_prob: Predicted probabilities of shape (n_samples, n_classes).
        metrics: List of metrics to compute. Default is ["accuracy", "f1_macro",
            "f1_micro"].
        y_predset: Optional prediction sets, one per sample. Only used by the
            prediction set metrics listed above. Default is None.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import multiclass_metrics_fn
        >>> y_true = np.array([0, 1, 2, 2])
        >>> y_prob = np.array([[0.9,  0.05, 0.05],
        ...                    [0.05, 0.9,  0.05],
        ...                    [0.05, 0.05, 0.9],
        ...                    [0.6,  0.2,  0.2]])
        >>> multiclass_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.75}
        >>> multiclass_metrics_fn(y_true, y_prob, metrics=["hits@1", "hits@3"])
        {'HITS@1': 0.75, 'HITS@3': 1.0}
    """
    if metrics is None:
        metrics = ["accuracy", "f1_macro", "f1_micro"]
    prediction_set_metrics = [
        "rejection_rate",
        "set_size",
        "miscoverage_mean_ps",
        "miscoverage_ps",
        "miscoverage_overall_ps",
        "error_mean_ps",
        "error_ps",
        "error_overall_ps",
    ]
    y_pred = np.argmax(y_prob, axis=-1)

    output = {}
    ranking = None
    for metric in metrics:
        if metric == "roc_auc_macro_ovo":
            y_prob_roc = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            roc_auc_macro_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob_roc, average="macro", multi_class="ovo"
            ) if y_prob.shape[1] > 2 else sklearn_metrics.roc_auc_score(y_true, y_prob_roc)
            output["roc_auc_macro_ovo"] = roc_auc_macro_ovo
        elif metric == "roc_auc_macro_ovr":
            y_prob_roc = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            roc_auc_macro_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob_roc, average="macro", multi_class="ovr"
            ) if y_prob.shape[1] > 2 else sklearn_metrics.roc_auc_score(y_true, y_prob_roc)
            output["roc_auc_macro_ovr"] = roc_auc_macro_ovr
        elif metric == "roc_auc_weighted_ovo":
            y_prob_roc = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            roc_auc_weighted_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob_roc, average="weighted", multi_class="ovo"
            ) if y_prob.shape[1] > 2 else sklearn_metrics.roc_auc_score(y_true, y_prob_roc)
            output["roc_auc_weighted_ovo"] = roc_auc_weighted_ovo
        elif metric == "roc_auc_weighted_ovr":
            y_prob_roc = y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob
            roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob_roc, average="weighted", multi_class="ovr"
            ) if y_prob.shape[1] > 2 else sklearn_metrics.roc_auc_score(y_true, y_prob_roc)
            output["roc_auc_weighted_ovr"] = roc_auc_weighted_ovr
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro")
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "jaccard_micro":
            jacard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jacard_micro
        elif metric == "jaccard_macro":
            jacard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jacard_macro
        elif metric == "jaccard_weighted":
            jacard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jacard_weighted
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "brier_top1":
            output[metric] = calib.brier_top1(y_prob, y_true)
        elif metric in {"ECE", "ECE_adapt"}:
            output[metric] = calib.ece_confidence_multiclass(
                y_prob, y_true, bins=20, adaptive=metric.endswith("_adapt")
            )
        elif metric in {"cwECEt", "cwECEt_adapt"}:
            thres = min(0.01, 1.0 / y_prob.shape[1])
            output[metric] = calib.ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=thres,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == "rejection_rate":
                output[metric] = pset.rejection_rate(y_predset)
            elif metric == "set_size":
                output[metric] = pset.size(y_predset)
            elif metric == "miscoverage_mean_ps":
                output[metric] = pset.miscoverage_ps(y_predset, y_true).mean()
            elif metric == "miscoverage_ps":
                output[metric] = pset.miscoverage_ps(y_predset, y_true)
            elif metric == "miscoverage_overall_ps":
                output[metric] = pset.miscoverage_overall_ps(y_predset, y_true)
            elif metric == "error_mean_ps":
                output[metric] = pset.error_ps(y_predset, y_true).mean()
            elif metric == "error_ps":
                output[metric] = pset.error_ps(y_predset, y_true)
            elif metric == "error_overall_ps":
                output[metric] = pset.error_overall_ps(y_predset, y_true)
        elif metric == "hits@n":
            if ranking is None:
                ranking = _true_label_ranks(y_true, y_prob)
            for k in (1, 5, 10):
                output[f"HITS@{k}"] = _hits_at_k(ranking, k)
        elif metric.startswith("hits@"):
            k = _parse_hits_cutoff(metric)
            if ranking is None:
                ranking = _true_label_ranks(y_true, y_prob)
            output[f"HITS@{k}"] = _hits_at_k(ranking, k)
        elif metric == "mean_rank":
            if ranking is None:
                ranking = _true_label_ranks(y_true, y_prob)
            output["mean_rank"] = np.mean(ranking)
            output["mean_reciprocal_rank"] = np.mean(1 / ranking)
        else:
            raise ValueError(f"Unknown metric for multiclass classification: {metric}")

    return output


def _true_label_ranks(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    """Returns the 1-based rank of the true class in each row of ``y_prob``."""
    y_true = np.asarray(y_true)
    num_classes = y_prob.shape[1]
    invalid = (y_true < 0) | (y_true >= num_classes)
    if np.any(invalid):
        bad = np.unique(y_true[invalid]).tolist()
        raise ValueError(
            f"y_true contains label {bad} outside the valid range "
            f"[0, {num_classes - 1}] for y_prob with {num_classes} classes."
        )
    argsort = np.argsort(-y_prob, axis=1)
    return np.argmax(argsort == np.asarray(y_true)[:, None], axis=1) + 1


def _hits_at_k(ranking: np.ndarray, k: int) -> float:
    """Fraction of samples whose true class is ranked within the top ``k``."""
    return np.count_nonzero(ranking <= k) / len(ranking)


def _parse_hits_cutoff(metric: str) -> int:
    """Parses ``k`` from a metric name such as ``"hits@3"``."""
    cutoff = metric.split("@", 1)[1]
    if not cutoff.isdigit() or int(cutoff) < 1:
        raise ValueError(
            f"Invalid metric {metric!r}: expected 'hits@K' with K a positive "
            "integer (e.g. 'hits@3'), or 'hits@n' for HITS@1/5/10."
        )
    return int(cutoff)


if __name__ == "__main__":  # pragma: no cover
    all_metrics = [
        "roc_auc_macro_ovo",
        "roc_auc_macro_ovr",
        "roc_auc_weighted_ovo",
        "roc_auc_weighted_ovr",
        "accuracy",
        "balanced_accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
        "cohen_kappa",
    ]
    all_metrics += ["brier_top1", "ECE", "ECE_adapt", "cwECEt", "cwECEt_adapt"]
    y_true = np.random.randint(4, size=100000)
    y_prob = np.random.randn(100000, 4)
    y_prob = np.exp(y_prob) / np.sum(np.exp(y_prob), axis=-1, keepdims=True)
    print(multiclass_metrics_fn(y_true, y_prob, metrics=all_metrics))
