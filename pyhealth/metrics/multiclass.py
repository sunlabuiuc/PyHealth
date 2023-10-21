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
    for metric in metrics:
        if metric == "roc_auc_macro_ovo":
            roc_auc_macro_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovo"
            )
            output["roc_auc_macro_ovo"] = roc_auc_macro_ovo
        elif metric == "roc_auc_macro_ovr":
            roc_auc_macro_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro", multi_class="ovr"
            )
            output["roc_auc_macro_ovr"] = roc_auc_macro_ovr
        elif metric == "roc_auc_weighted_ovo":
            roc_auc_weighted_ovo = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovo"
            )
            output["roc_auc_weighted_ovo"] = roc_auc_weighted_ovo
        elif metric == "roc_auc_weighted_ovr":
            roc_auc_weighted_ovr = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted", multi_class="ovr"
            )
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
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            output["HITS@1"] = np.count_nonzero(ranking <= 1) / len(ranking)
            output["HITS@5"] = np.count_nonzero(ranking <= 5) / len(ranking)
            output["HITS@10"] = np.count_nonzero(ranking <= 10) / len(ranking)
        elif metric == "mean_rank":
            argsort = np.argsort(-y_prob, axis=1)
            ranking = np.array([np.where(argsort[i] == y_true[i])[0][0] for i in range(len(y_true))]) + 1
            mean_rank = np.mean(ranking)
            mean_reciprocal_rank = np.mean(1/ranking)
            output["mean_rank"] = mean_rank
            output["mean_reciprocal_rank"] = mean_reciprocal_rank
            
        else:
            raise ValueError(f"Unknown metric for multiclass classification: {metric}")

    return output


if __name__ == "__main__":
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
