from typing import List, Optional, Dict

import numpy as np
import sklearn.metrics as sklearn_metrics


def multiclass_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
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
    y_true = np.random.randint(4, size=100000)
    y_prob = np.random.randn(100000, 4)
    y_prob = np.exp(y_prob) / np.sum(np.exp(y_prob), axis=-1, keepdims=True)
    print(multiclass_metrics_fn(y_true, y_prob, metrics=all_metrics))
