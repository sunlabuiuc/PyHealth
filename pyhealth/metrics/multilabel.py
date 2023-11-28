from typing import Dict, List, Optional
import os

import numpy as np
import sklearn.metrics as sklearn_metrics
from pyhealth.medcode import ATC
import pyhealth.metrics.calibration as calib
from pyhealth.metrics import ddi_rate_score
from pyhealth import BASE_CACHE_PATH as CACHE_PATH

def multilabel_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: Optional[List[str]] = None,
    threshold: float = 0.3,
    y_predset: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Computes metrics for multilabel classification.

    User can specify which metrics to compute by passing a list of metric names.
    The accepted metric names are:
        - roc_auc_micro: area under the receiver operating characteristic curve,
          micro averaged
        - roc_auc_macro: area under the receiver operating characteristic curve,
          macro averaged
        - roc_auc_weighted: area under the receiver operating characteristic curve,
          weighted averaged
        - roc_auc_samples: area under the receiver operating characteristic curve,
          samples averaged
        - pr_auc_micro: area under the precision recall curve, micro averaged
        - pr_auc_macro: area under the precision recall curve, macro averaged
        - pr_auc_weighted: area under the precision recall curve, weighted averaged
        - pr_auc_samples: area under the precision recall curve, samples averaged
        - accuracy: accuracy score
        - f1_micro: f1 score, micro averaged
        - f1_macro: f1 score, macro averaged
        - f1_weighted: f1 score, weighted averaged
        - f1_samples: f1 score, samples averaged
        - precision_micro: precision score, micro averaged
        - precision_macro: precision score, macro averaged
        - precision_weighted: precision score, weighted averaged
        - precision_samples: precision score, samples averaged
        - recall_micro: recall score, micro averaged
        - recall_macro: recall score, macro averaged
        - recall_weighted: recall score, weighted averaged
        - recall_samples: recall score, samples averaged
        - jaccard_micro: Jaccard similarity coefficient score, micro averaged
        - jaccard_macro: Jaccard similarity coefficient score, macro averaged
        - jaccard_weighted: Jaccard similarity coefficient score, weighted averaged
        - jaccard_samples: Jaccard similarity coefficient score, samples averaged
        - ddi: drug-drug interaction score (specifically for drug-related tasks, such as drug recommendation)
        - hamming_loss: Hamming loss
        - cwECE: classwise ECE (with 20 equal-width bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.
        - cwECE_adapt: classwise adaptive ECE (with 20 equal-size bins). Check :func:`pyhealth.metrics.calibration.ece_classwise`.

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:
        - fp: Number of false positives.
        - tp: Number of true positives.


    If no metrics are specified, pr_auc_samples is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples, n_labels).
        y_prob: Predicted probabilities of shape (n_samples, n_labels).
        metrics: List of metrics to compute. Default is ["pr_auc_samples"].
        threshold: Threshold to binarize the predicted probabilities. Default is 0.5.

    Returns:
        Dictionary of metrics whose keys are the metric names and values are
            the metric values.

    Examples:
        >>> from pyhealth.metrics import multilabel_metrics_fn
        >>> y_true = np.array([[0, 1, 1], [1, 0, 1]])
        >>> y_prob = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0.6]])
        >>> multilabel_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.5}
    """
    if metrics is None:
        metrics = ["pr_auc_samples"]
    prediction_set_metrics = ['tp', 'fp']

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "roc_auc_micro":
            roc_auc_micro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="micro"
            )
            output["roc_auc_micro"] = roc_auc_micro
        elif metric == "roc_auc_macro":
            roc_auc_macro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="macro"
            )
            output["roc_auc_macro"] = roc_auc_macro
        elif metric == "roc_auc_weighted":
            roc_auc_weighted = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="weighted"
            )
            output["roc_auc_weighted"] = roc_auc_weighted
        elif metric == "roc_auc_samples":
            roc_auc_samples = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average="samples"
            )
            output["roc_auc_samples"] = roc_auc_samples
        elif metric == "pr_auc_micro":
            pr_auc_micro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="micro"
            )
            output["pr_auc_micro"] = pr_auc_micro
        elif metric == "pr_auc_macro":
            pr_auc_macro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="macro"
            )
            output["pr_auc_macro"] = pr_auc_macro
        elif metric == "pr_auc_weighted":
            pr_auc_weighted = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="weighted"
            )
            output["pr_auc_weighted"] = pr_auc_weighted
        elif metric == "pr_auc_samples":
            pr_auc_samples = sklearn_metrics.average_precision_score(
                y_true, y_prob, average="samples"
            )
            output["pr_auc_samples"] = pr_auc_samples
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true.flatten(), y_pred.flatten())
            output["accuracy"] = accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average="micro")
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average="macro")
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average="weighted")
            output["f1_weighted"] = f1_weighted
        elif metric == "f1_samples":
            f1_samples = sklearn_metrics.f1_score(y_true, y_pred, average="samples")
            output["f1_samples"] = f1_samples
        elif metric == "precision_micro":
            precision_micro = sklearn_metrics.precision_score(
                y_true, y_pred, average="micro"
            )
            output["precision_micro"] = precision_micro
        elif metric == "precision_macro":
            precision_macro = sklearn_metrics.precision_score(
                y_true, y_pred, average="macro"
            )
            output["precision_macro"] = precision_macro
        elif metric == "precision_weighted":
            precision_weighted = sklearn_metrics.precision_score(
                y_true, y_pred, average="weighted"
            )
            output["precision_weighted"] = precision_weighted
        elif metric == "precision_samples":
            precision_samples = sklearn_metrics.precision_score(
                y_true, y_pred, average="samples"
            )
            output["precision_samples"] = precision_samples
        elif metric == "recall_micro":
            recall_micro = sklearn_metrics.recall_score(y_true, y_pred, average="micro")
            output["recall_micro"] = recall_micro
        elif metric == "recall_macro":
            recall_macro = sklearn_metrics.recall_score(y_true, y_pred, average="macro")
            output["recall_macro"] = recall_macro
        elif metric == "recall_weighted":
            recall_weighted = sklearn_metrics.recall_score(
                y_true, y_pred, average="weighted"
            )
            output["recall_weighted"] = recall_weighted
        elif metric == "recall_samples":
            recall_samples = sklearn_metrics.recall_score(
                y_true, y_pred, average="samples"
            )
            output["recall_samples"] = recall_samples
        elif metric == "jaccard_micro":
            jaccard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="micro"
            )
            output["jaccard_micro"] = jaccard_micro
        elif metric == "jaccard_macro":
            jaccard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="macro"
            )
            output["jaccard_macro"] = jaccard_macro
        elif metric == "jaccard_weighted":
            jaccard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="weighted"
            )
            output["jaccard_weighted"] = jaccard_weighted
        elif metric == "jaccard_samples":
            jaccard_samples = sklearn_metrics.jaccard_score(
                y_true, y_pred, average="samples"
            )
            output["jaccard_samples"] = jaccard_samples
        elif metric == "hamming_loss":
            hamming_loss = sklearn_metrics.hamming_loss(y_true, y_pred)
            output["hamming_loss"] = hamming_loss
        elif metric == "ddi":
            ddi_adj = np.load(os.path.join(CACHE_PATH, 'ddi_adj.npy'))
            y_pred = [np.where(item)[0] for item in y_pred]
            output["ddi_score"] = ddi_rate_score(y_pred, ddi_adj)
        elif metric in {"cwECE", "cwECE_adapt"}:
            output[metric] = calib.ece_classwise(
                y_prob,
                y_true,
                bins=20,
                adaptive=metric.endswith("_adapt"),
                threshold=0.0,
            )
        elif metric in prediction_set_metrics:
            if y_predset is None:
                continue
            if metric == 'tp':
                output[metric] = (y_true * y_predset).sum(1).mean()
            elif metric == 'fp':
                output[metric] = ((1-y_true) * y_predset).sum(1).mean()
        else:
            raise ValueError(f"Unknown metric for multilabel classification: {metric}")

    return output


if __name__ == "__main__":
    all_metrics = [
        "roc_auc_micro",
        "roc_auc_macro",
        "roc_auc_weighted",
        "roc_auc_samples",
        "pr_auc_micro",
        "pr_auc_macro",
        "pr_auc_weighted",
        "pr_auc_samples",
        "accuracy",
        "f1_micro",
        "f1_macro",
        "f1_weighted",
        "f1_samples",
        "precision_micro",
        "precision_macro",
        "precision_weighted",
        "precision_samples",
        "recall_micro",
        "recall_macro",
        "recall_weighted",
        "recall_samples",
        "jaccard_micro",
        "jaccard_macro",
        "jaccard_weighted",
        "jaccard_samples",
        "hamming_loss",
    ]
    y_true = np.random.randint(2, size=(100000, 100))
    y_prob = np.random.random(size=(100000, 100))
    print(multilabel_metrics_fn(y_true, y_prob, metrics=all_metrics))
