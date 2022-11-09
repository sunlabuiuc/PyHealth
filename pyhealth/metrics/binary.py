from typing import List, Optional, Dict
import numpy as np
import sklearn.metrics as sklearn_metrics


def binary_metrics_fn(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
) -> Dict[str, float]:
    if metrics is None:
        metrics = ["pr_auc", "roc_auc", "f1"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "pr_auc":
            pr_auc = sklearn_metrics.average_precision_score(y_true, y_prob)
            output["pr_auc"] = pr_auc
        elif metric == "roc_auc":
            roc_auc = sklearn_metrics.roc_auc_score(y_true, y_prob)
            output["roc_auc"] = roc_auc
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "balanced_accuracy":
            balanced_accuracy = sklearn_metrics.balanced_accuracy_score(y_true, y_pred)
            output["balanced_accuracy"] = balanced_accuracy
        elif metric == "f1":
            f1 = sklearn_metrics.f1_score(y_true, y_pred)
            output["f1"] = f1
        elif metric == "precision":
            precision = sklearn_metrics.precision_score(y_true, y_pred)
            output["precision"] = precision
        elif metric == "recall":
            recall = sklearn_metrics.recall_score(y_true, y_pred)
            output["recall"] = recall
        elif metric == "cohen_kappa":
            cohen_kappa = sklearn_metrics.cohen_kappa_score(y_true, y_pred)
            output["cohen_kappa"] = cohen_kappa
        elif metric == "jaccard":
            jaccard = sklearn_metrics.jaccard_score(y_true, y_pred)
            output["jaccard"] = jaccard
        else:
            raise ValueError(f"Unknown metric for binary classification: {metric}")
    return output


if __name__ == '__main__':
    all_metrics = [
        "pr_auc",
        "roc_auc",
        "accuracy",
        "balanced_accuracy",
        "f1",
        "precision",
        "recall",
        "cohen_kappa",
        "jaccard",
    ]
    y_true = np.random.randint(2, size=100000)
    y_prob = np.random.random(size=100000)
    print(binary_metrics_fn(y_true, y_prob, metrics=all_metrics))
