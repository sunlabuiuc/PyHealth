from typing import List, Optional, Dict

import numpy as np
import sklearn.metrics as sklearn_metrics


def multilabel_metrics_fn(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metrics: Optional[List[str]] = None,
        threshold: float = 0.5,
) -> Dict[str, float]:
    if metrics is None:
        metrics = ["roc_auc_macro", "pr_auc_macro"]

    y_pred = y_prob.copy()
    y_pred[y_pred >= threshold] = 1
    y_pred[y_pred < threshold] = 0

    output = {}
    for metric in metrics:
        if metric == "roc_auc_micro":
            roc_auc_micro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average='micro'
            )
            output["roc_auc_micro"] = roc_auc_micro
        elif metric == "roc_auc_macro":
            roc_auc_macro = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average='macro'
            )
            output["roc_auc_macro"] = roc_auc_macro
        elif metric == "roc_auc_weighted":
            roc_auc_weighted = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average='weighted'
            )
            output["roc_auc_weighted"] = roc_auc_weighted
        elif metric == "roc_auc_samples":
            roc_auc_samples = sklearn_metrics.roc_auc_score(
                y_true, y_prob, average='samples'
            )
            output["roc_auc_samples"] = roc_auc_samples
        elif metric == "pr_auc_micro":
            pr_auc_micro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average='micro'
            )
            output["pr_auc_micro"] = pr_auc_micro
        elif metric == "pr_auc_macro":
            pr_auc_macro = sklearn_metrics.average_precision_score(
                y_true, y_prob, average='macro'
            )
            output["pr_auc_macro"] = pr_auc_macro
        elif metric == "pr_auc_weighted":
            pr_auc_weighted = sklearn_metrics.average_precision_score(
                y_true, y_prob, average='weighted'
            )
            output["pr_auc_weighted"] = pr_auc_weighted
        elif metric == "pr_auc_samples":
            pr_auc_samples = sklearn_metrics.average_precision_score(
                y_true, y_prob, average='samples'
            )
            output["pr_auc_samples"] = pr_auc_samples
        elif metric == "accuracy":
            accuracy = sklearn_metrics.accuracy_score(y_true, y_pred)
            output["accuracy"] = accuracy
        elif metric == "f1_micro":
            f1_micro = sklearn_metrics.f1_score(y_true, y_pred, average='micro')
            output["f1_micro"] = f1_micro
        elif metric == "f1_macro":
            f1_macro = sklearn_metrics.f1_score(y_true, y_pred, average='macro')
            output["f1_macro"] = f1_macro
        elif metric == "f1_weighted":
            f1_weighted = sklearn_metrics.f1_score(y_true, y_pred, average='weighted')
            output["f1_weighted"] = f1_weighted
        elif metric == "f1_samples":
            f1_samples = sklearn_metrics.f1_score(y_true, y_pred, average='samples')
            output["f1_samples"] = f1_samples
        elif metric == "precision_micro":
            precision_micro = sklearn_metrics.precision_score(
                y_true, y_pred, average='micro'
            )
            output["precision_micro"] = precision_micro
        elif metric == "precision_macro":
            precision_macro = sklearn_metrics.precision_score(
                y_true, y_pred, average='macro'
            )
            output["precision_macro"] = precision_macro
        elif metric == "precision_weighted":
            precision_weighted = sklearn_metrics.precision_score(
                y_true, y_pred, average='weighted'
            )
            output["precision_weighted"] = precision_weighted
        elif metric == "precision_samples":
            precision_samples = sklearn_metrics.precision_score(
                y_true, y_pred, average='samples'
            )
            output["precision_samples"] = precision_samples
        elif metric == "recall_micro":
            recall_micro = sklearn_metrics.recall_score(
                y_true, y_pred, average='micro'
            )
            output["recall_micro"] = recall_micro
        elif metric == "recall_macro":
            recall_macro = sklearn_metrics.recall_score(
                y_true, y_pred, average='macro'
            )
            output["recall_macro"] = recall_macro
        elif metric == "recall_weighted":
            recall_weighted = sklearn_metrics.recall_score(
                y_true, y_pred, average='weighted'
            )
            output["recall_weighted"] = recall_weighted
        elif metric == "recall_samples":
            recall_samples = sklearn_metrics.recall_score(
                y_true, y_pred, average='samples'
            )
            output["recall_samples"] = recall_samples
        elif metric == "jaccard_micro":
            jaccard_micro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average='micro'
            )
            output["jaccard_micro"] = jaccard_micro
        elif metric == "jaccard_macro":
            jaccard_macro = sklearn_metrics.jaccard_score(
                y_true, y_pred, average='macro'
            )
            output["jaccard_macro"] = jaccard_macro
        elif metric == "jaccard_weighted":
            jaccard_weighted = sklearn_metrics.jaccard_score(
                y_true, y_pred, average='weighted'
            )
            output["jaccard_weighted"] = jaccard_weighted
        elif metric == "jaccard_samples":
            jaccard_samples = sklearn_metrics.jaccard_score(
                y_true, y_pred, average='samples'
            )
            output["jaccard_samples"] = jaccard_samples
        elif metric == "hamming_loss":
            hamming_loss = sklearn_metrics.hamming_loss(y_true, y_pred)
            output["hamming_loss"] = hamming_loss
        else:
            raise ValueError(f"Unknown metric for multilabel classification: {metric}")

    return output


if __name__ == '__main__':
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
