from typing import List, Optional, Dict

import numpy as np
import sklearn.metrics as sklearn_metrics


def multiclass_metrics_fn(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        metrics: Optional[List[str]] = None,
) -> Dict[str, float]:
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
