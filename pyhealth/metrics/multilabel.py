import os

import numpy as np
import sklearn.metrics as sklearn_metrics

import pyhealth.metrics.calibration as calib
import pyhealth.metrics.prediction_set as pset
from pyhealth import BASE_CACHE_PATH as CACHE_PATH
from pyhealth.metrics import ddi_rate_score
from pyhealth.metrics.ranking import multilabel_ranking_at_k


def _load_cached_ddi_adj() -> np.ndarray:
    """Loads the DDI matrix that a drug recommendation model cached."""
    path = os.path.join(CACHE_PATH, "ddi_adj.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"The 'ddi' metric needs a drug-drug interaction matrix, but {path} "
            "does not exist. Pass ddi_adj= (e.g. model.ddi_adj_matrix), or "
            "build an ARMR, GAMENet, MICRON, MoleRec or SafeDrug model first, which "
            "writes that file."
        )
    return np.load(path)


def multilabel_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: list[str] | None = None,
    threshold: float = 0.3,
    y_predset: np.ndarray | None = None,
    ddi_adj: np.ndarray | None = None,
    *,
    cost_weights: float | np.ndarray | list[float] = 1.0,
    value_weights: float | np.ndarray | list[float] = 1.0,
    target_cost: float | None = None,
) -> dict[str, float | np.ndarray]:
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
        - precision@K: fraction of the K highest-scored labels that are true
          (e.g. "precision@5")
        - recall@K: fraction of a sample's true labels among its K
          highest-scored labels
        - ndcg@K: normalized discounted cumulative gain of the K
          highest-scored labels (binary relevance). All three are averaged
          over the samples with at least one true label; see
          :func:`pyhealth.metrics.ranking.multilabel_ranking_at_k`.

    The following metrics related to the prediction sets are accepted as well, but will be ignored if y_predset is None:

        - fp: Mean unweighted false-positive count per sample.
        - tp: Mean unweighted true-positive count per sample.
        - fp_cost: Mean weighted false-positive cost per sample, using cost_weights.
        - fp_cost_violation_rate: Fraction with weighted cost strictly above
          target_cost, using the same cost_weights. Requires an explicit target_cost.
        - tp_value: Mean weighted true-positive value per sample, using value_weights.
        - set_size: Mean number of labels in each prediction set.
        - miscoverage_ps: Per-class miss rates. Classes without true labels return NaN.
        - miscoverage_mean_ps: Equal-weight average of defined class miss rates.
        - miscoverage_overall_ps: Mean fraction of true labels omitted from each
          prediction set, averaged over samples with at least one true label.
          Returns NaN if all targets are empty. Check
          :func:`pyhealth.metrics.prediction_set.miscoverage_overall_ps`.


    Undefined prediction-set measurements return NaN. Class averages exclude
    undefined classes; if every class is undefined, their average is NaN.
    fp/fp_cost and tp/tp_value share the number of samples as their denominator,
    including samples with no true labels. Empty batches return NaN.

    If no metrics are specified, pr_auc_samples is computed by default.

    This function calls sklearn.metrics functions to compute the metrics. For
    more information on the metrics, please refer to the documentation of the
    corresponding sklearn.metrics functions.

    Args:
        y_true: True target values of shape (n_samples, n_labels).
        y_prob: Predicted probabilities of shape (n_samples, n_labels).
        metrics: List of metrics to compute. Default is ["pr_auc_samples"].
        threshold: Threshold to binarize the predicted probabilities. Default is 0.3.
        y_predset: Binary prediction-set masks with shape (n_samples, n_labels).
        cost_weights: Finite non-negative scalar or class vector for fp_cost and
            fp_cost_violation_rate only.
            Defaults to unit costs. Vector entries use encoded class order.
        value_weights: Finite non-negative scalar or class vector for tp_value only.
            Defaults to unit values. Vector entries use encoded class order.
        target_cost: Finite non-negative limit for fp_cost_violation_rate. Required
            when requesting that metric with y_predset; ignored by other metrics.
            Zero-dimensional NumPy arrays and Torch tensors are accepted.
        ddi_adj: Optional drug-drug interaction matrix of shape
            (n_labels, n_labels) for the ``ddi`` metric, indexed like the
            label columns. When it is None, ``ddi`` loads ``ddi_adj.npy`` from
            the PyHealth cache directory, which holds the matrix of the drug
            recommendation model built most recently in any process. Pass
            the evaluated model's own matrix (``model.ddi_adj_matrix`` for
            ARMR, GAMENet, MICRON, MoleRec and SafeDrug) to avoid that shared file.

    Returns:
        Dictionary of metric names mapped to scalar values or per-class arrays.
            Undefined numerical results remain NaN; JSON conversion belongs to callers.

    Examples:
        >>> from pyhealth.metrics import multilabel_metrics_fn
        >>> y_true = np.array([[0, 1, 1], [1, 0, 1]])
        >>> y_prob = np.array([[0.1, 0.9, 0.8], [0.05, 0.95, 0.6]])
        >>> multilabel_metrics_fn(y_true, y_prob, metrics=["accuracy"])
        {'accuracy': 0.6666666666666666}
        >>> ddi_adj = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        >>> multilabel_metrics_fn(
        ...     y_true, y_prob, metrics=["ddi"], threshold=0.5, ddi_adj=ddi_adj
        ... )
        {'ddi_score': 0.0}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    if y_predset is not None:
        y_predset = np.asarray(y_predset)

    if metrics is None:
        metrics = ["pr_auc_samples"]
    prediction_set_metrics = [
        "tp",
        "fp",
        "fp_cost",
        "fp_cost_violation_rate",
        "tp_value",
        "set_size",
        "miscoverage_ps",
        "miscoverage_mean_ps",
        "miscoverage_overall_ps",
    ]

    # Reuse prediction_set validation and empty averages so direct helper calls
    # and this scorer enforce the same mask and undefined-denominator rules.
    if y_predset is not None and any(
        metric in prediction_set_metrics for metric in metrics
    ):
        pset._validate_multilabel_masks(y_predset, y_true)

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
            if ddi_adj is None:
                ddi_adj = _load_cached_ddi_adj()
            # Use a separate name to avoid mutating y_pred for subsequent metrics
            y_pred_ddi = [np.where(item)[0] for item in y_pred]
            output["ddi_score"] = ddi_rate_score(y_pred_ddi, ddi_adj)
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
                output[metric] = pset._mean_or_nan(
                    (y_true * y_predset).sum(axis=1)
                )
            elif metric == 'fp':
                output[metric] = pset._mean_or_nan(
                    ((1 - y_true) * y_predset).sum(axis=1)
                )
            elif metric == 'fp_cost':
                output[metric] = pset.fp_cost(y_predset, y_true, cost_weights)
            elif metric == 'fp_cost_violation_rate':
                output[metric] = pset.fp_cost_violation_rate(
                    y_predset, y_true, target_cost, cost_weights
                )
            elif metric == 'tp_value':
                output[metric] = pset.tp_value(y_predset, y_true, value_weights)
            elif metric == 'set_size':
                output[metric] = pset.size(y_predset)
            elif metric == 'miscoverage_ps':
                output[metric] = pset.miscoverage_ps(y_predset, y_true)
            elif metric == 'miscoverage_mean_ps':
                output[metric] = pset.miscoverage_mean_ps(y_predset, y_true)
            elif metric == 'miscoverage_overall_ps':
                output[metric] = pset.miscoverage_overall_ps(y_predset, y_true)
        elif metric.split("@", 1)[0] in ("precision", "recall", "ndcg") and (
            "@" in metric
        ):
            name, cutoff = metric.split("@", 1)
            if not cutoff.isdigit() or int(cutoff) < 1:
                raise ValueError(
                    f"Invalid metric {metric!r}: expected '{name}@K' with K a "
                    "positive integer (e.g. 'precision@5')."
                )
            output[metric] = multilabel_ranking_at_k(
                y_true, y_prob, int(cutoff)
            )[name]
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
