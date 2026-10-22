"""
Black box shift estimation (BBSE) for label shift.

BBSE estimates the label distribution of a target distribution from a
black box classifier's predictions, without target labels (Lipton et al.
2018). Under label shift, the class-conditional distribution of the
predictions does not change between the source and target distributions,
so the target prediction distribution is a mixture of the source
class-conditional prediction distributions, with the target label
distribution as the mixture weights. BBSE estimates the class-conditional
confusion matrix on labeled source data, measures the prediction
distribution on unlabeled target data, and solves the linear system
relating the two.

Paper:
    Lipton, Zachary C., Yu-Xiang Wang, and Alexander J. Smola.
    "Detecting and Correcting for Label Shift with Black Box Predictors."
    International Conference on Machine Learning (ICML), 2018.
    https://arxiv.org/abs/1802.03916
"""

import numpy as np

__all__ = ["bbse"]


def _as_label_array(y, name: str) -> np.ndarray:
    """Validate predicted or true labels and return them as int64."""
    y = np.asarray(y)
    if y.ndim != 1 or len(y) == 0:
        raise ValueError(
            f"{name} must be a non-empty one-dimensional array of labels."
        )
    if not np.issubdtype(y.dtype, np.integer):
        if not np.all(np.isfinite(y)) or not np.all(y == np.floor(y)):
            raise ValueError(f"{name} must contain integer labels.")
        y = y.astype(np.int64)
    if y.min() < 0:
        raise ValueError(f"{name} labels must be non-negative.")
    return y


def bbse(y_true_source, y_pred_source, y_pred_target, num_classes=None):
    """Estimate the target label distribution with BBSE.

    The estimate can be passed to
    :meth:`pyhealth.calib.predictionset.LabelShift.calibrate` as
    ``target_label_dist`` to build prediction sets that account for the
    shift. For a probabilistic model, the predictions are its predicted
    labels (the argmax of the predicted probabilities).

    Args:
        y_true_source: True labels of a labeled source dataset, with one
            entry per source example.
        y_pred_source: The model's predicted labels on the same source
            examples, in the same order.
        y_pred_target: The model's predicted labels on unlabeled target
            examples. No target labels are needed.
        num_classes: Number of classes. If None, it is inferred as one
            plus the largest label seen in any of the inputs.

    Returns:
        The estimated target label distribution, an array of shape
        (num_classes,) that is non-negative and sums to 1.

    Raises:
        ValueError: If the inputs are not valid label arrays, if a class
            never appears in the source labels (its target proportion
            cannot be estimated), or if the estimate has no positive mass.

    Examples:
        >>> import numpy as np
        >>> y_true_source = np.repeat([0, 1], 100)
        >>> y_pred_source = np.array(
        ...     [0] * 90 + [1] * 10 + [0] * 20 + [1] * 80)
        >>> y_pred_target = np.array([0] * 69 + [1] * 31)
        >>> np.round(bbse(y_true_source, y_pred_source, y_pred_target), 6)
        array([0.7, 0.3])
    """
    y_true_source = _as_label_array(y_true_source, "y_true_source")
    y_pred_source = _as_label_array(y_pred_source, "y_pred_source")
    y_pred_target = _as_label_array(y_pred_target, "y_pred_target")
    if len(y_true_source) != len(y_pred_source):
        raise ValueError(
            "y_true_source and y_pred_source must have the same length, "
            f"got {len(y_true_source)} and {len(y_pred_source)}."
        )
    if num_classes is None:
        num_classes = (
            int(
                max(
                    y_true_source.max(),
                    y_pred_source.max(),
                    y_pred_target.max(),
                )
            )
            + 1
        )
    if num_classes < 1:
        raise ValueError(f"num_classes must be positive, got {num_classes}.")
    for name, y in (
        ("y_true_source", y_true_source),
        ("y_pred_source", y_pred_source),
        ("y_pred_target", y_pred_target),
    ):
        if y.max() >= num_classes:
            raise ValueError(
                f"{name} contains label {y.max()}, which is outside "
                f"[0, {num_classes})."
            )

    # Class-conditional confusion matrix, C[i, j] = P(pred = i | true = j),
    # estimated on the labeled source data.
    confusion = np.zeros((num_classes, num_classes))
    np.add.at(confusion, (y_pred_source, y_true_source), 1.0)
    class_counts = confusion.sum(axis=0)
    if np.any(class_counts == 0):
        missing = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(
            "BBSE cannot estimate the target proportion of classes that "
            f"never appear in the source labels: {missing}."
        )
    confusion = confusion / class_counts

    # Prediction distribution on the unlabeled target data.
    target_pred_dist = np.bincount(y_pred_target, minlength=num_classes) / len(
        y_pred_target
    )

    # Under label shift, target_pred_dist = confusion @ target_label_dist.
    target_label_dist, *_ = np.linalg.lstsq(
        confusion, target_pred_dist, rcond=None
    )

    # Sampling noise can push the least squares solution off the simplex;
    # project back onto it.
    target_label_dist = np.clip(target_label_dist, 0.0, None)
    total = target_label_dist.sum()
    if total <= 0:
        raise ValueError(
            "BBSE produced an estimate with no positive mass; the "
            "classifier's source confusion matrix may be too close to "
            "singular for the observed target predictions."
        )
    return target_label_dist / total
