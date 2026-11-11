"""Metrics for pixel/voxel-level segmentation.

The functions accept one sample at a time or a list of per-case arrays. This
matches :class:`pyhealth.trainer.Trainer`, which keeps segmentation outputs as
per-sample arrays because masks can have different shapes. The core Dice/IoU
math is delegated to MONAI's ``compute_dice`` / ``compute_iou``; the NumPy API
and per-case aggregation remain here for Trainer compatibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value: Any) -> np.ndarray:
    """Convert tensors or array-like values to NumPy arrays."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _spatial_target(target: Any) -> np.ndarray:
    """Return a target mask without a singleton channel axis."""
    array = _to_numpy(target)
    if array.ndim >= 1 and array.shape[0] == 1 and array.ndim >= 3:
        array = array[0]
    return array.astype(np.int64, copy=False)


def _spatial_probability(prob: Any, mode: str, threshold: float) -> np.ndarray:
    """Return a class-index prediction from probabilities."""
    array = _to_numpy(prob).astype(np.float32, copy=False)
    if mode == "segmentation_binary":
        if array.ndim >= 1 and array.shape[0] == 1 and array.ndim >= 3:
            array = array[0]
        return (array >= threshold).astype(np.int64)
    if mode == "segmentation_multiclass":
        if array.ndim < 2:
            raise ValueError("Multiclass probabilities need a class axis.")
        return np.argmax(array, axis=0).astype(np.int64)
    raise ValueError(f"Unsupported segmentation mode: {mode!r}.")


def _valid_mask(
    target: np.ndarray,
    ignore_index: int | None,
) -> np.ndarray | None:
    if ignore_index is None:
        return None
    return target != int(ignore_index)


def _require_monai_metrics():
    """Import MONAI metric functions with an actionable optional error."""
    try:
        from monai.metrics import compute_dice, compute_iou
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Segmentation metrics require MONAI. Install it with "
            "`pip install 'monai>=1.6.1,<2'` or `pip install pyhealth[imaging]`."
        ) from exc
    return compute_dice, compute_iou


def _one_hot(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Convert a label map to a ``(1, C, *spatial)`` float one-hot tensor."""
    tensor = torch.as_tensor(labels, dtype=torch.long)
    return (
        F.one_hot(tensor, num_classes=num_classes)
        .movedim(-1, 0)
        .unsqueeze(0)
        .float()
    )


def _metric_arrays(
    y_true: Any,
    y_prob: Any,
    mode: str,
    threshold: float,
    ignore_index: int | None,
    num_classes: int | None,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return filtered target/prediction label maps and the class count."""
    target = _spatial_target(y_true)
    prediction = _spatial_probability(y_prob, mode, threshold)
    if mode == "segmentation_binary":
        # Binary predictions are always 0/1; a declared class count would
        # reject valid foreground pixels when it is 1.
        num_classes = None
    valid = _valid_mask(target, ignore_index)
    if valid is not None:
        target = target[valid]
        prediction = prediction[valid]
    if num_classes is not None:
        num_classes = int(num_classes)
        if prediction.size and int(prediction.max()) >= num_classes:
            raise ValueError(
                f"y_prob contains class index {int(prediction.max())}, "
                f"which is outside num_classes={num_classes}."
            )
        # ``ignore_index`` was already removed above. Any remaining
        # out-of-range label is a data error, not padding; dropping it would
        # silently inflate the metrics.
        invalid = (target < 0) | (target >= num_classes)
        if bool(invalid.any()):
            bad = np.unique(target[invalid])
            raise ValueError(
                f"Target contains labels outside num_classes={num_classes}: "
                f"{bad[:5].tolist()}. Pass ignore_index for padding values, "
                "or fix the mask."
            )
    if (
        num_classes is None
        and ignore_index is None
        and target.size
        and int(target.max()) >= 255
    ):
        raise ValueError(
            "Target contains a padding/unlabeled value >= 255; pass "
            "ignore_index explicitly, or use per_class_segmentation_metrics "
            "with num_classes."
        )
    if mode == "segmentation_binary":
        classes = 2
    elif num_classes is not None:
        classes = int(num_classes)
    else:
        classes = max(
            int(target.max()) + 1 if target.size else 0,
            int(prediction.max()) + 1 if prediction.size else 0,
        )
    return target, prediction, classes


def remap_class_mask(
    y_true: Any,
    y_prob: Any,
    class_mask: Any,
    ignore_index: int = 255,
) -> tuple[np.ndarray, np.ndarray]:
    """Compress probabilities and labels to the classes present in a mask.

    ``class_mask`` must include background (index 0).  Kept classes are
    renumbered contiguously so the metric's background exclusion still means
    the original background.  Labels for excluded classes become
    ``ignore_index``.

    Args:
        y_true: Integer mask of shape ``(*spatial)`` or ``(1, *spatial)``.
        y_prob: Probability map of shape ``(num_classes, *spatial)``.
        class_mask: Boolean sequence of length ``num_classes``.
        ignore_index: Label used for excluded target classes.

    Returns:
        A tuple ``(target, probability)`` with compressed class channels.

    Raises:
        ValueError: If the mask length does not match the probabilities, is
            all false, or does not include background.

    Examples:
        >>> import numpy as np
        >>> target = np.array([[0, 1]])
        >>> probability = np.zeros((3, 1, 2))
        >>> remapped, selected = remap_class_mask(
        ...     target, probability, [1, 0, 1]
        ... )
        >>> remapped.tolist()
        [[0, 255]]
        >>> selected.shape
        (2, 1, 2)
    """
    target = _spatial_target(y_true)
    probability = _to_numpy(y_prob)
    mask = _to_numpy(class_mask).astype(bool)
    if probability.ndim < 2:
        raise ValueError("y_prob must have a class axis.")
    if mask.ndim != 1 or mask.size != probability.shape[0]:
        raise ValueError(
            "class_mask must have one entry per probability class: "
            f"{mask.size} vs {probability.shape[0]}."
        )
    if not mask.any():
        raise ValueError("class_mask must select at least one class.")
    if not mask[0]:
        raise ValueError("class_mask must include background class 0.")

    valid = np.flatnonzero(mask)
    mapping = np.full(probability.shape[0], int(ignore_index), dtype=np.int64)
    mapping[valid] = np.arange(valid.size)
    in_range = (target >= 0) & (target < probability.shape[0])
    clipped = np.clip(target, 0, probability.shape[0] - 1)
    target = np.where(in_range, mapping[clipped], int(ignore_index))
    return target, probability[valid]


def _apply_empty_reference_rule(
    scores: torch.Tensor,
    prediction: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Replace MONAI's NaN for empty references with an explicit score.

    A class that is empty in both the reference and the prediction stays NaN,
    so it is left out of the case average instead of earning a free 1.0. An
    empty reference with predicted foreground scores 0.0, so a false positive
    cannot be reported as a perfect Dice or IoU.
    """
    scores = scores.reshape(-1)
    reference_has = reference.flatten(2).sum(-1).reshape(-1) > 0
    prediction_has = prediction.flatten(2).sum(-1).reshape(-1) > 0
    empty_reference = torch.where(
        prediction_has,
        torch.zeros_like(scores),
        torch.full_like(scores, float("nan")),
    )
    return torch.where(reference_has, scores, empty_reference)


def dice_score(
    y_true: Any,
    y_prob: Any,
    mode: str = "segmentation_multiclass",
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = False,
    num_classes: int | None = None,
) -> float:
    """Compute Dice for one segmentation case.

    Args:
        y_true: Integer mask of shape ``(*spatial)`` or ``(1, *spatial)``.
        y_prob: Probabilities of shape ``(1, *spatial)`` for binary or
            ``(num_classes, *spatial)`` for multiclass.
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        threshold: Probability threshold for binary predictions.
        ignore_index: Optional label value excluded from the score.
        include_background: Whether class 0 is included in the multiclass
            average. Defaults to False.
        num_classes: Optional fixed number of classes. When omitted, the class
            count is inferred from each case's target and prediction. Labels
            outside the range raise unless they equal ``ignore_index``.

    Returns:
        The mean Dice over classes present in the reference. An empty
        reference with predicted foreground scores 0.0; classes empty in both
        the reference and the prediction are left out. Returns NaN when no
        class is present in the reference.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.segmentation import dice_score
        >>> y = np.array([[0, 1], [1, 1]])
        >>> p = np.array([[0.0, 0.9], [0.8, 0.7]])
        >>> round(dice_score(y, p, mode="segmentation_binary"), 3)
        1.0
    """
    target, prediction, classes = _metric_arrays(
        y_true, y_prob, mode, threshold, ignore_index, num_classes
    )
    if target.size == 0:
        return float("nan")
    if mode == "segmentation_binary":
        include_background = False
        classes = 2
    elif classes == 0:
        return float("nan")
    prediction_one_hot = _one_hot(prediction, classes)
    target_one_hot = _one_hot(target, classes)
    compute_dice, _ = _require_monai_metrics()
    scores = compute_dice(
        prediction_one_hot,
        target_one_hot,
        include_background=include_background,
        ignore_empty=True,
    )
    if include_background:
        selected_prediction = prediction_one_hot
        selected_target = target_one_hot
    else:
        selected_prediction = prediction_one_hot[:, 1:]
        selected_target = target_one_hot[:, 1:]
    scores = _apply_empty_reference_rule(
        scores, selected_prediction, selected_target
    )
    finite = scores[~torch.isnan(scores)]
    return float(finite.mean()) if finite.numel() else float("nan")


def iou_score(
    y_true: Any,
    y_prob: Any,
    mode: str = "segmentation_multiclass",
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = False,
    num_classes: int | None = None,
) -> float:
    """Compute intersection-over-union for one segmentation case.

    Args:
        y_true: Integer mask of shape ``(*spatial)`` or ``(1, *spatial)``.
        y_prob: Probabilities for binary or multiclass segmentation.
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        threshold: Probability threshold for binary predictions.
        ignore_index: Optional label value excluded from the score.
        include_background: Whether class 0 is included in the multiclass
            average.
        num_classes: Optional fixed number of classes. When omitted, the class
            count is inferred from each case's target and prediction. Labels
            outside the range raise unless they equal ``ignore_index``.

    Returns:
        The mean IoU over classes present in the reference. An empty
        reference with predicted foreground scores 0.0; classes empty in both
        the reference and the prediction are left out. Returns NaN when no
        class is present in the reference.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.segmentation import iou_score
        >>> y = np.array([[0, 1], [1, 1]])
        >>> p = np.array([[0.0, 0.9], [0.8, 0.7]])
        >>> round(iou_score(y, p, mode="segmentation_binary"), 3)
        1.0
    """
    target, prediction, classes = _metric_arrays(
        y_true, y_prob, mode, threshold, ignore_index, num_classes
    )
    if target.size == 0:
        return float("nan")
    if mode == "segmentation_binary":
        include_background = False
        classes = 2
    elif classes == 0:
        return float("nan")
    prediction_one_hot = _one_hot(prediction, classes)
    target_one_hot = _one_hot(target, classes)
    _, compute_iou = _require_monai_metrics()
    scores = compute_iou(
        prediction_one_hot,
        target_one_hot,
        include_background=include_background,
        ignore_empty=True,
    )
    if include_background:
        selected_prediction = prediction_one_hot
        selected_target = target_one_hot
    else:
        selected_prediction = prediction_one_hot[:, 1:]
        selected_target = target_one_hot[:, 1:]
    scores = _apply_empty_reference_rule(
        scores, selected_prediction, selected_target
    )
    finite = scores[~torch.isnan(scores)]
    return float(finite.mean()) if finite.numel() else float("nan")


def _per_class_scores(
    y_true: Any,
    y_prob: Any,
    mode: str,
    num_classes: int | None,
    threshold: float,
    ignore_index: int | None,
    include_background: bool,
    metric: str,
) -> dict[int, float]:
    """Compute MONAI Dice or IoU separately for each class."""
    target, prediction, classes = _metric_arrays(
        y_true, y_prob, mode, threshold, ignore_index, num_classes
    )
    start = 0 if include_background else 1
    if mode == "segmentation_binary":
        classes = 2
    if target.size == 0 or classes <= start:
        return {
            cls: float("nan") for cls in range(start, max(start, classes))
        }
    compute_dice, compute_iou = _require_monai_metrics()
    score_fn = compute_dice if metric == "dice" else compute_iou
    prediction_one_hot = _one_hot(prediction, classes)
    target_one_hot = _one_hot(target, classes)
    scores = score_fn(
        prediction_one_hot,
        target_one_hot,
        include_background=include_background,
        ignore_empty=True,
    ).reshape(-1)
    scores = _apply_empty_reference_rule(
        scores,
        prediction_one_hot[:, start:],
        target_one_hot[:, start:],
    )
    return {
        cls: float(scores[index])
        for index, cls in enumerate(range(start, classes))
    }


def per_class_dice_score(
    y_true: Any,
    y_prob: Any,
    mode: str = "segmentation_multiclass",
    num_classes: int | None = None,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = True,
) -> dict[int, float]:
    """Compute Dice separately for every class in one case.

    Classes absent from both the target and the prediction are NaN and are
    left out of macro averages; a class with an empty target but predicted
    foreground scores 0.0.

    Args:
        y_true: Integer mask of shape ``(*spatial)`` or ``(1, *spatial)``.
        y_prob: Probabilities for binary or multiclass segmentation.
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        num_classes: Number of classes to report. Inferred when omitted.
        threshold: Probability threshold for binary predictions.
        ignore_index: Optional label value excluded from the score.
        include_background: Whether class 0 is included. Defaults to True
            because this function reports every class; aggregate metrics
            should keep it False.

    Returns:
        A mapping from class index to Dice score; empty classes are NaN.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.segmentation import per_class_dice_score
        >>> y = np.array([[0, 1], [2, 2]])
        >>> scores = per_class_dice_score(
        ...     y, np.zeros((3, 2, 2)), num_classes=3
        ... )
        >>> sorted(scores)
        [0, 1, 2]
    """
    return _per_class_scores(
        y_true,
        y_prob,
        mode,
        num_classes,
        threshold,
        ignore_index,
        include_background,
        "dice",
    )


def per_class_segmentation_metrics(
    y_true: Any,
    y_prob: Any,
    metrics: Sequence[str] | None = None,
    mode: str = "segmentation_multiclass",
    num_classes: int | None = None,
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = False,
) -> dict[str, float]:
    """Average per-class Dice (and optionally IoU) over cases.

    Args:
        y_true: One mask or a list of masks.
        y_prob: One probability map or a list of probability maps.
        metrics: Metric names. Currently supports ``"dice"`` and ``"iou"``.
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        num_classes: Number of classes to report.
        threshold: Probability threshold for binary predictions.
        ignore_index: Optional label value excluded from the metrics.
        include_background: Whether to report class 0. Defaults to False, so
            foreground averages do not get inflated by the large background
            class.

    Returns:
        A dictionary with keys such as ``dice_class_1`` and ``dice_class_2``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.segmentation import (
        ...     per_class_segmentation_metrics,
        ... )
        >>> y = [np.array([[0, 1]])]
        >>> p = [np.array([[[0.0, 0.9]]])]
        >>> scores = per_class_segmentation_metrics(
        ...     y, p, mode="segmentation_binary"
        ... )
        >>> round(scores["dice_class_1"], 3)
        1.0
    """
    if metrics is None:
        metrics = ["dice"]
    elif isinstance(metrics, str):
        metrics = [metrics]
    unknown = set(metrics) - {"dice", "iou"}
    if unknown:
        raise ValueError(f"Unsupported metric(s): {sorted(unknown)}.")

    true_cases = _as_case_list(y_true)
    prob_cases = _as_case_list(y_prob)
    if len(true_cases) != len(prob_cases):
        raise ValueError(
            "y_true and y_prob must contain the same number of cases."
        )

    per_case: dict[int, dict[str, list[float]]] = {}
    for true, prob in zip(true_cases, prob_cases):
        dice_scores = per_class_dice_score(
            true,
            prob,
            mode=mode,
            num_classes=num_classes,
            threshold=threshold,
            ignore_index=ignore_index,
            include_background=include_background,
        )
        iou_scores = (
            _per_class_scores(
                true,
                prob,
                mode,
                num_classes,
                threshold,
                ignore_index,
                include_background,
                "iou",
            )
            if "iou" in metrics
            else {}
        )
        for cls, dice in dice_scores.items():
            entry = per_case.setdefault(
                cls, {"dice": [], "iou": []}
            )
            entry["dice"].append(dice)
            if "iou" in metrics:
                entry["iou"].append(iou_scores.get(cls, float("nan")))

    output: dict[str, float] = {}
    for cls, values in sorted(per_case.items()):
        for metric in metrics:
            values_list = values[metric]
            finite = [value for value in values_list if not np.isnan(value)]
            output[f"{metric}_class_{cls}"] = (
                float(np.mean(finite)) if finite else float("nan")
            )
    return output


def _as_case_list(value: Any) -> list[Any]:
    """Treat a list/tuple as per-case data and an array as one case."""
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def segmentation_metrics_fn(
    y_true: Any,
    y_prob: Any,
    metrics: Sequence[str] | None = None,
    mode: str = "segmentation_multiclass",
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = False,
    num_classes: int | Sequence[int] | None = None,
) -> dict[str, float]:
    """Compute segmentation metrics over one case or a list of cases.

    Args:
        y_true: One mask or a list of masks.
        y_prob: One probability map or a list of probability maps.
        metrics: Metric names. Defaults to ``["dice", "iou"]``.
        mode: ``"segmentation_binary"`` or ``"segmentation_multiclass"``.
        threshold: Probability threshold for binary predictions.
        ignore_index: Optional label value excluded from the metrics.
        include_background: Whether class 0 is included in the multiclass
            average.
        num_classes: Optional fixed number of classes. A sequence gives one
            count per case, which is needed when ``class_mask`` remaps cases
            to different numbers of classes. When omitted, each case infers
            its own class count.

    Returns:
        A dictionary whose values are averaged over cases.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics.segmentation import segmentation_metrics_fn
        >>> y = [np.array([[0, 1], [1, 1]])]
        >>> p = [np.array([[0.0, 0.9], [0.8, 0.7]])]
        >>> scores = segmentation_metrics_fn(y, p, mode="segmentation_binary")
        >>> round(scores["dice"], 3)
        1.0
    """
    if metrics is None:
        metrics = ["dice", "iou"]
    elif isinstance(metrics, str):
        metrics = [metrics]
    unknown = set(metrics) - {"dice", "iou"}
    if unknown:
        raise ValueError(f"Unsupported segmentation metrics: {sorted(unknown)}.")
    true_cases = _as_case_list(y_true)
    prob_cases = _as_case_list(y_prob)
    if len(true_cases) != len(prob_cases):
        raise ValueError(
            "y_true and y_prob must contain the same number of cases."
        )
    if not true_cases:
        return {}

    per_case_classes = isinstance(num_classes, Sequence) and not isinstance(
        num_classes, (str, bytes)
    )
    if per_case_classes and len(num_classes) != len(true_cases):
        raise ValueError(
            "num_classes must have one entry per case: "
            f"{len(num_classes)} vs {len(true_cases)}."
        )
    scores: dict[str, list[float]] = {name: [] for name in metrics}
    for index, (true, prob) in enumerate(zip(true_cases, prob_cases)):
        case_num_classes = (
            num_classes[index] if per_case_classes else num_classes
        )
        if "dice" in scores:
            scores["dice"].append(
                dice_score(
                    true,
                    prob,
                    mode=mode,
                    threshold=threshold,
                    ignore_index=ignore_index,
                    include_background=include_background,
                    num_classes=case_num_classes,
                )
            )
        if "iou" in scores:
            scores["iou"].append(
                iou_score(
                    true,
                    prob,
                    mode=mode,
                    threshold=threshold,
                    ignore_index=ignore_index,
                    include_background=include_background,
                    num_classes=case_num_classes,
                )
            )
    return {
        name: (
            float(np.mean([value for value in values if not np.isnan(value)]))
            if any(not np.isnan(value) for value in values)
            else float("nan")
        )
        for name, values in scores.items()
    }


def case_level_segmentation_metrics(
    y_true: Any,
    y_prob: Any,
    metrics: Sequence[str] | None = None,
    mode: str = "segmentation_multiclass",
    threshold: float = 0.5,
    ignore_index: int | None = None,
    include_background: bool = False,
    num_classes: int | Sequence[int] | None = None,
) -> dict[str, float]:
    """Alias of :func:`segmentation_metrics_fn` for case-level evaluation.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.metrics import case_level_segmentation_metrics
        >>> y = [np.array([[0, 1]])]
        >>> p = [np.array([[0.0, 0.9]])]
        >>> score = case_level_segmentation_metrics(  # doctest: +SKIP
        ...     y, p, mode="segmentation_binary"
        ... )
    """
    return segmentation_metrics_fn(
        y_true,
        y_prob,
        metrics=metrics,
        mode=mode,
        threshold=threshold,
        ignore_index=ignore_index,
        include_background=include_background,
        num_classes=num_classes,
    )
