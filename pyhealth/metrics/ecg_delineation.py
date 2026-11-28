"""Ignore-aware ECG point and custom greedy segment metrics.

Contributors: Thi Van Trinh Tran and Zexi Li.
Reference: MedTsLLM, https://arxiv.org/abs/2408.07773.
The custom segment score below is NOT asserted to equal the paper's metric.
"""

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

CLASS_NAMES = ("background", "P", "QRS", "T")


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _validate(true, pred, valid, num_classes):
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    if true.shape != pred.shape or true.shape != valid.shape or true.ndim == 0:
        raise ValueError("Expected matching non-scalar label, prediction, mask arrays")
    for values in (true[valid], pred[valid]):
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("Labels and predictions must be integer class indices")
        if values.size and (values.min() < 0 or values.max() >= num_classes):
            raise ValueError("Class indices outside 0..num_classes-1")


def ecg_delineation_metrics(
    y_true: Any, y_pred: Any, ignore_mask: Any, *, num_classes: int = 4,
    sequences: Iterable[tuple[Any, Any, Any]] | None = None,
) -> dict:
    """Pool valid-point confusion counts; macro-average fixed classes.

    Arrays are [windows, samples] or a single [samples] sequence. Ignored
    points are excluded. All classes, including background, participate in
    macro precision, recall, F1 and point IoU; zero denominators score zero.
    ``sequences`` optionally supplies reconstructed patient/lead sequences
    (true, predicted, VALID mask), for custom greedy segment-wise mIoU.
    Without sequences no segment score is invented from flattened windows.
    Evaluation loss must come from the model, not class predictions.

    Examples:
        >>> scores = ecg_delineation_metrics([0, 1], [0, 2], [False, True])
        >>> scores["supervised_accuracy"], scores["ignored_points"]
        (1.0, 1)
    """
    true, pred = _numpy(y_true), _numpy(y_pred)
    ignored = _numpy(ignore_mask).astype(bool, copy=False)
    valid = ~ignored
    _validate(true, pred, valid, num_classes)
    windows = 1 if true.ndim == 1 else len(true)
    if true.ndim == 1:
        skipped = int(not valid.any())
    else:
        skipped = int(sum(not row.any() for row in valid))
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    if valid.any():
        np.add.at(confusion, (true[valid].astype(int), pred[valid].astype(int)), 1)
    tp = np.diag(confusion).astype(np.float64)
    fp, fn = confusion.sum(0) - tp, confusion.sum(1) - tp

    def ratio(numerator, denominator):
        return np.divide(numerator, denominator, out=np.zeros(num_classes),
                         where=denominator != 0)

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    f1, iou = ratio(2 * tp, 2 * tp + fp + fn), ratio(tp, tp + fp + fn)
    names = CLASS_NAMES if num_classes == 4 else tuple(
        f"class_{index}" for index in range(num_classes)
    )
    result = {
        "num_classes": num_classes, "class_names": list(names),
        "confusion_matrix": confusion.tolist(),
        "supervised_accuracy": float(tp.sum() / valid.sum()) if valid.any() else 0.0,
        "total_windows": windows, "skipped_windows": skipped,
        "supervised_points": int(valid.sum()), "ignored_points": int(ignored.sum()),
    }
    for name, values in (("precision", precision), ("recall", recall),
                         ("f1", f1), ("iou", iou)):
        result[f"macro_{name}"] = float(values.mean())
        result[f"per_class_{name}"] = dict(zip(names, map(float, values)))
    if sequences is not None:
        mean, per_class = segment_mean_iou(sequences, num_classes=num_classes)
        result["segment_miou"] = mean
        result["per_class_segment_miou"] = dict(zip(names, per_class))
    return result


def _segments(labels, valid, class_index):
    selected = valid & (labels == class_index)
    changes = np.diff(np.r_[False, selected, False].astype(np.int8))
    return list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))


def _interval_iou(left, right):
    intersection = max(0, min(left[1], right[1]) - max(left[0], right[0]))
    union = left[1] - left[0] + right[1] - right[0] - intersection
    return float(intersection / union) if union else 0.0


def segment_mean_iou(
    sequences: Iterable[tuple[Any, Any, Any]], *, num_classes: int = 4,
) -> tuple[float, list[float]]:
    """Return the research runner's deterministic greedy segment mIoU.

    Match true/predicted same-class intervals one-to-one by descending IoU
    (ties by true then predicted index). Zero-overlap pairs are ALSO matched,
    retaining the historical definition. Unmatched intervals add zeros. Pool
    scores across sequences within class, then macro-average all classes,
    including background and absent classes (zero). Invalid points break
    segments. This is not optimal matching, boundary-tolerance F1, or a verified
    reproduction of the paper's segment metric.

    Args:
        sequences: Iterable of (true, predicted, valid_mask) one-dimensional
            patient/lead sequences; never merge different patients or leads.
        num_classes: Fixed vocabulary size, including background.

    Examples:
        >>> segment_mean_iou([([0, 1], [0, 1], [True, True])], num_classes=2)
        (1.0, [1.0, 1.0])
    """
    if num_classes < 2:
        raise ValueError("num_classes must be at least two")
    pooled = [[] for _ in range(num_classes)]
    for true, pred, valid in sequences:
        true, pred = _numpy(true), _numpy(pred)
        valid = _numpy(valid).astype(bool, copy=False)
        _validate(true, pred, valid, num_classes)
        if true.ndim != 1:
            raise ValueError("Segment sequences must be one-dimensional")
        for index in range(num_classes):
            left = _segments(true, valid, index)
            right = _segments(pred, valid, index)
            candidates = [(_interval_iou(a, b), i, j)
                          for i, a in enumerate(left) for j, b in enumerate(right)]
            candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
            used_left, used_right = set(), set()
            scores = []
            for score, i, j in candidates:
                if i not in used_left and j not in used_right:
                    used_left.add(i)
                    used_right.add(j)
                    scores.append(score)
            scores.extend([0.0] * (
                len(left) - len(used_left) + len(right) - len(used_right)
            ))
            pooled[index].extend(scores)
    per_class = [float(np.mean(scores)) if scores else 0.0 for scores in pooled]
    return float(np.mean(per_class)), per_class


def assemble_ecg_sequences(parts: Mapping) -> list[tuple[np.ndarray, ...]]:
    """Join adjacent non-overlapping windows within each patient/lead only.

    ``parts[(patient_id, lead_index)]`` contains (start, end, true, predicted,
    valid_mask) entries. Gaps start a new sequence. Overlapping test windows
    raise an error; choosing an overlap-aggregation rule is out of scope.

    Examples:
        >>> parts = {(1, 0): [(0, 2, [0, 1], [0, 1], [True, True])]}
        >>> assemble_ecg_sequences(parts)[0][0].tolist()
        [0, 1]
    """
    sequences = []
    for key in sorted(parts):
        pending = ([], [], [])
        previous_end = None
        for start, end, true, pred, valid in sorted(parts[key], key=lambda x: x[:2]):
            values = tuple(_numpy(value) for value in (true, pred, valid))
            if end <= start or any(value.shape != (end - start,) for value in values):
                raise ValueError("Sequence bounds and window lengths disagree")
            if previous_end is not None and start < previous_end:
                raise ValueError(
                    "Overlapping evaluation windows require an explicit policy"
                )
            if previous_end is not None and start != previous_end:
                sequences.append(tuple(np.concatenate(value) for value in pending))
                pending = ([], [], [])
            for target, value in zip(pending, values):
                target.append(value)
            previous_end = end
        if pending[0]:
            sequences.append(tuple(np.concatenate(value) for value in pending))
    return sequences
