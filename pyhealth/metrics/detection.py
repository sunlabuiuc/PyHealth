"""Object detection metrics: COCO average precision for bounding boxes.

The numbers match ``pycocotools`` ``COCOeval`` with ``iouType="bbox"`` and
the area range ``"all"``: detections are matched greedily by score to the
unmatched ground-truth box with the highest IoU, precision is interpolated
at 101 recall points, and AP is averaged over classes (and, for ``"ap"``,
over the IoU thresholds 0.50:0.05:0.95). ``pycocotools`` is not needed.

Boxes are in COCO format, ``[x, y, width, height]`` in pixels.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

_IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
_METRICS = ("ap", "ap50", "ap75", "ar100")


def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU between two sets of COCO ``[x, y, w, h]`` boxes.

    Examples:
        >>> box_iou(np.array([[0, 0, 10, 10]]), np.array([[5, 0, 10, 10]]))
        array([[0.33333333]])
    """
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    bottom = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _match(
    gt: np.ndarray, dt: np.ndarray, iou_thresholds: np.ndarray
) -> np.ndarray:
    """Whether each detection (in score order) is a true positive, per IoU."""
    matched = np.zeros((len(iou_thresholds), len(dt)), dtype=bool)
    if len(gt) == 0 or len(dt) == 0:
        return matched
    ious = box_iou(dt, gt)
    for t, thresh in enumerate(iou_thresholds):
        taken = np.zeros(len(gt), dtype=bool)
        for d in range(len(dt)):
            best, best_iou = -1, min(thresh, 1 - 1e-10)
            for g in range(len(gt)):
                if taken[g] or ious[d, g] < best_iou:
                    continue
                best, best_iou = g, ious[d, g]
            if best >= 0:
                taken[best] = True
                matched[t, d] = True
    return matched


def _class_curves(
    y_true: Sequence[dict[str, Any]],
    y_pred: Sequence[dict[str, Any]],
    label: Any,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolated precision ``[T, R]`` and recall ``[T]`` of one class."""
    scores, hits, n_gt = [], [], 0
    for truth, pred in zip(y_true, y_pred):
        gt_labels = np.asarray(truth.get("labels", [0] * len(truth["boxes"])))
        gt = np.asarray(truth["boxes"], dtype=np.float64).reshape(-1, 4)[gt_labels == label]
        dt_labels = np.asarray(pred.get("labels", [0] * len(pred["boxes"])))
        keep = dt_labels == label
        dt = np.asarray(pred["boxes"], dtype=np.float64).reshape(-1, 4)[keep]
        dt_scores = np.asarray(pred["scores"], dtype=np.float64).reshape(-1)[keep]
        order = np.argsort(-dt_scores, kind="mergesort")[:max_detections]
        n_gt += len(gt)
        scores.append(dt_scores[order])
        hits.append(_match(gt, dt[order], _IOU_THRESHOLDS))
    precision = -np.ones((len(_IOU_THRESHOLDS), len(_RECALL_THRESHOLDS)))
    recall = -np.ones(len(_IOU_THRESHOLDS))
    if n_gt == 0:
        return precision, recall
    scores = np.concatenate(scores)
    order = np.argsort(-scores, kind="mergesort")
    hits = np.concatenate(hits, axis=1)[:, order]
    tp = np.cumsum(hits, axis=1, dtype=np.float64)
    fp = np.cumsum(~hits, axis=1, dtype=np.float64)
    for t in range(len(_IOU_THRESHOLDS)):
        rc = tp[t] / n_gt
        pr = tp[t] / (tp[t] + fp[t] + np.spacing(1))
        recall[t] = rc[-1] if len(rc) else 0.0
        pr = np.maximum.accumulate(pr[::-1])[::-1] if len(pr) else pr
        idx = np.searchsorted(rc, _RECALL_THRESHOLDS, side="left")
        q = np.zeros(len(_RECALL_THRESHOLDS))
        valid = idx < len(pr)
        q[valid] = pr[idx[valid]]
        precision[t] = q
    return precision, recall


def detection_metrics_fn(
    y_true: Sequence[dict[str, Any]],
    y_pred: Sequence[dict[str, Any]],
    metrics: Sequence[str] = ("ap", "ap50"),
    max_detections: int = 100,
) -> dict[str, float]:
    """COCO box AP of detections, one dict per image.

    Args:
        y_true: For each image, ``boxes`` (``[N, 4]``, COCO ``[x, y, w, h]``)
            and optionally ``labels`` (``[N]``; all one class if left out).
        y_pred: For each image, ``boxes`` (``[M, 4]``), ``scores`` (``[M]``)
            and optionally ``labels`` (``[M]``).
        metrics: Any of ``"ap"`` (mean over IoU 0.50:0.95), ``"ap50"``,
            ``"ap75"`` and ``"ar100"`` (recall averaged over IoU 0.50:0.95,
            at most ``max_detections`` detections per image and class).
        max_detections: Highest-scoring detections kept per image and class
            (COCO's ``maxDets``).

    Returns:
        The requested metrics, as fractions in ``[0, 1]``. A class without
        ground-truth boxes is left out of the averages, as in ``COCOeval``.

    Raises:
        ValueError: On an unknown metric or a different number of images.

    Examples:
        >>> y_true = [{"boxes": [[10, 10, 20, 20]]}]
        >>> y_pred = [{"boxes": [[10, 10, 20, 20], [50, 50, 5, 5]], "scores": [0.9, 0.8]}]
        >>> scores = detection_metrics_fn(y_true, y_pred)
        >>> {k: round(v, 6) for k, v in scores.items()}
        {'ap': 1.0, 'ap50': 1.0}
    """
    unknown = [m for m in metrics if m not in _METRICS]
    if unknown:
        raise ValueError(f"Unknown metrics {unknown}; choose from {_METRICS}.")
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} ground-truth images but {len(y_pred)} predictions.")
    labels = sorted(
        {label for truth in y_true for label in truth.get("labels", [0] * len(truth["boxes"]))},
        key=str,
    )
    curves = [_class_curves(y_true, y_pred, label, max_detections) for label in labels]
    precision = np.stack([p for p, _ in curves], axis=-1) if curves else np.empty((10, 101, 0))
    recall = np.stack([r for _, r in curves], axis=-1) if curves else np.empty((10, 0))

    def mean(values: np.ndarray) -> float:
        values = values[values > -1]
        return float(values.mean()) if values.size else float("nan")

    results = {
        "ap": lambda: mean(precision),
        "ap50": lambda: mean(precision[0]),
        "ap75": lambda: mean(precision[5]),
        "ar100": lambda: mean(recall),
    }
    return {m: results[m]() for m in metrics}
