"""
Segmentation metrics implementation.
"""

import numpy as np


def iou_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes the Intersection over Union (IoU) score.

    Args:
        y_true (np.ndarray): Ground truth mask of shape (N, H, W) or (N, C, H, W).
        y_prob (np.ndarray): Predicted probabilities of shape (N, H, W) or
            (N, C, H, W).
        threshold (float, optional): Threshold to binarize the predictions.
            Defaults to 0.5.

    Returns:
        float: The IoU score.
    """
    y_pred = (y_prob > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection

    return (intersection + 1e-7) / (union + 1e-7)


def dice_score(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> float:
    """
    Computes the Dice Coefficient (F1 score for segmentation).

    Args:
        y_true (np.ndarray): Ground truth mask of shape (N, H, W) or (N, C, H, W).
        y_prob (np.ndarray): Predicted probabilities of shape (N, H, W) or
            (N, C, H, W).
        threshold (float, optional): Threshold to binarize the predictions.
            Defaults to 0.5.

    Returns:
        float: The Dice score.
    """
    y_pred = (y_prob > threshold).astype(np.float32)
    y_true = y_true.astype(np.float32)

    intersection = np.sum(y_true * y_pred)
    dice = (2.0 * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-7)

    return dice


def segmentation_metrics_fn(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metrics: list[str] | None = None,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Computes a set of segmentation metrics.

    Args:
        y_true (np.ndarray): Ground truth mask of shape (N, H, W) or (N, C, H, W).
        y_prob (np.ndarray): Predicted probabilities of shape (N, H, W) or
            (N, C, H, W).
        metrics (list[str] | None, optional): List of metrics to compute.
            Available metrics are "iou" and "dice". If None, all available
            metrics are computed. Defaults to None.
        threshold (float, optional): Threshold to binarize the predictions.
            Defaults to 0.5.

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.
    """
    if metrics is None:
        metrics = ["iou", "dice"]

    results = {}
    for metric in metrics:
        if metric == "iou":
            results["iou"] = iou_score(y_true, y_prob, threshold)
        elif metric == "dice":
            results["dice"] = dice_score(y_true, y_prob, threshold)

    return results
