"""Training and evaluation helpers for Retina U-Net.

This module bridges the dataset / task pipeline
(``RetinaUNetCTDataset`` + ``RetinaUNetDetectionTask``) and the model
(``RetinaUNet``) in a way that drops into a standard PyTorch
``DataLoader`` + optimizer loop. It provides:

* :class:`RetinaUNetTorchDataset` — wraps a CT dataset + a detection
  task and emits ``(image_tensor, target_dict)`` pairs in the shape
  torchvision's RetinaNet expects (images are ``[C, H, W]`` float
  tensors; targets are dicts with ``boxes``/``labels``/``masks`` keys).
* :func:`collate_fn` — packs a batch into the ``(list[Tensor], list[dict])``
  structure the model consumes.
* :func:`train_one_epoch` — one pass over a training loader, returns
  the mean of every loss component for the epoch.
* :func:`evaluate` — detection F1 @ IoU 0.5 plus binary segmentation
  IoU on a validation loader.

None of this code is novel — it's scaffolding to turn the contribution
into something you can actually run end-to-end and compare against a
RetinaNet-only baseline via the ``seg_weight`` knob.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision.ops import box_iou

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask


class RetinaUNetTorchDataset(Dataset):
    """PyTorch ``Dataset`` wrapper around ``RetinaUNetCTDataset``.

    Each ``__getitem__`` call runs the underlying slice sample through
    the detection task and returns a tuple::

        (image: Tensor[C, H, W], target: Dict[str, Tensor])

    with ``target`` holding ``"boxes" (N, 4)``, ``"labels" (N,)`` and
    ``"masks" (H, W)``.

    Args:
        ct_dataset: An instantiated :class:`RetinaUNetCTDataset`.
        task: A detection task to apply to each raw sample. Defaults
            to :class:`RetinaUNetDetectionTask`.
        drop_empty: If True, silently skip samples whose mask yields
            zero bounding boxes. torchvision's RetinaNet training path
            handles empty boxes, but dropping them usually stabilizes
            the early epochs on small medical datasets.
    """

    def __init__(
        self,
        ct_dataset: RetinaUNetCTDataset,
        task: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        drop_empty: bool = True,
    ) -> None:
        self.ct_dataset = ct_dataset
        self.task = task if task is not None else RetinaUNetDetectionTask()
        self.drop_empty = drop_empty

        # Pre-filter empty samples so __len__ and __getitem__ align.
        self._indices: List[int] = []
        for i in range(len(self.ct_dataset)):
            if not drop_empty:
                self._indices.append(i)
                continue
            processed = self.task(self.ct_dataset[i])
            if processed["boxes"].shape[0] > 0:
                self._indices.append(i)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raw = self.ct_dataset[self._indices[idx]]
        processed = self.task(raw)

        image = processed["image"]
        # (H, W, C) -> (C, H, W)
        image_tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1))).float()

        target = {
            "boxes": torch.from_numpy(np.ascontiguousarray(processed["boxes"])).float(),
            "labels": torch.from_numpy(np.ascontiguousarray(processed["labels"])).long(),
            "masks": torch.from_numpy(np.ascontiguousarray(processed["mask"])).long(),
        }
        return image_tensor, target


def collate_fn(
    batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]
) -> Tuple[List[torch.Tensor], List[Dict[str, torch.Tensor]]]:
    """Collate variable-size detection samples for the RetinaUNet model."""
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


def _to_device(targets: List[Dict[str, torch.Tensor]], device: torch.device):
    return [{k: v.to(device) for k, v in t.items()} for t in targets]


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: Optional[float] = None,
) -> Dict[str, float]:
    """Run one training epoch; return mean loss components."""
    model.train()
    totals: Dict[str, float] = {}
    n_batches = 0

    for images, targets in loader:
        images = [img.to(device) for img in images]
        targets = _to_device(targets, device)

        losses = model(images, targets)
        loss = losses["loss_total"]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for k, v in losses.items():
            totals[k] = totals.get(k, 0.0) + float(v.item())
        n_batches += 1

    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}


def _match_predictions(
    pred_boxes: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> Tuple[int, int, int]:
    """Greedy one-to-one matching of predicted to GT boxes; return (tp, fp, fn).

    For each GT box, take the best-IoU unmatched prediction with IoU
    above threshold. Remaining predictions are false positives; unmatched
    GT boxes are false negatives.
    """
    if pred_boxes.numel() == 0 and gt_boxes.numel() == 0:
        return 0, 0, 0
    if pred_boxes.numel() == 0:
        return 0, 0, gt_boxes.shape[0]
    if gt_boxes.numel() == 0:
        return 0, pred_boxes.shape[0], 0

    ious = box_iou(pred_boxes, gt_boxes)  # [P, G]
    matched_pred = set()
    matched_gt = set()

    # For fair matching, sort GT by size descending (big lesions first)
    # and greedily pick the best unmatched prediction.
    gt_order = torch.argsort(
        (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1]),
        descending=True,
    )
    for g in gt_order.tolist():
        best_iou = iou_threshold
        best_p = -1
        for p in range(ious.shape[0]):
            if p in matched_pred:
                continue
            if ious[p, g].item() > best_iou:
                best_iou = ious[p, g].item()
                best_p = p
        if best_p >= 0:
            matched_pred.add(best_p)
            matched_gt.add(int(g))

    tp = len(matched_gt)
    fp = pred_boxes.shape[0] - len(matched_pred)
    fn = gt_boxes.shape[0] - len(matched_gt)
    return tp, fp, fn


def _binary_iou(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """Binary IoU between two 0/1 tensors of the same spatial size."""
    pred_b = pred.bool()
    gt_b = gt.bool()
    inter = (pred_b & gt_b).sum().item()
    union = (pred_b | gt_b).sum().item()
    if union == 0:
        return 1.0  # both empty = perfect match (no foreground)
    return inter / union


def _average_precision(
    per_image_preds: List[Dict[str, torch.Tensor]],
    iou_threshold: float,
) -> float:
    """PASCAL VOC-style all-point Average Precision.

    Args:
        per_image_preds: one dict per image, each with ``pred_boxes``
            ``[P, 4]``, ``pred_scores`` ``[P]`` and ``gt_boxes`` ``[G, 4]``.
            Pass every prediction — no score filtering — because AP
            integrates across the full precision-recall curve.
        iou_threshold: an anchor-to-GT match counts as a TP iff IoU
            meets this.

    Returns:
        AP as a float in ``[0, 1]``. Returns 0.0 if there are no GTs
        (undefined, but 0 is the conservative choice).
    """
    all_scores: List[float] = []
    all_tp: List[int] = []
    total_gts = 0

    for sample in per_image_preds:
        pred_boxes = sample["pred_boxes"]
        pred_scores = sample["pred_scores"]
        gt_boxes = sample["gt_boxes"]
        total_gts += gt_boxes.shape[0]

        if pred_boxes.numel() == 0:
            continue

        # Sort this image's predictions by score desc; match greedily.
        order = pred_scores.argsort(descending=True)
        pred_boxes = pred_boxes[order]
        pred_scores = pred_scores[order]

        if gt_boxes.numel() > 0:
            ious = box_iou(pred_boxes, gt_boxes)  # [P, G]
        else:
            ious = torch.zeros(pred_boxes.shape[0], 0)

        matched_gt = set()
        for i in range(pred_boxes.shape[0]):
            best_iou = 0.0
            best_gt = -1
            for g in range(gt_boxes.shape[0]):
                if g in matched_gt:
                    continue
                v = ious[i, g].item()
                if v > best_iou:
                    best_iou = v
                    best_gt = g
            is_tp = best_gt >= 0 and best_iou >= iou_threshold
            if is_tp:
                matched_gt.add(best_gt)
            all_tp.append(1 if is_tp else 0)
            all_scores.append(float(pred_scores[i].item()))

    if total_gts == 0 or not all_scores:
        return 0.0

    # Sort globally across all images by score descending.
    scores = np.asarray(all_scores)
    tp = np.asarray(all_tp, dtype=np.float64)
    order = np.argsort(-scores)
    tp = tp[order]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(1.0 - tp)
    recall = cum_tp / total_gts
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-10)

    # Prepend (0, 1), append (1, 0) to pin the PR curve endpoints.
    recall = np.concatenate([[0.0], recall, [1.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    # Precision envelope: monotone-decreasing in recall.
    for i in range(len(precision) - 1, 0, -1):
        precision[i - 1] = max(precision[i - 1], precision[i])

    # Integrate only over recall jumps.
    ap = 0.0
    for i in range(1, len(recall)):
        if recall[i] > recall[i - 1]:
            ap += (recall[i] - recall[i - 1]) * precision[i]
    return float(ap)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
    iou_threshold: float = 0.3,
    score_threshold: float = 0.01,
    ap_iou_thresholds: Tuple[float, ...] = (0.3, 0.5),
) -> Dict[str, float]:
    """Detection F1, AP@IoU, and binary segmentation IoU on a loader.

    Reports two families of detection metrics:

    * **F1 at a fixed score threshold** (``f1``, ``precision``, ``recall``)
      — the quick, thresholded view. Sensitive to calibration.
    * **Average Precision** (``ap_30``, ``ap_50`` by default) — PASCAL VOC
      all-point AP, integrates over the full precision-recall curve and
      is score-threshold-independent. This is the right number to
      compare across models and to the paper.
    """
    model.eval()
    tp = fp = fn = 0
    seg_ious: List[float] = []
    n_samples = 0
    total_preds_raw = 0
    total_preds_kept = 0
    max_score_seen = 0.0
    per_image_preds: List[Dict[str, torch.Tensor]] = []

    for images, targets in loader:
        images = [img.to(device) for img in images]
        out = model(images)
        detections = out["detections"]
        seg_logits = out["seg_logits"].cpu()

        for det, target, seg_logit in zip(detections, targets, seg_logits):
            pred_boxes_raw = det["boxes"].cpu()
            pred_scores = det["scores"].cpu()
            if pred_scores.numel() > 0:
                max_score_seen = max(max_score_seen, float(pred_scores.max().item()))
            keep = pred_scores > score_threshold
            pred_boxes = pred_boxes_raw[keep]

            total_preds_raw += pred_boxes_raw.shape[0]
            total_preds_kept += pred_boxes.shape[0]
            n_samples += 1

            gt_boxes = target["boxes"]

            tpi, fpi, fni = _match_predictions(
                pred_boxes, gt_boxes, iou_threshold=iou_threshold
            )
            tp += tpi
            fp += fpi
            fn += fni

            # AP uses all predictions — no score filtering.
            per_image_preds.append({
                "pred_boxes": pred_boxes_raw,
                "pred_scores": pred_scores,
                "gt_boxes": gt_boxes,
            })

            # Binary seg IoU at the logits' resolution (target mask is
            # resized to match so we compare apples to apples).
            seg_pred = (seg_logit.sigmoid() > 0.5).squeeze(0)
            gt_mask_full = (target["masks"] > 0).float().unsqueeze(0).unsqueeze(0)
            gt_mask_resized = torch.nn.functional.interpolate(
                gt_mask_full, size=seg_pred.shape, mode="nearest"
            ).squeeze()
            seg_ious.append(_binary_iou(seg_pred, gt_mask_resized))

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    mean_seg_iou = float(np.mean(seg_ious)) if seg_ious else 0.0

    ap_metrics = {}
    for iou_t in ap_iou_thresholds:
        key = f"ap_{int(round(iou_t * 100)):02d}"
        ap_metrics[key] = _average_precision(per_image_preds, iou_threshold=iou_t)

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "seg_iou": mean_seg_iou,
        "mean_preds_raw": total_preds_raw / max(n_samples, 1),
        "mean_preds_kept": total_preds_kept / max(n_samples, 1),
        "max_score": max_score_seen,
        **ap_metrics,
    }
