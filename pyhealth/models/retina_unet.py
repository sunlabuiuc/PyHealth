"""RetinaUNet model for PyHealth.

Contributor: Tuan Nguyen
NetID: tuanmn2
Paper: Retina U-Net: Embarrassingly Simple Exploitation of Segmentation
    Supervision for Medical Object Detection
Paper link: https://proceedings.mlr.press/v116/jaeger20a/jaeger20a.pdf
Description: Retina U-Net style medical object detection model with an
    auxiliary segmentation branch for PyHealth.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from pyhealth.datasets import SampleDataset
from pyhealth.models.base_model import BaseModel


def _box_area(boxes: torch.Tensor) -> torch.Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def _box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-6)


def _nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,), dtype=torch.long)

    order = scores.argsort(descending=True)
    keep: List[torch.Tensor] = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = _box_iou(boxes[current].unsqueeze(0), boxes[remaining]).squeeze(0)
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep)


def _weighted_box_clustering(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
    expected_num_predictions: float = 1.0,
    min_score: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    if boxes.numel() == 0:
        return boxes.new_zeros((0,)), boxes.new_zeros((0, 4))

    order = scores.argsort(descending=True)
    areas = _box_area(boxes).clamp(min=1e-6)
    keep_scores: List[torch.Tensor] = []
    keep_boxes: List[torch.Tensor] = []

    while order.numel() > 0:
        current = order[0]
        cluster_ious = _box_iou(boxes[current].unsqueeze(0), boxes[order]).squeeze(0)
        matches = cluster_ious > iou_threshold
        cluster_indices = order[matches]

        cluster_boxes = boxes[cluster_indices]
        cluster_scores = scores[cluster_indices]
        cluster_areas = areas[cluster_indices]
        cluster_overlap = cluster_ious[matches].clamp(min=1e-6)

        score_weights = cluster_overlap * cluster_areas
        weighted_scores = cluster_scores * score_weights
        expected = max(float(expected_num_predictions), 1.0)
        missing = max(0.0, expected - float(cluster_indices.numel()))
        mean_weight = score_weights.mean()
        denom = score_weights.sum() + missing * mean_weight
        if float(denom) <= 0.0:
            avg_score = cluster_scores.mean()
        else:
            avg_score = weighted_scores.sum() / denom

        coord_denom = weighted_scores.sum().clamp(min=1e-6)
        avg_box = (cluster_boxes * weighted_scores[:, None]).sum(dim=0) / coord_denom

        if float(avg_score) > min_score:
            keep_scores.append(avg_score)
            keep_boxes.append(avg_box)

        order = order[~matches]

    if not keep_scores:
        return boxes.new_zeros((0,)), boxes.new_zeros((0, 4))
    return torch.stack(keep_scores), torch.stack(keep_boxes)


def _encode_boxes(anchors: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    anchor_wh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-6)
    anchor_ctr = anchors[:, :2] + 0.5 * anchor_wh

    box_wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
    box_ctr = boxes[:, :2] + 0.5 * box_wh

    delta_ctr = (box_ctr - anchor_ctr) / anchor_wh
    delta_wh = torch.log(box_wh / anchor_wh)
    return torch.cat([delta_ctr, delta_wh], dim=1)


def _decode_boxes(anchors: torch.Tensor, deltas: torch.Tensor) -> torch.Tensor:
    anchor_wh = (anchors[:, 2:] - anchors[:, :2]).clamp(min=1e-6)
    anchor_ctr = anchors[:, :2] + 0.5 * anchor_wh

    pred_ctr = deltas[:, :2] * anchor_wh + anchor_ctr
    pred_wh = deltas[:, 2:].exp() * anchor_wh

    top_left = pred_ctr - 0.5 * pred_wh
    bottom_right = pred_ctr + 0.5 * pred_wh
    return torch.cat([top_left, bottom_right], dim=1)


def _clip_boxes(boxes: torch.Tensor, height: int, width: int) -> torch.Tensor:
    boxes = boxes.clone()
    boxes[:, 0] = boxes[:, 0].clamp(min=0, max=width)
    boxes[:, 1] = boxes[:, 1].clamp(min=0, max=height)
    boxes[:, 2] = boxes[:, 2].clamp(min=0, max=width)
    boxes[:, 3] = boxes[:, 3].clamp(min=0, max=height)
    return boxes


def _multiclass_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(
        target.long().clamp(min=0, max=num_classes - 1),
        num_classes=num_classes,
    ).permute(0, 3, 1, 2).to(dtype=probs.dtype)

    class_losses: List[torch.Tensor] = []
    for class_index in range(1, num_classes):
        class_probs = probs[:, class_index]
        class_target = one_hot[:, class_index]
        intersection = (class_probs * class_target).sum(dim=(-2, -1))
        denom = class_probs.sum(dim=(-2, -1)) + class_target.sum(dim=(-2, -1))
        dice = (2 * intersection + 1e-6) / (denom + 1e-6)
        class_losses.append(1 - dice.mean())

    if not class_losses:
        return logits.new_zeros(())
    return torch.stack(class_losses).mean()


def _multiclass_cross_entropy_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    target = target.long().clamp(min=0, max=logits.shape[1] - 1)
    gathered = torch.gather(log_probs, dim=1, index=target.unsqueeze(1)).squeeze(1)
    return -gathered.mean()


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class RetinaHead(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        layers: List[nn.Module] = []
        current = in_channels
        for _ in range(4):
            layers.extend(
                [
                    nn.Conv2d(current, hidden_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                ]
            )
            current = hidden_channels
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1))
        self.head = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


class RetinaUNet(BaseModel):
    """Detection-oriented Retina U-Net with auxiliary segmentation supervision.

    This follows the core idea of the original Retina U-Net paper and the
    MIC-DKFZ reference implementation:

    - multi-scale detection heads on coarse pyramid levels
    - a U-FPN decoder reaching full image resolution
    - segmentation loss used as auxiliary supervision

    The current implementation is 2D-only because the LIDC preprocessing in
    this project produces 2.5D slice stacks for 2D detectors.
    """

    def __init__(
        self,
        dataset: SampleDataset,
        in_channels: int = 1,
        num_classes: int = 1,
        base_channels: int = 32,
        anchor_sizes: Sequence[float] = (8.0, 16.0, 32.0, 64.0),
        anchor_scales: Sequence[float] = (1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
        aspect_ratios: Sequence[float] = (0.5, 1.0, 2.0),
        positive_iou_threshold: float = 0.5,
        negative_iou_threshold: float = 0.1,
        negative_to_positive_ratio: float = 1.0,
        seg_loss_weight: float = 1.0,
        bbox_loss_weight: float = 1.0,
        cls_loss_weight: float = 1.0,
        score_threshold: float = 0.1,
        nms_threshold: float = 1e-5,
        postprocess_method: str = "wbc",
        max_detections: int = 100,
    ):
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("RetinaUNet supports exactly one image-like feature key.")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if positive_iou_threshold <= negative_iou_threshold:
            raise ValueError("positive_iou_threshold must be greater than negative_iou_threshold.")
        if postprocess_method not in {"nms", "wbc"}:
            raise ValueError("postprocess_method must be one of {'nms', 'wbc'}.")

        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0] if self.label_keys else "label"
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_head_classes = num_classes + 1
        self.base_channels = base_channels
        self.anchor_sizes = tuple(anchor_sizes)
        self.anchor_scales = tuple(anchor_scales)
        self.aspect_ratios = tuple(aspect_ratios)
        self.positive_iou_threshold = positive_iou_threshold
        self.negative_iou_threshold = negative_iou_threshold
        self.negative_to_positive_ratio = negative_to_positive_ratio
        self.seg_loss_weight = seg_loss_weight
        self.bbox_loss_weight = bbox_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.postprocess_method = postprocess_method
        self.max_detections = max_detections
        self.num_anchors = len(self.anchor_scales) * len(self.aspect_ratios)
        self.pyramid_strides = (4, 8, 16, 32)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8
        c5 = base_channels * 16
        c6 = base_channels * 16

        self.stem = ConvBlock(in_channels, c1)
        self.enc1 = DownBlock(c1, c2)
        self.enc2 = DownBlock(c2, c3)
        self.enc3 = DownBlock(c3, c4)
        self.enc4 = DownBlock(c4, c5)
        self.bottleneck = DownBlock(c5, c6)

        self.lat5 = nn.Conv2d(c6, c5, kernel_size=1)
        self.lat4 = nn.Conv2d(c5, c5, kernel_size=1)
        self.lat3 = nn.Conv2d(c4, c5, kernel_size=1)
        self.lat2 = nn.Conv2d(c3, c5, kernel_size=1)
        self.lat1 = nn.Conv2d(c2, c5, kernel_size=1)
        self.lat0 = nn.Conv2d(c1, c5, kernel_size=1)

        self.out5 = nn.Conv2d(c5, c5, kernel_size=3, padding=1)
        self.out4 = nn.Conv2d(c5, c5, kernel_size=3, padding=1)
        self.out3 = nn.Conv2d(c5, c5, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(c5, c5, kernel_size=3, padding=1)
        self.out0 = nn.Conv2d(c5, c5, kernel_size=3, padding=1)

        self.cls_head = RetinaHead(
            in_channels=c5,
            hidden_channels=c5,
            out_channels=self.num_anchors * self.num_head_classes,
        )
        self.box_head = RetinaHead(
            in_channels=c5,
            hidden_channels=c5,
            out_channels=self.num_anchors * 4,
        )
        self.seg_head = nn.Conv2d(c5, self.num_head_classes, kernel_size=1)

    @staticmethod
    def _to_nchw(x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected 2D/3D/4D tensor, got shape {tuple(x.shape)}.")
        if x.dim() == 4 and x.shape[1] not in {1, 3} and x.shape[-1] in {1, 3}:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

    def _align_channels(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.in_channels:
            return x
        if self.in_channels == 1:
            return x.mean(dim=1, keepdim=True)
        if x.shape[1] == 1 and self.in_channels > 1:
            return x.repeat(1, self.in_channels, 1, 1)
        if x.shape[1] > self.in_channels:
            return x[:, : self.in_channels]
        repeats = (self.in_channels + x.shape[1] - 1) // x.shape[1]
        x = x.repeat(1, repeats, 1, 1)
        return x[:, : self.in_channels]

    @staticmethod
    def _resize_like(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def _build_pyramid(
        self, x: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        c0 = self.stem(x)
        c1 = self.enc1(c0)
        c2 = self.enc2(c1)
        c3 = self.enc3(c2)
        c4 = self.enc4(c3)
        c5 = self.bottleneck(c4)

        p5_pre = self.lat5(c5)
        p4_pre = self.lat4(c4) + F.interpolate(
            p5_pre, size=c4.shape[-2:], mode="bilinear", align_corners=False
        )
        p3_pre = self.lat3(c3) + F.interpolate(
            p4_pre, size=c3.shape[-2:], mode="bilinear", align_corners=False
        )
        p2_pre = self.lat2(c2) + F.interpolate(
            p3_pre, size=c2.shape[-2:], mode="bilinear", align_corners=False
        )
        p1_pre = self.lat1(c1) + F.interpolate(
            p2_pre, size=c1.shape[-2:], mode="bilinear", align_corners=False
        )
        p0_pre = self.lat0(c0) + F.interpolate(
            p1_pre, size=c0.shape[-2:], mode="bilinear", align_corners=False
        )

        p5 = self.out5(p5_pre)
        p4 = self.out4(p4_pre)
        p3 = self.out3(p3_pre)
        p2 = self.out2(p2_pre)
        p0 = self.out0(p0_pre)
        seg_logit = self.seg_head(p0)
        embed = F.adaptive_avg_pool2d(c5, output_size=(1, 1)).flatten(1)
        return [p2, p3, p4, p5], seg_logit, embed

    def _reshape_cls_output(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = x.view(b, self.num_anchors, self.num_head_classes, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.view(b, -1, self.num_head_classes)

    def _reshape_box_output(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = x.view(b, self.num_anchors, 4, h, w)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x.view(b, -1, 4)

    def _generate_level_anchors(
        self,
        feature_shape: tuple[int, int],
        stride: int,
        base_size: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        height, width = feature_shape
        shifts_y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * stride
        shifts_x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * stride
        grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
        centers = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

        anchor_shapes = []
        for scale in self.anchor_scales:
            scaled = base_size * scale
            for ratio in self.aspect_ratios:
                width_val = scaled * math.sqrt(1.0 / ratio)
                height_val = scaled * math.sqrt(ratio)
                anchor_shapes.append([width_val, height_val])

        wh = torch.tensor(anchor_shapes, device=device, dtype=dtype)
        centers = centers[:, None, :].expand(-1, wh.shape[0], -1)
        wh = wh[None, :, :].expand(centers.shape[0], -1, -1)

        top_left = centers - 0.5 * wh
        bottom_right = centers + 0.5 * wh
        return torch.cat([top_left, bottom_right], dim=-1).reshape(-1, 4)

    def _generate_anchors(
        self, features: Sequence[torch.Tensor], image_shape: tuple[int, int]
    ) -> torch.Tensor:
        _ = image_shape
        anchors = [
            self._generate_level_anchors(
                feature_shape=(feature.shape[-2], feature.shape[-1]),
                stride=stride,
                base_size=base_size,
                device=feature.device,
                dtype=feature.dtype,
            )
            for feature, stride, base_size in zip(features, self.pyramid_strides, self.anchor_sizes)
        ]
        return torch.cat(anchors, dim=0)

    def _normalize_box_targets(
        self,
        boxes: Sequence[torch.Tensor] | torch.Tensor | None,
        labels: Sequence[torch.Tensor] | torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if boxes is None:
            empty_boxes = [torch.zeros((0, 4), device=device) for _ in range(batch_size)]
            empty_labels = [torch.zeros((0,), dtype=torch.long, device=device) for _ in range(batch_size)]
            return empty_boxes, empty_labels

        if isinstance(boxes, torch.Tensor):
            box_list = [boxes[i].to(device=device, dtype=torch.float32) for i in range(boxes.shape[0])]
        else:
            box_list = [box.to(device=device, dtype=torch.float32) for box in boxes]

        if labels is None:
            label_list = [
                torch.ones((box.shape[0],), dtype=torch.long, device=device) for box in box_list
            ]
        elif isinstance(labels, torch.Tensor):
            label_list = [labels[i].to(device=device, dtype=torch.long) for i in range(labels.shape[0])]
        else:
            label_list = [label.to(device=device, dtype=torch.long) for label in labels]

        if len(box_list) != batch_size or len(label_list) != batch_size:
            raise ValueError("boxes and labels must provide one entry per batch element.")
        return box_list, label_list

    def _match_anchors(
        self, anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cls_targets = torch.full(
            (anchors.shape[0],), -1, dtype=torch.long, device=anchors.device
        )
        box_targets = torch.zeros((anchors.shape[0], 4), dtype=anchors.dtype, device=anchors.device)

        if gt_boxes.numel() == 0:
            cls_targets.fill_(0)
            return cls_targets, box_targets

        ious = _box_iou(anchors, gt_boxes)
        max_iou, matched_gt = ious.max(dim=1)

        cls_targets[max_iou < self.negative_iou_threshold] = 0
        positive = max_iou >= self.positive_iou_threshold
        cls_targets[positive] = gt_labels[matched_gt[positive]]
        box_targets[positive] = _encode_boxes(anchors[positive], gt_boxes[matched_gt[positive]])

        best_anchor_per_gt = ious.argmax(dim=0)
        gt_indices = torch.arange(gt_boxes.shape[0], device=anchors.device)
        cls_targets[best_anchor_per_gt] = gt_labels[gt_indices]
        box_targets[best_anchor_per_gt] = _encode_boxes(
            anchors[best_anchor_per_gt], gt_boxes[gt_indices]
        )
        return cls_targets, box_targets

    def _compute_cls_loss(
        self, cls_logits: torch.Tensor, cls_targets: torch.Tensor
    ) -> tuple[torch.Tensor, int, int]:
        pos_mask = cls_targets > 0
        neg_mask = cls_targets == 0

        zero = cls_logits.new_zeros(())
        pos_loss = zero
        neg_loss = zero

        pos_count = int(pos_mask.sum().item())
        neg_count = int(neg_mask.sum().item())

        if pos_count > 0:
            pos_loss = F.cross_entropy(cls_logits[pos_mask], cls_targets[pos_mask])

        if neg_count > 0:
            neg_losses = F.cross_entropy(
                cls_logits[neg_mask],
                cls_targets[neg_mask],
                reduction="none",
            )
            keep_neg = max(1, int(max(pos_count, 1) * self.negative_to_positive_ratio))
            keep_neg = min(keep_neg, neg_losses.shape[0])
            neg_loss = neg_losses.topk(keep_neg).values.mean()

        return (pos_loss + neg_loss) / 2, pos_count, neg_count

    def _compute_training_losses(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        anchors: torch.Tensor,
        boxes: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
        seg_logit: torch.Tensor,
        seg_target: torch.Tensor | None,
    ) -> Dict[str, torch.Tensor]:
        cls_loss = cls_logits.new_zeros(())
        bbox_loss = cls_logits.new_zeros(())
        total_pos = 0
        total_neg = 0

        for image_ix in range(cls_logits.shape[0]):
            cls_targets, box_targets = self._match_anchors(
                anchors=anchors,
                gt_boxes=boxes[image_ix],
                gt_labels=labels[image_ix],
            )
            image_cls_loss, pos_count, neg_count = self._compute_cls_loss(
                cls_logits=cls_logits[image_ix], cls_targets=cls_targets
            )
            cls_loss = cls_loss + image_cls_loss
            total_pos += pos_count
            total_neg += neg_count

            pos_mask = cls_targets > 0
            if pos_mask.any():
                bbox_loss = bbox_loss + F.smooth_l1_loss(
                    box_deltas[image_ix][pos_mask],
                    box_targets[pos_mask],
                )

        cls_loss = cls_loss / max(cls_logits.shape[0], 1)
        bbox_loss = bbox_loss / max(cls_logits.shape[0], 1)

        if seg_target is None:
            seg_loss = cls_logits.new_zeros(())
            seg_ce = cls_logits.new_zeros(())
            seg_dice = cls_logits.new_zeros(())
        else:
            if not isinstance(seg_target, torch.Tensor):
                seg_target = torch.as_tensor(seg_target)
            seg_target = seg_target.to(self.device, dtype=torch.long)
            if seg_target.dim() == 4 and seg_target.shape[1] == 1:
                seg_target = seg_target[:, 0]
            elif seg_target.dim() == 4 and seg_target.shape[1] > 1:
                seg_target = seg_target.argmax(dim=1)
            if seg_target.dim() == 2:
                seg_target = seg_target.unsqueeze(0)
            if seg_target.shape[-2:] != seg_logit.shape[-2:]:
                seg_target = F.interpolate(
                    seg_target.unsqueeze(1).to(dtype=torch.float32),
                    size=seg_logit.shape[-2:],
                    mode="nearest",
                ).squeeze(1).to(dtype=torch.long)
            seg_ce = _multiclass_cross_entropy_loss(seg_logit, seg_target)
            seg_dice = _multiclass_dice_loss(
                seg_logit,
                seg_target,
                num_classes=self.num_head_classes,
            )
            seg_loss = 0.5 * (seg_ce + seg_dice)

        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.bbox_loss_weight * bbox_loss
            + self.seg_loss_weight * seg_loss
        )
        return {
            "loss": total_loss,
            "cls_loss": cls_loss,
            "bbox_loss": bbox_loss,
            "seg_loss": seg_loss,
            "seg_ce_loss": seg_ce,
            "seg_bce_loss": seg_ce,
            "seg_dice_loss": seg_dice,
            "positive_anchors": cls_logits.new_tensor(float(total_pos)),
            "negative_anchors": cls_logits.new_tensor(float(total_neg)),
        }

    def _decode_detections(
        self,
        cls_logits: torch.Tensor,
        box_deltas: torch.Tensor,
        anchors: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> list[dict[str, torch.Tensor]]:
        probs = F.softmax(cls_logits, dim=-1)
        height, width = image_shape
        detections: list[dict[str, torch.Tensor]] = []

        for image_ix in range(cls_logits.shape[0]):
            class_scores = probs[image_ix][:, 1:]
            if class_scores.numel() == 0:
                detections.append(
                    {
                        "boxes": anchors.new_zeros((0, 4)),
                        "scores": anchors.new_zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.long, device=anchors.device),
                    }
                )
                continue

            scores, labels = class_scores.max(dim=1)
            labels = labels + 1
            keep = scores > self.score_threshold
            if keep.sum() == 0:
                detections.append(
                    {
                        "boxes": anchors.new_zeros((0, 4)),
                        "scores": anchors.new_zeros((0,)),
                        "labels": torch.zeros((0,), dtype=torch.long, device=anchors.device),
                    }
                )
                continue

            decoded = _decode_boxes(anchors[keep], box_deltas[image_ix][keep])
            decoded = _clip_boxes(decoded, height=height, width=width)
            kept_scores = scores[keep]
            kept_labels = labels[keep]

            detections.append(
                self.merge_detections(
                    detections=[
                        {
                            "boxes": decoded,
                            "scores": kept_scores,
                            "labels": kept_labels,
                        }
                    ],
                    expected_num_predictions=1.0,
                )
            )

        return detections

    def merge_detections(
        self,
        detections: Sequence[dict[str, torch.Tensor]],
        *,
        expected_num_predictions: float = 1.0,
    ) -> dict[str, torch.Tensor]:
        if not detections:
            device = self.device if isinstance(self.device, torch.device) else torch.device(self.device)
            return {
                "boxes": torch.zeros((0, 4), dtype=torch.float32, device=device),
                "scores": torch.zeros((0,), dtype=torch.float32, device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
            }

        boxes_chunks = [item["boxes"] for item in detections if item["boxes"].numel() > 0]
        if not boxes_chunks:
            reference = detections[0]["boxes"]
            return {
                "boxes": reference.new_zeros((0, 4)),
                "scores": reference.new_zeros((0,)),
                "labels": torch.zeros((0,), dtype=torch.long, device=reference.device),
            }

        boxes = torch.cat(boxes_chunks, dim=0)
        scores = torch.cat([item["scores"] for item in detections if item["boxes"].numel() > 0], dim=0)
        labels = torch.cat([item["labels"] for item in detections if item["boxes"].numel() > 0], dim=0)

        final_boxes: List[torch.Tensor] = []
        final_scores: List[torch.Tensor] = []
        final_labels: List[torch.Tensor] = []
        for class_id in torch.unique(labels):
            class_mask = labels == class_id
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            if self.postprocess_method == "wbc":
                merged_scores, merged_boxes = _weighted_box_clustering(
                    class_boxes,
                    class_scores,
                    iou_threshold=self.nms_threshold,
                    expected_num_predictions=expected_num_predictions,
                )
                class_labels = torch.full(
                    (merged_scores.shape[0],),
                    int(class_id.item()),
                    dtype=torch.long,
                    device=labels.device,
                )
                final_boxes.append(merged_boxes)
                final_scores.append(merged_scores)
                final_labels.append(class_labels)
            else:
                class_keep = _nms(
                    class_boxes,
                    class_scores,
                    iou_threshold=self.nms_threshold,
                )
                final_boxes.append(class_boxes[class_keep])
                final_scores.append(class_scores[class_keep])
                final_labels.append(labels[class_mask][class_keep])

        if final_boxes:
            boxes_out = torch.cat(final_boxes, dim=0)
            scores_out = torch.cat(final_scores, dim=0)
            labels_out = torch.cat(final_labels, dim=0)
            keep = scores_out > self.score_threshold
            boxes_out = boxes_out[keep]
            scores_out = scores_out[keep]
            labels_out = labels_out[keep]
            if scores_out.numel() > 0:
                order = scores_out.argsort(descending=True)[: self.max_detections]
                boxes_out = boxes_out[order]
                scores_out = scores_out[order]
                labels_out = labels_out[order]
            else:
                boxes_out = boxes.new_zeros((0, 4))
                scores_out = scores.new_zeros((0,))
                labels_out = torch.zeros((0,), dtype=torch.long, device=labels.device)
        else:
            boxes_out = boxes.new_zeros((0, 4))
            scores_out = scores.new_zeros((0,))
            labels_out = torch.zeros((0,), dtype=torch.long, device=labels.device)

        return {
            "boxes": boxes_out,
            "scores": scores_out,
            "labels": labels_out,
        }

    def forward(self, **kwargs) -> Dict[str, Any]:
        return_detections = bool(kwargs.pop("return_detections", True))
        return_seg_logit = bool(kwargs.pop("return_seg_logit", True))
        return_raw_outputs = bool(kwargs.pop("return_raw_outputs", True))
        x = kwargs[self.feature_key]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x)
        x = x.to(self.device, dtype=torch.float32)
        x = self._to_nchw(x)
        x = self._align_channels(x)

        pyramid_features, seg_logit, embed = self._build_pyramid(x)
        cls_logits = torch.cat(
            [self._reshape_cls_output(self.cls_head(feature)) for feature in pyramid_features],
            dim=1,
        )
        box_deltas = torch.cat(
            [self._reshape_box_output(self.box_head(feature)) for feature in pyramid_features],
            dim=1,
        )
        anchors = self._generate_anchors(
            features=pyramid_features,
            image_shape=(x.shape[-2], x.shape[-1]),
        )

        if kwargs.get("embed", False):
            return {"embed": embed}

        results: Dict[str, Any] = {}
        if return_raw_outputs:
            results.update(
                {
                    "logit": cls_logits,
                    "y_prob": F.softmax(cls_logits, dim=-1),
                    "cls_logits": cls_logits,
                    "bbox_deltas": box_deltas,
                    "anchors": anchors,
                }
            )
        if return_seg_logit:
            results["seg_logit"] = seg_logit
        if return_detections:
            results["detections"] = self._decode_detections(
                cls_logits=cls_logits,
                box_deltas=box_deltas,
                anchors=anchors,
                image_shape=(x.shape[-2], x.shape[-1]),
            )

        if "boxes" not in kwargs and "labels" not in kwargs and "seg_target" not in kwargs:
            return results

        boxes, labels = self._normalize_box_targets(
            boxes=kwargs.get("boxes"),
            labels=kwargs.get("labels"),
            batch_size=x.shape[0],
            device=self.device,
        )
        losses = self._compute_training_losses(
            cls_logits=cls_logits,
            box_deltas=box_deltas,
            anchors=anchors,
            boxes=boxes,
            labels=labels,
            seg_logit=seg_logit,
            seg_target=kwargs.get("seg_target"),
        )
        results.update(losses)
        return results
