"""Retina U-Net: RetinaNet detector with a U-Net segmentation decoder.

Reimplementation of the architecture from Jaeger et al., *Retina U-Net:
Embarrassingly Simple Exploitation of Segmentation Supervision for
Medical Object Detection* (https://arxiv.org/abs/1811.08661).

The model shares a single ResNet-FPN backbone between two heads:

* a RetinaNet detection head (classification + bbox regression over
  anchor-based dense predictions, trained with focal loss + smooth L1);
* a U-Net-style pyramidal decoder that consumes the same FPN feature
  maps and produces a pixel-level segmentation logit map at input
  resolution, trained with per-pixel binary cross-entropy (or
  cross-entropy for multi-class).

During training the combined loss is::

    L = L_cls + L_bbox + seg_weight * L_seg

where ``seg_weight`` (λ in the paper) is the auxiliary supervision
strength. At inference time the detection branch produces the usual
RetinaNet outputs; the segmentation branch is returned alongside so
callers can visualize or post-process it.

Implementation leans on torchvision's tested RetinaNet (backbone, FPN,
anchor generator, heads, and compute_loss) so the novel work stays in
the seg decoder + joint-loss machinery. A forward hook caches the
backbone's FPN output so the detection and segmentation paths share a
single backbone pass.
"""

from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNet, RetinaNetHead
from torchvision.ops.feature_pyramid_network import LastLevelP6P7

from pyhealth.models.base_model import BaseModel


class UNetDecoder(nn.Module):
    """U-Net-style decoder over FPN features.

    Takes the list of FPN outputs (from highest resolution to lowest),
    walks from the coarsest level upward concatenating each finer-level
    skip connection, and returns a per-class logit map upsampled to the
    target spatial size.

    Args:
        in_channels: Channel count of every FPN output (256 by default
            for torchvision's FPN).
        num_classes: Output channels of the final 1x1 conv. Use 1 for
            binary foreground segmentation (sigmoid + BCE at loss
            time); use ``K`` for K-class segmentation with cross-entropy.
        num_levels: Number of FPN levels the decoder walks. Must match
            the length of the ``features`` list passed to ``forward``.
        mid_channels: Internal channel count for the conv blocks.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 1,
        num_levels: int = 5,
        mid_channels: int = 128,
    ) -> None:
        super().__init__()
        self.num_levels = num_levels

        self.blocks = nn.ModuleList()
        for i in range(num_levels):
            block_in = in_channels if i == 0 else mid_channels + in_channels
            self.blocks.append(
                nn.Sequential(
                    nn.Conv2d(block_in, mid_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=min(32, mid_channels), num_channels=mid_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
                    nn.GroupNorm(num_groups=min(32, mid_channels), num_channels=mid_channels),
                    nn.ReLU(inplace=True),
                )
            )
        self.final = nn.Conv2d(mid_channels, num_classes, kernel_size=1)

    def forward(
        self,
        features: List[torch.Tensor],
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        if len(features) != self.num_levels:
            raise ValueError(
                f"UNetDecoder expected {self.num_levels} FPN levels, got {len(features)}"
            )

        # Process from lowest-res (last) to highest-res (first).
        feats = list(features)[::-1]

        x = self.blocks[0](feats[0])
        for i in range(1, self.num_levels):
            skip = feats[i]
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = self.blocks[i](x)

        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        return self.final(x)


class RetinaUNet(BaseModel):
    """Retina U-Net: RetinaNet + U-Net segmentation decoder.

    Args:
        num_classes: Number of foreground classes **plus** the implicit
            background class (torchvision's convention). For a
            single-class lesion detector, pass 2.
        seg_num_classes: Number of segmentation output channels. Use 1
            for binary foreground segmentation (the paper's default);
            use ``K`` for K-class semantic segmentation.
        in_channels: Number of input image channels. Medical slices
            are typically single-channel (``1``); the default ResNet
            ``conv1`` is replaced to match this.
        backbone_name: Any name supported by
            :func:`torchvision.models.detection.backbone_utils.resnet_fpn_backbone`.
            ``"resnet18"`` keeps CPU unit tests fast; swap to
            ``"resnet50"`` for real training.
        pretrained_backbone: If True, load ImageNet-pretrained weights
            for the backbone. Requires network access at first use.
        min_size / max_size: Passed to torchvision's
            ``GeneralizedRCNNTransform``. Defaults target small CT-slice
            resolutions; adjust for full-size volumes.
        seg_weight: Multiplier for the segmentation loss in the combined
            training objective (λ in the paper). ``0`` disables
            segmentation supervision (reduces to vanilla RetinaNet); the
            proposal calls for ablating this knob.
        seg_pos_weight: Optional positive-class weight for the binary
            segmentation BCE loss. Use when foreground occupies a tiny
            fraction of pixels (e.g. lung nodules at <1% coverage)
            so the seg head can't collapse to an all-background
            solution. Pass ``None`` (default) to disable weighting.
        anchor_sizes / aspect_ratios: Anchor box configuration per FPN
            level. Defaults are tuned for small medical lesions.
    """

    def __init__(
        self,
        num_classes: int = 2,
        seg_num_classes: int = 1,
        in_channels: int = 1,
        backbone_name: str = "resnet18",
        pretrained_backbone: bool = False,
        min_size: int = 128,
        max_size: int = 256,
        seg_weight: float = 1.0,
        seg_pos_weight: Optional[float] = None,
        anchor_sizes: Optional[Tuple[Tuple[int, ...], ...]] = None,
        aspect_ratios: Optional[Tuple[Tuple[float, ...], ...]] = None,
    ) -> None:
        super().__init__()

        if anchor_sizes is None:
            anchor_sizes = ((8,), (16,), (32,), (64,), (128,))
        if aspect_ratios is None:
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
        if len(anchor_sizes) != len(aspect_ratios):
            raise ValueError("anchor_sizes and aspect_ratios must be the same length")

        backbone = resnet_fpn_backbone(
            backbone_name=backbone_name,
            weights="DEFAULT" if pretrained_backbone else None,
            trainable_layers=5,
            returned_layers=[2, 3, 4],
            extra_blocks=LastLevelP6P7(256, 256),
        )

        if in_channels != 3:
            old = backbone.body.conv1
            backbone.body.conv1 = nn.Conv2d(
                in_channels,
                old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None,
            )

        anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)
        head = RetinaNetHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
            num_classes=num_classes,
        )

        self.detector = RetinaNet(
            backbone=backbone,
            num_classes=num_classes,
            anchor_generator=anchor_generator,
            head=head,
            min_size=min_size,
            max_size=max_size,
            image_mean=[0.0] * in_channels,
            image_std=[1.0] * in_channels,
            # Disable the detector's internal score filter so our
            # evaluate() can use its own (configurable) threshold.
            # torchvision defaults to 0.05 which silently eats
            # under-trained RetinaNet outputs on small medical data.
            score_thresh=0.0,
            detections_per_img=300,
            topk_candidates=1000,
        )

        self.seg_decoder = UNetDecoder(
            in_channels=backbone.out_channels,
            num_classes=seg_num_classes,
            num_levels=len(anchor_sizes),
        )

        self.seg_weight = float(seg_weight)
        self.seg_pos_weight = float(seg_pos_weight) if seg_pos_weight is not None else None
        self.seg_num_classes = seg_num_classes
        self.in_channels = in_channels

        # Forward hook caches FPN output so det + seg share one backbone pass.
        self._fpn_cache: Optional["OrderedDict[str, torch.Tensor]"] = None
        self.detector.backbone.register_forward_hook(self._cache_fpn)

    def _cache_fpn(self, _module, _inputs, output) -> None:
        self._fpn_cache = output

    @staticmethod
    def _split_seg_targets(
        targets: Optional[List[Dict[str, torch.Tensor]]]
    ) -> Tuple[Optional[List[Dict[str, torch.Tensor]]], Optional[List[torch.Tensor]]]:
        """Peel per-sample ``masks`` tensors off of the targets list.

        torchvision's RetinaNet rejects unknown target keys, so we
        strip ``"masks"`` before handing the list to the detector.
        Masks are returned as a **list** (not a stacked tensor) because
        per-sample spatial sizes can differ in a batch; resizing to a
        common grid happens in :meth:`_seg_loss`.
        """
        if targets is None or "masks" not in targets[0]:
            return targets, None

        seg = [t["masks"] for t in targets]
        det_targets = [
            {k: v for k, v in t.items() if k != "masks"} for t in targets
        ]
        return det_targets, seg

    def _seg_loss(
        self, logits: torch.Tensor, seg_targets: List[torch.Tensor]
    ) -> torch.Tensor:
        # Resize each per-sample target to the logit grid (nearest so
        # integer labels stay integer), then stack and compute the loss.
        target_hw = logits.shape[-2:]
        resized = [
            F.interpolate(
                m.unsqueeze(0).unsqueeze(0).float(),
                size=target_hw,
                mode="nearest",
            ).squeeze(0).squeeze(0)
            for m in seg_targets
        ]
        stacked = torch.stack(resized, dim=0)

        if self.seg_num_classes == 1:
            fg = (stacked > 0).float().unsqueeze(1)
            pos_weight = None
            if self.seg_pos_weight is not None:
                pos_weight = torch.tensor([self.seg_pos_weight], device=logits.device)
            return F.binary_cross_entropy_with_logits(logits, fg, pos_weight=pos_weight)
        return F.cross_entropy(logits, stacked.long())

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, Any]:
        """Forward pass.

        In training mode returns a dict of losses including
        ``loss_total`` (the combined objective actually worth
        backpropagating). In eval mode returns ``{"detections", "seg_logits"}``.
        """
        if self.training and targets is None:
            raise ValueError("targets are required in training mode")

        det_targets, seg_targets = self._split_seg_targets(targets)

        # Running the detector also populates self._fpn_cache via the hook.
        self._fpn_cache = None
        detector_output = self.detector(images, det_targets)

        features = self._fpn_cache
        if features is None:
            raise RuntimeError("backbone forward hook did not fire")
        features_list = list(features.values())

        # Target size = finest FPN level shape * its stride (8 for returned_layers=[2,3,4]).
        finest = features_list[0]
        target_hw = (finest.shape[-2] * 8, finest.shape[-1] * 8)
        seg_logits = self.seg_decoder(features_list, target_hw)

        if self.training:
            losses = dict(detector_output)
            if seg_targets is not None:
                losses["loss_segmentation"] = self._seg_loss(seg_logits, seg_targets)

            det_sum = losses["classification"] + losses["bbox_regression"]
            if "loss_segmentation" in losses:
                losses["loss_total"] = det_sum + self.seg_weight * losses["loss_segmentation"]
            else:
                losses["loss_total"] = det_sum
            return losses

        return {"detections": detector_output, "seg_logits": seg_logits}
