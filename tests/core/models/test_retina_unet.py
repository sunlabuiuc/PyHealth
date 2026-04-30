"""CPU-runnable unit tests for RetinaUNet.

All tests use tiny synthetic tensors (batch 2, 1 channel, 128x128) and
a resnet18 backbone so the full suite completes in seconds on CPU
without any training. They cover instantiation, forward shapes in both
modes, the joint-loss arithmetic, gradient flow, and the paper's λ
(``seg_weight``) knob.
"""

import pytest
import torch

from pyhealth.models.retina_unet import RetinaUNet, UNetDecoder


def _tiny_batch(n: int = 2, c: int = 1, h: int = 128, w: int = 128):
    imgs = [torch.randn(c, h, w) for _ in range(n)]
    targets = []
    for i in range(n):
        box = torch.tensor(
            [[10.0 + i, 20.0 + i, 60.0 + i, 70.0 + i]], dtype=torch.float32
        )
        labels = torch.tensor([1], dtype=torch.int64)
        mask = torch.zeros(h, w, dtype=torch.long)
        mask[20 + i : 70 + i, 10 + i : 60 + i] = 1
        targets.append({"boxes": box, "labels": labels, "masks": mask})
    return imgs, targets


def test_instantiates_with_defaults():
    model = RetinaUNet(min_size=128, max_size=128)
    assert model.seg_num_classes == 1
    assert model.in_channels == 1
    assert model.seg_weight == 1.0
    # has a detector (torchvision RetinaNet) and a seg decoder
    assert hasattr(model, "detector")
    assert isinstance(model.seg_decoder, UNetDecoder)


def test_eval_forward_shapes():
    model = RetinaUNet(min_size=128, max_size=128).eval()
    imgs, _ = _tiny_batch()
    with torch.no_grad():
        out = model(imgs)
    assert set(out.keys()) == {"detections", "seg_logits"}
    assert len(out["detections"]) == len(imgs)
    # seg logits at input resolution
    assert out["seg_logits"].shape == (2, 1, 128, 128)
    # each detection dict has the standard torchvision keys
    for det in out["detections"]:
        assert {"boxes", "scores", "labels"} <= set(det.keys())


def test_train_forward_returns_combined_loss():
    model = RetinaUNet(min_size=128, max_size=128).train()
    imgs, targets = _tiny_batch()
    losses = model(imgs, targets)

    assert {"classification", "bbox_regression", "loss_segmentation", "loss_total"} <= set(
        losses.keys()
    )
    # loss_total = det + seg_weight * seg
    expected = (
        losses["classification"]
        + losses["bbox_regression"]
        + model.seg_weight * losses["loss_segmentation"]
    )
    assert torch.allclose(losses["loss_total"], expected, atol=1e-6)


def test_train_forward_without_masks_omits_seg_loss():
    model = RetinaUNet(min_size=128, max_size=128).train()
    imgs, targets = _tiny_batch()
    # strip masks
    targets = [{k: v for k, v in t.items() if k != "masks"} for t in targets]

    losses = model(imgs, targets)
    assert "loss_segmentation" not in losses
    assert torch.allclose(
        losses["loss_total"],
        losses["classification"] + losses["bbox_regression"],
        atol=1e-6,
    )


def test_seg_pos_weight_changes_seg_loss():
    imgs, targets = _tiny_batch()

    torch.manual_seed(0)
    model_a = RetinaUNet(min_size=128, max_size=128, seg_pos_weight=None).train()
    torch.manual_seed(0)
    model_b = RetinaUNet(min_size=128, max_size=128, seg_pos_weight=50.0).train()

    with torch.no_grad():
        loss_a = model_a(imgs, targets)["loss_segmentation"].item()
        loss_b = model_b(imgs, targets)["loss_segmentation"].item()

    # Same model init + same inputs; only the positive-class weight
    # differs. pos_weight > 1 multiplies the positive-pixel contribution,
    # so the loss must differ.
    assert loss_a != pytest.approx(loss_b, rel=1e-4)
    # And pos_weight = 50 on tiny foregrounds should produce a larger loss
    # than the unweighted version.
    assert loss_b > loss_a


def test_seg_weight_zero_drops_seg_contribution():
    model = RetinaUNet(min_size=128, max_size=128, seg_weight=0.0).train()
    imgs, targets = _tiny_batch()
    losses = model(imgs, targets)
    # seg_loss still computed, but zero-weighted in the total
    assert "loss_segmentation" in losses
    assert torch.allclose(
        losses["loss_total"],
        losses["classification"] + losses["bbox_regression"],
        atol=1e-6,
    )


def test_backward_populates_grads_on_shared_backbone():
    model = RetinaUNet(min_size=128, max_size=128).train()
    imgs, targets = _tiny_batch()
    losses = model(imgs, targets)
    losses["loss_total"].backward()

    # Backbone params and seg decoder params should both have grads
    backbone_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.detector.backbone.parameters()
    )
    seg_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.seg_decoder.parameters()
    )
    assert backbone_has_grad, "backbone received no gradient — hook may have failed"
    assert seg_has_grad, "seg decoder received no gradient"


def test_train_without_targets_raises():
    model = RetinaUNet(min_size=128, max_size=128).train()
    imgs, _ = _tiny_batch()
    with pytest.raises(ValueError):
        model(imgs, None)


def test_multiclass_segmentation_head():
    model = RetinaUNet(
        seg_num_classes=3, min_size=128, max_size=128
    ).eval()
    imgs, _ = _tiny_batch()
    with torch.no_grad():
        out = model(imgs)
    assert out["seg_logits"].shape == (2, 3, 128, 128)


def test_unet_decoder_standalone_shapes():
    decoder = UNetDecoder(in_channels=256, num_classes=1, num_levels=5)
    # simulate FPN outputs at strides 8, 16, 32, 64, 128 for a 128x128 input
    feats = [
        torch.randn(2, 256, 16, 16),
        torch.randn(2, 256, 8, 8),
        torch.randn(2, 256, 4, 4),
        torch.randn(2, 256, 2, 2),
        torch.randn(2, 256, 1, 1),
    ]
    out = decoder(feats, target_size=(128, 128))
    assert out.shape == (2, 1, 128, 128)


def test_unet_decoder_wrong_level_count_raises():
    decoder = UNetDecoder(in_channels=256, num_classes=1, num_levels=5)
    with pytest.raises(ValueError):
        decoder([torch.randn(1, 256, 4, 4)], target_size=(32, 32))
