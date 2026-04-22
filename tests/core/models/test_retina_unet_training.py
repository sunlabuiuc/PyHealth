"""CPU-runnable tests for the training helpers.

Uses tiny synthetic volumes (same fixture style as
``tests/datasets``) so the full suite remains fast.
"""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets.retina_unet_ct_dataset import RetinaUNetCTDataset
from pyhealth.models.retina_unet import RetinaUNet
from pyhealth.models.retina_unet_training import (
    RetinaUNetTorchDataset,
    _average_precision,
    _binary_iou,
    _match_predictions,
    collate_fn,
    evaluate,
    train_one_epoch,
)


def _tiny_ct_dataset(n_patients: int = 2, depth: int = 4, hw: int = 64, seed: int = 0):
    rng = np.random.default_rng(seed)
    volumes, masks = {}, {}
    for p in range(n_patients):
        vol = rng.standard_normal((depth, hw, hw)).astype(np.float32)
        mask = np.zeros((depth, hw, hw), dtype=np.int32)
        for z in range(depth):
            # put one 10x10 lesion on every slice so none get dropped
            y, x = 20 + p, 20 + z
            mask[z, y : y + 10, x : x + 10] = 1
        volumes[f"p{p}"] = vol
        masks[f"p{p}"] = mask
    return RetinaUNetCTDataset(volumes=volumes, masks=masks)


def test_adapter_emits_tensor_image_and_target_dict():
    ct = _tiny_ct_dataset()
    ds = RetinaUNetTorchDataset(ct)
    img, target = ds[0]

    assert img.dtype == torch.float32
    assert img.shape == (1, 64, 64)  # (C, H, W)
    assert set(target.keys()) == {"boxes", "labels", "masks"}
    assert target["boxes"].shape[1] == 4
    assert target["labels"].dtype == torch.int64
    assert target["masks"].shape == (64, 64)


def test_collate_fn_returns_lists():
    ct = _tiny_ct_dataset()
    ds = RetinaUNetTorchDataset(ct)
    batch = [ds[0], ds[1]]
    images, targets = collate_fn(batch)

    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2
    assert images[0].shape == (1, 64, 64)


def test_dataloader_roundtrip():
    ct = _tiny_ct_dataset()
    ds = RetinaUNetTorchDataset(ct)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
    batches = list(loader)

    assert len(batches) == len(ds) // 2
    images, targets = batches[0]
    assert len(images) == 2


def test_match_predictions_counts_tp_fp_fn():
    pred = torch.tensor([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 60.0, 60.0]])
    gt = torch.tensor([[10.0, 10.0, 20.0, 20.0]])  # one match, one FP
    tp, fp, fn = _match_predictions(pred, gt, iou_threshold=0.5)
    assert (tp, fp, fn) == (1, 1, 0)

    # no predictions, one GT -> 1 FN
    tp, fp, fn = _match_predictions(torch.zeros(0, 4), gt)
    assert (tp, fp, fn) == (0, 0, 1)

    # predictions but no GT -> all FP
    tp, fp, fn = _match_predictions(pred, torch.zeros(0, 4))
    assert (tp, fp, fn) == (0, 2, 0)

    # both empty -> all zeros
    assert _match_predictions(torch.zeros(0, 4), torch.zeros(0, 4)) == (0, 0, 0)


def test_binary_iou_edges():
    a = torch.zeros(4, 4)
    b = torch.zeros(4, 4)
    assert _binary_iou(a, b) == 1.0  # both empty -> perfect

    a[0:2, 0:2] = 1
    b[0:2, 0:2] = 1
    assert _binary_iou(a, b) == 1.0  # full overlap

    b.zero_()
    b[1:3, 1:3] = 1
    # intersection 1 pixel, union 7 -> 1/7
    assert _binary_iou(a, b) == pytest.approx(1 / 7)


def test_train_one_epoch_decreases_loss_on_toy_overfit():
    ct = _tiny_ct_dataset(n_patients=1, depth=2, hw=64)
    ds = RetinaUNetTorchDataset(ct)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    torch.manual_seed(0)
    model = RetinaUNet(min_size=64, max_size=64)
    opt = torch.optim.Adam(model.parameters(), lr=5e-4)
    device = torch.device("cpu")
    model.to(device)

    loss_0 = train_one_epoch(model, loader, opt, device)["loss_total"]
    # two more epochs to let it actually move
    for _ in range(2):
        last = train_one_epoch(model, loader, opt, device)["loss_total"]

    # Not a rigorous convergence test — just that the loop runs and
    # produces a finite, generally-decreasing loss on a tiny set.
    assert np.isfinite(loss_0)
    assert np.isfinite(last)
    assert last < loss_0 + 1e-3  # allow tiny noise


def test_average_precision_perfect_ranking():
    # Two GT boxes, two predictions matching them exactly, scored in
    # any order → AP must be 1.0 because every recall level hits P=1.
    samples = [{
        "pred_boxes": torch.tensor([[10., 10., 20., 20.], [50., 50., 60., 60.]]),
        "pred_scores": torch.tensor([0.9, 0.8]),
        "gt_boxes": torch.tensor([[10., 10., 20., 20.], [50., 50., 60., 60.]]),
    }]
    assert _average_precision(samples, iou_threshold=0.5) == pytest.approx(1.0)


def test_average_precision_no_predictions():
    samples = [{
        "pred_boxes": torch.zeros(0, 4),
        "pred_scores": torch.zeros(0),
        "gt_boxes": torch.tensor([[10., 10., 20., 20.]]),
    }]
    assert _average_precision(samples, iou_threshold=0.5) == 0.0


def test_average_precision_no_gt_returns_zero():
    samples = [{
        "pred_boxes": torch.tensor([[10., 10., 20., 20.]]),
        "pred_scores": torch.tensor([0.9]),
        "gt_boxes": torch.zeros(0, 4),
    }]
    # No GTs anywhere → AP undefined; we return 0 as a conservative default.
    assert _average_precision(samples, iou_threshold=0.5) == 0.0


def test_average_precision_non_overlapping_all_fp():
    samples = [{
        "pred_boxes": torch.tensor([[100., 100., 110., 110.]]),
        "pred_scores": torch.tensor([0.9]),
        "gt_boxes": torch.tensor([[10., 10., 20., 20.]]),
    }]
    assert _average_precision(samples, iou_threshold=0.5) == 0.0


def test_average_precision_low_score_tp_still_counts():
    # 9 high-score FPs ahead of one low-score TP → precision at recall 1
    # is 1/10 = 0.1 exactly (all-point AP).
    fp_boxes = torch.tensor(
        [[100. + i * 20, 100., 110. + i * 20, 110.] for i in range(9)]
    )
    samples = [{
        "pred_boxes": torch.cat([fp_boxes, torch.tensor([[10., 10., 20., 20.]])]),
        "pred_scores": torch.tensor([0.9] * 9 + [0.1]),
        "gt_boxes": torch.tensor([[10., 10., 20., 20.]]),
    }]
    assert _average_precision(samples, iou_threshold=0.5) == pytest.approx(0.1, abs=1e-6)


def test_average_precision_monotonic_in_ranking_quality():
    # Same predictions, same GT — but good prediction ranked first vs last.
    gt = torch.tensor([[10., 10., 20., 20.]])
    preds = torch.cat(
        [torch.tensor([[10., 10., 20., 20.]]),
         torch.tensor([[100., 100., 110., 110.]])]
    )

    good_first = [{
        "pred_boxes": preds,
        "pred_scores": torch.tensor([0.9, 0.1]),
        "gt_boxes": gt,
    }]
    good_last = [{
        "pred_boxes": preds,
        "pred_scores": torch.tensor([0.1, 0.9]),
        "gt_boxes": gt,
    }]
    ap_good = _average_precision(good_first, iou_threshold=0.5)
    ap_bad = _average_precision(good_last, iou_threshold=0.5)
    assert ap_good > ap_bad
    assert ap_good == pytest.approx(1.0)


def test_evaluate_reports_ap_keys():
    ct = _tiny_ct_dataset()
    ds = RetinaUNetTorchDataset(ct)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    from pyhealth.models.retina_unet import RetinaUNet
    model = RetinaUNet(min_size=64, max_size=64)
    metrics = evaluate(model, loader, torch.device("cpu"))

    assert "ap_30" in metrics
    assert "ap_50" in metrics
    assert 0.0 <= metrics["ap_30"] <= 1.0
    assert 0.0 <= metrics["ap_50"] <= 1.0


def test_evaluate_returns_full_metric_dict():
    ct = _tiny_ct_dataset()
    ds = RetinaUNetTorchDataset(ct)
    loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

    model = RetinaUNet(min_size=64, max_size=64)
    metrics = evaluate(model, loader, torch.device("cpu"))

    assert {
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "seg_iou",
        "mean_preds_raw",
        "mean_preds_kept",
    } <= set(metrics.keys())
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0
    assert 0.0 <= metrics["seg_iou"] <= 1.0
