"""UNet3Plus tests on tiny synthetic tensors."""

import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import UNet3Plus, segmentation_iou
from pyhealth.models.unet3plus import dice_loss

LENGTH = 50  # not a power of two, to exercise the resizing paths


def _make_dataset(in_channels: int = 1, n_masks: int = 3):
    rng = np.random.default_rng(0)
    samples = []
    for i in range(4):
        mask = np.zeros((n_masks, LENGTH), dtype=np.float32)
        mask[i % n_masks, 10:20] = 1.0
        samples.append(
            {
                "patient_id": f"p{i}",
                "signal": rng.standard_normal((in_channels, LENGTH)).astype(np.float32),
                "mask": mask,
            }
        )
    return create_sample_dataset(
        samples, {"signal": "tensor"}, {"mask": "tensor"}, dataset_name="toy"
    )


class TestUNet3Plus(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _make_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=4)))

    def _model(self, **kwargs) -> UNet3Plus:
        params = {"base_channels": 4, "depth": 3, "kernel_size": 3}
        params.update(kwargs)
        return UNet3Plus(self.dataset, **params)

    def test_infers_channels(self):
        model = self._model()
        self.assertEqual((model.in_channels, model.n_masks), (1, 3))

    def test_forward_shapes_and_loss(self):
        out = self._model()(**self.batch)
        self.assertEqual(tuple(out["logit"].shape), (4, 3, LENGTH))
        self.assertEqual(out["y_prob"].shape, out["y_true"].shape)
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        self.assertEqual(out["loss"].dim(), 0)

    def test_backward_reaches_every_parameter(self):
        for deep_supervision in (True, False):
            with self.subTest(deep_supervision=deep_supervision):
                model = self._model(deep_supervision=deep_supervision)
                model(**self.batch)["loss"].backward()
                missing = [
                    name
                    for name, param in model.named_parameters()
                    # _dummy_param is BaseModel's device probe.
                    if param.grad is None and name != "_dummy_param"
                ]
                self.assertEqual(missing, [])

    def test_forward_without_target(self):
        out = self._model()(signal=self.batch["signal"])
        self.assertNotIn("loss", out)
        self.assertEqual(tuple(out["y_prob"].shape), (4, 3, LENGTH))

    def test_variants(self):
        for kwargs in (
            {"deep_supervision": False},
            {"depth": 4},
            {"encoder_convs": 1},
            {"bce_weight": 1.0, "dice_weight": 0.0},
        ):
            with self.subTest(**kwargs):
                out = self._model(**kwargs)(**self.batch)
                self.assertEqual(tuple(out["logit"].shape), (4, 3, LENGTH))
                self.assertTrue(torch.isfinite(out["loss"]))

    def test_multichannel_input(self):
        dataset = _make_dataset(in_channels=2, n_masks=2)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        model = UNet3Plus(dataset, base_channels=4, depth=3, kernel_size=3)
        self.assertEqual(tuple(model(**batch)["logit"].shape), (4, 2, LENGTH))

    def test_invalid_arguments(self):
        for kwargs in (
            {"depth": 1},
            {"kernel_size": 4},
            {"base_channels": 0},
            {"bce_weight": 0.0, "dice_weight": 0.0},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                self._model(**kwargs)


class TestLossAndMetric(unittest.TestCase):
    def test_dice_loss_bounds(self):
        target = torch.zeros(1, 1, 10)
        target[..., 2:5] = 1
        perfect = dice_loss((target * 2 - 1) * 50, target)
        wrong = dice_loss((1 - target * 2) * 50, target)
        self.assertLess(perfect.item(), 1e-3)
        self.assertGreater(wrong.item(), 0.7)

    def test_segmentation_iou(self):
        y_true = np.zeros((2, 2, 10))
        y_true[0, 0, 0:4] = 1  # channel A: 4 true steps
        y_true[1, 1, 5:10] = 1  # channel B: 5 true steps
        y_prob = np.zeros((2, 2, 10))
        y_prob[0, 0, 2:6] = 0.9  # 2 hit, 2 false positive
        y_prob[1, 1, 5:10] = 0.8  # exact
        scores = segmentation_iou(y_true, y_prob, names=("A", "B"))
        self.assertAlmostEqual(scores["iou_A"], 2 / 6)
        self.assertAlmostEqual(scores["iou_B"], 1.0)
        self.assertAlmostEqual(scores["miou"], (2 / 6 + 1) / 2)

    def test_segmentation_iou_annotated_only(self):
        y_true = np.zeros((2, 2, 10))
        y_true[0, 0, 3:5] = 1  # annotated span of sample 0 is 3..6
        y_true[0, 1, 6] = 1
        y_prob = np.zeros((2, 2, 10))
        y_prob[0, 0, 0:5] = 0.9  # 3 false positives before the span
        y_prob[0, 1, 6] = 0.9
        y_prob[1, 0, :] = 0.9  # sample 1 has no annotation at all
        full = segmentation_iou(y_true, y_prob, names=("A", "B"))
        span = segmentation_iou(y_true, y_prob, names=("A", "B"), annotated_only=True)
        self.assertAlmostEqual(full["iou_A"], 2 / 15)
        self.assertAlmostEqual(span["iou_A"], 1.0)
        self.assertAlmostEqual(span["iou_B"], 1.0)

    def test_segmentation_iou_errors(self):
        with self.assertRaises(ValueError):
            segmentation_iou(np.zeros((1, 3, 5)), np.zeros((1, 3, 4)))
        with self.assertRaises(ValueError):
            segmentation_iou(np.zeros((1, 2, 5)), np.zeros((1, 2, 5)))


if __name__ == "__main__":
    unittest.main()
