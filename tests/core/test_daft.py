# Authors: Christopher Perez, Kai Cheung (kaicc2), Sarah Lee
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
#   Health Records and Chest X-Ray Scans (Elsharief et al., CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/elsharief25a.html
# Description: Tests for the DAFT model, synthetic tensors only.

import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DAFT, DAFTBlock, DAFTLayer

try:
    import torchvision  # noqa: F401

    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

RESNET18_CHANNELS = [64, 64, 128, 256, 512]


def _samples(labels, with_mask=True, seed=0):
    generator = torch.Generator().manual_seed(seed)
    samples = []
    for index, label in enumerate(labels):
        sample = {
            "patient_id": f"p{index}",
            "visit_id": f"v{index}",
            "ehr": torch.randn(6, 5, generator=generator).tolist(),
            "cxr": torch.randn(3, 32, 32, generator=generator).tolist(),
            "label": label,
        }
        if with_mask:
            sample["cxr_mask"] = float(index % 2)
        samples.append(sample)
    return samples


class TestDAFTBlock(unittest.TestCase):
    def test_sequence_and_feature_map_shapes(self):
        block = DAFTBlock(channels=8, cond_dim=4, bottleneck_dim=3)
        cond = torch.randn(2, 4)
        self.assertEqual(block(torch.randn(2, 5, 8), cond).shape, (2, 5, 8))
        self.assertEqual(block(torch.randn(2, 8, 3, 3), cond).shape, (2, 8, 3, 3))

    def test_zero_shift_only_block_is_identity(self):
        block = DAFTBlock(channels=8, cond_dim=4, bottleneck_dim=3, scale=False)
        torch.nn.init.zeros_(block.aux.aux_out.weight)
        target = torch.randn(2, 5, 8)
        self.assertTrue(torch.equal(block(target, torch.randn(2, 4)), target))

    def test_scale_activation_bounds(self):
        block = DAFTBlock(
            channels=8, cond_dim=4, bottleneck_dim=3, shift=False, activation="sigmoid"
        )
        v_scale, v_shift = block.film_params(torch.randn(2, 5, 8), torch.randn(2, 4))
        self.assertTrue(((v_scale > 0) & (v_scale < 1)).all())
        self.assertTrue(torch.equal(v_shift, torch.zeros_like(v_shift)))

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            DAFTBlock(
                channels=8, cond_dim=4, bottleneck_dim=3, scale=False, shift=False
            )
        with self.assertRaises(ValueError):
            DAFTBlock(channels=8, cond_dim=4, bottleneck_dim=3, activation="relu")
        block = DAFTBlock(channels=8, cond_dim=4, bottleneck_dim=3)
        with self.assertRaises(ValueError):
            block(torch.randn(2, 8), torch.randn(2, 4))


@unittest.skipUnless(HAS_TORCHVISION, "torchvision is required for DAFT")
class TestDAFTLayer(unittest.TestCase):
    @staticmethod
    def _layer(**kwargs):
        options = dict(
            ehr_input_dim=5,
            ehr_hidden_dim=8,
            cxr_backbone="resnet18",
            cxr_pretrained=False,
            layer_after=1,
        )
        options.update(kwargs)
        return DAFTLayer(**options)

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.ehr = torch.randn(4, 6, 5)
        cls.cxr = torch.randn(4, 3, 32, 32)
        cls.mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
        cls.layers = {
            direction: cls._layer(direction=direction, num_labels=3).eval()
            for direction in DAFTLayer.DIRECTIONS
        }

    def test_output_shape_both_directions(self):
        for direction, layer in self.layers.items():
            with self.subTest(direction=direction):
                self.assertEqual(layer(self.ehr, self.cxr, self.mask).shape, (4, 3))

    def test_stage_channels_and_truncated_backbone(self):
        stages, channels = DAFTLayer._build_cxr_stages(
            "resnet18", False, "ehr_to_cxr", 4
        )
        found = [DAFTLayer._stage_channels(stage) for stage in stages]
        self.assertEqual(found, RESNET18_CHANNELS)
        self.assertEqual(channels, 512)
        self.assertEqual(len(self.layers["cxr_to_ehr"].cxr_stages), 2)
        self.assertEqual(len(self.layers["ehr_to_cxr"].cxr_stages), 5)

    def test_medmod_bottleneck_default(self):
        layer = self.layers["cxr_to_ehr"]
        self.assertEqual(layer.daft.aux.aux_base.out_features, (16 + 64) // 7)

    def test_missing_image_does_not_affect_output(self):
        changed = self.cxr.clone()
        changed[self.mask == 0] = 100.0
        for direction, layer in self.layers.items():
            with self.subTest(direction=direction):
                out_a = layer(self.ehr, self.cxr, self.mask)
                out_b = layer(self.ehr, changed, self.mask)
                self.assertTrue(torch.allclose(out_a, out_b))

    def test_skip_matches_ehr_only_when_no_image(self):
        layer = self.layers["cxr_to_ehr"]
        no_image = layer(self.ehr, self.cxr, torch.zeros(4))
        self.assertTrue(torch.allclose(no_image, layer(self.ehr, None, None)))

    def test_learned_missing_embedding_gets_gradient(self):
        for direction in DAFTLayer.DIRECTIONS:
            with self.subTest(direction=direction):
                layer = self._layer(direction=direction, missing_cxr="learned")
                layer(self.ehr, self.cxr, self.mask).sum().backward()
                self.assertIsNotNone(layer.missing_embedding.grad)
                self.assertGreater(layer.missing_embedding.grad.abs().sum(), 0)

    def test_single_image_in_training_batch(self):
        mask = torch.tensor([1.0, 0.0, 0.0, 0.0])
        for direction in DAFTLayer.DIRECTIONS:
            with self.subTest(direction=direction):
                layer = self._layer(direction=direction, layer_after=4).train()
                layer(self.ehr, self.cxr, mask).sum().backward()
                norms = [
                    m for m in layer.modules() if isinstance(m, torch.nn.BatchNorm2d)
                ]
                self.assertTrue(all(m.training for m in norms))

    def test_grayscale_image_is_accepted(self):
        out = self.layers["cxr_to_ehr"](self.ehr, self.cxr[:, :1], self.mask)
        self.assertEqual(out.shape, (4, 3))

    def test_invalid_arguments(self):
        for bad in (
            dict(direction="both"),
            dict(cxr_backbone="vgg16"),
            dict(layer_after=5),
            dict(missing_cxr="zero"),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                self._layer(**bad)
        layer = self.layers["cxr_to_ehr"]
        with self.assertRaises(ValueError):
            layer(self.ehr[0], self.cxr, self.mask)
        with self.assertRaises(ValueError):
            layer(self.ehr, self.cxr[:2], self.mask)
        with self.assertRaises(ValueError):
            layer(self.ehr, self.cxr, torch.ones(3))


@unittest.skipUnless(HAS_TORCHVISION, "torchvision is required for DAFT")
class TestDAFT(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = create_sample_dataset(
            samples=_samples([0, 1, 0, 1]),
            input_schema={"ehr": "tensor", "cxr": "tensor", "cxr_mask": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="daft_binary_test",
        )
        cls.multilabel = create_sample_dataset(
            samples=_samples([[0], [1], [0, 1], []]),
            input_schema={"ehr": "tensor", "cxr": "tensor", "cxr_mask": "tensor"},
            output_schema={"label": "multilabel"},
            dataset_name="daft_multilabel_test",
        )

    @staticmethod
    def _model(dataset, **kwargs):
        return DAFT(
            dataset,
            ehr_hidden_dim=8,
            cxr_backbone="resnet18",
            cxr_pretrained=False,
            layer_after=1,
            **kwargs,
        )

    @staticmethod
    def _batch(dataset):
        return next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))

    def test_forward_outputs(self):
        model = self._model(self.binary)
        out = model(**self._batch(self.binary))
        self.assertEqual(set(out), {"logit", "y_prob", "loss", "y_true"})
        self.assertEqual(out["y_prob"].shape, (4, 1))
        self.assertTrue(((out["y_prob"] >= 0) & (out["y_prob"] <= 1)).all())
        self.assertEqual(out["loss"].dim(), 0)

    def test_backward_reaches_both_branches(self):
        model = self._model(self.binary)
        model(**self._batch(self.binary))["loss"].backward()
        self.assertIsNotNone(model.layer.daft.aux.aux_base.weight.grad)
        self.assertIsNotNone(model.layer.ehr_lstm_1.weight_ih_l0.grad)
        self.assertIsNotNone(model.layer.cxr_stages[0][0].weight.grad)

    def test_multilabel_output_size(self):
        model = self._model(self.multilabel, direction="ehr_to_cxr")
        out = model(**self._batch(self.multilabel))
        self.assertEqual(out["y_prob"].shape, (4, 2))

    def test_without_labels_returns_predictions_only(self):
        model = self._model(self.binary)
        batch = self._batch(self.binary)
        batch.pop("label")
        self.assertEqual(set(model(**batch)), {"logit", "y_prob"})

    def test_ehr_only_dataset(self):
        dataset = create_sample_dataset(
            samples=[
                {k: s[k] for k in ("patient_id", "visit_id", "ehr", "label")}
                for s in _samples([0, 1])
            ],
            input_schema={"ehr": "tensor"},
            output_schema={"label": "binary"},
            dataset_name="daft_ehr_only_test",
        )
        model = self._model(dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertEqual(model(**batch)["y_prob"].shape, (2, 1))

    def test_missing_ehr_key_raises(self):
        with self.assertRaises(ValueError):
            self._model(self.binary, ehr_feature_key="labs")
        model = self._model(self.binary)
        batch = self._batch(self.binary)
        batch.pop("ehr")
        with self.assertRaises(ValueError):
            model(**batch)


if __name__ == "__main__":
    unittest.main()
