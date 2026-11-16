"""Unit tests for the GradCAM interpretability method
(Selvaraju et al., 2017, https://arxiv.org/abs/1610.02391)."""

import importlib.util
import os
import unittest

import numpy as np
import torch
from torch import nn

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.interpret.methods import GradCAM
from pyhealth.models import (
    CNN,
    MLP,
    BaseModel,
    ContraWR,
    SparcNet,
    TorchvisionModel,
)

IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "../../test-resources/core/cameraman.tif"
)


def _batch(dataset, batch_size=3):
    return next(iter(get_dataloader(dataset, batch_size=batch_size, shuffle=False)))


def _signal_dataset(n=3, channels=2, length=16, label="multiclass", shape=None):
    rng = np.random.RandomState(0)
    shape = shape or (channels, length)
    samples = [
        {
            "patient_id": f"p{i}",
            "visit_id": "v0",
            "signal": rng.randn(*shape).astype(np.float32),
            "label": i % 3 if label == "multiclass" else i % 2,
        }
        for i in range(n)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": label},
        dataset_name="grad_cam_signal",
    )


def _image_dataset():
    samples = [
        {"patient_id": f"p{i}", "visit_id": "v0", "image": IMAGE_PATH, "label": i}
        for i in range(2)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"image": "image"},
        output_schema={"label": "binary"},
        dataset_name="grad_cam_image",
    )


class ToyConvNet(BaseModel):
    """One convolution, global average pooling and a linear head.

    Its Grad-CAM has a closed form: with ``A = conv(x)`` over ``T`` positions
    and head weights ``W``, the gradient of logit ``c`` w.r.t. ``A[k, t]`` is
    ``W[c, k] / T``, so ``cam[t] = ReLU(sum_k W[c, k] / T * A[k, t])``.
    """

    def __init__(self, dataset, channels=2, hidden=4, spatial_dim=1):
        super().__init__(dataset)
        conv = {1: nn.Conv1d, 2: nn.Conv2d}[spatial_dim]
        # registered first, so the default target (last convolution) is self.conv
        self.unused = conv(channels, hidden, kernel_size=1)
        self.conv = conv(channels, hidden, kernel_size=3, padding=1)
        self.fc = nn.Linear(hidden, self.label_processor.output_size())

    def forward(self, **kwargs):
        x = kwargs[self.feature_keys[0]].float()
        act = self.conv(x)
        logit = self.fc(act.flatten(2).mean(-1))
        return {"logit": logit, "y_prob": self.label_processor.activation(logit)}

    def expected_cam(self, x, target):
        """The closed-form Grad-CAM for class indices ``target`` ([batch])."""
        with torch.no_grad():
            act = self.conv(x.float())
            positions = act.flatten(2).shape[-1]
            weights = self.fc.weight[target] / positions  # [batch, hidden]
            return (weights.view(*weights.shape, *[1] * (act.dim() - 2)) * act).sum(1)


class TestGradCAMKnownAnswer(unittest.TestCase):
    """Exact values on a model whose Grad-CAM can be computed by hand."""

    def setUp(self):
        torch.manual_seed(0)
        self.dataset = _signal_dataset()
        self.model = ToyConvNet(self.dataset)
        self.batch = _batch(self.dataset)
        self.x = self.batch["signal"]

    def test_predicted_class_by_default(self):
        cam = GradCAM(self.model).attribute(**self.batch)["signal"]
        predicted = self.model(**self.batch)["logit"].argmax(-1)
        expected = torch.relu(self.model.expected_cam(self.x, predicted))
        # one score per time step, repeated over the input channels
        self.assertEqual(cam.shape, self.x.shape)
        for channel in range(self.x.shape[1]):
            torch.testing.assert_close(cam[:, channel], expected)

    def test_target_class_idx(self):
        grad_cam = GradCAM(self.model, upsample=False)
        maps = {
            c: grad_cam.attribute(target_class_idx=c, **self.batch)["signal"]
            for c in (0, 2)
        }
        for c, cam in maps.items():
            target = torch.full((self.x.shape[0],), c)
            expected = torch.relu(self.model.expected_cam(self.x, target))
            torch.testing.assert_close(cam, expected)
        self.assertFalse(torch.allclose(maps[0], maps[2]))

    def test_relu_false_keeps_negative_evidence(self):
        cam = GradCAM(self.model, relu=False, upsample=False).attribute(
            target_class_idx=1, **self.batch
        )["signal"]
        target = torch.full((self.x.shape[0],), 1)
        torch.testing.assert_close(cam, self.model.expected_cam(self.x, target))
        self.assertTrue((cam < 0).any())

    def test_2d_input(self):
        dataset = _signal_dataset(shape=(2, 5, 6))
        model = ToyConvNet(dataset, spatial_dim=2)
        batch = _batch(dataset)
        cam = GradCAM(model, upsample=False).attribute(target_class_idx=0, **batch)
        target = torch.zeros(3, dtype=torch.long)
        expected = torch.relu(model.expected_cam(batch["signal"], target))
        torch.testing.assert_close(cam["signal"], expected)

    def test_in_place_op_after_layer_does_not_change_activation(self):
        """An in-place ReLU after the target layer (as in DenseNet) runs on a
        copy: the map uses the layer's real, signed output."""

        class InPlaceReluNet(ToyConvNet):
            def forward(self, **kwargs):
                act = self.conv(kwargs[self.feature_keys[0]].float())
                act.relu_()
                logit = self.fc(act.mean(-1))
                return {
                    "logit": logit,
                    "y_prob": self.label_processor.activation(logit),
                }

        torch.manual_seed(0)
        model = InPlaceReluNet(self.dataset)
        cam = GradCAM(model, relu=False, upsample=False).attribute(
            target_class_idx=0, **self.batch
        )["signal"]

        with torch.no_grad():
            act = model.conv(self.x.float())  # before the ReLU
            positions = act.shape[-1]
            # d logit / d act[k, t] = W[0, k] / T where act > 0, else 0
            grad = model.fc.weight[0].view(1, -1, 1) / positions * (act > 0)
            expected = (grad.mean(-1, keepdim=True) * act).sum(1)
        self.assertTrue((act < 0).any())
        torch.testing.assert_close(cam, expected)

    def test_frozen_model(self):
        """Frozen parameters before the target layer don't block the gradient."""
        expected = GradCAM(self.model).attribute(**self.batch)["signal"]
        for p in self.model.parameters():
            p.requires_grad_(False)
        frozen = GradCAM(self.model).attribute(**self.batch)["signal"]
        torch.testing.assert_close(frozen, expected)

    def test_works_under_no_grad_and_leaves_model_clean(self):
        expected = GradCAM(self.model).attribute(**self.batch)["signal"]
        with torch.no_grad():
            cam = GradCAM(self.model).attribute(**self.batch)["signal"]
        torch.testing.assert_close(cam, expected)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.assertEqual(len(self.model.conv._forward_hooks), 0)

    def test_hooks_removed_after_error(self):
        grad_cam = GradCAM(self.model, target_layers="unused")
        with self.assertRaises(RuntimeError):
            grad_cam.attribute(**self.batch)
        self.assertEqual(len(self.model.unused._forward_hooks), 0)

    def test_binary_ignores_target_class_idx(self):
        dataset = _signal_dataset(label="binary")
        model = ToyConvNet(dataset)
        batch = _batch(dataset)
        grad_cam = GradCAM(model)
        torch.testing.assert_close(
            grad_cam.attribute(**batch)["signal"],
            grad_cam.attribute(target_class_idx=1, **batch)["signal"],
        )

    def test_callable_matches_attribute(self):
        grad_cam = GradCAM(self.model)
        torch.testing.assert_close(
            grad_cam(**self.batch)["signal"], grad_cam.attribute(**self.batch)["signal"]
        )


class TestGradCAMModels(unittest.TestCase):
    """Default target layers and input-shape mapping on PyHealth's CNNs."""

    def test_cnn_sequence_tensor_and_multi_hot(self):
        torch.manual_seed(0)
        samples = [
            {
                "patient_id": f"p{i}",
                "visit_id": "v0",
                "codes": ["A", "B", "C"][: 3 - i],
                "labs": [[1.0, 2.0, 3.0]] * (4 - i),
                "flags": ["x", "y"],
                "label": i,
            }
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence", "labs": "tensor", "flags": "multi_hot"},
            output_schema={"label": "binary"},
            dataset_name="grad_cam_cnn",
        )
        model = CNN(dataset, embedding_dim=8, hidden_dim=4, num_layers=2)
        batch = _batch(dataset, batch_size=2)

        cams = GradCAM(model).attribute(**batch)

        self.assertEqual(set(cams), {"codes", "labs", "flags"})
        for key, cam in cams.items():
            self.assertEqual(cam.shape, batch[key].shape)
            self.assertTrue(torch.all(cam >= 0))
        # one score per time step, the same for every lab
        labs = cams["labs"]
        torch.testing.assert_close(labs, labs[..., :1].expand_as(labs))
        # a multi-hot vector is a single position for the CNN
        flags = cams["flags"]
        torch.testing.assert_close(flags, flags[:, :1].expand_as(flags))

    def test_cnn_target_layers_subset(self):
        dataset = _signal_dataset(label="binary")
        model = CNN(dataset, embedding_dim=8, hidden_dim=4, num_layers=2)
        cams = GradCAM(model, target_layers={"signal": "cnn.signal.cnn.0"}).attribute(
            **_batch(dataset)
        )
        self.assertEqual(cams["signal"].shape, _batch(dataset)["signal"].shape)

    def test_sparcnet_signal(self):
        torch.manual_seed(0)
        dataset = _signal_dataset(length=256)
        model = SparcNet(dataset=dataset)
        batch = _batch(dataset)

        cam = GradCAM(model).attribute(**batch)["signal"]
        raw = GradCAM(model, upsample=False).attribute(**batch)["signal"]

        self.assertEqual(cam.shape, batch["signal"].shape)
        self.assertTrue(torch.isfinite(cam).all() and torch.all(cam >= 0))
        torch.testing.assert_close(cam[:, 0], cam[:, 1])
        self.assertEqual(raw.dim(), 2)
        self.assertLess(raw.shape[1], batch["signal"].shape[-1])

    def test_torchvision_resnet(self):
        torch.manual_seed(0)
        dataset = _image_dataset()
        model = TorchvisionModel(dataset, "resnet18", {"weights": None})
        batch = _batch(dataset, batch_size=2)

        cam = GradCAM(model, target_layers="model.layer4").attribute(**batch)["image"]
        raw = GradCAM(model, target_layers="model.layer4", upsample=False).attribute(
            **batch
        )

        self.assertEqual(cam.shape, batch["image"].shape)
        self.assertEqual(raw["image"].shape, (2, 7, 7))
        self.assertTrue(torch.all(cam >= 0))
        # the default (last convolution) also runs
        self.assertEqual(GradCAM(model).attribute(**batch)["image"].shape, cam.shape)

    def test_torchvision_densenet(self):
        torch.manual_seed(0)
        dataset = _image_dataset()
        model = TorchvisionModel(dataset, "densenet121", {"weights": None})
        batch = _batch(dataset, batch_size=2)

        raw = GradCAM(model, target_layers="model.features", relu=False, upsample=False)
        unfrozen = raw.attribute(**batch)["image"]
        self.assertEqual(unfrozen.shape, (2, 7, 7))
        # DenseNet runs an in-place ReLU right after model.features; with a
        # frozen backbone that used to raise, and it must not change the map
        for p in model.model.features.parameters():
            p.requires_grad_(False)
        torch.testing.assert_close(raw.attribute(**batch)["image"], unfrozen)
        for p in model.model.features.parameters():
            p.requires_grad_(True)
        cam = GradCAM(model, target_layers="model.features").attribute(**batch)
        self.assertEqual(cam["image"].shape, batch["image"].shape)
        self.assertEqual(
            GradCAM(model).attribute(**batch)["image"].shape, cam["image"].shape
        )

    def test_cnn_image(self):
        dataset = _image_dataset()
        model = CNN(dataset, hidden_dim=4)
        batch = _batch(dataset, batch_size=2)
        self.assertEqual(
            GradCAM(model).attribute(**batch)["image"].shape, batch["image"].shape
        )


class TestGradCAMErrors(unittest.TestCase):
    def setUp(self):
        self.dataset = _signal_dataset()
        self.model = ToyConvNet(self.dataset)

    def test_vit_backbone_rejected(self):
        model = TorchvisionModel(_image_dataset(), "vit_b_16", {"weights": None})
        with self.assertRaises(TypeError):
            GradCAM(model)

    def test_model_without_convolutions_rejected(self):
        samples = [
            {"patient_id": f"p{i}", "visit_id": "v0", "codes": ["A", "B"], "label": i}
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"codes": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="grad_cam_mlp",
        )
        with self.assertRaises(TypeError):
            GradCAM(MLP(dataset))

    def test_bad_target_layers(self):
        with self.assertRaises(ValueError):
            GradCAM(self.model, target_layers="no.such.layer")
        with self.assertRaises(ValueError):
            GradCAM(self.model, target_layers={"other": "conv"})
        with self.assertRaises(ValueError):
            GradCAM(self.model, target_layers=nn.Conv1d(2, 2, 1))
        with self.assertRaises(TypeError):
            GradCAM(self.model, target_layers=3)

    def test_spectrogram_map_is_not_stretched_over_signal(self):
        """A 2-D spectrogram map can't line up with a [channels, time] input."""
        dataset = _signal_dataset(length=256)
        model = ContraWR(dataset=dataset)
        batch = _batch(dataset)
        with self.assertRaises(ValueError):
            GradCAM(model).attribute(**batch)
        raw = GradCAM(model, upsample=False).attribute(**batch)["signal"]
        self.assertEqual(raw.dim(), 3)  # [batch, freq, time]

    def test_input_without_value_tensor_rejected(self):
        """A feature the map can't be lined up with raises unless upsample=False."""
        batch = _batch(self.dataset)
        batch["signal"] = {"value": None}
        with self.assertRaises(ValueError):
            GradCAM(self.model)._map_to_input_shapes(
                {"signal": torch.ones(3, 16)}, batch
            )
        dict_batch = dict(batch, signal={"value": torch.zeros(3, 2, 16)})
        mapped = GradCAM(self.model)._map_to_input_shapes(
            {"signal": torch.ones(3, 16)}, dict_batch
        )
        self.assertEqual(mapped["signal"].shape, (3, 2, 16))

    def test_layer_without_spatial_axes_rejected(self):
        grad_cam = GradCAM(self.model, target_layers="fc")
        with self.assertRaises(ValueError):
            grad_cam.attribute(**_batch(self.dataset))


@unittest.skipUnless(
    importlib.util.find_spec("pytorch_grad_cam"), "pytorch-grad-cam not installed"
)
class TestGradCAMMatchesReference(unittest.TestCase):
    """Same maps as jacobgil/pytorch-grad-cam, up to its min-max scaling."""

    def test_matches_pytorch_grad_cam(self):
        from pytorch_grad_cam import GradCAM as ReferenceGradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        torch.manual_seed(0)
        dataset = _signal_dataset(shape=(2, 5, 6))
        model = ToyConvNet(dataset, spatial_dim=2).eval()
        batch = _batch(dataset)

        class Logits(nn.Module):
            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def forward(self, x):
                return self.inner(signal=x)["logit"]

        reference = ReferenceGradCAM(model=Logits(model), target_layers=[model.conv])
        expected = reference(
            input_tensor=batch["signal"], targets=[ClassifierOutputTarget(1)] * 3
        )
        ours = GradCAM(model, upsample=False).attribute(target_class_idx=1, **batch)
        ours = ours["signal"].flatten(1)
        ours = (ours - ours.min(1, keepdim=True).values) / (
            ours.max(1, keepdim=True).values - ours.min(1, keepdim=True).values + 1e-7
        )
        np.testing.assert_allclose(
            ours.reshape(expected.shape).numpy(), expected, atol=1e-4
        )


if __name__ == "__main__":
    unittest.main()
