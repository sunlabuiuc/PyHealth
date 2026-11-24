"""Unit tests for TorchvisionModel."""

import os
import unittest

import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import TorchvisionModel

IMAGE_PATH = os.path.join(
    os.path.dirname(__file__), "../../test-resources/core/cameraman.tif"
)


def _image_dataset(label="binary"):
    samples = [
        {"patient_id": f"p{i}", "visit_id": "v0", "image": IMAGE_PATH, "label": i % 2}
        for i in range(2)
    ]
    return create_sample_dataset(
        samples=samples,
        input_schema={"image": "image"},
        output_schema={"label": label},
        dataset_name="torchvision_test",
    )


class TestTorchvisionModel(unittest.TestCase):
    """One model per backbone family, randomly initialized (no downloads)."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.dataset = _image_dataset()
        cls.batch = next(iter(get_dataloader(cls.dataset, batch_size=2, shuffle=False)))
        cls.models = {
            name: TorchvisionModel(cls.dataset, name, {"weights": None}).eval()
            for name in ("resnet18", "densenet121", "vit_b_16", "swin_t")
        }

    def test_forward_outputs(self):
        """forward returns loss, y_prob, y_true and logit for every backbone."""
        for name, model in self.models.items():
            with self.subTest(name), torch.no_grad():
                out = model(**self.batch)
                self.assertEqual(set(out), {"loss", "y_prob", "y_true", "logit"})
                self.assertEqual(out["logit"].shape, (2, 1))
                torch.testing.assert_close(out["y_prob"], torch.sigmoid(out["logit"]))
                self.assertEqual(out["loss"].dim(), 0)

    def test_grayscale_input_is_repeated_to_rgb(self):
        """A single-channel image runs, and matches its 3-channel copy."""
        self.assertEqual(self.batch["image"].shape[1], 1)
        rgb = dict(self.batch, image=self.batch["image"].repeat(1, 3, 1, 1))
        model = self.models["resnet18"]
        with torch.no_grad():
            torch.testing.assert_close(
                model(**self.batch)["logit"], model(**rgb)["logit"]
            )

    def test_embeddings_reproduce_logits(self):
        """embed=True gives the head's input: the head on it gives the logits."""
        for name, model in self.models.items():
            with self.subTest(name), torch.no_grad():
                embed = model(embed=True, **self.batch)["embed"]
                self.assertEqual(embed.shape, (2, model.hidden_dim))
                from_embed = model.forward_from_embedding(embed, **self.batch)
                torch.testing.assert_close(
                    from_embed["logit"],
                    model(**self.batch)["logit"],
                    atol=1e-4,
                    rtol=1e-4,
                )

    def test_unsupported_model_name(self):
        with self.assertRaises(AssertionError):
            TorchvisionModel(self.dataset, "alexnet", {"weights": None})

    def test_multiclass_output_size(self):
        samples = [
            {"patient_id": f"p{i}", "visit_id": "v0", "image": IMAGE_PATH, "label": i}
            for i in range(3)
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"image": "image"},
            output_schema={"label": "multiclass"},
            dataset_name="torchvision_multiclass",
        )
        model = TorchvisionModel(dataset, "resnet18", {"weights": None}).eval()
        batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
        with torch.no_grad():
            out = model(**batch)
        self.assertEqual(out["logit"].shape, (3, 3))
        torch.testing.assert_close(out["y_prob"].sum(-1), torch.ones(3))


class TestTorchvisionModelViTAttention(unittest.TestCase):
    """The ViT attention helpers used for attention-based interpretability."""

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        dataset = _image_dataset()
        cls.batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=False)))
        cls.vit = TorchvisionModel(dataset, "vit_b_16", {"weights": None}).eval()
        cls.resnet = TorchvisionModel(dataset, "resnet18", {"weights": None}).eval()

    def test_patch_geometry(self):
        self.assertTrue(self.vit.is_vit_model())
        self.assertFalse(self.resnet.is_vit_model())
        self.assertEqual(self.vit.get_patch_size(), 16)
        self.assertEqual(self.vit.get_num_patches(224), (14, 14))
        self.assertEqual(len(self.vit._vit_blocks), 12)

    def test_forward_with_attention_maps(self):
        """One row-stochastic attention map per encoder block."""
        with torch.no_grad():
            logits, maps = self.vit.forward_with_attention(
                self.batch["image"], register_hook=False
            )
        self.assertEqual(logits.shape, (2, 1))
        self.assertEqual(len(maps), 12)
        self.assertEqual(maps[0].shape, (2, 12, 197, 197))
        row_sums = maps[0].sum(-1)
        torch.testing.assert_close(row_sums, torch.ones_like(row_sums))
        self.assertIs(self.vit.get_attention_maps(), maps)

    @unittest.expectedFailure
    def test_forward_with_attention_matches_forward(self):
        """Known issue: forward_with_attention skips the position embeddings
        that torchvision's Encoder.forward adds, so its logits differ from
        forward's. Remove expectedFailure once that is fixed."""
        with torch.no_grad():
            logits, _ = self.vit.forward_with_attention(
                self.batch["image"], register_hook=False
            )
            expected = self.vit(**self.batch)["logit"]
        torch.testing.assert_close(logits, expected, atol=1e-4, rtol=1e-4)

    def test_attention_gradients_are_captured(self):
        logits, _ = self.vit.forward_with_attention(self.batch["image"])
        logits.sum().backward()
        grads = self.vit.get_attention_gradients()
        self.assertEqual(len(grads), 12)
        self.assertEqual(grads[0].shape, (2, 12, 197, 197))
        self.vit.zero_grad(set_to_none=True)

        self.vit.clear_attention_storage()
        self.assertEqual(self.vit.get_attention_maps(), [])
        self.assertEqual(self.vit.get_attention_gradients(), [])

    def test_vit_only_methods_reject_other_models(self):
        with self.assertRaises(ValueError):
            self.resnet.forward_with_attention(self.batch["image"])
        with self.assertRaises(ValueError):
            self.resnet.get_patch_size()


if __name__ == "__main__":
    unittest.main()
