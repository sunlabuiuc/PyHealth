# Author: Josh Steier
# Description: Tests for VisionEmbeddingModel

"""Test cases for VisionEmbeddingModel.

Run with: python -m pytest test_vision_embedding.py -v
"""

import tempfile
import shutil
import os
import unittest

import numpy as np
from PIL import Image

import torch
import torch.nn as nn

from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.utils import get_dataloader
from pyhealth.models.vision_embedding import VisionEmbeddingModel


class TestVisionEmbeddingModel(unittest.TestCase):
    """Test cases for VisionEmbeddingModel."""

    @classmethod
    def setUpClass(cls):
        """Create synthetic image dataset for all tests."""
        cls.temp_dir = tempfile.mkdtemp()
        cls.samples = []

        for i in range(20):
            img_path = os.path.join(cls.temp_dir, f"img_{i}.png")
            img = Image.fromarray(
                np.random.randint(0, 255, (224, 224), dtype=np.uint8), mode="L"
            )
            img.save(img_path)
            cls.samples.append({
                "patient_id": f"p{i // 2}",
                "visit_id": f"v{i}",
                "chest_xray": img_path,
                "label": i % 2,
            })

        cls.dataset = create_sample_dataset(
            samples=cls.samples,
            input_schema={"chest_xray": ("image", {"image_size": 224, "mode": "L"})},
            output_schema={"label": "binary"},
            dataset_name="test_vision",
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        shutil.rmtree(cls.temp_dir)

    def test_patch_backbone(self):
        """Test patch backbone produces correct output shape."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            patch_size=16,
            backbone="patch",
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        outputs = model({"chest_xray": batch["chest_xray"]})

        # 224/16 = 14, 14*14 = 196 patches
        self.assertEqual(outputs["chest_xray"].shape, (4, 196, 128))

    def test_cnn_backbone(self):
        """Test CNN backbone produces correct output shape."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="cnn",
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        outputs = model({"chest_xray": batch["chest_xray"]})

        # CNN outputs 7x7 = 49 spatial positions
        self.assertEqual(outputs["chest_xray"].shape, (4, 49, 128))

    def test_resnet18_backbone(self):
        """Test ResNet18 backbone produces correct output shape."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="resnet18",
            pretrained=False,  # Faster for tests
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        outputs = model({"chest_xray": batch["chest_xray"]})

        # ResNet outputs 7x7 = 49 spatial positions
        self.assertEqual(outputs["chest_xray"].shape, (4, 49, 128))

    def test_cls_token(self):
        """Test CLS token adds one position."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="cnn",
            use_cls_token=True,
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        outputs = model({"chest_xray": batch["chest_xray"]})

        # 49 patches + 1 CLS = 50
        self.assertEqual(outputs["chest_xray"].shape, (4, 50, 128))

    def test_output_mask(self):
        """Test mask output is all True for images."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="cnn",
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        outputs, masks = model({"chest_xray": batch["chest_xray"]}, output_mask=True)

        self.assertEqual(masks["chest_xray"].shape, (4, 49))
        self.assertTrue(masks["chest_xray"].all())

    def test_get_output_info(self):
        """Test output info metadata."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=256,
            backbone="patch",
            patch_size=16,
            use_cls_token=True,
        )

        info = model.get_output_info("chest_xray")

        self.assertEqual(info["num_patches"], 196)
        self.assertEqual(info["num_tokens"], 197)
        self.assertEqual(info["embedding_dim"], 256)
        self.assertTrue(info["has_cls_token"])

    def test_gradient_flow(self):
        """Test gradients flow through the model."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=64,
            backbone="cnn",
        )

        loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)
        batch = next(iter(loader))

        inputs = batch["chest_xray"].clone().requires_grad_(True)
        outputs = model({"chest_xray": inputs})

        loss = outputs["chest_xray"].sum()
        loss.backward()

        self.assertIsNotNone(inputs.grad)
        self.assertFalse(torch.all(inputs.grad == 0))

    def test_invalid_backbone(self):
        """Test invalid backbone raises error."""
        with self.assertRaises(ValueError):
            VisionEmbeddingModel(
                dataset=self.dataset,
                embedding_dim=128,
                backbone="invalid",
            )

    def test_feature_keys(self):
        """Test feature_keys inherited from BaseModel."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="cnn",
        )

        self.assertIn("chest_xray", model.feature_keys)

    def test_device_property(self):
        """Test device property from BaseModel."""
        model = VisionEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=128,
            backbone="cnn",
        )

        self.assertIsInstance(model.device, torch.device)


if __name__ == "__main__":
    unittest.main(verbosity=2)