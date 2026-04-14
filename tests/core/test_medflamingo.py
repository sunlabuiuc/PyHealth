"""Test cases for the MedFlamingo model stub."""

import unittest

import torch

from pyhealth.models.base_model import BaseModel
from pyhealth.models.medflamingo import MedFlamingo, MedFlamingoLayer


class TestMedFlamingoLayer(unittest.TestCase):
    """Test cases for MedFlamingoLayer."""

    def test_layer_initialization_defaults(self):
        """Test that MedFlamingoLayer initializes with default params."""
        layer = MedFlamingoLayer()
        self.assertEqual(layer.vision_dim, 768)
        self.assertEqual(layer.lang_dim, 1024)
        self.assertEqual(layer.num_resampler_tokens, 64)
        self.assertEqual(layer.num_resampler_layers, 6)
        self.assertEqual(layer.num_heads, 8)
        self.assertEqual(layer.dropout, 0.0)

    def test_layer_custom_params(self):
        """Test MedFlamingoLayer with custom dimensions."""
        layer = MedFlamingoLayer(
            vision_dim=512,
            lang_dim=2048,
            num_resampler_tokens=32,
            num_resampler_layers=4,
            num_heads=16,
            dropout=0.1,
        )
        self.assertEqual(layer.vision_dim, 512)
        self.assertEqual(layer.lang_dim, 2048)
        self.assertEqual(layer.num_resampler_tokens, 32)
        self.assertEqual(layer.num_resampler_layers, 4)
        self.assertEqual(layer.num_heads, 16)
        self.assertEqual(layer.dropout, 0.1)

    def test_layer_forward_raises(self):
        """Test that forward raises NotImplementedError (stub)."""
        layer = MedFlamingoLayer()
        lang_hidden = torch.randn(2, 10, 1024)
        vision_features = torch.randn(2, 196, 768)
        with self.assertRaises(NotImplementedError):
            layer(lang_hidden, vision_features)

    def test_layer_is_nn_module(self):
        """Test that MedFlamingoLayer is an nn.Module."""
        layer = MedFlamingoLayer()
        self.assertIsInstance(layer, torch.nn.Module)


class TestMedFlamingo(unittest.TestCase):
    """Test cases for the MedFlamingo model."""

    def test_model_initialization_standalone(self):
        """Test MedFlamingo initializes without a dataset."""
        model = MedFlamingo(dataset=None)
        self.assertIsInstance(model, MedFlamingo)
        self.assertEqual(model.vision_model_name, "openai/clip-vit-large-patch14")
        self.assertEqual(model.lang_model_name, "facebook/opt-6.7b")
        self.assertIsNone(model.medflamingo_checkpoint)
        self.assertEqual(model.cross_attn_every_n_layers, 4)
        self.assertEqual(model.num_resampler_tokens, 64)
        self.assertTrue(model.freeze_vision)
        self.assertTrue(model.freeze_lm)

    def test_model_custom_params(self):
        """Test MedFlamingo with custom model names and config."""
        model = MedFlamingo(
            dataset=None,
            vision_model_name="openai/clip-vit-base-patch32",
            lang_model_name="facebook/opt-1.3b",
            cross_attn_every_n_layers=2,
            num_resampler_tokens=32,
            freeze_vision=False,
        )
        self.assertEqual(model.vision_model_name, "openai/clip-vit-base-patch32")
        self.assertEqual(model.lang_model_name, "facebook/opt-1.3b")
        self.assertEqual(model.cross_attn_every_n_layers, 2)
        self.assertEqual(model.num_resampler_tokens, 32)
        self.assertFalse(model.freeze_vision)

    def test_forward_raises(self):
        """Test that forward raises NotImplementedError (stub)."""
        model = MedFlamingo(dataset=None)
        with self.assertRaises(NotImplementedError):
            model.forward()

    def test_generate_raises(self):
        """Test that generate raises NotImplementedError (stub)."""
        model = MedFlamingo(dataset=None)
        dummy_image = torch.randn(3, 224, 224)
        with self.assertRaises(NotImplementedError):
            model.generate(images=[dummy_image], prompt="What is shown?")

    def test_inherits_base_model(self):
        """Test that MedFlamingo inherits from BaseModel."""
        model = MedFlamingo(dataset=None)
        self.assertIsInstance(model, BaseModel)

    def test_standalone_has_empty_keys(self):
        """Test that standalone model has empty feature/label keys."""
        model = MedFlamingo(dataset=None)
        self.assertEqual(model.feature_keys, [])
        self.assertEqual(model.label_keys, [])

    def test_device_property(self):
        """Test that the device property works (inherited from BaseModel)."""
        model = MedFlamingo(dataset=None)
        self.assertIsInstance(model.device, torch.device)


if __name__ == "__main__":
    unittest.main()
