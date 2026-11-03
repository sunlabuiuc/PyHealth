"""Tests for SaliencyVisualizer and visualize_attribution."""

import os
import shutil
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
import numpy as np
import torch

from pyhealth.interpret.methods.saliency_visualization import (
    SaliencyVisualizer,
    visualize_attribution,
)


class TestSaliencyVisualizerHelpers(unittest.TestCase):
    """Tensor conversion, shape processing and normalization helpers."""

    def setUp(self):
        self.viz = SaliencyVisualizer()

    def test_to_numpy_accepts_tensor_and_array_like(self):
        t = torch.ones(2, 2, requires_grad=True)
        self.assertIsInstance(self.viz._to_numpy(t), np.ndarray)
        out = self.viz._to_numpy([[1.0, 2.0]])
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (1, 2))

    def test_process_image_shapes(self):
        # Batched RGB CHW -> HWC
        self.assertEqual(self.viz._process_image(np.random.rand(1, 3, 8, 8)).shape, (8, 8, 3))
        # Single-channel CHW -> HW
        self.assertEqual(self.viz._process_image(np.random.rand(1, 8, 8)).shape, (8, 8))
        # Already HWC with one channel -> HW
        self.assertEqual(self.viz._process_image(np.random.rand(8, 8, 1)).shape, (8, 8))
        # Grayscale stays 2-D
        self.assertEqual(self.viz._process_image(np.random.rand(8, 8)).shape, (8, 8))

    def test_process_image_rescales_uint8_range(self):
        img = np.full((4, 4), 255.0)
        self.assertAlmostEqual(self.viz._process_image(img).max(), 1.0)

    def test_process_saliency_shapes(self):
        self.assertEqual(self.viz._process_saliency(np.random.rand(1, 3, 8, 8)).shape, (8, 8))
        self.assertEqual(self.viz._process_saliency(np.random.rand(3, 8, 8)).shape, (8, 8))
        self.assertEqual(self.viz._process_saliency(np.random.rand(8, 8)).shape, (8, 8))

    def test_process_saliency_aggregates_channel_magnitudes(self):
        sal = np.stack([np.ones((2, 2)), -np.ones((2, 2))])
        np.testing.assert_allclose(self.viz._process_saliency(sal), np.full((2, 2), 2.0))

    def test_normalize_range(self):
        out = self.viz._normalize_saliency(np.random.rand(16, 16))
        self.assertGreaterEqual(out.min(), 0.0)
        self.assertLessEqual(out.max(), 1.0)
        self.assertAlmostEqual(out.max(), 1.0)

    def test_normalize_constant_returns_zeros(self):
        out = self.viz._normalize_saliency(np.full((4, 4), 3.0))
        np.testing.assert_array_equal(out, np.zeros((4, 4)))

    def test_normalize_signed_uses_magnitude(self):
        # Strongly negative and strongly positive pixels should both be bright.
        sal = np.zeros((4, 4))
        sal[0, 0] = 5.0
        sal[3, 3] = -5.0
        out = self.viz._normalize_saliency(sal, percentile_clip=0)
        self.assertAlmostEqual(out[0, 0], 1.0)
        self.assertAlmostEqual(out[3, 3], 1.0)
        self.assertAlmostEqual(out[1, 1], 0.0)

    def test_normalize_without_clipping_is_min_max(self):
        sal = np.array([[0.0, 1.0], [2.0, 4.0]])
        out = self.viz._normalize_saliency(sal, percentile_clip=0)
        np.testing.assert_allclose(out, sal / 4.0)


class TestSaliencyVisualizerPlots(unittest.TestCase):
    """Smoke tests for each plotting method on the Agg backend."""

    def setUp(self):
        self.viz = SaliencyVisualizer(default_cmap="jet", default_alpha=0.5, figure_size=(4, 4))
        self.rgb = torch.rand(3, 8, 8)
        self.gray = torch.rand(1, 8, 8)
        self.saliency = torch.randn(8, 8)
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        plt.close("all")
        shutil.rmtree(self.temp_dir)

    def _path(self, name):
        return os.path.join(self.temp_dir, name)

    def test_overlay_rgb_and_gray(self):
        for img in (self.rgb, self.gray):
            self.viz.plot_saliency_overlay(plt, img, self.saliency, title="t", show=False)
        self.assertGreaterEqual(len(plt.get_fignums()), 2)

    def test_overlay_save_and_show(self):
        path = self._path("overlay.png")
        self.viz.plot_saliency_overlay(
            plt, self.rgb.numpy(), self.saliency.numpy(),
            alpha=0.2, cmap="hot", normalize=False, show=True, save_path=path,
        )
        self.assertTrue(os.path.exists(path))

    def test_multiple_attributions(self):
        attributions = {"grad": torch.randn(8, 8), "lrp": torch.randn(3, 8, 8)}
        path = self._path("multi.png")
        self.viz.plot_multiple_attributions(
            plt, self.rgb, attributions,
            method_names={"grad": "Gradient"}, save_path=path,
        )
        self.assertTrue(os.path.exists(path))
        # One panel for the image plus one per method (colorbars add more axes)
        self.assertGreaterEqual(len(plt.gcf().axes), 3)

    def test_multiple_attributions_gray_unnormalized(self):
        self.viz.plot_multiple_attributions(
            plt, self.gray, {"grad": self.saliency}, normalize=False,
        )
        self.assertGreaterEqual(len(plt.gcf().axes), 2)

    def test_heatmap(self):
        path = self._path("heatmap.png")
        self.viz.plot_saliency_heatmap(plt, self.saliency, title="Heatmap", save_path=path)
        self.assertTrue(os.path.exists(path))
        self.viz.plot_saliency_heatmap(plt, self.saliency, cmap="viridis", normalize=False, show=False)

    def test_attribution_distribution(self):
        path = self._path("hist.png")
        self.viz.plot_attribution_distribution(plt, self.saliency, title="Dist", bins=10, save_path=path)
        self.assertTrue(os.path.exists(path))
        self.viz.plot_attribution_distribution(plt, self.saliency.numpy(), show=False)
        self.assertEqual(plt.gca().get_title(), "Attribution Value Distribution")

    def test_top_k_features(self):
        path = self._path("topk.png")
        self.viz.plot_top_k_features(plt, self.rgb, self.saliency, k=5, title="Top", save_path=path)
        self.assertTrue(os.path.exists(path))
        self.viz.plot_top_k_features(plt, self.gray, self.saliency, k=3, show=False)


class TestVisualizeAttribution(unittest.TestCase):
    """The visualize_attribution convenience dispatcher."""

    def setUp(self):
        self.image = torch.rand(3, 8, 8)
        self.attr = torch.randn(8, 8)

    def tearDown(self):
        plt.close("all")

    def test_dispatches_each_method(self):
        visualize_attribution(plt, self.image, self.attr, title="o", method="overlay", show=False)
        visualize_attribution(plt, self.image, self.attr, title="h", method="heatmap", show=False)
        visualize_attribution(plt, self.image, self.attr, title="k", method="top_k", k=4, show=False)
        self.assertGreaterEqual(len(plt.get_fignums()), 3)

    def test_unknown_method_raises(self):
        with self.assertRaises(ValueError):
            visualize_attribution(plt, self.image, self.attr, method="bogus")


if __name__ == "__main__":
    unittest.main()
