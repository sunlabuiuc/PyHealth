# Author: Joshua Steier
# Description: Unit tests for TimeImageProcessor.

import os
import shutil
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from pyhealth.processors.time_image_processor import (
    TimeImageProcessor,
)


class TestTimeImageProcessor(unittest.TestCase):
    """Tests for TimeImageProcessor."""

    def setUp(self):
        """Create temp directory with synthetic test images."""
        self.temp_dir = tempfile.mkdtemp()

        self.rgb_paths = []
        for i in range(5):
            path = os.path.join(
                self.temp_dir, f"rgb_{i}.png"
            )
            img = Image.new(
                "RGB",
                (100 + i * 10, 100 + i * 10),
                color=(255, i * 50, 0),
            )
            img.save(path)
            self.rgb_paths.append(path)

        self.gray_path = os.path.join(
            self.temp_dir, "gray.png"
        )
        Image.new("L", (80, 80), color=128).save(
            self.gray_path
        )

        self.rgba_path = os.path.join(
            self.temp_dir, "rgba.png"
        )
        Image.new(
            "RGBA", (90, 90), color=(255, 0, 0, 128)
        ).save(self.rgba_path)

        self.times = [0.0, 1.5, 3.0, 7.0, 14.0]

    def tearDown(self):
        """Remove temporary test directory."""
        shutil.rmtree(self.temp_dir)

    # ---- Initialization ----

    def test_init_default(self):
        """Default init sets expected attributes."""
        proc = TimeImageProcessor()
        self.assertEqual(proc.image_size, 224)
        self.assertTrue(proc.to_tensor)
        self.assertFalse(proc.normalize)
        self.assertIsNone(proc.mean)
        self.assertIsNone(proc.std)
        self.assertIsNone(proc.mode)
        self.assertIsNone(proc.max_images)

    def test_init_custom(self):
        """Custom init stores all arguments."""
        proc = TimeImageProcessor(
            image_size=128,
            to_tensor=True,
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            mode="RGB",
            max_images=3,
        )
        self.assertEqual(proc.image_size, 128)
        self.assertEqual(proc.max_images, 3)
        self.assertTrue(proc.normalize)
        self.assertEqual(proc.mode, "RGB")

    def test_init_normalize_without_mean_std_raises(self):
        """ValueError when normalize=True but no mean/std."""
        with self.assertRaises(ValueError):
            TimeImageProcessor(normalize=True)

    def test_init_mean_std_without_normalize_raises(self):
        """ValueError when mean/std given but normalize=False."""
        with self.assertRaises(ValueError):
            TimeImageProcessor(mean=[0.5], std=[0.5])

    # ---- Core process() ----

    def test_process_basic_rgb(self):
        """Basic RGB returns correct shapes and tag."""
        proc = TimeImageProcessor(image_size=64)
        paths = self.rgb_paths[:3]
        times = self.times[:3]

        images, timestamps, tag = proc.process(
            (paths, times)
        )

        self.assertEqual(images.shape, (3, 3, 64, 64))
        self.assertEqual(timestamps.shape, (3,))
        self.assertEqual(tag, "image")
        self.assertIsInstance(images, torch.Tensor)
        self.assertIsInstance(timestamps, torch.Tensor)

    def test_process_single_image(self):
        """Single image works correctly."""
        proc = TimeImageProcessor(image_size=32)
        images, timestamps, tag = proc.process(
            ([self.rgb_paths[0]], [0.0])
        )

        self.assertEqual(images.shape, (1, 3, 32, 32))
        self.assertEqual(timestamps.shape, (1,))
        self.assertEqual(tag, "image")

    def test_process_grayscale(self):
        """Grayscale mode produces single-channel output."""
        proc = TimeImageProcessor(image_size=64, mode="L")
        images, timestamps, tag = proc.process(
            ([self.gray_path], [0.0])
        )

        self.assertEqual(images.shape, (1, 1, 64, 64))

    def test_process_rgba_to_rgb(self):
        """RGBA converted to RGB via mode parameter."""
        proc = TimeImageProcessor(image_size=64, mode="RGB")
        images, timestamps, tag = proc.process(
            ([self.rgba_path], [1.0])
        )

        self.assertEqual(images.shape, (1, 3, 64, 64))

    def test_process_with_normalization(self):
        """Normalization applies without errors."""
        proc = TimeImageProcessor(
            image_size=64,
            normalize=True,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )
        images, timestamps, tag = proc.process(
            (self.rgb_paths[:2], self.times[:2])
        )

        self.assertEqual(images.shape, (2, 3, 64, 64))
        self.assertIsInstance(images, torch.Tensor)

    # ---- Chronological sorting ----

    def test_process_sorts_by_timestamp(self):
        """Images reordered chronologically by timestamp."""
        proc = TimeImageProcessor(image_size=32)

        paths = list(reversed(self.rgb_paths[:3]))
        times = [10.0, 5.0, 1.0]

        _, timestamps, _ = proc.process((paths, times))

        expected = torch.tensor([1.0, 5.0, 10.0])
        self.assertTrue(torch.equal(timestamps, expected))

    # ---- max_images truncation ----

    def test_max_images_truncates_to_most_recent(self):
        """max_images keeps the N most recent images."""
        proc = TimeImageProcessor(
            image_size=32, max_images=2
        )
        images, timestamps, tag = proc.process(
            (self.rgb_paths[:4], self.times[:4])
        )

        self.assertEqual(images.shape[0], 2)
        self.assertEqual(timestamps.shape, (2,))
        self.assertAlmostEqual(timestamps[0].item(), 3.0)
        self.assertAlmostEqual(timestamps[1].item(), 7.0)

    def test_max_images_no_truncation_when_under(self):
        """No truncation when count is under max_images."""
        proc = TimeImageProcessor(
            image_size=32, max_images=10
        )
        images, timestamps, _ = proc.process(
            (self.rgb_paths[:3], self.times[:3])
        )

        self.assertEqual(images.shape[0], 3)

    # ---- Error handling ----

    def test_process_mismatched_lengths_raises(self):
        """ValueError for mismatched paths and times."""
        proc = TimeImageProcessor()
        with self.assertRaises(ValueError):
            proc.process(
                (self.rgb_paths[:3], self.times[:2])
            )

    def test_process_empty_paths_raises(self):
        """ValueError for empty image list."""
        proc = TimeImageProcessor()
        with self.assertRaises(ValueError):
            proc.process(([], []))

    def test_process_invalid_path_raises(self):
        """FileNotFoundError for nonexistent image."""
        proc = TimeImageProcessor()
        with self.assertRaises(FileNotFoundError):
            proc.process(
                (["/nonexistent/img.png"], [0.0])
            )

    # ---- Path types ----

    def test_process_accepts_path_objects(self):
        """Path objects accepted alongside strings."""
        proc = TimeImageProcessor(image_size=32)
        paths = [Path(self.rgb_paths[0])]
        images, _, _ = proc.process((paths, [0.0]))

        self.assertEqual(images.shape, (1, 3, 32, 32))

    # ---- fit() ----

    def test_fit_infers_channels_from_mode(self):
        """fit() infers n_channels from mode parameter."""
        proc = TimeImageProcessor(mode="L")
        proc.fit([], "xray")
        self.assertEqual(proc.size(), 1)

        proc2 = TimeImageProcessor(mode="RGB")
        proc2.fit([], "xray")
        self.assertEqual(proc2.size(), 3)

    def test_fit_infers_channels_from_sample(self):
        """fit() infers n_channels from actual image data."""
        proc = TimeImageProcessor(image_size=32)
        samples = [
            {
                "xray": (
                    [self.rgb_paths[0]],
                    [0.0],
                ),
            }
        ]
        proc.fit(samples, "xray")
        self.assertEqual(proc.size(), 3)

    def test_fit_defaults_to_3_channels(self):
        """fit() defaults to 3 channels if nothing found."""
        proc = TimeImageProcessor()
        proc.fit([], "xray")
        self.assertEqual(proc.size(), 3)

    # ---- size() ----

    def test_size_none_before_fit(self):
        """size() returns None before fit or process."""
        proc = TimeImageProcessor()
        self.assertIsNone(proc.size())

    def test_size_set_after_process(self):
        """size() returns channels after process()."""
        proc = TimeImageProcessor(image_size=32)
        proc.process(([self.rgb_paths[0]], [0.0]))
        self.assertEqual(proc.size(), 3)

    # ---- Repr ----

    def test_repr(self):
        """__repr__ contains key parameter values."""
        proc = TimeImageProcessor(
            image_size=128, max_images=5, mode="L"
        )
        r = repr(proc)
        self.assertIn("TimeImageProcessor", r)
        self.assertIn("image_size=128", r)
        self.assertIn("max_images=5", r)
        self.assertIn("mode=L", r)

    # ---- Tensor properties ----

    def test_output_values_in_valid_range(self):
        """Without normalization, pixels are in [0, 1]."""
        proc = TimeImageProcessor(image_size=32)
        images, _, _ = proc.process(
            (self.rgb_paths[:2], self.times[:2])
        )

        self.assertTrue(torch.all(images >= 0))
        self.assertTrue(torch.all(images <= 1))

    def test_timestamp_dtype_is_float32(self):
        """Timestamps are always float32 tensors."""
        proc = TimeImageProcessor(image_size=32)
        _, timestamps, _ = proc.process(
            (self.rgb_paths[:2], self.times[:2])
        )

        self.assertEqual(timestamps.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()