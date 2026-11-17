import unittest
import tempfile
import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from pyhealth.processors.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for test images
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test RGB image
        self.rgb_image = Image.new('RGB', (100, 100), color=(255, 0, 0))  # Red image
        self.rgb_path = os.path.join(self.temp_dir, 'test_rgb.png')
        self.rgb_image.save(self.rgb_path)
        
        # Create a test grayscale image
        self.gray_image = Image.new('L', (50, 50), color=128)
        self.gray_path = os.path.join(self.temp_dir, 'test_gray.png')
        self.gray_image.save(self.gray_path)
        
        # Create a test RGBA image
        self.rgba_image = Image.new('RGBA', (75, 75), color=(255, 0, 0, 128))
        self.rgba_path = os.path.join(self.temp_dir, 'test_rgba.png')
        self.rgba_image.save(self.rgba_path)

    def tearDown(self):
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_init_default(self):
        processor = ImageProcessor()
        self.assertEqual(processor.image_size, 224)
        self.assertTrue(processor.to_tensor)
        self.assertFalse(processor.normalize)
        self.assertIsNone(processor.mean)
        self.assertIsNone(processor.std)
        self.assertIsNone(processor.mode)

    def test_init_custom(self):
        processor = ImageProcessor(
            image_size=128,
            to_tensor=False,
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5],
            mode='L'
        )
        self.assertEqual(processor.image_size, 128)
        self.assertFalse(processor.to_tensor)
        self.assertTrue(processor.normalize)
        self.assertEqual(processor.mean, [0.5, 0.5, 0.5])
        self.assertEqual(processor.std, [0.5, 0.5, 0.5])
        self.assertEqual(processor.mode, 'L')

    def test_init_normalize_without_mean_std(self):
        with self.assertRaises(ValueError):
            ImageProcessor(normalize=True)

    def test_init_mean_std_without_normalize(self):
        with self.assertRaises(ValueError):
            ImageProcessor(mean=[0.5], std=[0.5])

    def test_process_rgb_image_default(self):
        processor = ImageProcessor()
        result = processor.process(self.rgb_path)
        
        # Should be tensor
        self.assertIsInstance(result, torch.Tensor)
        # Should be resized to 224x224
        self.assertEqual(result.shape, (3, 224, 224))
        # Should be in [0, 1] range
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 1))

    def test_process_rgb_image_no_tensor(self):
        processor = ImageProcessor(to_tensor=False)
        result = processor.process(self.rgb_path)
        
        # Should be PIL Image
        self.assertIsInstance(result, Image.Image)
        # Should be resized
        self.assertEqual(result.size, (224, 224))
        self.assertEqual(result.mode, 'RGB')

    def test_process_rgb_image_custom_size(self):
        processor = ImageProcessor(image_size=64)
        result = processor.process(self.rgb_path)
        
        self.assertEqual(result.shape, (3, 64, 64))

    def test_process_rgb_image_normalize(self):
        processor = ImageProcessor(
            normalize=True,
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
        result = processor.process(self.rgb_path)
        
        # After normalization, values should be around 1.0 (since original is [1, 0, 0])
        # But let's check it's a tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 224, 224))

    def test_process_grayscale_image(self):
        processor = ImageProcessor(mode='L')
        result = processor.process(self.gray_path)
        
        # Should convert to grayscale and then to tensor
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (1, 224, 224))  # Grayscale has 1 channel

    def test_process_rgba_image(self):
        processor = ImageProcessor(mode='RGB')
        result = processor.process(self.rgba_path)
        
        # Should convert RGBA to RGB
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 224, 224))

    def test_process_invalid_path(self):
        processor = ImageProcessor()
        with self.assertRaises(FileNotFoundError):
            processor.process('/nonexistent/path/image.png')

    def test_process_path_object(self):
        processor = ImageProcessor()
        result = processor.process(Path(self.rgb_path))
        
        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 224, 224))

    def test_repr(self):
        processor = ImageProcessor()
        repr_str = repr(processor)
        self.assertIn('ImageLoadingProcessor', repr_str)
        self.assertIn('image_size=224', repr_str)
        self.assertIn('to_tensor=True', repr_str)
        self.assertIn('normalize=False', repr_str)

    def test_transform_build(self):
        processor = ImageProcessor()
        transform = processor.transform
        self.assertIsInstance(transform, transforms.Compose)

    def test_no_resize(self):
        # Test with image_size=None - but wait, default is 224, and if None, it might not resize
        # Looking at code: if self.image_size is not None: resize
        # So if image_size=None, no resize
        processor = ImageProcessor(image_size=None)
        result = processor.process(self.rgb_path)
        
        # Original image is 100x100, should remain that size
        self.assertEqual(result.shape, (3, 100, 100))


class TestImageProcessorFit(unittest.TestCase):
    """fit() records the widest (C, H, W) the processor will output."""

    setUp = TestImageProcessor.setUp
    tearDown = TestImageProcessor.tearDown

    def _samples(self, *paths):
        return [{"image": path} for path in paths]

    def test_widest_mode_without_a_mode(self):
        processor = ImageProcessor(image_size=32)
        processor.fit(self._samples(self.gray_path, self.rgb_path, self.rgba_path), "image")
        self.assertEqual(processor._shape, (4, 32, 32))
        processor.fit(self._samples(self.gray_path, self.rgb_path), "image")
        self.assertEqual(processor._shape, (3, 32, 32))

    def test_widest_size_without_an_image_size(self):
        processor = ImageProcessor(image_size=None, mode="L")
        processor.fit(self._samples(self.gray_path, self.rgba_path, self.rgb_path), "image")
        self.assertEqual(processor._shape, (1, 100, 100))
        wide = os.path.join(self.temp_dir, "wide.png")
        Image.new("RGB", (120, 40)).save(wide)
        processor.fit(self._samples(self.gray_path, wide), "image")
        self.assertEqual(processor._shape, (1, 50, 120))  # (C, H, W)

    def test_mode_and_image_size_need_no_images(self):
        processor = ImageProcessor(image_size=16, mode="RGBA")
        processor.fit(self._samples("/does/not/exist.png"), "image")
        self.assertEqual(processor._shape, (4, 16, 16))

    def test_missing_values_are_skipped(self):
        processor = ImageProcessor(image_size=8)
        processor.fit([{"image": None}, {}], "image")
        self.assertIsNone(processor._shape)
        processor.fit([{"image": None}, {"image": self.gray_path}], "image")
        self.assertEqual(processor._shape, (1, 8, 8))

    def test_fitted_shape_matches_process(self):
        for mode in (None, "L", "RGB"):
            with self.subTest(mode=mode):
                processor = ImageProcessor(image_size=24, mode=mode)
                processor.fit(self._samples(self.rgb_path), "image")
                self.assertEqual(
                    processor._shape, tuple(processor.process(self.rgb_path).shape)
                )

    def test_processors_pickled_before_fit_have_no_shape(self):
        processor = ImageProcessor(image_size=8)
        processor.__dict__.pop("_shape", None)
        self.assertIsNone(processor._shape)


if __name__ == '__main__':
    unittest.main()