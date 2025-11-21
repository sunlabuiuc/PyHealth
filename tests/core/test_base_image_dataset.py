"""
Unit tests for the BaseImageDataset class.

Author:
    Kilo Code
"""
import unittest
from unittest.mock import patch, MagicMock

from pyhealth.datasets import BaseImageDataset
from pyhealth.processors import ImageProcessor


class TestBaseImageDataset(unittest.TestCase):
    def setUp(self):
        # Mock the BaseDataset __init__ to avoid dependencies
        with patch('pyhealth.datasets.base_dataset.BaseDataset.__init__', return_value=None):
            self.dataset = BaseImageDataset(root="/fake/root")

    def test_set_task_adds_image_processor_when_missing(self):
        """Test that set_task adds ImageProcessor when 'image' key is missing from input_processors."""
        # Mock BaseDataset.set_task to capture arguments
        with patch('pyhealth.datasets.base_dataset.BaseDataset.set_task') as mock_super_set_task:
            mock_super_set_task.return_value = MagicMock()

            result = self.dataset.set_task()

            # Check that super().set_task was called
            mock_super_set_task.assert_called_once()
            args, kwargs = mock_super_set_task.call_args

            # input_processors is args[4]
            input_processors = args[4]
            self.assertIsNotNone(input_processors)
            self.assertIn('image', input_processors)
            self.assertIsInstance(input_processors['image'], ImageProcessor)

            # Check default values
            image_proc = input_processors['image']
            self.assertEqual(image_proc.image_size, 299)
            self.assertEqual(image_proc.mode, "L")

    def test_set_task_does_not_override_existing_image_processor(self):
        """Test that set_task does not override existing ImageProcessor."""
        custom_processor = ImageProcessor(image_size=128, mode="RGB")

        with patch('pyhealth.datasets.base_dataset.BaseDataset.set_task') as mock_super_set_task:
            mock_super_set_task.return_value = MagicMock()

            result = self.dataset.set_task(input_processors={'image': custom_processor})

            mock_super_set_task.assert_called_once()
            args, kwargs = mock_super_set_task.call_args

            # input_processors is args[4]
            input_processors = args[4]
            self.assertIsNotNone(input_processors)
            self.assertIn('image', input_processors)
            self.assertIs(input_processors['image'], custom_processor)

    def test_set_task_preserves_other_processors(self):
        """Test that set_task preserves other processors while adding image processor."""
        other_processor = MagicMock()

        with patch('pyhealth.datasets.base_dataset.BaseDataset.set_task') as mock_super_set_task:
            mock_super_set_task.return_value = MagicMock()

            result = self.dataset.set_task(input_processors={'other': other_processor})

            mock_super_set_task.assert_called_once()
            args, kwargs = mock_super_set_task.call_args

            # input_processors is args[4]
            input_processors = args[4]
            self.assertIsNotNone(input_processors)
            self.assertIn('image', input_processors)
            self.assertIn('other', input_processors)
            self.assertIs(input_processors['other'], other_processor)
            self.assertIsInstance(input_processors['image'], ImageProcessor)


if __name__ == "__main__":
    unittest.main()