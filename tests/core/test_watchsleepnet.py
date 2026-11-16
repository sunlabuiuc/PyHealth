import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import (
    WatchSleepNet,
)


class TestWatchSleepNet(unittest.TestCase):
    """Test cases for the WatchSleepNet model."""

    def setUp(self):
        """Set up test data and model."""

        def _make_sample(patient_id, label):
            pass

        self.samples = [
            _make_sample("patient-0", 1),
            _make_sample("patient-1", 0),
        ]

        self.input_schema = {
            "ibi": "tensor",
        }
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="test",
        )

        self.model = WatchSleepNet(
            dataset=self.dataset,
        )

    def test_model_initialization(self):
        pass

    def test_model_forward(self):
        pass

    def test_model_backward(self):
        pass


if __name__ == "__main__":
    unittest.main()
