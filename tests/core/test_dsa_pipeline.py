"""End-to-end test: DSADataset -> set_task -> SensorLSTM on synthetic data."""

import os
import shutil
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import DSADataset, get_dataloader
from pyhealth.models.sensor_lstm import SensorLSTM
from pyhealth.tasks.dsa_activity_classification import DSAActivityClassification

SUBJECTS = (1, 2)
ACTIVITIES = (1, 2)
SEGMENTS = (1, 2)
NUM_SAMPLES = len(SUBJECTS) * len(ACTIVITIES) * len(SEGMENTS)


def _write_synthetic_dsa(root: str) -> None:
    """Writes 2 subjects x 2 activities x 2 segments of random data."""
    rng = np.random.default_rng(0)
    for act in ACTIVITIES:
        for sub in SUBJECTS:
            folder = os.path.join(root, f"a{act:02d}", f"p{sub}")
            os.makedirs(folder)
            for seg in SEGMENTS:
                data = rng.normal(size=(125, 45))
                np.savetxt(
                    os.path.join(folder, f"s{seg:02d}.txt"), data, delimiter=","
                )


class TestDSAPipeline(unittest.TestCase):
    """Runs the dataset, the task and the model together."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.cache_dir = tempfile.mkdtemp()
        _write_synthetic_dsa(cls.root)
        dataset = DSADataset(root=cls.root, cache_dir=cls.cache_dir)
        task = DSAActivityClassification(source_sensors=["torso"])
        cls.samples = dataset.set_task(task)
        loader = get_dataloader(cls.samples, batch_size=NUM_SAMPLES, shuffle=False)
        cls.batch = next(iter(loader))

    @classmethod
    def tearDownClass(cls):
        cls.samples.close()
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_task_produces_one_sample_per_segment(self):
        self.assertEqual(len(self.samples), NUM_SAMPLES)

    def test_model_forward_and_backward_on_the_target_stream(self):
        torch.manual_seed(0)
        model = SensorLSTM(self.samples, hidden_dim=4)
        out = model(**self.batch)
        self.assertEqual(tuple(out["logit"].shape), (NUM_SAMPLES, len(ACTIVITIES)))
        self.assertTrue(torch.isfinite(out["loss"]))
        out["loss"].backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(grads)

    def test_model_reads_a_source_stream(self):
        torch.manual_seed(0)
        model = SensorLSTM(self.samples, input_key="source_torso", hidden_dim=4)
        out = model(**self.batch)
        self.assertEqual(tuple(out["y_prob"].shape), (NUM_SAMPLES, len(ACTIVITIES)))


if __name__ == "__main__":
    unittest.main()
