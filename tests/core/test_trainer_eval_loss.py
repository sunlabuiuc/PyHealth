"""Trainer.evaluate reports the mean loss over samples, whatever the batch size.

The loss used to be the plain mean of the per-batch mean losses, so a small
last batch counted as much as a full one (#859).
"""

import random
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.trainer import Trainer, _batch_size


def _samples(n=23):
    rng = random.Random(0)
    return [
        {
            "patient_id": f"p{i}",
            "visit_id": f"v{i}",
            "conditions": [f"c{rng.randrange(8)}" for _ in range(rng.randrange(1, 4))],
            "label": i % 2,
        }
        for i in range(n)
    ]


class _MeanOfX(nn.Module):
    """A model with no label processor whose batch loss is the mean of ``x``."""

    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return {"loss": x.float().mean() + 0 * self.dummy.sum()}


class TestEvaluateLoss(unittest.TestCase):
    def test_loss_does_not_depend_on_batch_size(self):
        torch.manual_seed(0)
        dataset = create_sample_dataset(
            samples=_samples(),
            input_schema={"conditions": "sequence"},
            output_schema={"label": "binary"},
            dataset_name="eval_loss",
        )
        trainer = Trainer(model=RNN(dataset=dataset), enable_logging=False)
        losses = {
            bs: trainer.evaluate(get_dataloader(dataset, batch_size=bs))["loss"]
            for bs in (23, 8, 22)  # one batch; 8 + 8 + 7; 22 + 1
        }
        self.assertAlmostEqual(losses[8], losses[23], places=5)
        self.assertAlmostEqual(losses[22], losses[23], places=5)

    def test_loss_without_label_processor_is_weighted_by_batch(self):
        values = [1.0, 1.0, 1.0, 10.0]  # batches of 3 and 1
        loader = DataLoader([{"x": torch.tensor(v)} for v in values], batch_size=3)
        trainer = Trainer(model=_MeanOfX(), enable_logging=False)
        # sample mean 13 / 4, not the mean of batch means (1 + 10) / 2
        self.assertAlmostEqual(trainer.evaluate(loader)["loss"], 3.25, places=5)

    def test_batch_size_of_collated_batches(self):
        self.assertEqual(_batch_size({"patient_id": ["a", "b"], "x": [1]}), 2)
        self.assertEqual(_batch_size({"x": torch.zeros(5, 3)}), 5)
        self.assertEqual(_batch_size({"s": torch.tensor(1.0), "codes": [[1], [2], [3]]}),
                         3)
        self.assertEqual(_batch_size({"flag": True}), 1)


if __name__ == "__main__":
    unittest.main()
