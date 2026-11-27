"""Trainer.evaluate reports the mean loss over samples, whatever the batch size.

The loss used to be the plain mean of the per-batch mean losses, so a small
last batch counted as much as a full one (#859).
"""

import os
import random
import tempfile
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

    def forward(self, x, **kwargs):
        x = x.float()
        return {"loss": x.mean() + 0 * self.dummy.sum(), "y_true": x, "y_prob": x}


def _rnn_dataset(name):
    return create_sample_dataset(
        samples=_samples(),
        input_schema={"conditions": "sequence"},
        output_schema={"label": "binary"},
        dataset_name=name,
    )


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
        loader = DataLoader(
            [{"patient_id": f"p{i}", "x": torch.tensor(v)} for i, v in enumerate(values)],
            batch_size=3,
        )
        trainer = Trainer(model=_MeanOfX(), enable_logging=False)
        # sample mean 13 / 4, not the mean of batch means (1 + 10) / 2
        self.assertAlmostEqual(trainer.evaluate(loader)["loss"], 3.25, places=5)

    def test_inference_returns_extra_outputs_and_patient_ids(self):
        torch.manual_seed(0)
        dataset = _rnn_dataset("eval_loss_inference")
        trainer = Trainer(model=RNN(dataset=dataset), enable_logging=False)
        y_true, _, loss, extra, pids = trainer.inference(
            get_dataloader(dataset, batch_size=10),
            additional_outputs=["logit"],
            return_patient_ids=True,
        )
        self.assertEqual(len(y_true), 23)
        self.assertEqual(extra["logit"].shape[0], 23)
        self.assertEqual(sorted(pids), sorted(f"p{i}" for i in range(23)))
        self.assertAlmostEqual(
            loss, trainer.evaluate(get_dataloader(dataset, batch_size=23))["loss"],
            places=5,
        )

    def test_inference_without_label_processor_concatenates_batches(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        loader = DataLoader(
            [{"patient_id": f"p{i}", "x": torch.tensor(v)} for i, v in enumerate(values)],
            batch_size=2,
        )
        trainer = Trainer(model=_MeanOfX(), enable_logging=False)
        y_true, y_prob, loss = trainer.inference(loader)
        self.assertEqual(y_true.tolist(), values)
        self.assertEqual(y_prob.tolist(), values)
        self.assertAlmostEqual(loss, 3.0, places=5)

    def test_train_loads_a_checkpoint_and_reports_the_test_split(self):
        torch.manual_seed(0)
        dataset = _rnn_dataset("eval_loss_ckpt")
        source = Trainer(model=RNN(dataset=dataset), enable_logging=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.ckpt")
            source.save_ckpt(path)
            target = Trainer(
                model=RNN(dataset=dataset), checkpoint_path=path, enable_logging=False
            )
            for a, b in zip(source.model.parameters(), target.model.parameters()):
                self.assertTrue(torch.equal(a, b))
            loader = get_dataloader(dataset, batch_size=8)
            with self.assertLogs("pyhealth.trainer", level="INFO") as logs:
                target.train(loader, test_dataloader=loader, epochs=1)
        self.assertTrue(any("--- Test ---" in line for line in logs.output))

    def test_batch_size_of_collated_batches(self):
        self.assertEqual(_batch_size({"patient_id": ["a", "b"], "x": [1]}), 2)
        self.assertEqual(_batch_size({"visit_id": ["v1", "v2", "v3"]}), 3)
        # flattened fields (jagged codes, timeseries records) are not batch-first,
        # so without an id list every batch gets the same weight
        jagged = {"codes": torch.zeros(7, 4), "offsets": [0, 3, 7]}
        self.assertEqual(_batch_size(jagged), 1)
        self.assertEqual(_batch_size({"x": torch.zeros(5, 3)}), 1)


if __name__ == "__main__":
    unittest.main()
