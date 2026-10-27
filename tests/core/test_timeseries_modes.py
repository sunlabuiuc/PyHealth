import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import (
    timeseries_binary_metrics_fn,
    timeseries_regression_metrics_fn,
)
from pyhealth.models import RNN
from pyhealth.processors.base_processor import FeatureProcessor
from pyhealth.trainer import Trainer

T0 = datetime(2020, 1, 1)


class _StepLabelProcessor(FeatureProcessor):
    """Minimal per-step label processor: ``(values, mask)`` lists to tensors."""

    def process(self, value):
        values, mask = value
        return (
            torch.tensor(values, dtype=torch.float),
            torch.tensor(mask, dtype=torch.float),
        )

    def size(self):
        return 1

    def is_token(self):
        return False

    def schema(self):
        return ("value", "mask")

    def dim(self):
        return (1, 1)

    def spatial(self):
        return (True,)


def _dataset(mode="timeseries_binary", unlabeled=()):
    rng = np.random.default_rng(0)
    samples = []
    for i, steps in enumerate([5, 3, 6, 4]):
        values = [float((i + h) % 2) for h in range(steps)]
        mask = [0.0 if (i, h) in unlabeled else 1.0 for h in range(steps)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": (
                    [T0 + timedelta(hours=h) for h in range(steps)],
                    rng.normal(size=(steps, 2)),
                ),
                "static": [float(i), 1.0],
                "label": (values, mask),
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"dynamic": "timeseries", "static": "tensor"},
        output_schema={"label": mode},
        output_processors={"label": _StepLabelProcessor()},
        dataset_name="timeseries_modes",
    )


class TestTimeseriesModes(unittest.TestCase):
    def test_modes_are_separate_from_binary_and_regression(self):
        for mode in ("timeseries_binary", "timeseries_regression"):
            self.assertEqual(RNN(dataset=_dataset(mode)).mode, mode)

    def test_loss_ignores_unlabeled_and_padded_steps(self):
        dataset = _dataset(unlabeled={(1, 0)})
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8).eval()
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        with torch.no_grad():
            out = model(**batch)
        value, mask = out["y_true"]
        self.assertEqual(tuple(out["y_prob"].shape), (4, 6, 1))
        self.assertEqual(int(mask.sum()), 17)  # 5 + 3 + 6 + 4 steps, minus one

        logits = out["logit"].clone()
        logits[mask == 0] = 1e3
        loss = model.get_loss_function()(logits, (value, mask))
        torch.testing.assert_close(loss, out["loss"])

    def test_predictions_do_not_see_future_steps(self):
        dataset = _dataset()
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8).eval()
        batch = next(iter(get_dataloader(dataset, batch_size=1)))
        with torch.no_grad():
            before = model(**batch)["y_prob"][0, :, 0]
            batch["dynamic"] = batch["dynamic"].clone()
            batch["dynamic"][:, -1] += 100.0
            after = model(**batch)["y_prob"][0, :, 0]
        torch.testing.assert_close(before[:-1], after[:-1])
        self.assertFalse(torch.allclose(before[-1], after[-1]))

    def test_bidirectional_is_rejected(self):
        with self.assertRaises(ValueError):
            RNN(dataset=_dataset(), bidirectional=True)

    def test_trainer_scores_labeled_steps_per_sample(self):
        dataset = _dataset()
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8)
        trainer = Trainer(model=model, metrics=["roc_auc"], enable_logging=False)
        loader = get_dataloader(dataset, batch_size=3)
        y_true, y_prob, _ = trainer.inference(loader)
        self.assertEqual(sorted(len(y) for y in y_true), [3, 4, 5, 6])
        self.assertEqual([len(p) for p in y_prob], [len(y) for y in y_true])
        self.assertIn("roc_auc", trainer.evaluate(loader))

        dataset = _dataset("timeseries_regression")
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8)
        trainer = Trainer(model=model, metrics=["mae"], enable_logging=False)
        self.assertIn("mae", trainer.evaluate(get_dataloader(dataset, batch_size=3)))


class TestTimeseriesMetrics(unittest.TestCase):
    def test_binary_pools_steps_of_all_samples(self):
        y_true = [np.array([0, 0, 1]), np.array([1, 0])]
        y_prob = [np.array([0.1, 0.4, 0.8]), np.array([0.7, 0.2])]
        scores = timeseries_binary_metrics_fn(y_true, y_prob, metrics=["roc_auc"])
        self.assertEqual(scores, {"roc_auc": 1.0})

    def test_regression_pools_steps_of_all_samples(self):
        y_true = [np.array([3.0, 2.0]), np.array([1.0])]
        y_pred = [np.array([2.0, 2.0]), np.array([2.0])]
        scores = timeseries_regression_metrics_fn(y_true, y_pred, metrics=["mae"])
        self.assertAlmostEqual(scores["mae"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
