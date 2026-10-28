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


def _dataset(mode="timeseries_binary", unlabeled=(), unlabeled_value=None):
    rng = np.random.default_rng(0)
    samples = []
    for i, steps in enumerate([5, 3, 6, 4]):
        values = [float((i + h) % 2) for h in range(steps)]
        mask = [0.0 if (i, h) in unlabeled else 1.0 for h in range(steps)]
        if unlabeled_value is not None:
            values = [unlabeled_value if m == 0 else v for v, m in zip(values, mask)]
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

    def test_nan_on_unlabeled_step_is_ignored(self):
        for mode in ("timeseries_binary", "timeseries_regression"):
            nan = _dataset(mode, unlabeled={(1, 0)}, unlabeled_value=float("nan"))
            zero = _dataset(mode, unlabeled={(1, 0)}, unlabeled_value=0.0)
            torch.manual_seed(0)
            model = RNN(dataset=nan, hidden_dim=8, embedding_dim=8)
            batch = next(iter(get_dataloader(nan, batch_size=4)))
            out = model(**batch)
            self.assertTrue(torch.isfinite(out["loss"]))
            out["loss"].backward()
            for parameter in model.parameters():
                if parameter.grad is not None:
                    self.assertTrue(torch.isfinite(parameter.grad).all())

            # Same loss as with any other value on the unlabeled step.
            loss_fn = model.get_loss_function()
            logits = out["logit"].detach()
            zero_batch = next(iter(get_dataloader(zero, batch_size=4)))
            torch.testing.assert_close(
                loss_fn(logits, batch["label"]), loss_fn(logits, zero_batch["label"])
            )

    def test_label_longer_than_input_is_rejected(self):
        model = RNN(dataset=_dataset(), hidden_dim=8, embedding_dim=8)
        logits = torch.zeros(1, 3, 1)
        label = (torch.zeros(1, 4), torch.ones(1, 4))
        with self.assertRaises(ValueError):
            model.get_loss_function()(logits, label)

    def test_time_series_inputs_must_share_the_time_grid(self):
        samples = [
            {
                "patient_id": "p0",
                "a": ([T0 + timedelta(hours=h) for h in range(3)], np.zeros((3, 1))),
                "b": ([T0 + timedelta(hours=h) for h in range(5)], np.zeros((5, 1))),
                "label": ([0.0] * 3, [1.0] * 3),
            }
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"a": "timeseries", "b": "timeseries"},
            output_schema={"label": "timeseries_binary"},
            output_processors={"label": _StepLabelProcessor()},
            dataset_name="timeseries_grids",
        )
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8)
        batch = next(iter(get_dataloader(dataset, batch_size=1)))
        with self.assertRaises(ValueError):
            model(**batch)

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
