import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.processors import (
    TimeseriesBinaryLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from pyhealth.trainer import Trainer

T0 = datetime(2020, 1, 1)
NAN = float("nan")


def _times(n):
    return [T0 + timedelta(hours=h) for h in range(n)]


def _samples(label_fn):
    rng = np.random.default_rng(0)
    samples = []
    for i, n in enumerate([5, 3, 6, 4]):
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": (_times(n), rng.normal(size=(n, 2))),
                "static": [float(i), 1.0],
                "label": (_times(n), [label_fn(i, h) for h in range(n)]),
            }
        )
    return samples


def _dataset(output="timeseries_binary", label_fn=lambda i, h: float((i + h) % 2)):
    return create_sample_dataset(
        samples=_samples(label_fn),
        input_schema={"dynamic": "timeseries", "static": "tensor"},
        output_schema={"label": output},
        dataset_name="per_step",
    )


class TestTimeseriesLabelProcessors(unittest.TestCase):
    def test_value_and_mask(self):
        value, mask = TimeseriesBinaryLabelProcessor().process(
            (_times(4), [0, NAN, 1, 1])
        )
        np.testing.assert_array_equal(value.numpy(), [0, 0, 1, 1])
        np.testing.assert_array_equal(mask.numpy(), [1, 0, 1, 1])

    def test_grid_follows_sampling_rate(self):
        times = [T0, T0 + timedelta(hours=3)]
        value, mask = TimeseriesRegressionLabelProcessor().process((times, [2.5, 4.0]))
        np.testing.assert_array_equal(value.numpy(), [2.5, 0, 0, 4.0])
        np.testing.assert_array_equal(mask.numpy(), [1, 0, 0, 1])

    def test_binary_rejects_other_values(self):
        with self.assertRaises(ValueError):
            TimeseriesBinaryLabelProcessor().process((_times(2), [0, 2]))


class TestPerStepModel(unittest.TestCase):
    def test_modes_are_separate_from_binary_and_regression(self):
        model = RNN(dataset=_dataset("timeseries_binary"))
        self.assertEqual(model.mode, "timeseries_binary")
        model = RNN(dataset=_dataset("timeseries_regression"))
        self.assertEqual(model.mode, "timeseries_regression")

    def test_loss_ignores_unlabeled_and_padded_steps(self):
        # Stay 1 has an unlabeled step and is padded; changing what the
        # model predicts there must not change the loss.
        dataset = _dataset(label_fn=lambda i, h: NAN if (i, h) == (1, 0) else float(h % 2))
        model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8).eval()
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        with torch.no_grad():
            out = model(**batch)
        value, mask = out["y_true"]
        self.assertEqual(tuple(out["y_prob"].shape), (4, 6, 1))
        self.assertEqual(int(mask.sum()), 17)  # 5 + 3 + 6 + 4 steps, minus one

        logits = out["logit"].clone()
        loss_fn = model.get_loss_function()
        logits[mask == 0] = 1e3
        torch.testing.assert_close(loss_fn(logits, (value, mask)), out["loss"])

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

if __name__ == "__main__":
    unittest.main()
