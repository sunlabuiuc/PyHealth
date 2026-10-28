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

    def test_timestamps_between_steps_are_floored(self):
        times = [T0, T0 + timedelta(minutes=90)]
        value, mask = TimeseriesRegressionLabelProcessor().process((times, [1.0, 2.0]))
        np.testing.assert_array_equal(value.numpy(), [1.0, 2.0])
        np.testing.assert_array_equal(mask.numpy(), [1, 1])

    def test_two_labels_in_one_step_are_rejected(self):
        times = [T0, T0 + timedelta(minutes=20), T0 + timedelta(hours=1)]
        with self.assertRaisesRegex(ValueError, "time step 0"):
            TimeseriesBinaryLabelProcessor().process((times, [0, 1, 0]))


class TestPerStepEndToEnd(unittest.TestCase):
    """The label processors with a model and the Trainer.

    Model-level behaviour (masked loss, causality, bidirectional guard) is
    covered with a mock label processor in test_timeseries_modes.py.
    """

    def test_label_processors_train_and_evaluate(self):
        for mode, metric in (
            ("timeseries_binary", "roc_auc"),
            ("timeseries_regression", "mae"),
        ):
            dataset = _dataset(mode)
            model = RNN(dataset=dataset, hidden_dim=8, embedding_dim=8)
            self.assertEqual(model.mode, mode)
            batch = next(iter(get_dataloader(dataset, batch_size=4)))
            out = model(**batch)
            self.assertEqual(tuple(out["y_prob"].shape), (4, 6, 1))
            self.assertTrue(torch.isfinite(out["loss"]))

            trainer = Trainer(model=model, metrics=[metric], enable_logging=False)
            loader = get_dataloader(dataset, batch_size=3)
            y_true, _, _ = trainer.inference(loader)
            self.assertEqual(sorted(len(y) for y in y_true), [3, 4, 5, 6])
            self.assertIn(metric, trainer.evaluate(loader))

if __name__ == "__main__":
    unittest.main()
