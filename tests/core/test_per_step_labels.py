import unittest
from collections.abc import Callable
from itertools import product
from typing import Any

import numpy as np
import torch

from pyhealth.datasets import SampleDataset, create_sample_dataset, get_dataloader
from pyhealth.models import BaseModel, Unified_RNN
from pyhealth.processors import (
    LevelConfig,
    SequenceProcessorV2,
    TimeseriesBinaryLabelProcessor,
    TimeseriesLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from pyhealth.trainer import Trainer
from tests.core.label_record_model_utils import LabelRecordModel

NAN = float("nan")
DYNAMIC: list[LevelConfig] = [{"ordering": "series", "values": [2]}]


def _samples(label_fn) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    samples = []
    for i, n in enumerate([5, 3, 6, 4]):
        hours = [float(h) for h in range(n)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": {"time": hours, "values": rng.normal(size=(n, 2))},
                "static": [float(i), 1.0],
                "label": {"time": hours, "values": [label_fn(i, h) for h in range(n)]},
            }
        )
    return samples


def _dataset(output="timeseries_binary", label_fn=lambda i, h: float((i + h) % 2)):
    return create_sample_dataset(
        samples=_samples(label_fn),
        input_schema={
            "dynamic": ("sequence_v2", {"levels": DYNAMIC}),
            "static": "tensor",
        },
        output_schema={"label": output},
        dataset_name="per_step",
    )


class TestTimeseriesLabelProcessors(unittest.TestCase):
    def test_are_restricted_sequence_v2_fields(self):
        processor = TimeseriesBinaryLabelProcessor()
        self.assertIsInstance(processor, SequenceProcessorV2)
        self.assertIsInstance(processor, TimeseriesLabelProcessor)
        self.assertEqual(processor.schema(), ((0, "values"), (0, "time")))
        self.assertEqual(processor.size(), 1)
        self.assertFalse(processor.learns_statistics)
        self.assertEqual(repr(processor), "TimeseriesBinaryLabelProcessor()")

    def test_values_and_time(self):
        values, time = TimeseriesRegressionLabelProcessor().process(
            {"time": [0.0, 3.0, 4.5], "values": [[2.5], [4.0], [1.0]]}
        )
        self.assertEqual(values.tolist(), [[2.5], [4.0], [1.0]])
        self.assertEqual(time.tolist(), [0.0, 3.0, 4.5])

    def test_needs_no_fit(self):
        self.assertEqual(
            TimeseriesBinaryLabelProcessor()
            .process({"time": [1.0], "values": [1]})[0]
            .tolist(),
            [[1.0]],
        )

    def test_rejects(self):
        bad: list[tuple[Any, Any]] = [
            (TimeseriesBinaryLabelProcessor(), {"time": [0.0, 1.0], "values": [0, 2]}),
            (
                TimeseriesBinaryLabelProcessor(),
                {"time": [0.0, 0.0], "values": [0, 1]},
            ),  # one time twice
            (
                TimeseriesRegressionLabelProcessor(),
                {"time": [1.0, 0.0], "values": [1.0, 2.0]},
            ),
            (
                TimeseriesRegressionLabelProcessor(),
                {"time": [0.0, 1.0], "values": [1.0, NAN]},
            ),
            (TimeseriesRegressionLabelProcessor(), {"values": [1.0]}),  # no time
            (
                TimeseriesRegressionLabelProcessor(),
                {"time": [0.0], "values": [[1.0, 2.0]]},
            ),
        ]
        for processor, records in bad:
            with self.subTest(records=records), self.assertRaises(ValueError):
                processor.process(records)

    def test_batches_are_packed(self):
        dataset = _dataset()
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        values, time, offsets = batch["label"]
        self.assertEqual(values.shape, (18, 1))
        self.assertEqual(time.shape, (18,))
        self.assertEqual(offsets.tolist(), [0, 5, 8, 14, 18])


class TestPerStepEndToEnd(unittest.TestCase):
    """The label processors with a per-step model and the Trainer.

    The loss over label records is covered in test_timeseries_modes.py.
    """

    def test_label_processors_train_and_evaluate(self):
        models: dict[str, Callable[[SampleDataset], BaseModel]] = {
            "LabelRecordModel": LabelRecordModel,
            "Unified_RNN": lambda dataset: Unified_RNN(
                dataset, embedding_dim={"dynamic": [{"values": 8}]}, hidden_dim=8
            ),
        }
        for (mode, metric), (name, make_model) in product(
            (("timeseries_binary", "roc_auc"), ("timeseries_regression", "mae")),
            models.items(),
        ):
            with self.subTest(mode=mode, model=name):
                dataset = _dataset(mode)
                model = make_model(dataset)
                self.assertEqual(model.mode, mode)
                batch = next(iter(get_dataloader(dataset, batch_size=4)))
                out = model(**batch)
                self.assertEqual(tuple(out["y_prob"].shape), (18, 1))
                self.assertTrue(torch.isfinite(out["loss"]))

                trainer = Trainer(model=model, metrics=[metric], enable_logging=False)
                loader = get_dataloader(dataset, batch_size=3)
                trainer.train(train_dataloader=loader, epochs=1)
                y_true, _, _ = trainer.inference(loader)
                self.assertEqual(sorted(len(y) for y in y_true), [3, 4, 5, 6])
                self.assertIn(metric, trainer.evaluate(loader))


if __name__ == "__main__":
    unittest.main()
