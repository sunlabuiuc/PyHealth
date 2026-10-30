import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.metrics import (
    timeseries_binary_metrics_fn,
    timeseries_regression_metrics_fn,
)
from pyhealth.models import RNN, MultimodalRNN
from pyhealth.processors import (
    LevelConfig,
    TimeseriesBinaryLabelProcessor,
    TimeseriesLabelProcessor,
    TimeseriesRegressionLabelProcessor,
)
from pyhealth.trainer import Trainer
from tests.core.label_record_model_utils import LabelRecordModel

DYNAMIC: list[LevelConfig] = [{"ordering": "series", "values": [2]}]


def _dataset(
    mode: str | type[TimeseriesLabelProcessor] = "timeseries_binary",
    labeled=lambda i, h: True,
):
    """Hourly dynamic series; labels at the hours ``labeled`` keeps."""
    rng = np.random.default_rng(0)
    samples = []
    for i, steps in enumerate([5, 3, 6, 4]):
        hours = [float(h) for h in range(steps)]
        kept = [h for h in range(steps) if labeled(i, h)]
        samples.append(
            {
                "patient_id": f"p{i}",
                "dynamic": {"time": hours, "values": rng.normal(size=(steps, 2))},
                "static": [float(i), 1.0],
                "label": {
                    "time": [float(h) for h in kept],
                    "values": [float((i + h) % 2) for h in kept],
                },
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={
            "dynamic": ("sequence_v2", {"levels": DYNAMIC}),
            "static": "tensor",
        },
        output_schema={"label": mode},
        dataset_name="timeseries_modes",
    )


def _model(dataset) -> LabelRecordModel:
    model = LabelRecordModel(dataset)
    model.eval()
    return model


class TestTimeseriesModes(unittest.TestCase):
    def test_modes_are_separate_from_binary_and_regression(self):
        for mode in ("timeseries_binary", "timeseries_regression"):
            self.assertEqual(_model(_dataset(mode)).mode, mode)

    def test_label_processor_class_resolves_to_its_own_mode(self):
        # The label processors subclass SequenceProcessorV2; the most specific
        # registered class decides the mode.
        for processor, mode in (
            (TimeseriesBinaryLabelProcessor, "timeseries_binary"),
            (TimeseriesRegressionLabelProcessor, "timeseries_regression"),
        ):
            with self.subTest(mode=mode):
                self.assertEqual(_model(_dataset(processor)).mode, mode)

    def test_one_prediction_per_label_record(self):
        # Sample 1 has no label at hour 0: it is left out, not masked.
        dataset = _dataset(labeled=lambda i, h: (i, h) != (1, 0))
        model = _model(dataset)
        batch = next(iter(get_dataloader(dataset, batch_size=4)))
        with torch.no_grad():
            out = model(**batch)
        values, time, offsets = out["y_true"]
        self.assertEqual(tuple(out["logit"].shape), (17, 1))  # 5 + 2 + 6 + 4
        self.assertEqual(offsets.tolist(), [0, 5, 7, 13, 17])
        self.assertEqual(time[5:7].tolist(), [1.0, 2.0])  # sample 1 from hour 1
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            out["logit"], values
        )
        torch.testing.assert_close(loss, out["loss"])

    def test_loss_needs_one_prediction_per_label(self):
        model = _model(_dataset())
        label = (torch.zeros(4, 1), torch.arange(4.0), torch.tensor([0, 4]))
        with self.assertRaisesRegex(ValueError, "one prediction per label record"):
            model.get_loss_function()(torch.zeros(1, 3, 1), label)

    def test_v1_rnns_reject_per_step_labels(self):
        rng = np.random.default_rng(0)
        t0 = datetime(2020, 1, 1)
        samples = [
            {
                "patient_id": f"p{i}",
                "dynamic": (
                    [t0 + timedelta(hours=h) for h in range(3)],
                    rng.normal(size=(3, 2)),
                ),
                "label": {"time": [0.0, 1.0, 2.0], "values": [0.0, 1.0, float(i)]},
            }
            for i in range(2)
        ]
        dataset = create_sample_dataset(
            samples, {"dynamic": "timeseries"}, {"label": "timeseries_binary"}
        )
        for model in (RNN, MultimodalRNN):
            with (
                self.subTest(model=model.__name__),
                self.assertRaisesRegex(ValueError, "does not support per-step"),
            ):
                model(dataset=dataset)

    def test_trainer_scores_label_records_per_sample(self):
        dataset = _dataset()
        trainer = Trainer(
            model=_model(dataset), metrics=["roc_auc"], enable_logging=False
        )
        loader = get_dataloader(dataset, batch_size=3)
        y_true, y_prob, _ = trainer.inference(loader)
        self.assertEqual(sorted(len(y) for y in y_true), [3, 4, 5, 6])
        self.assertEqual([len(p) for p in y_prob], [len(y) for y in y_true])
        self.assertIn("roc_auc", trainer.evaluate(loader))

        dataset = _dataset("timeseries_regression")
        trainer = Trainer(model=_model(dataset), metrics=["mae"], enable_logging=False)
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
