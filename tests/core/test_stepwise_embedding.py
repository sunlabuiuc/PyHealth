from datetime import datetime, timedelta
import unittest

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import StepWiseEmbeddingModel
from pyhealth.tasks import StepWiseMortalityPredictionMIMICExtract


class TestStepWiseMortalityPredictionMIMICExtract(unittest.TestCase):
    """Task tests using a synthetic MIMIC-Extract style patient timeline."""

    def _build_patient(self) -> Patient:
        base_time = datetime(2020, 1, 1, 0, 0, 0)
        rows = [
            {
                "patient_id": "p-1",
                "timestamp": base_time,
                "event_type": "admissions",
                "admissions/visit_id": "stay-1",
                "admissions/hospital_expire_flag": 1,
            },
            {
                "patient_id": "p-1",
                "timestamp": base_time,
                "event_type": "vitals_labs",
                "vitals_labs/visit_id": "stay-1",
                "vitals_labs/hours_in": 0,
                "vitals_labs/code": "50824",
                "vitals_labs/mean": 138.0,
            },
            {
                "patient_id": "p-1",
                "timestamp": base_time,
                "event_type": "vitals_labs",
                "vitals_labs/visit_id": "stay-1",
                "vitals_labs/hours_in": 0,
                "vitals_labs/code": "50822",
                "vitals_labs/mean": 4.2,
            },
            {
                "patient_id": "p-1",
                "timestamp": base_time + timedelta(hours=1),
                "event_type": "vitals_labs",
                "vitals_labs/visit_id": "stay-1",
                "vitals_labs/hours_in": 1,
                "vitals_labs/code": "50824",
                "vitals_labs/mean": 139.5,
            },
            {
                "patient_id": "p-1",
                "timestamp": base_time + timedelta(hours=2),
                "event_type": "vitals_labs",
                "vitals_labs/visit_id": "stay-1",
                "vitals_labs/hours_in": 2,
                "vitals_labs/code": "50960",
                "vitals_labs/mean": 1.9,
            },
        ]
        return Patient(patient_id="p-1", data_source=pl.DataFrame(rows))

    def test_task_generates_stepwise_sample(self):
        task = StepWiseMortalityPredictionMIMICExtract(
            observation_window_hours=48,
            min_observed_steps=2,
        )

        samples = task(self._build_patient())
        self.assertEqual(len(samples), 1)

        sample = samples[0]
        self.assertEqual(sample["mortality"], 1)
        self.assertEqual(sample["visit_id"], "stay-1")
        self.assertEqual(sample["hours"], [0.0, 1.0, 2.0])
        self.assertEqual(len(sample["step_wise_inputs"]), 3)
        self.assertEqual(sample["step_wise_inputs"][0]["codes"], ["50822", "50824"])

    def test_task_applies_window(self):
        task = StepWiseMortalityPredictionMIMICExtract(observation_window_hours=2)
        samples = task(self._build_patient())
        self.assertEqual(samples[0]["hours"], [0.0, 1.0])


class TestStepWiseEmbeddingModel(unittest.TestCase):
    """Model tests using raw step-wise sample datasets."""

    def setUp(self):
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "hours": [0.0, 1.0, 3.0],
                "step_wise_inputs": [
                    {"codes": ["50822", "50824"], "values": [4.0, 138.0]},
                    {"codes": ["50824"], "values": [139.0]},
                    {"codes": ["50960"], "values": [1.8]},
                ],
                "mortality": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "hours": [0.0, 2.0],
                "step_wise_inputs": [
                    {"codes": ["50822"], "values": [5.1]},
                    {"codes": ["50824", "50960"], "values": [142.0, 2.4]},
                ],
                "mortality": 1,
            },
        ]
        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema={"step_wise_inputs": "raw", "hours": "raw"},
            output_schema={"mortality": "binary"},
            dataset_name="stepwise-test",
        )
        self.train_loader = get_dataloader(self.dataset, batch_size=2, shuffle=False)
        self.batch = next(iter(self.train_loader))

    def test_model_forward_grouped(self):
        model = StepWiseEmbeddingModel(dataset=self.dataset, group_mode="grouped")

        with torch.no_grad():
            outputs = model(**self.batch)

        self.assertIn("loss", outputs)
        self.assertIn("y_prob", outputs)
        self.assertIn("y_true", outputs)
        self.assertIn("logit", outputs)
        self.assertEqual(outputs["y_prob"].shape[0], 2)
        self.assertEqual(outputs["logit"].shape, (2, 1))
        self.assertEqual(outputs["loss"].dim(), 0)

    def test_model_forward_flat(self):
        model = StepWiseEmbeddingModel(dataset=self.dataset, group_mode="flat")
        batch = dict(self.batch)
        batch["embed"] = True

        with torch.no_grad():
            outputs = model(**batch)

        self.assertIn("embed", outputs)
        self.assertEqual(outputs["embed"].shape, (2, model.hidden_dim))

    def test_model_backward(self):
        model = StepWiseEmbeddingModel(dataset=self.dataset, group_mode="grouped")
        outputs = model(**self.batch)
        outputs["loss"].backward()

        has_gradient = any(
            param.grad is not None for param in model.parameters() if param.requires_grad
        )
        self.assertTrue(has_gradient)

    def test_model_lstm_forward(self):
        model = StepWiseEmbeddingModel(
            dataset=self.dataset,
            rnn_type="LSTM",
            hidden_dim=64,
        )
        with torch.no_grad():
            outputs = model(**self.batch)
        self.assertEqual(outputs["logit"].shape, (2, 1))
        self.assertEqual(model.hidden_dim, 64)

    def test_model_multilayer_configuration(self):
        model = StepWiseEmbeddingModel(
            dataset=self.dataset,
            embedding_dim=64,
            hidden_dim=32,
            num_layers=2,
            dropout=0.2,
            group_mode="grouped",
        )
        with torch.no_grad():
            outputs = model(**self.batch)
        self.assertEqual(model.embedding_dim, 64)
        self.assertEqual(model.hidden_dim, 32)
        self.assertEqual(model.num_layers, 2)
        self.assertEqual(outputs["y_prob"].shape, (2, 1))

    def test_invalid_group_mode_raises(self):
        with self.assertRaises(ValueError):
            StepWiseEmbeddingModel(dataset=self.dataset, group_mode="bad-mode")

    def test_invalid_rnn_type_raises(self):
        with self.assertRaises(ValueError):
            StepWiseEmbeddingModel(dataset=self.dataset, rnn_type="RNN")

    def test_end_to_end_optimization_step(self):
        model = StepWiseEmbeddingModel(dataset=self.dataset, group_mode="grouped")
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        optimizer.zero_grad()
        outputs = model(**self.batch)
        outputs["loss"].backward()
        optimizer.step()

        self.assertTrue(torch.isfinite(outputs["loss"]).all())

    def test_variable_length_batch_with_empty_step(self):
        samples = self.samples + [
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "hours": [0.0, 1.0, 4.0, 8.0],
                "step_wise_inputs": [
                    {"codes": ["50822"], "values": [3.9]},
                    {"codes": [], "values": []},
                    {"codes": ["50960", "50824"], "values": [2.0, 141.0]},
                    {"codes": ["50824"], "values": [140.5]},
                ],
                "mortality": 0,
            }
        ]
        dataset = create_sample_dataset(
            samples=samples,
            input_schema={"step_wise_inputs": "raw", "hours": "raw"},
            output_schema={"mortality": "binary"},
            dataset_name="stepwise-varlen",
        )
        batch = next(iter(get_dataloader(dataset, batch_size=3, shuffle=False)))
        model = StepWiseEmbeddingModel(dataset=dataset, group_mode="flat")

        with torch.no_grad():
            outputs = model(**batch)

        self.assertEqual(outputs["logit"].shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
