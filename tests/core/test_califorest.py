import unittest
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import CaliForest


class TestCaliForest(unittest.TestCase):
    """Test cases for the CaliForest model."""

    def setUp(self):
        """Set up synthetic data, dataset, and model."""
        self.samples = [
            {
                "patient_id": "patient-0",
                "visit_id": "visit-0",
                "features": [1.0, 2.0, 3.0, 4.0],
                "label": 0,
            },
            {
                "patient_id": "patient-1",
                "visit_id": "visit-1",
                "features": [2.0, 1.5, 0.5, 3.0],
                "label": 1,
            },
            {
                "patient_id": "patient-2",
                "visit_id": "visit-2",
                "features": [0.5, 0.7, 1.2, 1.8],
                "label": 0,
            },
            {
                "patient_id": "patient-3",
                "visit_id": "visit-3",
                "features": [3.1, 2.9, 4.0, 1.2],
                "label": 1,
            },
        ]

        self.input_schema = {"features": "tensor"}
        self.output_schema = {"label": "binary"}

        self.dataset = create_sample_dataset(
            samples=self.samples,
            input_schema=self.input_schema,
            output_schema=self.output_schema,
            dataset_name="califorest_test",
        )

        self.model = CaliForest(
            dataset=self.dataset,
            n_estimators=10,
            calibration="isotonic",
            random_state=42,
        )

        self.loader = get_dataloader(self.dataset, batch_size=4, shuffle=False)

    def test_model_initialization(self):
        """Test that the model initializes correctly."""
        self.assertIsInstance(self.model, CaliForest)
        self.assertEqual(self.model.n_estimators, 10)
        self.assertEqual(self.model.calibration, "isotonic")
        self.assertEqual(self.model.label_key, "label")
        self.assertFalse(self.model.is_fitted)

    def test_model_forward(self):
        """Test that forward pass works and returns expected keys."""
        batch = next(iter(self.loader))
        self.model.fit(self.loader)

        with torch.no_grad():
            ret = self.model(**batch)

        self.assertIn("loss", ret)
        self.assertIn("y_prob", ret)
        self.assertIn("y_true", ret)
        self.assertIn("logit", ret)

        self.assertEqual(ret["y_prob"].shape, (4, 1))
        self.assertEqual(ret["y_true"].shape, (4, 1))
        self.assertEqual(ret["logit"].shape, (4, 1))
        self.assertEqual(ret["loss"].dim(), 0)

    def test_probability_range(self):
        """Test that predicted probabilities are in [0, 1]."""
        batch = next(iter(self.loader))
        self.model.fit(self.loader)

        with torch.no_grad():
            ret = self.model(**batch)

        y_prob = ret["y_prob"]
        self.assertTrue(torch.all(y_prob >= 0.0).item())
        self.assertTrue(torch.all(y_prob <= 1.0).item())

    def test_forward_before_fit_raises(self):
        """Test that calling forward before fit raises a clear error."""
        batch = next(iter(self.loader))

        with self.assertRaises(RuntimeError):
            self.model(**batch)

    def test_logistic_calibration(self):
        """Test the logistic calibration option."""
        model = CaliForest(
            dataset=self.dataset,
            n_estimators=10,
            calibration="logistic",
            random_state=42,
        )

        batch = next(iter(self.loader))
        model.fit(self.loader)

        with torch.no_grad():
            ret = model(**batch)

        self.assertIn("y_prob", ret)
        self.assertEqual(ret["y_prob"].shape, (4, 1))

    def test_isotonic_and_logistic_differ(self):
        """Test that isotonic and logistic calibration produce different outputs."""
        iso_model = CaliForest(
            dataset=self.dataset,
            n_estimators=10,
            calibration="isotonic",
            random_state=42,
        )
        log_model = CaliForest(
            dataset=self.dataset,
            n_estimators=10,
            calibration="logistic",
            random_state=42,
        )

        iso_model.fit(self.loader)
        log_model.fit(self.loader)

        batch = next(iter(self.loader))

        with torch.no_grad():
            iso_probs = iso_model(**batch)["y_prob"]
            log_probs = log_model(**batch)["y_prob"]

        self.assertFalse(torch.allclose(iso_probs, log_probs))


if __name__ == "__main__":
    unittest.main()
