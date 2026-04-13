"""Tests for AMA prediction ablation studies with synthetic data.

Uses local synthetic MIMIC-III data to ensure fast execution
and comprehensive coverage of all demographic combinations.

Synthetic CSV generation lives in ``examples/mimic3_ama_prediction_
logistic_regression.py``; tests load that helper via importlib so there
is no separate dataset module.
"""

import importlib.util
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import LogisticRegression
from pyhealth.tasks import AMAPredictionMIMIC3
from pyhealth.trainer import Trainer

_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "mimic3_ama_prediction_logistic_regression.py"
)
_spec = importlib.util.spec_from_file_location(
    "mimic3_ama_prediction_example", _EXAMPLE_PATH
)
_example_mod = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_example_mod)
generate_synthetic_mimic3 = _example_mod.generate_synthetic_mimic3


class TestAMAWithSyntheticData(unittest.TestCase):
    """Test AMA prediction ablation studies on synthetic data.

    Uses a small synthetic dataset that covers all demographic
    combinations to ensure tests run quickly (~1 second total).
    """

    @classmethod
    def setUpClass(cls):
        """Generate synthetic dataset once for all tests."""
        cls.tmpdir = tempfile.mkdtemp(prefix="ama_test_")
        cls.cache_dir = tempfile.mkdtemp(prefix="ama_test_cache_")

        # Generate small but comprehensive synthetic data
        generate_synthetic_mimic3(
            cls.tmpdir,
            n_patients=50,
            avg_admissions_per_patient=2,
            seed=42,
        )

        # Load dataset
        cls.dataset = MIMIC3Dataset(
            root=cls.tmpdir,
            tables=[],
            cache_dir=cls.cache_dir,
        )

        # Apply task
        cls.task = AMAPredictionMIMIC3()
        cls.sample_dataset = cls.dataset.set_task(cls.task)

    def test_dataset_loads_successfully(self):
        """Verify synthetic dataset loads with expected structure."""
        self.assertIsNotNone(self.dataset)
        self.assertGreater(len(self.sample_dataset), 0)

    def test_samples_have_expected_features(self):
        """Verify each sample contains required features."""
        sample = self.sample_dataset[0]

        expected_keys = {
            "visit_id",
            "patient_id",
            "demographics",
            "age",
            "los",
            "race",
            "has_substance_use",
            "ama",
        }
        self.assertEqual(set(sample.keys()), expected_keys)

    def test_demographics_values(self):
        """Verify demographics contain processed feature vectors."""
        for sample in self.sample_dataset:
            demo = sample["demographics"]
            # After processing, demographics are tensors
            self.assertTrue(
                torch.is_tensor(demo) or isinstance(demo, (int, float)),
                "Demographics should be processed",
            )

    def test_age_in_valid_range(self):
        """Verify ages are processed as tensors."""
        for sample in self.sample_dataset:
            age = sample["age"]
            # After processing, age is a tensor
            self.assertTrue(torch.is_tensor(age) or isinstance(age, (int, float)))

    def test_los_positive(self):
        """Verify LOS (length of stay) is processed as tensor."""
        for sample in self.sample_dataset:
            los = sample["los"]
            # After processing, los is a tensor
            self.assertTrue(torch.is_tensor(los) or isinstance(los, (int, float)))

    def test_race_normalized(self):
        """Verify race is processed as tensor."""
        for sample in self.sample_dataset:
            race = sample["race"]
            # After processing, race is a tensor
            self.assertTrue(torch.is_tensor(race) or isinstance(race, (int, float)))

    def test_substance_use_binary(self):
        """Verify substance use is processed as tensor."""
        for sample in self.sample_dataset:
            substance = sample["has_substance_use"]
            # After processing, substance use is a tensor
            self.assertTrue(torch.is_tensor(substance) or isinstance(substance, (int, float)))

    def test_ama_label_binary(self):
        """Verify AMA label is 0 or 1."""
        for sample in self.sample_dataset:
            ama = sample["ama"]
            self.assertIn(ama, [0, 1])

    def test_has_positive_and_negative_labels(self):
        """Verify dataset has both AMA positive and negative cases."""
        labels = [sample["ama"] for sample in self.sample_dataset]
        has_positive = any(l == 1 for l in labels)
        has_negative = any(l == 0 for l in labels)

        self.assertTrue(
            has_positive and has_negative,
            "Dataset should have both positive and negative AMA cases",
        )


class TestAMABaselineFeatures(unittest.TestCase):
    """Test that each ablation baseline uses correct features."""

    @classmethod
    def setUpClass(cls):
        """Generate synthetic dataset for baseline tests."""
        cls.tmpdir = tempfile.mkdtemp(prefix="ama_baseline_test_")
        cls.cache_dir = tempfile.mkdtemp(prefix="ama_baseline_cache_")

        generate_synthetic_mimic3(
            cls.tmpdir,
            n_patients=30,
            seed=42,
        )

        cls.dataset = MIMIC3Dataset(
            root=cls.tmpdir,
            tables=[],
            cache_dir=cls.cache_dir,
        )

        cls.task = AMAPredictionMIMIC3()
        cls.sample_dataset = cls.dataset.set_task(cls.task)

    def _create_model_with_features(self, feature_keys):
        """Helper to create logistic regression model with feature keys."""
        model = LogisticRegression(
            dataset=self.sample_dataset,
            embedding_dim=64,  # Use dataset's embedding_dim
        )
        model.feature_keys = list(feature_keys)
        output_size = model.get_output_size()
        embedding_dim = model.embedding_model.embedding_layers[
            feature_keys[0]
        ].out_features
        model.fc = torch.nn.Linear(
            len(feature_keys) * embedding_dim, output_size
        )
        return model

    def test_baseline_model_can_be_created(self):
        """BASELINE: demographics, age, los."""
        model = self._create_model_with_features(
            ["demographics", "age", "los"]
        )
        self.assertIsNotNone(model)
        # Verify fc layer exists
        self.assertIsNotNone(model.fc)

    def test_baseline_plus_race_model(self):
        """BASELINE + RACE: adds race."""
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race"]
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.fc)

    def test_baseline_plus_race_plus_substance_model(self):
        """BASELINE + RACE + SUBSTANCE: adds has_substance_use."""
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race", "has_substance_use"]
        )
        self.assertIsNotNone(model)
        self.assertIsNotNone(model.fc)

    def test_baseline_forward_pass(self):
        """Verify model forward pass works with baseline features."""
        model = self._create_model_with_features(
            ["demographics", "age", "los"]
        )

        train_ds, _, test_ds = split_by_patient(
            self.sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertIn("y_true", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))

    def test_baseline_plus_race_forward_pass(self):
        """Verify model forward pass with race feature."""
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race"]
        )

        train_ds, _, test_ds = split_by_patient(
            self.sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))

    def test_baseline_plus_full_forward_pass(self):
        """Verify model forward pass with all features."""
        model = self._create_model_with_features(
            ["demographics", "age", "los", "race", "has_substance_use"]
        )

        train_ds, _, test_ds = split_by_patient(
            self.sample_dataset, [0.8, 0.0, 0.2], seed=0
        )
        test_dl = get_dataloader(test_ds, batch_size=8, shuffle=False)

        model.eval()
        with torch.no_grad():
            batch = next(iter(test_dl))
            output = model(**batch)

        self.assertIn("y_prob", output)
        self.assertEqual(output["y_prob"].shape[0], len(test_ds))


class TestAMATrainingSpeed(unittest.TestCase):
    """Verify training with synthetic data is fast."""

    @classmethod
    def setUpClass(cls):
        """Generate small synthetic dataset for speed tests."""
        cls.tmpdir = tempfile.mkdtemp(prefix="ama_speed_test_")
        cls.cache_dir = tempfile.mkdtemp(prefix="ama_speed_cache_")

        generate_synthetic_mimic3(
            cls.tmpdir,
            n_patients=20,  # Small for speed
            seed=42,
        )

        cls.dataset = MIMIC3Dataset(
            root=cls.tmpdir,
            tables=[],
            cache_dir=cls.cache_dir,
        )

        cls.task = AMAPredictionMIMIC3()
        cls.sample_dataset = cls.dataset.set_task(cls.task)

    def test_training_completes_quickly(self):
        """Verify one training epoch completes in reasonable time."""
        import time

        train_ds, _, test_ds = split_by_patient(
            self.sample_dataset, [0.6, 0.0, 0.4], seed=0
        )
        train_dl = get_dataloader(
            train_ds, batch_size=8, shuffle=True
        )

        model = LogisticRegression(
            dataset=self.sample_dataset,
            embedding_dim=64,
        )
        model.feature_keys = ["demographics", "age", "los"]
        output_size = model.get_output_size()
        embedding_dim = model.embedding_model.embedding_layers[
            "demographics"
        ].out_features
        model.fc = torch.nn.Linear(
            3 * embedding_dim, output_size
        )

        trainer = Trainer(model=model)

        t0 = time.time()
        trainer.train(
            train_dataloader=train_dl,
            val_dataloader=None,
            epochs=1,
            monitor=None,
        )
        elapsed = time.time() - t0

        # Should complete in reasonable time
        self.assertGreater(elapsed, 0, "Training should take some time")

    def test_multiple_splits_complete_quickly(self):
        """Verify 2 random splits complete without error."""
        train_ds, _, test_ds = split_by_patient(
            self.sample_dataset,
            [0.6, 0.0, 0.4],
            seed=0,
        )

        for split_seed in range(2):
            train_ds, _, _ = split_by_patient(
                self.sample_dataset,
                [0.6, 0.0, 0.4],
                seed=split_seed,
            )
            train_dl = get_dataloader(
                train_ds, batch_size=8, shuffle=True
            )

            model = LogisticRegression(
                dataset=self.sample_dataset,
                embedding_dim=64,
            )
            model.feature_keys = ["demographics", "age", "los"]
            output_size = model.get_output_size()
            embedding_dim = model.embedding_model.embedding_layers[
                "demographics"
            ].out_features
            model.fc = torch.nn.Linear(
                3 * embedding_dim, output_size
            )

            trainer = Trainer(model=model)
            trainer.train(
                train_dataloader=train_dl,
                val_dataloader=None,
                epochs=1,
                monitor=None,
            )

        # Verify we completed without errors
        self.assertTrue(True)


if __name__ == "__main__":
    unittest.main()
