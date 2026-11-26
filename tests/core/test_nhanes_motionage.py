"""Integration tests for the NHANES MotionAge pipeline.

Contributors: Inoh Pak (inohpak2), Yejin Kim (yejink6)

Paper: Song et al., MotionAge: A Deep Learning Framework for Biological Age
    Prediction from Wearable Activity, MLHC 2026 (PMLR 340).
    https://proceedings.mlr.press/v340/song26a.html
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MotionAge
from pyhealth.tasks import NHANESMortalityPrediction


class TestNHANESMotionAge(unittest.TestCase):
    """Tests task features and MotionAge model integration."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        wear = np.full(2016, 5)
        wear[:1008] = 0
        wear[1008:1012] = [0, 1, 2, 5]

        epochs = pd.DataFrame(
            {
                "epoch": np.arange(1, 2017),
                "day": np.arange(2016) // 288 + 1,
                "hour": (np.arange(2016) // 12) % 24,
                "intensity": np.full(2016, 100, dtype=np.float32),
                "wear_minutes": wear,
                "wear_fraction": wear / 5,
            }
        )

        epochs.to_parquet(self.root / "activity.parquet")

        self.patients = []

        for label in (0, 1):
            event = Mock(
                eligstat=1,
                mortstat=label,
                permth_int=24 if label else 120,
                activity_file="activity.parquet",
                age=60,
                gender=1,
                bmi=25.0,
                systolic_bp=np.nan,
                diastolic_bp=80.0,
            )

            patient = Mock(patient_id=str(label))
            patient.get_events.return_value = [event]
            self.patients.append(patient)

    def make_task(self, **kwargs) -> NHANESMortalityPrediction:
        """Creates a MotionAge-compatible mortality task."""
        return NHANESMortalityPrediction(
            root=self.root,
            feature_format="motionage",
            **kwargs,
        )

    def test_temporal_alignment(self) -> None:
        """Filtered windows must retain their temporal alignment."""
        sample = self.make_task(wear_representation="binary")(self.patients[0])[0]

        self.assertEqual(sample["window_id"], 2)
        self.assertEqual(sample["hour"][0], 12)
        self.assertEqual(sample["day"][0], 3)
        self.assertEqual(sample["intensity"].shape, (1008,))

    def test_wear_representation(self) -> None:
        """Only pooling weights should differ between representations."""
        binary = self.make_task(wear_representation="binary")(self.patients[0])[0]

        fraction = self.make_task(wear_representation="fraction")(self.patients[0])[0]

        for key in ("intensity", "hour", "day"):
            np.testing.assert_array_equal(binary[key], fraction[key])

        np.testing.assert_array_equal(binary["wear_mask"][:4], [0, 1, 1, 1])

        np.testing.assert_allclose(fraction["wear_mask"][:4], [0, 0.2, 0.4, 1])

    def test_fraction_respects_threshold(self) -> None:
        """Invalid epochs must receive zero pooling weight."""
        sample = self.make_task(
            wear_representation="fraction",
            min_wear_minutes=2,
        )(self.patients[0])[0]

        np.testing.assert_allclose(sample["wear_mask"][:4], [0, 0, 0.4, 1])

    def test_static_features(self) -> None:
        """Static covariates preserve missing values."""
        sample = self.make_task(include_static=True)(self.patients[0])[0]

        self.assertEqual(sample["static"].shape, (5,))
        self.assertTrue(np.isnan(sample["static"][3]))

    def test_invalid_configuration(self) -> None:
        """Unsupported feature configurations should raise errors."""
        with self.assertRaises(ValueError):
            NHANESMortalityPrediction(
                root=self.root,
                feature_format="invalid",
            )

        with self.assertRaises(ValueError):
            self.make_task(include_wear_fraction=False)

    def test_model_forward_backward(self) -> None:
        """Task samples must support weighted training in MotionAge."""
        task = self.make_task(
            window_size=12,
            wear_representation="binary",
            pos_weight=2.0,
        )

        samples = [task(patient)[0] for patient in self.patients]

        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
        )

        loader = get_dataloader(
            dataset,
            batch_size=2,
            shuffle=False,
        )

        model = MotionAge(
            dataset=dataset,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
        )

        output = model(**next(iter(loader)))

        self.assertEqual(output["y_prob"].shape, (2, 1))
        self.assertTrue(torch.isfinite(output["loss"]).item())

        output["loss"].backward()

        gradients = [p.grad for p in model.parameters() if p.grad is not None]

        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all().item() for g in gradients))


if __name__ == "__main__":
    unittest.main()
