"""
Name: Hyunsoo Lee
NetId: hyunsoo2

Description: Test case for SHy model and DiagnosisPredictionMIMIC3 task
"""

import unittest
import torch
import polars as pl

from pyhealth.datasets import SampleDataset
from pyhealth.models import SHy
from pyhealth.tasks import DiagnosisPredictionMIMIC3

# ----------------------------------------------------------------------
# Helpers for the task test: fake Patient + Admissions
# ----------------------------------------------------------------------
class _SimpleAdmission:
    """Minimal admission object with only `hadm_id`."""
    def __init__(self, hadm_id):
        self.hadm_id = hadm_id


class _FakePatientMIMIC3:
    """Minimal fake Patient object to test DiagnosisPredictionMIMIC3."""

    def __init__(self, patient_id: str):
        self.patient_id = patient_id

        # Two admissions so we can form (current, next) pairs.
        self._admissions = [
            _SimpleAdmission(hadm_id=1),
            _SimpleAdmission(hadm_id=2),
        ]

    def get_events(self, event_type: str, filters=None, return_df: bool = False):
        """Mimic the interface used in DiagnosisPredictionMIMIC3.__call__."""

        if event_type == "admissions":
            # For admissions they expect a list of objects with .hadm_id
            return self._admissions

        # All non-admission events are returned as Polars DataFrames.
        # We only care about filtering by hadm_id == current_hadm_id.
        hadm_id = None
        if filters:
            for col, op, value in filters:
                if col == "hadm_id" and op == "==":
                    hadm_id = value

        # Minimal synthetic contents: we just ensure non-empty lists.
        if event_type == "diagnoses_icd":
            if hadm_id == 1:
                df = pl.DataFrame(
                    {"diagnoses_icd/icd9_code": ["41071", "25000"]}
                )
            else:
                df = pl.DataFrame(
                    {"diagnoses_icd/icd9_code": ["4280"]}
                )
            return df if return_df else df.to_dicts()

        if event_type == "procedures_icd":
            df = pl.DataFrame(
                {"procedures_icd/icd9_code": ["1234", "5678"]}
            )
            return df if return_df else df.to_dicts()

        if event_type == "prescriptions":
            df = pl.DataFrame(
                {"prescriptions/drug": ["DRUGA123", "DRUGB999"]}
            )
            return df if return_df else df.to_dicts()

        raise ValueError(f"Unknown event_type: {event_type}")


# ----------------------------------------------------------------------
# Tests for DiagnosisPredictionMIMIC3
# ----------------------------------------------------------------------
class TestDiagnosisPredictionMIMIC3(unittest.TestCase):
    def setUp(self):
        self.task = DiagnosisPredictionMIMIC3()
        self.patient = _FakePatientMIMIC3(patient_id="P1")

    def test_task_returns_non_empty_samples(self):
        samples = self.task(self.patient)
        self.assertGreater(len(samples), 0)

        sample = samples[0]
        # Check required keys
        for key in [
            "patient_id",
            "visit_id",
            "conditions",
            "procedures",
            "drugs",
            "label",
        ]:
            self.assertIn(key, sample)

        # Check simple type expectations
        self.assertEqual(sample["patient_id"], "P1")
        self.assertIsInstance(sample["conditions"], list)
        self.assertIsInstance(sample["label"], list)
        self.assertGreater(len(sample["conditions"]), 0)
        self.assertGreater(len(sample["label"]), 0)


# ----------------------------------------------------------------------
# Tests for SHy model: minimal SampleDataset + forward pass
# ----------------------------------------------------------------------
class TestSHyModel(unittest.TestCase):
    def setUp(self):
        # Build tiny synthetic samples that match the task schema.
        # IMPORTANT: "conditions" must be a flat sequence for SampleDataset.
        samples = [
            {
                "patient_id": "P1",
                "visit_id": "V1",
                "conditions": [1, 2, 3, 4, 5],
                "label": [1, 4],
            },
            {
                "patient_id": "P2",
                "visit_id": "V2",
                "conditions": [2, 3, 6],
                "label": [3],
            },
        ]

        self.dataset = SampleDataset(
            samples=samples,
            input_schema={"conditions": "sequence"},
            output_schema={"label": "multilabel"},
            dataset_name="toy_diagnosis",
            task_name="DiagnosisPredictionMIMIC3",
        )

    def test_forward_pass_shapes_and_keys(self):
        # Small model so tests are fast.
        model = SHy(
            dataset=self.dataset,
            feature_keys=["conditions"],
            label_key="label",
            mode="multilabel",
            embedding_dim=8,
            hgnn_dim=8,
            num_temporal_phenotypes=2,
            hgnn_layers=1,
            nhead=2,
            n_c=4,
            key_dim=8,
            sa_head=2,
        )

        # Build a minimal batch emulating what DataLoader would yield.
        # The model expects diagnosis sequences as LongTensor.
        batch = {
            "conditions": torch.randint(
                low=0,
                high=5,
                size=(2, 2, 3),  # (batch, num_visits, max_codes)
            ),
            "label": torch.randint(
                low=0,
                high=2,
                size=(2, 5),  # (batch, num_classes) — arbitrary small toy size
            ).float(),
        }

        outputs = model(**batch)

        # Check that expected keys exist
        for key in ["loss", "y_prob", "y_true", "logit"]:
            self.assertIn(key, outputs)

        # Shapes: batch size should match input batch size
        self.assertEqual(outputs["y_prob"].shape[0], 2)
        self.assertEqual(outputs["logit"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()