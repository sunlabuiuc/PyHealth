"""Fast synthetic unit tests for TCGACRCkDataset and TCGACRCkMSIClassification.

This test file uses only synthetic data generated inside
test-resources/tcga_crck during the test run.
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from pyhealth.datasets import TCGACRCkDataset
from pyhealth.tasks import TCGACRCkMSIClassification


class _SyntheticTCGACRCkData:
    """Shared helpers for a tiny synthetic TCGA-CRCk fixture."""

    @classmethod
    def setUpClass(cls):
        cls.test_dir = (
            Path(__file__).parent.parent.parent / "test-resources" / "tcga_crck"
        )
        cls.test_dir.mkdir(parents=True, exist_ok=True)

        cls._remove_raw_fixture()
        cls._build_raw_fixture()

        # Overwrite the CSV from the fresh synthetic raw files each run.
        TCGACRCkDataset.prepare_metadata(str(cls.test_dir))

    @classmethod
    def tearDownClass(cls):
        """Remove only synthetic image folders, keep root folder and CSV."""
        cls._remove_raw_fixture()

    @classmethod
    def _remove_raw_fixture(cls) -> None:
        """Deletes only raw image directories and keeps the metadata CSV."""
        for dirname in ["CRC_DX_TRAIN", "CRC_DX_TEST"]:
            dir_path = cls.test_dir / dirname
            if dir_path.exists():
                shutil.rmtree(dir_path, ignore_errors=True)

    @classmethod
    def _make_image(cls, path: Path) -> None:
        """Creates a tiny synthetic RGB PNG image."""
        path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(
            np.random.randint(0, 255, (8, 8, 3), dtype=np.uint8),
            mode="RGB",
        )
        image.save(path)

    @classmethod
    def _tile_filename(cls, random_prefix: str, slide_id: str) -> str:
        """Returns a tile filename matching the dataset regex convention."""
        return f"blk-{random_prefix}-{slide_id}.png"

    @classmethod
    def _build_raw_fixture(cls) -> None:
        """Builds a tiny synthetic raw TCGA-CRCk-style folder tree.

        Current dataset convention:
            1 -> MSIMUT
            0 -> MSS
        """
        # 3 patients total, 5 images total
        slide_specs = [
            (
                "TCGA-AA-0001",
                "TCGA-AA-0001-01Z-00-DX1",
                "test",
                0,
                ["YYNNMCRNWWEP"],
            ),
            (
                "TCGA-BB-0002",
                "TCGA-BB-0002-01Z-00-DX1",
                "train",
                0,
                ["YYIVRGPNDWWN"],
            ),
            (
                "TCGA-BB-0002",
                "TCGA-BB-0002-01Z-00-DX2",
                "train",
                0,
                ["YYKVVRTRMCKG"],
            ),
            (
                "TCGA-CC-0003",
                "TCGA-CC-0003-01Z-00-DX1",
                "train",
                1,
                ["WWKFIIMTFPSG"],
            ),
            (
                "TCGA-CC-0003",
                "TCGA-CC-0003-01Z-00-DX2",
                "train",
                1,
                ["WWMIEPLQMACE"],
            ),
        ]

        for _, slide_id, data_split, label, random_prefixes in slide_specs:
            split_dir = "CRC_DX_TRAIN" if data_split == "train" else "CRC_DX_TEST"
            label_dir = "MSIMUT" if label == 1 else "MSS"

            for random_prefix in random_prefixes:
                filename = cls._tile_filename(random_prefix, slide_id)
                image_path = cls.test_dir / split_dir / label_dir / filename
                cls._make_image(image_path)


class TestTCGACRCkDataset(_SyntheticTCGACRCkData, unittest.TestCase):
    """Fast test cases for TCGACRCkDataset."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = TCGACRCkDataset(root=str(cls.test_dir))

    def test_dataset_initialization(self):
        """Test that the dataset initializes correctly."""
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.dataset_name, "tcga_crck")

    def test_num_patients(self):
        """Test the number of unique synthetic patient IDs."""
        self.assertEqual(len(self.dataset.unique_patient_ids), 3)

    def test_get_patient_single_slide(self):
        """Test retrieving a patient record with a single slide."""
        patient = self.dataset.get_patient("TCGA-AA-0001")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "TCGA-AA-0001")

        events = patient.get_events(event_type="tcga_crck")
        self.assertEqual(len(events), 1)

        event = events[0]
        self.assertEqual(event["slide_id"], "TCGA-AA-0001-01Z-00-DX1")
        self.assertEqual(event["data_split"], "test")
        self.assertEqual(event["label"], "0")
        self.assertTrue(str(event["tile_path"]).endswith(".png"))

    def test_get_patient_multi_slide(self):
        """Test retrieving a patient record with multiple slides."""
        patient = self.dataset.get_patient("TCGA-BB-0002")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "TCGA-BB-0002")

        events = patient.get_events(event_type="tcga_crck")
        self.assertEqual(len(events), 2)

        slide_ids = {event["slide_id"] for event in events}
        self.assertEqual(
            slide_ids,
            {
                "TCGA-BB-0002-01Z-00-DX1",
                "TCGA-BB-0002-01Z-00-DX2",
            },
        )

    def test_event_fields_exist(self):
        """Test that event records contain the expected fields."""
        patient = self.dataset.get_patient("TCGA-CC-0003")
        events = patient.get_events(event_type="tcga_crck")

        self.assertGreater(len(events), 0)
        event = events[0]
        self.assertIn("slide_id", event)
        self.assertIn("tile_path", event)
        self.assertIn("tile_index", event)
        self.assertIn("data_split", event)
        self.assertIn("label", event)


class TestTCGACRCkMSIClassification(_SyntheticTCGACRCkData, unittest.TestCase):
    """Fast test cases for TCGACRCkMSIClassification."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.dataset = TCGACRCkDataset(root=str(cls.test_dir))
        cls.task = TCGACRCkMSIClassification(max_tiles=1)

    def test_default_task(self):
        """Test that the dataset exposes the default task."""
        self.assertIsInstance(self.dataset.default_task, TCGACRCkMSIClassification)

    def test_task_raw_output_single_slide_patient(self):
        """Test raw task output on a patient with one slide."""
        patient = self.dataset.get_patient("TCGA-AA-0001")
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)

        sample = samples[0]
        self.assertEqual(sample["patient_id"], "TCGA-AA-0001")
        self.assertEqual(sample["visit_id"], "TCGA-AA-0001")
        self.assertEqual(sample["label"], 0)

        tile_paths, tile_times = sample["tile_bag"]
        self.assertEqual(len(tile_paths), 1)
        self.assertEqual(len(tile_times), 1)
        self.assertEqual(tile_times, [0.0])

    def test_set_task_runs(self):
        """Single end-to-end smoke test for set_task()."""
        sample_ds = self.dataset.set_task(self.task)
        self.assertGreater(len(sample_ds), 0)


if __name__ == "__main__":
    unittest.main()