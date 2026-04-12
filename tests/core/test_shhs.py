import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
from pyhealth.datasets.shhs import SHHSDataset


class TestSHHSPrepareMetadata(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

        poly = Path(self.root) / "polysomnography"
        self.edf_dir1 = poly / "edfs" / "shhs1"
        self.edf_dir2 = poly / "edfs" / "shhs2"
        self.ann_dir1 = (
            poly / "annotations-events-profusion" / "shhs1"
        )
        self.ann_dir2 = (
            poly / "annotations-events-profusion" / "shhs2"
        )

        for d in (
            self.edf_dir1, self.edf_dir2,
            self.ann_dir1, self.ann_dir2,
        ):
            d.mkdir(parents=True)

        for nsrrid in ("200001", "200002", "200003"):
            (self.edf_dir1 / f"shhs1-{nsrrid}.edf").touch()
        for nsrrid in ("200001", "200002"):
            (self.ann_dir1 / f"shhs1-{nsrrid}-profusion.xml").touch()

        for nsrrid in ("200077", "200078"):
            (self.edf_dir2 / f"shhs2-{nsrrid}.edf").touch()
            (self.ann_dir2 / f"shhs2-{nsrrid}-profusion.xml").touch()

    def tearDown(self):
        self.tmpdir.cleanup()

    def _read_metadata(self) -> pd.DataFrame:
        return pd.read_csv(
            Path(self.root) / "shhs-metadata.csv",
            dtype={"patient_id": str},
        )

    def test_generates_metadata_csv(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        expected_cols = {
            "patient_id", "visitnumber", "signal_file",
            "annotation_file", "ecg_sample_rate",
        }
        self.assertTrue(expected_cols.issubset(set(df.columns)))

    def test_patient_ids_from_filenames(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        ids = set(df["patient_id"])
        self.assertEqual(
            ids, {"200001", "200002", "200077", "200078"}
        )

    def test_row_count_excludes_unmatched(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        self.assertEqual(len(df), 4)

    def test_visit_numbers(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        shhs1 = df[df["patient_id"].isin(["200001", "200002"])]
        shhs2 = df[df["patient_id"].isin(["200077", "200078"])]
        self.assertTrue((shhs1["visitnumber"] == 1).all())
        self.assertTrue((shhs2["visitnumber"] == 2).all())

    def test_ecg_sample_rates(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        for _, row in df.iterrows():
            expected = 125 if row["visitnumber"] == 1 else 256
            self.assertEqual(row["ecg_sample_rate"], expected)

    def test_signal_and_annotation_paths(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        for _, row in df.iterrows():
            self.assertTrue(Path(row["signal_file"]).exists())
            self.assertTrue(
                Path(row["annotation_file"]).exists()
            )

    def test_no_matched_pairs_raises(self):
        empty_root = tempfile.mkdtemp()
        poly = (
            Path(empty_root) / "polysomnography" / "edfs" / "shhs1"
        )
        poly.mkdir(parents=True)
        ann = (
            Path(empty_root)
            / "polysomnography"
            / "annotations-events-profusion"
            / "shhs1"
        )
        ann.mkdir(parents=True)
        (poly / "shhs1-999999.edf").touch()

        with self.assertRaises(FileNotFoundError):
            SHHSDataset.prepare_metadata(empty_root)

    def test_demographics_merged(self):
        datasets_dir = Path(self.root) / "datasets"
        datasets_dir.mkdir()
        demo = pd.DataFrame({
            "nsrrid": ["200001", "200002", "200077", "200078"],
            "visitnumber": [1, 1, 2, 2],
            "nsrr_age": [63.0, 55.0, 71.0, 68.0],
            "nsrr_sex": ["Male", "Female", "Female", "Male"],
            "nsrr_bmi": [28.5, 24.1, 25.1, 30.2],
            "nsrr_ahi_hp3r_aasm15": [12.3, 5.6, 8.7, 15.4],
        })
        demo.to_csv(
            datasets_dir / "shhs-harmonized-dataset-0.21.0.csv",
            index=False,
        )

        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()

        self.assertIn("age", df.columns)
        self.assertIn("sex", df.columns)
        self.assertIn("bmi", df.columns)
        self.assertIn("ahi", df.columns)

        row = df[df["patient_id"] == "200001"].iloc[0]
        self.assertAlmostEqual(row["age"], 63.0)
        self.assertEqual(row["sex"], "Male")
        self.assertAlmostEqual(row["bmi"], 28.5)

    def test_no_harmonized_csv_fills_none(self):
        SHHSDataset.prepare_metadata(self.root)
        df = self._read_metadata()
        for col in ("age", "sex", "bmi", "ahi"):
            self.assertIn(col, df.columns)
            self.assertTrue(df[col].isna().all())


class TestSHHSDataset(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_tmpdir = tempfile.TemporaryDirectory()
        self.root = self.tmpdir.name

        metadata = pd.DataFrame({
            "patient_id": ["200001", "200001", "200077"],
            "visitnumber": [1, 2, 2],
            "signal_file": [
                "/fake/shhs1-200001.edf",
                "/fake/shhs2-200001.edf",
                "/fake/shhs2-200077.edf",
            ],
            "annotation_file": [
                "/fake/shhs1-200001-profusion.xml",
                "/fake/shhs2-200001-profusion.xml",
                "/fake/shhs2-200077-profusion.xml",
            ],
            "ecg_sample_rate": [125, 256, 256],
            "age": [63.0, 63.0, 71.0],
            "sex": ["Male", "Male", "Female"],
            "bmi": [28.5, 28.5, 25.1],
            "ahi": [12.3, 12.3, 8.7],
        })
        metadata.to_csv(
            os.path.join(self.root, "shhs-metadata.csv"),
            index=False,
        )

        with patch(
            "pyhealth.datasets.base_dataset.platformdirs"
            ".user_cache_dir",
            return_value=self.cache_tmpdir.name,
        ):
            self.dataset = SHHSDataset(root=self.root)

    def tearDown(self):
        self.tmpdir.cleanup()
        self.cache_tmpdir.cleanup()

    def test_dataset_loads(self):
        self.assertIsInstance(self.dataset, SHHSDataset)

    def test_patient_ids(self):
        ids = set(self.dataset.unique_patient_ids)
        self.assertEqual(ids, {"200001", "200077"})

    def test_event_attributes(self):
        patient = self.dataset.get_patient("200001")
        events = patient.get_events(event_type="shhs_sleep")
        self.assertGreaterEqual(len(events), 1)

        event = events[0]
        for attr in (
            "visitnumber", "signal_file", "annotation_file",
            "ecg_sample_rate", "age", "sex", "bmi", "ahi",
        ):
            self.assertIn(attr, event, f"Missing: {attr}")

    def test_ecg_sample_rate_values(self):
        patient = self.dataset.get_patient("200001")
        events = patient.get_events(event_type="shhs_sleep")
        rates = {
            str(int(float(e["ecg_sample_rate"])))
            for e in events
        }
        self.assertTrue(rates.issubset({"125", "256"}))

    def test_default_task(self):
        from pyhealth.tasks.shhs_sleep_staging import (
            SleepStagingSHHS,
        )
        task = self.dataset.default_task
        self.assertIsInstance(task, SleepStagingSHHS)


if __name__ == "__main__":
    unittest.main()
