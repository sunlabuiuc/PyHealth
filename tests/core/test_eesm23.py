"""EESM23Dataset metadata, using a synthetic BIDS tree."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets.eesm23 import EESM23Dataset, _metadata_cache_dir
from pyhealth.tasks import SleepStagingEarEEG


class TestEESM23Dataset(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def _write_fixture(self, root: Path) -> None:
        for sub, ses, scored in [
            ("sub-001", "ses-001", True),
            ("sub-001", "ses-003", False),
            ("sub-002", "ses-001", True),
        ]:
            eeg_dir = root / sub / ses / "eeg"
            set_name = f"{sub}_{ses}_task-sleep_acq-earEEG_eeg.set"
            self._touch(eeg_dir / set_name)
            failures = f"{sub}_{ses}_task-sleep_acq-earEEGRecordingFailures_events.tsv"
            self._touch(eeg_dir / failures)
            if scored:
                scoring = f"{sub}_{ses}_task-sleep_acq-scoring_events.tsv"
                (eeg_dir / scoring).write_text(
                    "onset\tduration\tscoring_idx\tscoring\n0\t30\t1\tWake\n"
                )

    def test_prepare_metadata_marks_scored_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            ds = EESM23Dataset.__new__(EESM23Dataset)
            ds.root = str(root)
            ds.prepare_metadata()

            csv_path = root / "eesm23-pyhealth.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            self.assertEqual(len(df), 3)

            scored = df[df["session_id"] == "ses-001"].sort_values("patient_id")
            self.assertTrue(all(scored["has_scoring"] == "1"))
            self.assertTrue(
                all(scored["scoring_file"].str.endswith("_acq-scoring_events.tsv"))
            )

            unscored = df[df["session_id"] == "ses-003"].iloc[0]
            self.assertEqual(unscored["has_scoring"], "0")
            self.assertEqual(unscored["scoring_file"], "")

    def test_ids_come_from_file_names_not_the_root_path(self):
        # A root under ``sub-study/ses-pilot`` must not give every night the
        # same patient and session.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "sub-study" / "ses-pilot" / "ds005178"
            self._write_fixture(root)
            ds = EESM23Dataset.__new__(EESM23Dataset)
            ds.root = str(root)
            ds.prepare_metadata()
            df = pd.read_csv(root / "eesm23-pyhealth.csv", dtype=str)
            self.assertEqual(
                sorted(zip(df["patient_id"], df["session_id"])),
                [
                    ("sub-001", "ses-001"),
                    ("sub-001", "ses-003"),
                    ("sub-002", "ses-001"),
                ],
            )

    def test_file_without_bids_ids_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            self._touch(root / "sub-003" / "ses-001" / "eeg" / "x_acq-earEEG_eeg.set")
            ds = EESM23Dataset.__new__(EESM23Dataset)
            ds.root = str(root)
            ds.prepare_metadata()
            df = pd.read_csv(root / "eesm23-pyhealth.csv", dtype=str)
            self.assertNotIn("sub-003", set(df["patient_id"]))
            self.assertEqual(len(df), 3)

    def test_prepare_metadata_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            ds = EESM23Dataset.__new__(EESM23Dataset)
            ds.root = str(root)
            ds.prepare_metadata()
            csv_path = root / "eesm23-pyhealth.csv"
            first = csv_path.read_bytes()
            self._touch(root / "sub-002" / "ses-002" / "eeg" / "x.set")
            ds.prepare_metadata()
            self.assertEqual(csv_path.read_bytes(), first)

    def test_default_task_and_default_split(self):
        ds = EESM23Dataset.__new__(EESM23Dataset)
        self.assertIsInstance(ds.default_task, SleepStagingEarEEG)
        split = EESM23Dataset.default_subject_split()
        self.assertEqual([len(part) for part in split.subjects], [6, 2, 2])
        self.assertEqual(split.subjects[1], ("sub-007", "sub-008"))
        self.assertEqual(split.subjects[2], ("sub-009", "sub-010"))
        self.assertIn("sub-001", split.subjects[0])
        self.assertIn("sub-006", split.subjects[0])

    def test_home_cache_is_keyed_by_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            self._write_fixture(root_b)
            with patch("pyhealth.datasets.eesm23.Path.home", return_value=home):
                cache_a = _metadata_cache_dir(root_a)
                self.assertNotEqual(cache_a, _metadata_cache_dir(root_b))
                cache_a.mkdir(parents=True)
                (cache_a / "eesm23-pyhealth.csv").write_text("patient_id\nsub-999\n")
                ds = EESM23Dataset.__new__(EESM23Dataset)
                ds.root = str(root_b)
                ds.prepare_metadata()
            df = pd.read_csv(root_b / "eesm23-pyhealth.csv", dtype=str)
            self.assertEqual(set(df["patient_id"]), {"sub-001", "sub-002"})

    def test_prepare_metadata_empty_root_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = EESM23Dataset.__new__(EESM23Dataset)
            ds.root = tmp
            ds.prepare_metadata()
            self.assertFalse((Path(tmp) / "eesm23-pyhealth.csv").exists())


if __name__ == "__main__":
    unittest.main()
