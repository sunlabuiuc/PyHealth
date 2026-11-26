"""CHBMITDataset metadata, using synthetic summary files and empty EDFs."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from pyhealth.datasets.chbmit import CHBMITDataset, _metadata_cache_dir
from pyhealth.tasks.seizure_detection_chbmit import SeizureDetectionCHBMIT


class TestCHBMITDataset(unittest.TestCase):
    def _touch(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")

    def _write_fixture(self, root: Path) -> None:
        (root / "SUBJECT-INFO").write_text(
            "Case\tGender\tAge (years)\nchb01\tF\t11\nchb02\tM\t 6\n"
        )
        chb01 = root / "chb01"
        self._touch(chb01 / "chb01_01.edf")
        self._touch(chb01 / "chb01_02.edf")  # on disk, absent from the summary
        self._touch(chb01 / "chb01_03.edf")
        (chb01 / "chb01-summary.txt").write_text(
            "\n".join(
                [
                    "Data Sampling Rate: 256 Hz",
                    "Channels in EDF Files:",
                    "Channel 1: FP1-F7",
                    "File Name: chb01_03.edf",
                    "Number of Seizures in File: 2",
                    "Seizure 1 Start Time: 2996 seconds",
                    "Seizure 1 End Time: 3036 seconds",
                    "Seizure 2 Start Time: 4000 seconds",
                    "Seizure 2 End Time: 4010 seconds",
                    "Channels changed:",
                    "Channel 1: F7-CS2",
                    "File Name: chb01_01.edf",
                    "Number of Seizures in File: 0",
                    "File Name: chb01_99.edf",
                    "Number of Seizures in File: 1",
                    "Seizure Start Time: 10 seconds",
                    "Seizure End Time: 20 seconds",
                    "Seizure Start Time: not-a-number seconds",
                    "",
                ]
            )
        )
        chb02 = root / "chb02"
        self._touch(chb02 / "chb02_01.edf")
        (chb02 / "chb02-summary.txt").write_text(
            "\n".join(
                [
                    "File Name: chb02_01.edf",
                    "Number of Seizures in File: 1",
                    "Seizure Start Time: 100 seconds",
                    "Seizure End Time: 140 seconds",
                    "",
                ]
            )
        )

    def test_prepare_metadata_parses_both_seizure_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            ds = CHBMITDataset.__new__(CHBMITDataset)
            ds.root = str(root)
            ds.prepare_metadata()

            csv_path = root / "chbmit-pyhealth.csv"
            self.assertTrue(csv_path.exists())
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            self.assertEqual(
                df["record_id"].tolist(),
                ["chb01_01", "chb01_02", "chb01_03", "chb02_01"],
            )
            # chb01_99 is named in the summary but has no EDF.
            self.assertNotIn("chb01_99", df["record_id"].tolist())

            row = df[df["record_id"] == "chb01_03"].iloc[0]
            self.assertEqual(row["patient_id"], "chb01")
            self.assertEqual(row["num_seizures"], "2")
            self.assertEqual(row["seizure_starts"], "2996;4000")
            self.assertEqual(row["seizure_ends"], "3036;4010")
            self.assertEqual(row["gender"], "F")
            self.assertEqual(row["age"], "11")
            self.assertTrue(row["signal_file"].endswith("chb01_03.edf"))

            background = df[df["record_id"] == "chb01_01"].iloc[0]
            self.assertEqual(background["num_seizures"], "0")
            self.assertEqual(background["seizure_starts"], "")

            # On disk but not in the summary: kept, with no seizures.
            extra = df[df["record_id"] == "chb01_02"].iloc[0]
            self.assertEqual(extra["num_seizures"], "0")

            chb02 = df[df["record_id"] == "chb02_01"].iloc[0]
            self.assertEqual(chb02["seizure_starts"], "100")
            self.assertEqual(chb02["seizure_ends"], "140")
            self.assertEqual(chb02["gender"], "M")
            self.assertEqual(chb02["age"], "6")

    def test_prepare_metadata_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_fixture(root)
            ds = CHBMITDataset.__new__(CHBMITDataset)
            ds.root = str(root)
            ds.prepare_metadata()
            csv_path = root / "chbmit-pyhealth.csv"
            first = csv_path.read_bytes()
            # A second call must not rewrite an existing CSV.
            (root / "chb02" / "chb02_02.edf").write_bytes(b"")
            ds.prepare_metadata()
            self.assertEqual(csv_path.read_bytes(), first)

    def test_default_task_and_default_split(self):
        ds = CHBMITDataset.__new__(CHBMITDataset)
        self.assertIsInstance(ds.default_task, SeizureDetectionCHBMIT)
        split = CHBMITDataset.default_subject_split()
        self.assertEqual([len(part) for part in split.subjects], [20, 2, 2])
        self.assertEqual(split.subjects[1], ("chb21", "chb22"))
        self.assertEqual(split.subjects[2], ("chb23", "chb24"))
        self.assertIn("chb01", split.subjects[0])
        self.assertIn("chb20", split.subjects[0])

    def test_home_cache_is_keyed_by_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp) / "home"
            root_a = Path(tmp) / "a"
            root_b = Path(tmp) / "b"
            root_a.mkdir()
            root_b.mkdir()
            self._write_fixture(root_b)
            with patch("pyhealth.datasets.chbmit.Path.home", return_value=home):
                cache_a = _metadata_cache_dir(root_a)
                self.assertNotEqual(cache_a, _metadata_cache_dir(root_b))
                # Metadata cached for root A must not stop root B from
                # writing its own.
                cache_a.mkdir(parents=True)
                (cache_a / "chbmit-pyhealth.csv").write_text("patient_id\nchbxx\n")
                ds = CHBMITDataset.__new__(CHBMITDataset)
                ds.root = str(root_b)
                ds.prepare_metadata()
            df = pd.read_csv(root_b / "chbmit-pyhealth.csv", dtype=str)
            self.assertEqual(set(df["patient_id"]), {"chb01", "chb02"})

    def test_prepare_metadata_empty_root_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            ds = CHBMITDataset.__new__(CHBMITDataset)
            ds.root = tmp
            ds.prepare_metadata()
            self.assertFalse((Path(tmp) / "chbmit-pyhealth.csv").exists())


if __name__ == "__main__":
    unittest.main()
