"""SHHSDataset metadata, using a synthetic NSRR ``polysomnography`` tree."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import SHHSDataset
from pyhealth.tasks import SleepStagingSHHS


def _write_recording(root: Path, visit: str, patient_id: str, *, annotation_dir=None):
    """Create an empty EDF and its Profusion XML under the NSRR layout."""
    edf_dir = root / "edfs" / visit
    edf_dir.mkdir(parents=True, exist_ok=True)
    record = f"{visit}-{patient_id}"
    (edf_dir / f"{record}.edf").write_bytes(b"")

    subdir = annotation_dir or visit
    ann_dir = root / "annotations-events-profusion" / subdir
    ann_dir.mkdir(parents=True, exist_ok=True)
    (ann_dir / f"{record}-profusion.xml").write_text("<PSGAnnotation/>")


class TestSHHSDatasetMetadata(unittest.TestCase):
    def _prepare(self, root: Path, visits=None) -> Path:
        """Run prepare_metadata without building the 2.0 event cache."""
        dataset = SHHSDataset.__new__(SHHSDataset)
        dataset.root = str(root)
        dataset.visits = list(visits or ("shhs1", "shhs2"))
        dataset.prepare_metadata()
        return root

    def test_pairs_edfs_with_profusion_annotations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = self._prepare(Path(tmp))
            csv_path = root / "shhs1-pyhealth.csv"
            self.assertFalse(csv_path.exists())

            _write_recording(root, "shhs1", "200001")
            self._prepare(root)

            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["patient_id"], "200001")
            self.assertEqual(row["visit"], "shhs1")
            self.assertEqual(row["record_id"], "shhs1-200001")
            self.assertTrue(row["signal_file"].endswith("shhs1-200001.edf"))
            self.assertTrue(
                row["annotation_file"].endswith("shhs1-200001-profusion.xml")
            )

    def test_shared_participant_id_across_visits(self):
        """Both nights of one person share patient_id so they stay one Patient."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs1", "200001")
            _write_recording(root, "shhs2", "200001")
            self._prepare(root)

            df = pd.read_csv(root / "shhs1-pyhealth.csv", dtype=str)
            df2 = pd.read_csv(root / "shhs2-pyhealth.csv", dtype=str)
            self.assertEqual(set(df["patient_id"]), {"200001"})
            self.assertEqual(set(df2["patient_id"]), {"200001"})
            self.assertEqual(set(df["record_id"]), {"shhs1-200001"})
            self.assertEqual(set(df2["record_id"]), {"shhs2-200001"})

    def test_shhs2_annotations_resolve_under_legacy_label_dir(self):
        """The 1.x loader looked for shhs2 under ``label/``; both are accepted."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs2", "200002", annotation_dir="label")
            with self.assertLogs("pyhealth.datasets.shhs", level="WARNING") as logs:
                self._prepare(root, visits=("shhs2",))
            self.assertTrue(any("label/" in message for message in logs.output))

            df = pd.read_csv(root / "shhs2-pyhealth.csv", dtype=str)
            self.assertEqual(len(df), 1)

    def test_recordings_without_annotation_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs1", "200001")
            (root / "edfs" / "shhs1" / "shhs1-200003.edf").write_bytes(b"")
            self._prepare(root, visits=("shhs1",))

            df = pd.read_csv(root / "shhs1-pyhealth.csv", dtype=str)
            self.assertEqual(set(df["patient_id"]), {"200001"})

    def test_unparsable_file_names_are_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs1", "200001")
            (root / "edfs" / "shhs1" / "notes.edf").write_bytes(b"")
            self._prepare(root, visits=("shhs1",))

            df = pd.read_csv(root / "shhs1-pyhealth.csv", dtype=str)
            self.assertEqual(len(df), 1)

    def test_prepare_metadata_is_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs1", "200001")
            self._prepare(root)
            csv_path = root / "shhs1-pyhealth.csv"
            first = csv_path.read_bytes()

            _write_recording(root, "shhs1", "200002")
            self._prepare(root)
            self.assertEqual(csv_path.read_bytes(), first)

    def test_only_requested_visits_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write_recording(root, "shhs1", "200001")
            _write_recording(root, "shhs2", "200001")
            self._prepare(root, visits=("shhs2",))

            self.assertFalse((root / "shhs1-pyhealth.csv").exists())
            self.assertTrue((root / "shhs2-pyhealth.csv").exists())

    def test_empty_root_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._prepare(root)
            self.assertEqual(list(root.glob("*.csv")), [])

    def test_unknown_visit_raises(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(ValueError):
            SHHSDataset(root=tmp, visits=("shhs3",))

    def test_empty_root_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(FileNotFoundError):
            SHHSDataset(root=tmp)

    def test_default_task_is_sleep_staging(self):
        dataset = SHHSDataset.__new__(SHHSDataset)
        self.assertIsInstance(dataset.default_task, SleepStagingSHHS)
        self.assertEqual(dataset.default_task.epoch_seconds, 30)


class TestSHHSModuleHelpers(unittest.TestCase):
    def test_parse_patient_id_reads_the_numeric_part(self):
        from pyhealth.datasets.shhs import parse_patient_id

        self.assertEqual(parse_patient_id("shhs1-200001.edf"), "200001")
        self.assertEqual(
            parse_patient_id(Path("/data/edfs/shhs2/shhs2-200001.edf")), "200001"
        )
        with self.assertRaises(ValueError):
            parse_patient_id("notes.edf")

    def test_read_sleep_stages_returns_integer_codes(self):
        from pyhealth.datasets.shhs import read_sleep_stages

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "shhs1-200001-profusion.xml"
            xml.write_text(
                "<PSGAnnotation><SleepStages>"
                "<SleepStage>0</SleepStage><SleepStage>2</SleepStage>"
                "<SleepStage>5</SleepStage><SleepStage>9</SleepStage>"
                "</SleepStages></PSGAnnotation>"
            )
            self.assertEqual(read_sleep_stages(xml), [0, 2, 5, 9])

    def test_read_sleep_stages_rejects_missing_stages(self):
        from pyhealth.datasets.shhs import read_sleep_stages

        with tempfile.TemporaryDirectory() as tmp:
            xml = Path(tmp) / "bad.xml"
            xml.write_text("<PSGAnnotation/>")
            with self.assertRaises(ValueError):
                read_sleep_stages(xml)

    def test_rem_is_a_valid_stage_and_movement_is_not(self):
        from pyhealth.datasets.shhs import VALID_SLEEP_STAGES

        self.assertIn(5, VALID_SLEEP_STAGES)
        self.assertNotIn(9, VALID_SLEEP_STAGES)


if __name__ == "__main__":
    unittest.main()
