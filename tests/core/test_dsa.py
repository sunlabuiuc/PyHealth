"""Tests for Daily and Sports Activities (DSA) dataset."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyhealth.datasets import DSADataset

EXPECTED_MANIFEST_COLUMNS = (
    "subject_id",
    "activity_name",
    "activity_code",
    "segment_path",
)

LOAD_TABLE_COLUMNS = frozenset(
    {
        "patient_id",
        "event_type",
        "timestamp",
        "segments/segment_path",
        "segments/activity_name",
        "segments/activity_code",
    }
)

ACTIVITY_RECORD_KEYS = frozenset({"id", "segments"})

SEGMENT_RECORD_KEYS = frozenset(
    {
        "activity",
        "data",
        "file_path",
        "num_samples",
        "sampling_rate",
        "segment_filename",
        "subject_id",
    }
)

EXPECTED_UNIT_KEYS_IN_ORDER = ("T", "RA", "LA", "RL", "LL")
EXPECTED_SENSOR_KEYS_IN_ORDER = (
    "xacc", "yacc", "zacc",
    "xgyro", "ygyro", "zgyro",
    "xmag", "ymag", "zmag",
)


def _write_segment(path: Path, n_rows: int = 125, n_cols: int = 45) -> None:
    """Write a synthetic DSA segment file."""
    line = ",".join(["0.0"] * n_cols)
    path.write_text("\n".join([line] * n_rows) + "\n", encoding="utf-8")


def _make_minimal_dsa_tree(root: Path, activities=None, subjects=None, segments=None) -> Path:
    """Create minimal DSA directory structure with configurable layout."""
    activities = activities or ["a01"]
    subjects = subjects or ["p1"]
    segments = segments or ["s01.txt"]

    first_seg = None
    for activity in activities:
        for subject in subjects:
            for segment in segments:
                seg_dir = root / activity / subject
                seg_dir.mkdir(parents=True, exist_ok=True)
                seg_path = seg_dir / segment
                _write_segment(seg_path)
                if first_seg is None:
                    first_seg = seg_path
    return first_seg


class TestDSADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root_path = cls._tmpdir.name
        # Create minimal tree; DSADataset creates manifest if missing
        seg_dir = Path(cls.root_path) / "a01" / "p1"
        seg_dir.mkdir(parents=True)
        _write_segment(seg_dir / "s01.txt")

        cls.dataset = DSADataset(root=cls.root_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_dataset_initialization(self):
        self.assertIsNotNone(self.dataset)
        self.assertEqual(self.dataset.dataset_name, "dsa")
        self.assertIsNotNone(self.dataset.config)
        manifest = Path(self.root_path) / "dsa-pyhealth.csv"
        self.assertTrue(manifest.is_file())

    def test_config_attributes_loaded(self):
        """Verify loader attributes (SleepEDF-style constants) are populated."""
        ds = self.dataset
        self.assertIsInstance(ds.label_mapping, dict)
        self.assertIsInstance(ds.units, list)
        self.assertIsInstance(ds.sensors, list)
        self.assertEqual(ds.sampling_frequency, 25)
        self.assertEqual(ds._num_columns, 45)
        self.assertEqual(ds._num_rows, 125)

    def test_get_subject_ids(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertIsInstance(subject_ids, list)
        self.assertEqual(subject_ids, ["p1"])

    def test_get_activity_labels(self):
        activity_labels = self.dataset.get_activity_labels()
        self.assertIsInstance(activity_labels, dict)
        self.assertEqual(len(activity_labels), 19)
        self.assertEqual(activity_labels.get("sitting"), 0)

    def test_subject_data_loading(self):
        subject_ids = self.dataset.get_subject_ids()
        self.assertTrue(subject_ids)
        subject_id = subject_ids[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        self.assertIsInstance(subject_data, dict)
        self.assertEqual(subject_data["id"], subject_id)
        self.assertIn("activities", subject_data)
        self.assertIn("sitting", subject_data["activities"])

        activity_data = subject_data["activities"]["sitting"]
        self.assertIsInstance(activity_data["segments"], list)
        self.assertTrue(activity_data["segments"])

        segment = activity_data["segments"][0]
        self.assertIsInstance(segment["data"], np.ndarray)
        self.assertEqual(segment["sampling_rate"], 25)
        self.assertEqual(segment["data"].shape, (125, 45))

    def test_segment_schema(self):
        """Each segment dict exposes a stable key schema."""
        subject_id = self.dataset.get_subject_ids()[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        for activity_name, activity_data in subject_data["activities"].items():
            self.assertEqual(frozenset(activity_data.keys()), ACTIVITY_RECORD_KEYS)
            self.assertIsInstance(activity_data["id"], str)
            self.assertIsInstance(activity_data["segments"], list)

            for seg in activity_data["segments"]:
                self.assertEqual(frozenset(seg.keys()), SEGMENT_RECORD_KEYS)
                self.assertEqual(seg["subject_id"], subject_id)
                self.assertEqual(seg["activity"], activity_name)
                self.assertEqual(seg["segment_filename"], seg["file_path"].name)
                self.assertIsInstance(seg["file_path"], Path)
                arr = seg["data"]
                self.assertIsInstance(arr, np.ndarray)
                self.assertEqual(arr.ndim, 2)
                self.assertEqual(arr.shape[1], 45, "45 channels per DSA segment row")
                self.assertEqual(arr.shape[0], seg["num_samples"])

    def test_sensor_and_unit_channel_metadata(self):
        """Sensors/units lists match module metadata."""
        ds = self.dataset

        self.assertEqual(len(ds.units), len(EXPECTED_UNIT_KEYS_IN_ORDER))
        self.assertEqual(len(ds.sensors), len(EXPECTED_SENSOR_KEYS_IN_ORDER))

        unit_keys = [list(u.keys())[0] for u in ds.units]
        self.assertEqual(tuple(unit_keys), EXPECTED_UNIT_KEYS_IN_ORDER)

        sensor_keys = [list(s.keys())[0] for s in ds.sensors]
        self.assertEqual(tuple(sensor_keys), EXPECTED_SENSOR_KEYS_IN_ORDER)

    def test_manifest_csv_columns(self):
        manifest = Path(self.root_path) / "dsa-pyhealth.csv"
        df = pd.read_csv(manifest)
        self.assertEqual(list(df.columns), list(EXPECTED_MANIFEST_COLUMNS))
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["subject_id"], "p1")
        self.assertEqual(row["activity_name"], "sitting")
        self.assertEqual(row["activity_code"], "A01")
        self.assertEqual(row["segment_path"], "a01/p1/s01.txt")

    def test_load_table_manifest_via_base_dataset(self):
        df = self.dataset.load_table("segments").compute()
        self.assertEqual(frozenset(df.columns), LOAD_TABLE_COLUMNS)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["patient_id"], "p1")
        self.assertEqual(row["event_type"], "segments")
        self.assertEqual(row["segments/activity_name"], "sitting")
        self.assertEqual(row["segments/activity_code"], "A01")
        self.assertEqual(row["segments/segment_path"], "a01/p1/s01.txt")
        self.assertTrue(pd.isna(row["timestamp"]))

    def test_segment_raises_on_wrong_row_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            _write_segment(seg_path, n_rows=124, n_cols=45)
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "has 124 rows, expected 125"):
                ds.get_subject_data("p1")

    def test_segment_raises_on_non_numeric_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            bad_line = ",".join(["oops"] + ["0.0"] * 44)
            seg_path.write_text("\n".join([bad_line] * 125) + "\n", encoding="utf-8")
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "Failed to parse DSA segment"):
                ds.get_subject_data("p1")

    def test_segment_raises_on_non_finite_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            nan_line = ",".join(["nan"] + ["0.0"] * 44)
            seg_path.write_text("\n".join([nan_line] * 125) + "\n", encoding="utf-8")
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "contains non-finite values"):
                ds.get_subject_data("p1")

    def test_multiple_subjects_and_activities(self):
        """Manifest correctly scans multiple activities and subjects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _make_minimal_dsa_tree(
                Path(tmpdir),
                activities=["a01", "a02"],
                subjects=["p1", "p2"],
                segments=["s01.txt", "s02.txt"],
            )
            ds = DSADataset(root=tmpdir)

            subjects = ds.get_subject_ids()
            self.assertEqual(sorted(subjects), ["p1", "p2"])

            manifest = Path(tmpdir) / "dsa-pyhealth.csv"
            df = pd.read_csv(manifest)
            self.assertEqual(len(df), 8)  # 2 activities × 2 subjects × 2 segments

            # Check activity names are mapped correctly
            self.assertIn("sitting", df["activity_name"].values)
            self.assertIn("standing", df["activity_name"].values)
            self.assertIn("A01", df["activity_code"].values)
            self.assertIn("A02", df["activity_code"].values)

    def test_subject_not_found_raises(self):
        """Requesting data for a non-existent subject raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Subject 'nonexistent' not found"):
            self.dataset.get_subject_data("nonexistent")

    def test_manifest_raises_on_empty_directory(self):
        """Empty directory with no valid segments raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(ValueError, "No DSA segments under"):
                DSADataset(root=tmpdir)

    def test_segment_column_count_mismatch(self):
        """Segment with wrong number of columns raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            seg_path = _make_minimal_dsa_tree(Path(tmpdir))
            _write_segment(seg_path, n_rows=125, n_cols=44)
            ds = DSADataset(root=tmpdir)
            with self.assertRaisesRegex(ValueError, "has 44 columns, expected 45"):
                ds.get_subject_data("p1")

    def test_prepare_metadata_scans_standard_layout(self):
        """prepare_metadata finds segments using the built-in layout rules."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a03" / "p5").mkdir(parents=True)
            _write_segment(root / "a03" / "p5" / "s10.txt")

            manifest_path = root / "dsa-pyhealth.csv"
            config_path = Path(__file__).parent.parent.parent / "pyhealth" / "datasets" / "configs" / "dsa.yaml"

            DSADataset(root=str(root), config_path=str(config_path))
            df = pd.read_csv(manifest_path)
            self.assertEqual(len(df), 1)
            row = df.iloc[0]
            self.assertEqual(row["subject_id"], "p5")
            self.assertEqual(row["activity_code"], "A03")
            self.assertEqual(row["activity_name"], "lying_on_back")


if __name__ == "__main__":
    unittest.main()
