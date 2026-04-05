"""Tests for Daily and Sports Activities (DSA) dataset."""

import os
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
        "dsa_segments/segment_path",
        "dsa_segments/activity_name",
        "dsa_segments/activity_code",
    }
)

# Keys on each value in ``subject_data["activities"][activity_name]``.
ACTIVITY_RECORD_KEYS = frozenset({"id", "segments"})

# Keys on each entry in ``activity_data["segments"]`` (one time-series segment).
SEGMENT_RECORD_KEYS = frozenset(
    {
        "activity",
        "data",
        "file_path",
        "num_samples",
        "sampling_rate",
        "segment_filename",
        "sensors",
        "subject_id",
        "units",
    }
)

# ``configs/dsa.yaml`` — one IMU unit block is 9 consecutive channels; five blocks → 45.
EXPECTED_UNIT_KEYS_IN_ORDER = ("T", "RA", "LA", "RL", "LL")
EXPECTED_UNIT_DESCRIPTIONS_IN_ORDER = (
    "Torso",
    "Right Arm",
    "Left Arm",
    "Right Leg",
    "Left Leg",
)

EXPECTED_SENSOR_KEYS_IN_ORDER = (
    "xacc",
    "yacc",
    "zacc",
    "xgyro",
    "ygyro",
    "zgyro",
    "xmag",
    "ymag",
    "zmag",
)
EXPECTED_SENSOR_DESCRIPTIONS_IN_ORDER = (
    "X-axis Accelerometer",
    "Y-axis Accelerometer",
    "Z-axis Accelerometer",
    "X-axis Gyroscope",
    "Y-axis Gyroscope",
    "Z-axis Gyroscope",
    "X-axis Magnetometer",
    "Y-axis Magnetometer",
    "Z-axis Magnetometer",
)


def _single_key_dict(d: dict) -> tuple[str, str]:
    """Return (key, value) for a one-entry mapping; assert exactly one key."""
    if not isinstance(d, dict) or len(d) != 1:
        raise AssertionError(f"expected single-key dict, got {d!r}")
    key = next(iter(d))
    return key, str(d[key])


def _write_segment(path: Path, n_rows: int = 125, n_cols: int = 45) -> None:
    line = ",".join(["0.0"] * n_cols)
    path.write_text("\n".join([line] * n_rows) + "\n", encoding="utf-8")


class TestDSADataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.root_path = cls._tmpdir.name
        # Minimal tree; DSADataset creates dsa_manifest.csv if missing (see COVID).
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
        manifest = Path(self.root_path) / "dsa_manifest.csv"
        self.assertTrue(manifest.is_file())

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

    def test_nested_activity_and_segment_schema(self):
        """Every activity bucket and every segment dict exposes a stable key schema.

        The second axis of ``data`` is the 45 sensor channels (DSA layout).
        """
        subject_id = self.dataset.get_subject_ids()[0]
        subject_data = self.dataset.get_subject_data(subject_id)

        label_to_code = {
            name: code for code, name in self.dataset.label_mapping.items()
        }

        for activity_name, activity_data in subject_data["activities"].items():
            self.assertEqual(frozenset(activity_data.keys()), ACTIVITY_RECORD_KEYS)
            self.assertIsInstance(activity_data["id"], str)
            self.assertEqual(activity_data["id"], label_to_code[activity_name])
            self.assertIsInstance(activity_data["segments"], list)

            for seg in activity_data["segments"]:
                self.assertEqual(frozenset(seg.keys()), SEGMENT_RECORD_KEYS)
                self.assertEqual(seg["subject_id"], subject_id)
                self.assertEqual(seg["activity"], activity_name)
                self.assertEqual(seg["segment_filename"], seg["file_path"].name)
                self.assertIsInstance(seg["file_path"], Path)
                self.assertListEqual(seg["units"], self.dataset.units)
                self.assertListEqual(seg["sensors"], self.dataset.sensors)
                arr = seg["data"]
                self.assertIsInstance(arr, np.ndarray)
                self.assertEqual(arr.ndim, 2)
                self.assertEqual(arr.shape[1], 45, "45 channels per DSA segment row")
                self.assertEqual(arr.shape[0], seg["num_samples"])

    def test_sensor_and_unit_channel_metadata(self):
        """Sensors/units lists match YAML: 9 channels × 5 placements = 45 CSV columns.

        Column ``j`` in each segment row is channel ``j % 9`` on body unit ``j // 9``.
        """
        ds = self.dataset

        self.assertEqual(len(ds.units), len(EXPECTED_UNIT_KEYS_IN_ORDER))
        self.assertEqual(len(ds.sensors), len(EXPECTED_SENSOR_KEYS_IN_ORDER))

        unit_pairs = [_single_key_dict(u) for u in ds.units]
        self.assertEqual(
            tuple(k for k, _ in unit_pairs), EXPECTED_UNIT_KEYS_IN_ORDER
        )
        self.assertEqual(
            tuple(v for _, v in unit_pairs), EXPECTED_UNIT_DESCRIPTIONS_IN_ORDER
        )

        sensor_pairs = [_single_key_dict(s) for s in ds.sensors]
        self.assertEqual(
            tuple(k for k, _ in sensor_pairs), EXPECTED_SENSOR_KEYS_IN_ORDER
        )
        self.assertEqual(
            tuple(v for _, v in sensor_pairs), EXPECTED_SENSOR_DESCRIPTIONS_IN_ORDER
        )

        n_u, n_s = len(ds.units), len(ds.sensors)
        self.assertEqual(
            n_u * n_s,
            45,
            "DSA segment rows use 5 IMU placements × 9 channels",
        )

        unit_keys = [k for k, _ in unit_pairs]
        sensor_keys = [k for k, _ in sensor_pairs]
        # CSV column index ``c`` maps to unit ``c // 9`` and within-unit channel ``c % 9``.
        spots = tuple(
            (unit_keys[c // n_s], sensor_keys[c % n_s]) for c in (0, 8, 9, 17, 44)
        )
        self.assertEqual(
            spots,
            (
                ("T", "xacc"),
                ("T", "zmag"),
                ("RA", "xacc"),
                ("RA", "zmag"),
                ("LL", "zmag"),
            ),
        )

    def test_manifest_csv_columns(self):
        manifest = Path(self.root_path) / "dsa_manifest.csv"
        df = pd.read_csv(manifest)
        self.assertEqual(list(df.columns), list(EXPECTED_MANIFEST_COLUMNS))
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["subject_id"], "p1")
        self.assertEqual(row["activity_name"], "sitting")
        self.assertEqual(row["activity_code"], "A1")
        self.assertEqual(row["segment_path"], "a01/p1/s01.txt")

    def test_load_table_manifest_via_base_dataset(self):
        df = self.dataset.load_table("dsa_segments").compute()
        self.assertEqual(frozenset(df.columns), LOAD_TABLE_COLUMNS)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["patient_id"], "p1")
        self.assertEqual(row["event_type"], "dsa_segments")
        self.assertEqual(row["dsa_segments/activity_name"], "sitting")
        self.assertEqual(row["dsa_segments/activity_code"], "A1")
        self.assertEqual(row["dsa_segments/segment_path"], "a01/p1/s01.txt")
        self.assertTrue(pd.isna(row["timestamp"]))

    def test_data_consistency(self):
        self.dataset.get_subject_ids()
        self.assertIsNotNone(self.dataset._metadata)
        for _subject_id, subject_info in self.dataset._metadata["subjects"].items():
            for _activity_name, activity_info in subject_info["activities"].items():
                for segment_file in activity_info["segments"]:
                    file_path = os.path.join(activity_info["path"], segment_file)
                    with open(file_path, encoding="utf-8") as f:
                        line = f.readline()
                    self.assertEqual(len(line.strip().split(",")), 45)


if __name__ == "__main__":
    unittest.main()
