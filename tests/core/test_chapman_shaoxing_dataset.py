"""Tests for ChapmanShaoxingDataset on a tiny synthetic WFDB tree."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyhealth.datasets import ChapmanShaoxingDataset
from pyhealth.datasets.chapman_shaoxing import parse_chapman_header

LEADS = ("I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6")


def write_chapman_record(
    folder: Path,
    name: str,
    dx: str,
    age: str = "60",
    sex: str = "Male",
    n_samples: int = 50,
    with_mat: bool = True,
) -> None:
    """Write a WFDB header and a ``.mat`` (format 16, 24-byte offset)."""
    folder.mkdir(parents=True, exist_ok=True)
    lines = [f"{name} 12 500 {n_samples}"]
    for lead in LEADS:
        lines.append(f"{name}.mat 16+24 1000/mV 16 0 0 0 0 {lead}")
    lines += [f"#Age: {age}", f"#Sex: {sex}", f"#Dx: {dx}", "#Rx: Unknown"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")
    if with_mat:
        rng = np.random.default_rng(len(name) + n_samples)
        data = rng.integers(-500, 500, size=(n_samples, 12)).astype("<i2")
        (folder / f"{name}.mat").write_bytes(b"\0" * 24 + data.tobytes())


def build_tree(root: Path) -> None:
    """Four records; one without ``.mat`` and one beyond the CLIMB subset."""
    sub = root / "WFDBRecords" / "01" / "010"
    write_chapman_record(sub, "JS00001", "164889003,59118001,164934002", age="85")
    write_chapman_record(sub, "JS00002", "426783006", age="NaN", sex="Female")
    write_chapman_record(sub, "JS00004", "713427006", with_mat=False)
    write_chapman_record(
        root / "WFDBRecords" / "22" / "220", "JS20001", "426177001,713427006"
    )


class TestChapmanShaoxingDataset(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.root = self.tmp / "ecg-arrhythmia"
        build_tree(self.root)
        self.meta = self.tmp / "meta"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _dataset(self, **kwargs):
        return ChapmanShaoxingDataset(
            root=str(self.root),
            metadata_cache_dir=str(self.meta),
            cache_dir=str(self.tmp / "cache"),
            **kwargs,
        )

    def test_parse_header(self):
        row = parse_chapman_header(
            self.root / "WFDBRecords" / "01" / "010" / "JS00002.hea"
        )
        self.assertEqual(row["record_id"], "JS00002")
        self.assertEqual(row["sampling_rate"], 500)
        self.assertEqual(row["n_samples"], 50)
        self.assertEqual(row["age"], -1)
        self.assertEqual(row["sex"], "Female")
        self.assertEqual(row["dx_codes"], "426783006")

    def test_loads_records_and_writes_metadata_outside_root(self):
        ds = self._dataset()
        self.assertEqual(
            sorted(ds.unique_patient_ids), ["JS00001", "JS00002", "JS20001"]
        )
        csvs = list(self.meta.glob("chapman_shaoxing-pyhealth-full-*.csv"))
        self.assertEqual(len(csvs), 1)
        self.assertEqual(list(self.root.rglob("*.csv")), [])
        event = ds.get_patient("JS00001").get_events(event_type="records")[0]
        self.assertEqual(str(event.dx_codes), "164889003;59118001;164934002")
        self.assertTrue(str(event.signal_file).endswith("JS00001"))
        self.assertEqual(int(event.age), 85)

    def test_climb_subset_drops_late_records(self):
        ds = self._dataset(subset="climb")
        self.assertEqual(sorted(ds.unique_patient_ids), ["JS00001", "JS00002"])

    def test_bad_subset_and_missing_data(self):
        with self.assertRaises(ValueError):
            self._dataset(subset="nope")
        empty = self.tmp / "empty"
        empty.mkdir()
        with self.assertRaises(FileNotFoundError):
            ChapmanShaoxingDataset(root=str(empty), metadata_cache_dir=str(self.meta))

    def test_unmapped_codes_are_logged(self):
        with self.assertLogs("pyhealth.datasets.chapman_shaoxing", "INFO") as logs:
            self._dataset()
        self.assertTrue(any("713427006 (1)" in m for m in logs.output))

    def test_default_task(self):
        ds = self._dataset()
        self.assertEqual(
            ds.default_task.task_name, "ChapmanShaoxingSuperclassClassification"
        )


if __name__ == "__main__":
    unittest.main()
