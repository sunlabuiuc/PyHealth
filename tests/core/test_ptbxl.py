"""Tests for the PTB-XL dataset and its multi-label classification task.

All tests use a tiny synthetic PTB-XL-like folder (5 ECGs from 4 patients)
written to a temporary directory; no real data is read.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from pyhealth.datasets import PTBXLDataset
from pyhealth.tasks import PTBXLMultilabelClassification
from pyhealth.tasks.ptbxl_multilabel_classification import read_wfdb_signal

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# Chosen to cover: likelihood filtering, super/sub-diagnostic aggregation, and
# the edge case where a record has rhythm codes only (ECG 5, no diagnostic label).
ECGS = [
    (1, 10, {"NORM": 100.0, "SR": 0.0}, 1),
    (2, 10, {"IMI": 80.0, "ABQRS": 0.0}, 1),
    (3, 11, {"LVH": 50.0, "SR": 0.0}, 9),
    (4, 12, {"NORM": 100.0}, 10),
    (5, 13, {"SR": 0.0}, 2),  # rhythm only: no diagnostic labels
]

STATEMENTS = pd.DataFrame(
    {
        "diagnostic": [1, 1, 1, 0, 0],
        "form": [0, 0, 0, 0, 1],
        "rhythm": [0, 0, 0, 1, 0],
        "diagnostic_class": ["NORM", "MI", "HYP", None, None],
        "diagnostic_subclass": ["NORM", "IMI", "LVH", None, None],
    },
    index=pd.Index(["NORM", "IMI", "LVH", "SR", "ABQRS"], name="scp_code"),
)


def write_record(path_without_ext: str, signal: np.ndarray) -> None:
    """Writes a 16-bit WFDB record (PTB-XL layout, gain 1000/mV)."""
    os.makedirs(os.path.dirname(path_without_ext), exist_ok=True)
    num_samples = signal.shape[0]
    digital = np.round(signal * 1000).astype("<i2")
    digital.tofile(path_without_ext + ".dat")
    name = os.path.basename(path_without_ext)
    lines = [f"{name} 12 100 {num_samples}"]
    for lead in LEADS:
        lines.append(f"{name}.dat 16 1000/mV 16 0 0 0 0 {lead}")
    with open(path_without_ext + ".hea", "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def build_synthetic_ptbxl(root: str) -> None:
    """Creates a minimal PTB-XL folder under ``root``."""
    rng = np.random.default_rng(0)
    rows = []
    for ecg_id, patient_id, codes, fold in ECGS:
        rel = f"records100/00000/{ecg_id:05d}_lr"
        write_record(os.path.join(root, rel), rng.normal(size=(1000, 12)))
        rows.append(
            {
                "ecg_id": ecg_id,
                "patient_id": float(patient_id),
                "age": 60.0,
                "sex": 1,
                "scp_codes": str(codes),
                "strat_fold": fold,
                "filename_lr": rel,
                "filename_hr": rel.replace("records100", "records500"),
            }
        )
    pd.DataFrame(rows).to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)
    STATEMENTS.to_csv(os.path.join(root, "scp_statements.csv"))


class TestPTBXL(unittest.TestCase):
    """Dataset, task and reader tests on synthetic PTB-XL files."""

    @classmethod
    def setUpClass(cls):
        cls.root = tempfile.mkdtemp()
        cls.cache = tempfile.mkdtemp()
        build_synthetic_ptbxl(cls.root)
        cls.dataset = PTBXLDataset(root=cls.root, cache_dir=cls.cache)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.root, ignore_errors=True)
        shutil.rmtree(cls.cache, ignore_errors=True)

    def test_metadata_file_and_label_strings(self):
        meta = pd.read_csv(os.path.join(self.root, "ptbxl-metadata-pyhealth.csv"))
        self.assertEqual(len(meta), 5)
        self.assertEqual(meta.loc[0, "patient_id"], 10)
        # IMI aggregates to the MI super-class and the IMI sub-class.
        second = meta.set_index("ecg_id").loc[2]
        self.assertEqual(second["labels_superdiagnostic"], "MI:80.0")
        self.assertEqual(second["labels_subdiagnostic"], "IMI:80.0")
        self.assertEqual(second["labels_form"], "ABQRS:0.0")
        self.assertEqual(second["labels_all"], "ABQRS:0.0;IMI:80.0")

    def test_missing_files_raise(self):
        with tempfile.TemporaryDirectory() as empty, self.assertRaises(FileNotFoundError):
            PTBXLDataset(root=empty)

    def test_patients_and_events(self):
        self.assertEqual(len(self.dataset.unique_patient_ids), 4)
        patient = self.dataset.get_patient("10")
        self.assertEqual(len(patient.get_events(event_type="ptbxl")), 2)

    def test_default_task(self):
        self.assertIsInstance(
            self.dataset.default_task, PTBXLMultilabelClassification
        )

    def test_task_samples_superdiagnostic(self):
        task = PTBXLMultilabelClassification(label_set="superdiagnostic")
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 4)  # ECG 5 has no diagnostic label
        item = samples[0]
        self.assertEqual(tuple(item["signal"].shape), (12, 1000))
        self.assertEqual(item["labels"].shape[0], 3)  # NORM, MI, HYP vocab
        self.assertAlmostEqual(float(item["signal"].mean()), 0.0, places=4)

    def test_task_rhythm_skips_unlabelled(self):
        task = PTBXLMultilabelClassification(label_set="rhythm")
        samples = self.dataset.set_task(task)
        self.assertEqual(len(samples), 3)  # ECGs 1, 3 and 5 have SR

    def test_parse_labels_likelihood_filter(self):
        parse = PTBXLMultilabelClassification._parse_labels
        self.assertEqual(parse("A:100.0;B:0.0", 0.0), ["A", "B"])
        self.assertEqual(parse("A:100.0;B:0.0", 50.0), ["A"])
        self.assertEqual(parse(None, 0.0), [])
        self.assertEqual(parse(float("nan"), 0.0), [])

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(label_set="nope")
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(sampling_rate=250)
        with self.assertRaises(ValueError):
            PTBXLMultilabelClassification(standardize="dataset")

    def test_read_wfdb_signal_matches_written_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            signal = np.linspace(-1.0, 1.0, 24).reshape(2, 12)
            path = os.path.join(tmp, "rec")
            write_record(path, signal)
            read = read_wfdb_signal(path)
        self.assertEqual(read.shape, (2, 12))
        np.testing.assert_allclose(read, signal, atol=1e-3)

    def test_read_wfdb_signal_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            read_wfdb_signal("/nonexistent/record")

    def test_prepare_signal_pads_and_standardizes(self):
        task = PTBXLMultilabelClassification(remove_baseline=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "short")
            write_record(path, np.random.default_rng(1).normal(size=(500, 12)))
            out = task._prepare_signal(path)
        self.assertEqual(out.shape, (12, 1000))
        self.assertEqual(out.dtype, np.float32)
        self.assertTrue(np.isfinite(out).all())


if __name__ == "__main__":
    unittest.main()
