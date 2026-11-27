"""CLIMBECGClassification and read_climb_ecg_splits, on synthetic Challenge files."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from scipy.io import savemat

from pyhealth.datasets import CardiologyDataset, ChapmanShaoxingDataset
from pyhealth.tasks import (
    CLIMB_ECG_CLASSES,
    CLIMBECGClassification,
    read_climb_ecg_splits,
)
from pyhealth.tasks.climb_ecg import ECG_JEPA_LEADS, ECG_LEADS

HEADER = "{name}.mat 16x1+24 1000.0(0)/mV 16 0 0 0 0 {lead}"


def write_record(folder, name, fs=500, n_samples=5000, signal=None, chapman=False):
    """A 12-lead Challenge record (MATLAB ``.mat`` + ``.hea``); lead i is i + 1."""
    folder.mkdir(parents=True, exist_ok=True)
    if signal is None:
        signal = np.tile(np.arange(1, 13, dtype=np.int16)[:, None], (1, n_samples))
    savemat(folder / f"{name}.mat", {"val": signal})
    lines = [f"{name} 12 {fs} {signal.shape[1]}"]
    lines += [HEADER.format(name=name, lead=lead) for lead in ECG_LEADS]
    sep = "#" if chapman else "# "
    lines += [f"{sep}Age: 60", f"{sep}Sex: Male", f"{sep}Dx: 426783006"]
    (folder / f"{name}.hea").write_text("\n".join(lines) + "\n")


def write_splits(path, rows):
    header = ",file_name," + ",".join(CLIMB_ECG_CLASSES) + ",patient,split,label\n"
    body = "".join(
        f"{i},x/{name}.hea,{','.join(str(int(c in classes)) for c in CLIMB_ECG_CLASSES)},{name},{split},0\n"
        for i, (name, split, classes) in enumerate(rows)
    )
    path.write_text(header + body)


class TestReadSplits(unittest.TestCase):
    def test_reads_split_and_classes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "Ga_splits.csv"
            write_splits(path, [("E00001", "train", ["cd", "hyp"]), ("E00002", "val", [])])
            self.assertEqual(
                read_climb_ecg_splits(path),
                {"E00001": ("train", ["cd", "hyp"]), "E00002": ("val", [])},
            )


class TestSignal(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _signal(self, task, **kwargs):
        write_record(self.tmp, "R", **kwargs)
        return task.signal(str(self.tmp / "R.mat"), kwargs.get("fs", 500))

    def test_default_is_eight_leads_of_2500_scaled_by_range(self):
        x = self._signal(CLIMBECGClassification({}))
        self.assertEqual(x.shape, (8, 2500))
        expected = np.array([ECG_LEADS.index(lead) + 1 for lead in ECG_JEPA_LEADS], float)
        np.testing.assert_allclose(x[:, 100], expected / (expected.max() - expected.min()), rtol=1e-5)

    def test_all_leads_at_500_hz_without_scaling(self):
        task = CLIMBECGClassification({}, num_samples=5000, leads=ECG_LEADS, normalize=False)
        x = self._signal(task)
        self.assertEqual(x.shape, (12, 5000))
        np.testing.assert_allclose(x[:, 0], np.arange(1, 13))

    def test_other_rates_are_resampled_and_short_records_padded(self):
        task = CLIMBECGClassification({}, num_samples=5000, normalize=False)
        x = self._signal(task, fs=1000, n_samples=6000)  # 6 s at 1 kHz
        self.assertEqual(x.shape, (8, 5000))
        np.testing.assert_allclose(x[0, 1000:2000], 1, atol=1e-3)
        np.testing.assert_allclose(x[:, 3100:], 0, atol=1e-3)  # padded after 6 s

    def test_records_starting_with_zeros_are_dropped(self):
        signal = np.ones((12, 5000), dtype=np.int16)
        signal[:, :15] = 0
        self.assertIsNone(self._signal(CLIMBECGClassification({}), signal=signal))
        signal[3, 14] = 5
        self.assertIsNotNone(self._signal(CLIMBECGClassification({}), signal=signal))

    def test_record_ids_with_an_extension_still_match(self):
        write_record(self.tmp, "E00001")
        event = type("Event", (), {
            "record_id": "E00001.mat", "event_type": "georgia",
            "attr_dict": {"signal_file": str(self.tmp / "E00001.mat"), "fs": 500},
        })()
        patient = type("Patient", (), {"patient_id": "E00001.mat", "get_events": lambda self: [event]})()
        samples = CLIMBECGClassification({"E00001": ("val", ["cd"])})(patient)
        self.assertEqual([s["record_id"] for s in samples], ["E00001"])

    def test_rejects_bad_settings(self):
        for kwargs in ({"duration": 0}, {"num_samples": 0}, {"leads": ["V7"]}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                CLIMBECGClassification({}, **kwargs)


class TestWithDatasets(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.tmp / "training"
        write_record(root / "ptb-xl" / "g1", "HR00001")
        write_record(root / "georgia" / "g1", "E00001")
        write_record(root / "georgia" / "g1", "E00002")  # not in the splits
        write_record(root / "chapman_shaoxing" / "g1", "JS00001", chapman=True)
        cls.labels = {
            "HR00001": ("train", ["normal", "other"]),
            "E00001": ("val", []),
            "JS00001": ("val", ["afib"]),
        }
        cls.root = root

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_cardiology_dataset(self):
        dataset = CardiologyDataset(
            root=str(self.root), sources=["ptb-xl", "georgia"], cache_dir=str(self.tmp / "c1")
        )
        samples = dataset.set_task(CLIMBECGClassification(self.labels))
        by_id = {samples[i]["record_id"]: samples[i] for i in range(len(samples))}
        self.assertEqual(sorted(by_id), ["E00001", "HR00001"])
        self.assertEqual(by_id["HR00001"]["split"], "train")
        self.assertEqual(by_id["HR00001"]["signal"].shape, (8, 2500))
        # Label columns follow CLIMB_ECG_CLASSES: normal and other.
        torch.testing.assert_close(by_id["HR00001"]["label"], torch.tensor([1.0, 0, 0, 0, 1, 0, 0]))
        torch.testing.assert_close(by_id["E00001"]["label"], torch.zeros(7))

    def test_chapman_shaoxing_dataset(self):
        dataset = ChapmanShaoxingDataset(
            root=str(self.root / "chapman_shaoxing"),
            metadata_cache_dir=str(self.tmp / "meta"),
            cache_dir=str(self.tmp / "c2"),
        )
        samples = dataset.set_task(CLIMBECGClassification(self.labels))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["record_id"], "JS00001")
        self.assertEqual(samples[0]["signal"].shape, (8, 2500))
        self.assertEqual(samples[0]["label"][CLIMB_ECG_CLASSES.index("afib")].item(), 1.0)


if __name__ == "__main__":
    unittest.main()
