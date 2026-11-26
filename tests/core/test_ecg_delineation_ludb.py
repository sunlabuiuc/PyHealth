"""ECGDelineationLUDB tests on synthetic WFDB records (no real LUDB data)."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pyhealth.datasets import LUDBDataset
from pyhealth.datasets.ludb import load_ludb_record
from pyhealth.tasks import ECGDelineationLUDB
from pyhealth.tasks.ecg_delineation_ludb import WAVES, filter_ecg, segments_to_mask
from tests.core.test_ludb import HAS_WFDB, N_SAMPLES, write_synthetic_record

FS = 500


class TestSegmentsToMask(unittest.TestCase):
    def test_channels_and_inclusive_bounds(self):
        mask = segments_to_mask([("P", 1, 2), ("QRS", 4, 5), ("T", 7, 8)], 10)
        self.assertEqual(mask.shape, (len(WAVES), 10))
        self.assertEqual(mask.dtype, np.float32)
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [1, 2])
        self.assertEqual(np.flatnonzero(mask[1]).tolist(), [4, 5])
        self.assertEqual(np.flatnonzero(mask[2]).tolist(), [7, 8])

    def test_segment_past_the_end_is_clipped(self):
        mask = segments_to_mask([("T", 8, 20)], 10)
        self.assertEqual(np.flatnonzero(mask[2]).tolist(), [8, 9])

    def test_segment_before_the_start_is_ignored(self):
        mask = segments_to_mask([("P", -5, -3), ("P", -2, 1)], 10)
        self.assertEqual(np.flatnonzero(mask[0]).tolist(), [0, 1])

    def test_no_segments_gives_empty_mask(self):
        self.assertEqual(segments_to_mask([], 5).sum(), 0)


class TestFilterECG(unittest.TestCase):
    def setUp(self):
        t = np.arange(2 * FS) / FS
        self.beat = np.exp(-(((t - 1.0) / 0.02) ** 2))  # one narrow peak at 1 s
        self.offset = 2.0
        self.noise = 0.2 * np.sin(2 * np.pi * 200 * t)  # above the 150 Hz cut-off

    def test_none_returns_input(self):
        x = self.beat + self.offset
        np.testing.assert_allclose(filter_ecg(x, mode="none"), x, rtol=1e-6)

    def test_highpass_removes_offset_without_shifting_peak(self):
        y = filter_ecg(self.beat + self.offset, mode="highpass", sampling_rate=FS)
        self.assertLess(abs(np.median(y)), 0.1)
        self.assertEqual(int(np.argmax(y)), FS)  # zero phase: peak stays at 1 s

    def test_lowpass_removes_high_frequency_noise(self):
        y = filter_ecg(self.beat + self.noise, mode="lowpass", sampling_rate=FS)
        residual = np.abs(y - self.beat)[100:-100].max()
        self.assertLess(residual, 0.05)

    def test_bandpass_keeps_shape_and_dtype(self):
        x = np.stack([self.beat, self.beat + self.offset])
        y = filter_ecg(x, mode="bandpass", sampling_rate=FS)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float32)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            filter_ecg(self.beat, mode="notch")


class TestTaskArguments(unittest.TestCase):
    def test_schema(self):
        task = ECGDelineationLUDB()
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"mask": "tensor"})

    def test_single_lead_string(self):
        self.assertEqual(ECGDelineationLUDB(leads="ii").leads, ("ii",))

    def test_invalid_arguments_raise(self):
        bad = [
            {"filter_mode": "notch"},
            {"low_cut_hz": 0.0},
            {"low_cut_hz": 200.0, "high_cut_hz": 150.0},
            {"high_cut_hz": 300.0},
            {"filter_order": 0},
            {"leads": ["x"]},
            {"leads": []},
            {"leads": ["i", "i"]},
        ]
        for kwargs in bad:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                ECGDelineationLUDB(**kwargs)


@unittest.skipUnless(HAS_WFDB, "wfdb optional extra not installed")
class TestECGDelineationLUDB(unittest.TestCase):
    """Runs the task on two synthetic records: one with and one without P."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        root = cls.root = cls.tmp / "ludb"
        data_dir = root / "data"
        data_dir.mkdir(parents=True)
        write_synthetic_record(data_dir, "1", has_p=True)
        write_synthetic_record(data_dir, "2", has_p=False)
        pd.DataFrame(
            {
                "ID": [1, 2],
                "Sex": ["F", "M"],
                "Age": ["51", "64"],
                "Rhythms": ["Sinus rhythm", "Atrial fibrillation"],
                "Electric axis of the heart": [
                    "Electric axis of the heart: normal",
                    "Electric axis of the heart: left axis deviation",
                ],
            }
        ).to_csv(root / "ludb.csv", index=False)
        cls.dataset = LUDBDataset(root=str(root), cache_dir=str(cls.tmp / "cache"))
        # Pay the one-off costs (event table, SciPy import) here, not in a test.
        cls.dataset.get_patient("1")
        filter_ecg(np.zeros((1, N_SAMPLES), dtype=np.float32))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_default_task(self):
        self.assertIsInstance(self.dataset.default_task, ECGDelineationLUDB)
        self.assertEqual(self.dataset.default_task.filter_mode, "bandpass")

    def test_one_sample_per_lead(self):
        samples = ECGDelineationLUDB()(self.dataset.get_patient("1"))
        self.assertEqual(len(samples), 12)
        self.assertEqual(samples[1]["lead"], "ii")
        self.assertEqual(samples[0]["record_id"], "1")
        self.assertEqual(samples[0]["signal"].shape, (1, N_SAMPLES))
        self.assertEqual(samples[0]["mask"].shape, (3, N_SAMPLES))

    def test_mask_matches_annotations(self):
        # The synthetic beat is P 50-80, QRS 120-150, T 220-280.
        mask = ECGDelineationLUDB()(self.dataset.get_patient("1"))[0]["mask"]
        self.assertEqual(mask[0].sum(), 31)
        self.assertEqual(np.flatnonzero(mask[1])[[0, -1]].tolist(), [120, 150])
        self.assertEqual(np.flatnonzero(mask[2])[[0, -1]].tolist(), [220, 280])
        self.assertEqual((mask.sum(axis=0) > 1).sum(), 0)  # waves never overlap

    def test_record_without_p_wave(self):
        sample = ECGDelineationLUDB(leads=["v1"])(self.dataset.get_patient("2"))[0]
        self.assertEqual(sample["has_p_wave"], 0)
        self.assertEqual(sample["rhythm"], "Atrial fibrillation")
        self.assertEqual(sample["mask"][0].sum(), 0)
        self.assertEqual(sample["mask"][1].sum(), 31)  # QRS 120-150
        # The trailing T wave has no offset, so it ends at its peak (400-450).
        self.assertEqual(
            np.flatnonzero(sample["mask"][2])[[0, -1]].tolist(), [400, 450]
        )

    def test_filter_mode_changes_signal_not_mask(self):
        patient = self.dataset.get_patient("1")
        raw = ECGDelineationLUDB(filter_mode="none", leads=["i"])(patient)[0]
        band = ECGDelineationLUDB(filter_mode="bandpass", leads=["i"])(patient)[0]
        self.assertFalse(np.allclose(raw["signal"], band["signal"]))
        np.testing.assert_array_equal(raw["mask"], band["mask"])

    def test_normalize(self):
        patient = self.dataset.get_patient("1")
        z = ECGDelineationLUDB(leads=["ii"])(patient)[0]["signal"]
        self.assertAlmostEqual(float(z.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(z.std()), 1.0, places=3)
        raw = ECGDelineationLUDB(filter_mode="none", normalize=False, leads=["ii"])
        signal, _ = load_ludb_record(self.root / "data" / "1", ["ii"])
        np.testing.assert_allclose(raw(patient)[0]["signal"], signal, rtol=1e-6)

    def test_set_task_builds_tensors(self):
        samples = self.dataset.set_task(ECGDelineationLUDB(leads=["i", "ii"]))
        self.assertEqual(len(samples), 4)
        sample = samples[0]
        self.assertIsInstance(sample["signal"], torch.Tensor)
        self.assertEqual(tuple(sample["signal"].shape), (1, N_SAMPLES))
        self.assertEqual(tuple(sample["mask"].shape), (3, N_SAMPLES))
        self.assertIn("record_id", sample)
        self.assertIn("has_p_wave", sample)


if __name__ == "__main__":
    unittest.main()
