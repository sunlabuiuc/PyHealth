"""Unit tests for PTBXLBinaryClassification.

The task is called on stand-in patients, so no dataset is built. The label
map is the synthetic fixture test-resources/core/ptbxl/scp_statements.csv
(not real PhysioNet data), and waveforms are generated at test time.

Authors:
    Bjorn Elvar (bthor5@illinois.edu), Rohan Vanjani
"""

import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from pyhealth.tasks import PTBXLBinaryClassification

try:
    import wfdb  # noqa: F401

    HAS_WFDB = True
except ImportError:
    HAS_WFDB = False

FIXTURE_ROOT = Path(__file__).resolve().parents[2] / "test-resources" / "core" / "ptbxl"
SCP_STATEMENTS = FIXTURE_ROOT / "scp_statements.csv"
N_SAMPLES = 250  # half a second at 500 Hz


def _write_wfdb(record_base: Path, n_samples: int = N_SAMPLES, fs: int = 500) -> None:
    """Writes a 12-lead WFDB record whose lead ``i`` is the ramp ``t + i`` uV."""
    record_base.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(n_samples, dtype=np.int16)[:, None] + np.arange(12, dtype=np.int16)
    Path(f"{record_base}.dat").write_bytes(data.tobytes())
    name = record_base.name
    lines = [f"{name} 12 {fs} {n_samples}"]
    lines += [f"{name}.dat 16 1000.0(0)/uV 16 0 0 0 0 {i}" for i in range(12)]
    Path(f"{record_base}.hea").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _patient(record_base: Path, scp_codes: str, sampling_rate: int = 500):
    """A stand-in for a PTBXLDataset patient with one ECG."""
    event = SimpleNamespace(
        record_id=1,
        scp_codes=scp_codes,
        strat_fold="3.0",
        signal_file=str(record_base),
        sampling_rate=sampling_rate,
    )
    return SimpleNamespace(patient_id="42", get_events=lambda event_type: [event])


class TestConfiguration(unittest.TestCase):
    def test_schema(self):
        task = PTBXLBinaryClassification(SCP_STATEMENTS)
        self.assertEqual(task.task_name, "PTBXLBinaryClassification")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "binary"})
        self.assertEqual(
            (task.target, task.sampling_rate, task.length), ("MI", 250, 2496)
        )

    def test_invalid_arguments(self):
        for kwargs in ({"target": "AFIB"}, {"sampling_rate": 0}, {"length": 0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                PTBXLBinaryClassification(SCP_STATEMENTS, **kwargs)

    def test_missing_scp_statements(self):
        task = PTBXLBinaryClassification("/nonexistent/scp_statements.csv")
        with self.assertRaises(FileNotFoundError):
            task(_patient(Path("/nonexistent/record"), "{'NORM': 100.0}"))


@unittest.skipUnless(HAS_WFDB, "needs the optional wfdb dependency")
class TestSamples(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        cls.record = cls.tmp / "00001_hr"
        _write_wfdb(cls.record)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def _sample(self, scp_codes="{'IMI': 80.0, 'SR': 0.0}", **kwargs):
        kwargs.setdefault("length", None)
        task = PTBXLBinaryClassification(SCP_STATEMENTS, **kwargs)
        (sample,) = task(_patient(self.record, scp_codes))
        return sample

    def test_sample_fields(self):
        sample = self._sample()
        self.assertEqual(
            sorted(sample), ["label", "patient_id", "record_id", "signal", "strat_fold"]
        )
        self.assertEqual(sample["patient_id"], "42")
        self.assertEqual(sample["record_id"], "1")
        self.assertEqual(sample["strat_fold"], 3)
        self.assertEqual(sample["signal"].dtype, np.float32)

    def test_label_per_target(self):
        cases = [
            ("{'IMI': 80.0, 'SR': 0.0}", "MI", 1),
            ("{'IMI': 0.0}", "MI", 1),  # likelihood 0 means unknown, not absent
            ("{'NORM': 100.0}", "MI", 0),
            ("{'NORM': 100.0}", "NORM", 1),
            ("{'IMI': 100.0, 'LVH': 80.0}", "HYP", 1),
            ("{'IMI': 100.0, 'LVH': 80.0}", "STTC", 0),
            ("{'PACE': 100.0}", "MI", 0),  # no diagnostic statement: a negative
            ("{}", "CD", 0),
        ]
        for scp_codes, target, expected in cases:
            with self.subTest(scp_codes=scp_codes, target=target):
                self.assertEqual(
                    self._sample(scp_codes, target=target)["label"], expected
                )

    def test_decimation_keeps_every_second_sample(self):
        signal = self._sample()["signal"]
        self.assertEqual(signal.shape, (12, N_SAMPLES // 2))
        ramp = np.arange(0, N_SAMPLES, 2, dtype=np.float32) / 1000  # uV to mV
        np.testing.assert_allclose(signal[0], ramp, atol=1e-6)
        np.testing.assert_allclose(signal[11], ramp + 0.011, atol=1e-6)

    def test_other_rates(self):
        self.assertEqual(
            self._sample(sampling_rate=500)["signal"].shape, (12, N_SAMPLES)
        )
        self.assertEqual(self._sample(sampling_rate=100)["signal"].shape, (12, 50))

    def test_rate_must_divide_dataset_rate(self):
        for rate in (300, 1000):
            with self.subTest(rate=rate), self.assertRaisesRegex(ValueError, "divide"):
                self._sample(sampling_rate=rate)

    def test_length_crops_and_pads(self):
        full = self._sample()["signal"]
        cropped = self._sample(length=100)["signal"]
        self.assertEqual(cropped.shape, (12, 100))
        np.testing.assert_array_equal(cropped, full[:, :100])
        padded = self._sample(length=200)["signal"]
        self.assertEqual(padded.shape, (12, 200))
        np.testing.assert_array_equal(padded[:, : N_SAMPLES // 2], full)
        self.assertTrue((padded[:, N_SAMPLES // 2 :] == 0).all())


if __name__ == "__main__":
    unittest.main()
