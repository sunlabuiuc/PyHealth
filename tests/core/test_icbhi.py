"""
Unit tests for ICBHIDataset and RespiratoryAbnormalityPredictionICBHI.

These tests intentionally bypass ``BaseDataset.__init__`` and
``dataset.set_task(...)`` — the PyHealth pipeline (parquet caching,
processor fitting, litdata chunk serialization) dominates runtime and is
integration-tested elsewhere. Here we test:

1. ``ICBHIDataset.prepare_metadata()`` — by calling it on a stub instance
   and inspecting the CSVs it writes.
2. ``RespiratoryAbnormalityPredictionICBHI.__call__`` — by invoking it on
   a mock Patient, which exercises the full sample-emission logic without
   touching PyHealth's event-store / processor machinery.
3. Signal-processing helpers (``_pad_or_trim``, ``_resample``).
4. Module constants (``_LABEL_MAP``, ``LABEL_NAMES``).
5. Cheap validation paths (invalid subset, invalid label_mode).

Author:
    Andrew Zhao (andrew.zhao@aeroseal.com)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    import scipy.io.wavfile

    scipy.io.wavfile.write(str(path), sample_rate, data)


def _make_fixture(root: Path) -> None:
    """Populate *root* with raw-ICBHI-shaped synthetic data."""
    sample_rate = 4000
    n_samples = sample_rate * 4
    audio = np.zeros(n_samples, dtype=np.int16)

    recordings = [
        ("101", "1b1", "Al", "sc", "Meditron", "train"),
        ("101", "2b1", "Ar", "sc", "Meditron", "train"),
        ("102", "1b1", "Ar", "sc", "Meditron", "test"),
    ]

    train_stems, test_stems = [], []
    for pid, rec, loc, mode, equip, split in recordings:
        stem = f"{pid}_{rec}_{loc}_{mode}_{equip}"
        _write_wav(root / f"{stem}.wav", sample_rate, audio)
        (root / f"{stem}.txt").write_text("0.0\t1.5\t0\t0\n1.5\t3.0\t1\t0\n")
        (train_stems if split == "train" else test_stems).append(stem)

    (root / "ICBHI_challenge_diagnosis.txt").write_text(
        "101 Healthy\n102 URTI\n"
    )
    (root / "ICBHI_Challenge_demographic_information.txt").write_text(
        "101 45 M 22.5 NA NA\n102 5 F NA 18.2 108.0\n"
    )
    split_dir = root / "ICBHI_challenge_train_and_test_txt"
    split_dir.mkdir()
    (split_dir / "ICBHI_challenge_train.txt").write_text(
        "\n".join(train_stems) + "\n"
    )
    (split_dir / "ICBHI_challenge_test.txt").write_text(
        "\n".join(test_stems) + "\n"
    )


# ---------------------------------------------------------------------------
# Mock Patient / Event objects — lets us exercise task.__call__ directly.
# ---------------------------------------------------------------------------

class _MockEvent:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _MockPatient:
    def __init__(self, patient_id, train=None, test=None):
        self.patient_id = patient_id
        self._events = {"train": train or [], "test": test or []}

    def get_events(self, split):
        return self._events.get(split, [])


def _mock_cycle_event(
    audio_path,
    cycle_id,
    cycle_start,
    cycle_end,
    has_crackles=0,
    has_wheezes=0,
    metadata_text="Patient 101; diagnosis: Healthy.",
):
    return _MockEvent(
        recording_id=Path(audio_path).stem,
        cycle_id=cycle_id,
        audio_path=str(audio_path),
        cycle_start=cycle_start,
        cycle_end=cycle_end,
        duration=cycle_end - cycle_start,
        has_crackles=has_crackles,
        has_wheezes=has_wheezes,
        metadata_text=metadata_text,
    )


# ---------------------------------------------------------------------------
# prepare_metadata() — call it on a stub instance, inspect the CSVs.
# ---------------------------------------------------------------------------

class TestICBHIPrepareMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import ICBHIDataset

        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmpdir.name)
        _make_fixture(cls.root)

        # Bypass BaseDataset.__init__ — we only want the parser.
        stub = ICBHIDataset.__new__(ICBHIDataset)
        stub.root = str(cls.root)
        stub.prepare_metadata()

        cls.train_df = pd.read_csv(cls.root / "icbhi-train-pyhealth.csv")
        cls.test_df = pd.read_csv(cls.root / "icbhi-test-pyhealth.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_train_csv_created(self):
        self.assertTrue((self.root / "icbhi-train-pyhealth.csv").exists())

    def test_test_csv_created(self):
        self.assertTrue((self.root / "icbhi-test-pyhealth.csv").exists())

    def test_train_row_count(self):
        # 2 recordings × 2 cycles each
        self.assertEqual(len(self.train_df), 4)

    def test_test_row_count(self):
        # 1 recording × 2 cycles
        self.assertEqual(len(self.test_df), 2)

    def test_patient_ids_train(self):
        self.assertEqual(set(self.train_df["patient_id"].astype(str)), {"101"})

    def test_patient_ids_test(self):
        self.assertEqual(set(self.test_df["patient_id"].astype(str)), {"102"})

    def test_columns(self):
        expected = {
            "patient_id", "recording_id", "cycle_id", "audio_path",
            "cycle_start", "cycle_end", "duration",
            "has_crackles", "has_wheezes", "diagnosis",
            "chest_location", "acquisition_mode", "equipment",
            "age", "sex", "adult_bmi", "child_weight", "child_height",
            "metadata_text",
        }
        self.assertEqual(set(self.train_df.columns), expected)

    def test_crackle_wheeze_values(self):
        # Each recording has Normal (0,0) then Crackle (1,0)
        self.assertEqual(set(self.train_df["has_crackles"]), {0, 1})
        self.assertEqual(set(self.train_df["has_wheezes"]), {0})

    def test_diagnosis_lookup(self):
        self.assertEqual(set(self.train_df["diagnosis"]), {"Healthy"})
        self.assertEqual(set(self.test_df["diagnosis"]), {"URTI"})

    def test_metadata_text_nonempty(self):
        for text in self.train_df["metadata_text"]:
            self.assertIsInstance(text, str)
            self.assertTrue(len(text) > 0)
            self.assertIn("Patient", text)
            self.assertIn("Healthy", text)

    def test_cycle_ids_zero_indexed(self):
        for _, group in self.train_df.groupby("recording_id"):
            self.assertEqual(sorted(group["cycle_id"].tolist()), [0, 1])


# ---------------------------------------------------------------------------
# Task __call__ — invoked against a mock Patient. No pipeline involved.
# ---------------------------------------------------------------------------

class TestRespiratoryAbnormalityPredictionICBHI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI

        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.wav = Path(cls.tmpdir.name) / "101_1b1_Al_sc_Meditron.wav"
        _write_wav(cls.wav, 4000, np.zeros(4000 * 4, dtype=np.int16))

        # Two cycles: Normal then Crackle.
        cls.patient = _MockPatient(
            "101",
            train=[
                _mock_cycle_event(cls.wav, 0, 0.0, 1.5, 0, 0),
                _mock_cycle_event(cls.wav, 1, 1.5, 3.0, 1, 0),
            ],
        )

        cls.task = RespiratoryAbnormalityPredictionICBHI(
            label_mode="any_abnormal", resample_rate=4000, target_length=2.0,
        )
        cls.samples = cls.task(cls.patient)

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_sample_count(self):
        self.assertEqual(len(self.samples), 2)

    def test_signal_shape(self):
        import torch

        sample = self.samples[0]
        self.assertIsInstance(sample["signal"], torch.Tensor)
        self.assertEqual(tuple(sample["signal"].shape), (1, 8000))

    def test_required_fields_present(self):
        required = {
            "patient_id", "recording_id", "cycle_id", "audio_path",
            "segment_start", "segment_end", "duration", "split",
            "signal", "label", "label_name", "metadata_text",
        }
        self.assertTrue(required.issubset(set(self.samples[0].keys())))

    def test_any_abnormal_labels(self):
        labels = [s["label"] for s in self.samples]
        self.assertEqual(labels, [0, 1])

    def test_any_abnormal_names(self):
        names = [s["label_name"] for s in self.samples]
        self.assertEqual(names, ["normal", "abnormal"])

    def test_split_is_train(self):
        self.assertTrue(all(s["split"] == "train" for s in self.samples))

    def test_cycle_id_matches_event(self):
        self.assertEqual([s["cycle_id"] for s in self.samples], [0, 1])

    def test_segment_bounds_match_event(self):
        self.assertEqual(self.samples[0]["segment_start"], 0.0)
        self.assertEqual(self.samples[0]["segment_end"], 1.5)
        self.assertEqual(self.samples[0]["duration"], 1.5)


class TestRespiratoryAbnormalityLabelModes(unittest.TestCase):
    """Each mode's label computation — cheap, no signal processing needed."""

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.wav = Path(cls.tmpdir.name) / "101_1b1_Al_sc_Meditron.wav"
        _write_wav(cls.wav, 4000, np.zeros(4000 * 2, dtype=np.int16))

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def _run(self, label_mode, cycle_flags):
        """Build a mock patient with the given (crackle, wheeze) flag pairs."""
        from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI

        events = [
            _mock_cycle_event(self.wav, i, 0.0, 1.0, c, w)
            for i, (c, w) in enumerate(cycle_flags)
        ]
        patient = _MockPatient("101", train=events)
        task = RespiratoryAbnormalityPredictionICBHI(
            label_mode=label_mode, resample_rate=4000, target_length=1.0,
        )
        return task(patient)

    def test_any_abnormal_mode(self):
        # (0,0) -> 0, (1,0) -> 1, (0,1) -> 1, (1,1) -> 1
        samples = self._run("any_abnormal", [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertEqual([s["label"] for s in samples], [0, 1, 1, 1])
        self.assertEqual(
            [s["label_name"] for s in samples],
            ["normal", "abnormal", "abnormal", "abnormal"],
        )

    def test_crackle_only_mode(self):
        # (0,0) -> 0, (1,0) -> 1, (0,1) -> 0, (1,1) -> 1
        samples = self._run("crackle_only", [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertEqual([s["label"] for s in samples], [0, 1, 0, 1])
        self.assertEqual(
            [s["label_name"] for s in samples],
            ["no_crackle", "crackle", "no_crackle", "crackle"],
        )

    def test_wheeze_only_mode(self):
        # (0,0) -> 0, (1,0) -> 0, (0,1) -> 1, (1,1) -> 1
        samples = self._run("wheeze_only", [(0, 0), (1, 0), (0, 1), (1, 1)])
        self.assertEqual([s["label"] for s in samples], [0, 0, 1, 1])
        self.assertEqual(
            [s["label_name"] for s in samples],
            ["no_wheeze", "no_wheeze", "wheeze", "wheeze"],
        )

    def test_invalid_label_mode_raises(self):
        from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI

        with self.assertRaises(ValueError):
            RespiratoryAbnormalityPredictionICBHI(label_mode="banana")


# ---------------------------------------------------------------------------
# Cheap validation — no filesystem.
# ---------------------------------------------------------------------------

class TestICBHIDatasetValidation(unittest.TestCase):
    def test_invalid_subset_raises(self):
        from pyhealth.datasets import ICBHIDataset

        # Raises before prepare_metadata() runs, so the root path is
        # never touched.
        with self.assertRaises(ValueError):
            ICBHIDataset(root="/does/not/exist", subset="bogus")

    def test_backward_compat_alias_is_class(self):
        from pyhealth.tasks import (
            ICBHIRespiratoryTask,
            RespiratoryAbnormalityPredictionICBHI,
        )

        self.assertIs(ICBHIRespiratoryTask, RespiratoryAbnormalityPredictionICBHI)


# ---------------------------------------------------------------------------
# Module constants — retained for reference / backward compatibility.
# ---------------------------------------------------------------------------

class TestICBHIRespiratoryTaskLabelMap(unittest.TestCase):
    def test_label_map_values(self):
        from pyhealth.tasks.icbhi_respiratory_classification import _LABEL_MAP

        self.assertEqual(_LABEL_MAP[(0, 0)], 0)  # Normal
        self.assertEqual(_LABEL_MAP[(1, 0)], 1)  # Crackle
        self.assertEqual(_LABEL_MAP[(0, 1)], 2)  # Wheeze
        self.assertEqual(_LABEL_MAP[(1, 1)], 3)  # Both

    def test_label_names_length(self):
        from pyhealth.tasks.icbhi_respiratory_classification import LABEL_NAMES

        self.assertEqual(len(LABEL_NAMES), 4)


# ---------------------------------------------------------------------------
# Signal helpers — pure numpy, fast.
# ---------------------------------------------------------------------------

class TestICBHIRespiratoryTaskSignalProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.tasks import RespiratoryAbnormalityPredictionICBHI

        cls.task = RespiratoryAbnormalityPredictionICBHI(
            resample_rate=4000, target_length=2.0,
        )

    def test_pad_short_signal(self):
        short = np.zeros(1000, dtype=np.float32)
        self.assertEqual(len(self.task._pad_or_trim(short)), 8000)

    def test_trim_long_signal(self):
        long_sig = np.zeros(20000, dtype=np.float32)
        self.assertEqual(len(self.task._pad_or_trim(long_sig)), 8000)

    def test_resample_identity(self):
        data = np.random.rand(8000).astype(np.float32)
        result = self.task._resample(data, 4000, 4000)
        np.testing.assert_array_equal(result, data)

    def test_resample_changes_length(self):
        data = np.zeros(8000, dtype=np.float32)
        result = self.task._resample(data, 8000, 4000)
        self.assertEqual(len(result), 4000)


if __name__ == "__main__":
    unittest.main()
