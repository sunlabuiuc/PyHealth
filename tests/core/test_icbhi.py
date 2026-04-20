"""
Unit tests for ICBHIDataset and ICBHIRespiratoryTask.

The fixture drops raw ICBHI-shaped files into a temp directory and lets
ICBHIDataset.prepare_metadata() build the CSVs. This exercises the real
ingestion path rather than pre-baking cycle-level CSVs.

Author:
    Andrew Zhao (andrew.zhao@aeroseal.com)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    """Write a minimal WAV file without requiring scipy at import time."""
    import scipy.io.wavfile

    scipy.io.wavfile.write(str(path), sample_rate, data)


def _make_fixture(root: Path) -> None:
    """Populate *root* with raw-ICBHI-shaped synthetic data.

    Layout created:

    - 3 WAV files (2 for patient 101 train, 1 for patient 102 test)
    - Paired annotation .txt files (2 cycles each: Normal then Crackle)
    - ICBHI_challenge_diagnosis.txt (patient -> diagnosis)
    - ICBHI_challenge_train_and_test_txt/ICBHI_challenge_{train,test}.txt
    - ICBHI_Challenge_demographic_information.txt
    """
    sample_rate = 4000  # small for test speed
    n_samples = sample_rate * 4  # 4 second silent clip
    audio = np.zeros(n_samples, dtype=np.int16)

    recordings = [
        # (patient_id, rec_index, chest_location, mode, equipment, split)
        ("101", "1b1", "Al", "sc", "Meditron", "train"),
        ("101", "2b1", "Ar", "sc", "Meditron", "train"),
        ("102", "1b1", "Ar", "sc", "Meditron", "test"),
    ]

    train_stems, test_stems = [], []
    for pid, rec, loc, mode, equip, split in recordings:
        stem = f"{pid}_{rec}_{loc}_{mode}_{equip}"
        wav_path = root / f"{stem}.wav"
        txt_path = root / f"{stem}.txt"

        _write_wav(wav_path, sample_rate, audio)
        # Two cycles: Normal (0,0) then Crackle (1,0)
        txt_path.write_text("0.0\t1.5\t0\t0\n1.5\t3.0\t1\t0\n")

        if split == "train":
            train_stems.append(stem)
        else:
            test_stems.append(stem)

    # Diagnosis file (whitespace-separated per real ICBHI)
    (root / "ICBHI_challenge_diagnosis.txt").write_text(
        "101 Healthy\n102 URTI\n"
    )

    # Demographic file (real ICBHI has NA entries; test both paths)
    (root / "ICBHI_Challenge_demographic_information.txt").write_text(
        "101 45 M 22.5 NA NA\n102 5 F NA 18.2 108.0\n"
    )

    # Official split lists
    split_dir = root / "ICBHI_challenge_train_and_test_txt"
    split_dir.mkdir()
    (split_dir / "ICBHI_challenge_train.txt").write_text(
        "\n".join(train_stems) + "\n"
    )
    (split_dir / "ICBHI_challenge_test.txt").write_text(
        "\n".join(test_stems) + "\n"
    )


class TestICBHIDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import ICBHIDataset
        from pyhealth.tasks import ICBHIRespiratoryTask

        cls.data_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(cls.data_dir.name)
        _make_fixture(root)

        cls.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.dataset = ICBHIDataset(
            root=str(root),
            subset="both",
            cache_dir=cls.cache_dir.name,
        )
        cls.task = ICBHIRespiratoryTask(resample_rate=4000, target_length=2.0)
        cls.samples = cls.dataset.set_task(cls.task)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.samples.close()
        except Exception:
            pass
        cls.cache_dir.cleanup()
        cls.data_dir.cleanup()

    # -- Dataset-level tests --

    def test_num_patients(self):
        """Fixture has 2 unique patients."""
        self.assertEqual(len(self.dataset.unique_patient_ids), 2)

    def test_patient_ids(self):
        ids = set(self.dataset.unique_patient_ids)
        self.assertEqual(ids, {"101", "102"})

    def test_patient_101_has_four_cycle_events(self):
        """Patient 101 has 2 recordings × 2 cycles = 4 cycle events."""
        events = self.dataset.get_patient("101").get_events()
        self.assertEqual(len(events), 4)

    def test_patient_102_has_two_cycle_events(self):
        """Patient 102 has 1 recording × 2 cycles = 2 cycle events."""
        events = self.dataset.get_patient("102").get_events()
        self.assertEqual(len(events), 2)

    def test_event_has_audio_path(self):
        events = self.dataset.get_patient("101").get_events()
        self.assertIn("audio_path", events[0])

    def test_event_has_recording_id(self):
        events = self.dataset.get_patient("101").get_events()
        self.assertIn("recording_id", events[0])

    def test_event_has_cycle_boundaries(self):
        events = self.dataset.get_patient("101").get_events()
        self.assertIn("cycle_start", events[0])
        self.assertIn("cycle_end", events[0])
        self.assertIn("duration", events[0])

    def test_event_has_crackle_wheeze_flags(self):
        """Crackle/wheeze supervision must flow through events (RespLLM req)."""
        events = self.dataset.get_patient("101").get_events()
        self.assertIn("has_crackles", events[0])
        self.assertIn("has_wheezes", events[0])

    def test_event_crackle_wheeze_values(self):
        """Two cycles per recording: (0,0) Normal then (1,0) Crackle."""
        events = self.dataset.get_patient("101").get_events()
        crackles = {int(e["has_crackles"]) for e in events}
        wheezes = {int(e["has_wheezes"]) for e in events}
        self.assertEqual(crackles, {0, 1})
        self.assertEqual(wheezes, {0})

    def test_event_has_diagnosis(self):
        events = self.dataset.get_patient("101").get_events()
        self.assertEqual(events[0]["diagnosis"], "Healthy")

    def test_event_has_metadata_text(self):
        events = self.dataset.get_patient("101").get_events()
        text = events[0]["metadata_text"]
        self.assertIsInstance(text, str)
        self.assertTrue(len(text) > 0)
        # Real fields should appear; nothing should be invented.
        self.assertIn("101", text)
        self.assertIn("Healthy", text)

    def test_event_has_acquisition_fields(self):
        events = self.dataset.get_patient("101").get_events()
        self.assertIn("chest_location", events[0])
        self.assertIn("acquisition_mode", events[0])
        self.assertIn("equipment", events[0])

    def test_stats(self):
        self.dataset.stats()

    # -- Task-level tests --

    def test_sample_count(self):
        """3 recordings × 2 cycles each = 6 samples total."""
        self.assertEqual(len(self.samples), 6)

    def test_sample_has_signal(self):
        sample = self.samples[0]
        self.assertIn("signal", sample)

    def test_signal_shape(self):
        """Signal shape should be (1, target_length * resample_rate)."""
        import torch

        sample = self.samples[0]
        self.assertIsInstance(sample["signal"], torch.Tensor)
        # target_length=2.0, resample_rate=4000 → 8000 samples
        self.assertEqual(sample["signal"].shape, (1, 8000))

    def test_sample_has_label(self):
        sample = self.samples[0]
        self.assertIn("label", sample)

    def test_label_values(self):
        """Labels must be in {0, 1, 2, 3}."""
        for sample in self.samples:
            self.assertIn(int(sample["label"]), {0, 1, 2, 3})

    def test_normal_and_crackle_labels_present(self):
        """Each annotation file has one Normal (0) and one Crackle (1) cycle."""
        labels = {int(sample["label"]) for sample in self.samples}
        self.assertIn(0, labels)  # Normal
        self.assertIn(1, labels)  # Crackle

    def test_sample_has_patient_id(self):
        self.assertIn("patient_id", self.samples[0])

    def test_sample_has_record_id(self):
        self.assertIn("record_id", self.samples[0])

    def test_sample_has_split(self):
        self.assertIn("split", self.samples[0])

    def test_begin_end_times_present(self):
        sample = self.samples[0]
        self.assertIn("begin_time", sample)
        self.assertIn("end_time", sample)


class TestICBHIRespiratoryTaskSubsets(unittest.TestCase):
    """Test train-only and test-only subsets."""

    def setUp(self):
        self.data_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        root = Path(self.data_dir.name)
        _make_fixture(root)
        self.cache_dir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = root

    def tearDown(self):
        self.cache_dir.cleanup()
        self.data_dir.cleanup()

    def _load(self, subset: str):
        from pyhealth.datasets import ICBHIDataset
        from pyhealth.tasks import ICBHIRespiratoryTask

        ds = ICBHIDataset(
            root=str(self.root),
            subset=subset,
            cache_dir=self.cache_dir.name,
        )
        task = ICBHIRespiratoryTask(resample_rate=4000, target_length=2.0)
        return ds, ds.set_task(task)

    def _close(self, samples) -> None:
        try:
            samples.close()
        except Exception:
            pass

    def test_train_subset_patients(self):
        ds, samples = self._load("train")
        self.assertEqual(set(ds.unique_patient_ids), {"101"})
        self._close(samples)

    def test_test_subset_patients(self):
        ds, samples = self._load("test")
        self.assertEqual(set(ds.unique_patient_ids), {"102"})
        self._close(samples)

    def test_train_sample_count(self):
        """2 train recordings × 2 cycles = 4 samples."""
        ds, samples = self._load("train")
        self.assertEqual(len(samples), 4)
        self._close(samples)

    def test_invalid_subset_raises(self):
        from pyhealth.datasets import ICBHIDataset

        with self.assertRaises(ValueError):
            ICBHIDataset(root=str(self.root), subset="invalid")


class TestICBHIRespiratoryTaskLabelMap(unittest.TestCase):
    """Unit tests for the crackle/wheeze → integer label mapping."""

    def test_label_map_values(self):
        from pyhealth.tasks.icbhi_respiratory_classification import _LABEL_MAP

        self.assertEqual(_LABEL_MAP[(0, 0)], 0)  # Normal
        self.assertEqual(_LABEL_MAP[(1, 0)], 1)  # Crackle
        self.assertEqual(_LABEL_MAP[(0, 1)], 2)  # Wheeze
        self.assertEqual(_LABEL_MAP[(1, 1)], 3)  # Both

    def test_label_names_length(self):
        from pyhealth.tasks.icbhi_respiratory_classification import LABEL_NAMES

        self.assertEqual(len(LABEL_NAMES), 4)


class TestICBHIRespiratoryTaskSignalProcessing(unittest.TestCase):
    """Unit tests for audio helper methods."""

    def setUp(self):
        from pyhealth.tasks import ICBHIRespiratoryTask

        self.task = ICBHIRespiratoryTask(resample_rate=4000, target_length=2.0)

    def test_pad_short_signal(self):
        short = np.zeros(1000, dtype=np.float32)
        result = self.task._pad_or_trim(short)
        self.assertEqual(len(result), 4000 * 2)  # 8000

    def test_trim_long_signal(self):
        long = np.zeros(20000, dtype=np.float32)
        result = self.task._pad_or_trim(long)
        self.assertEqual(len(result), 8000)

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
