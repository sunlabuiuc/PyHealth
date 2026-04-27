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
    Andrew Zhao (NetID: aazhao2, aazhao2@illinois.edu)
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


def _write_wav(path: Path, sample_rate: int, data: np.ndarray) -> None:
    import scipy.io.wavfile

    scipy.io.wavfile.write(str(path), sample_rate, data)


def _make_fixture(
    root: Path,
    *,
    write_splits: bool = True,
    write_diagnosis: bool = True,
    write_demographics: bool = True,
) -> None:
    """Populate *root* with raw-ICBHI-shaped synthetic data.

    Optional flags mirror the real-world failure modes the loader has
    to handle: any of the sidecar metadata files may be absent in a
    bare ICBHI distribution, and the loader should still produce valid
    train/test CSVs.
    """
    sample_rate = 4000
    n_samples = sample_rate * 4
    audio = np.zeros(n_samples, dtype=np.int16)

    recordings = [
        ("101", "1b1", "Al", "sc", "Meditron", "train"),
        ("101", "2b1", "Ar", "sc", "Meditron", "train"),
        ("102", "1b1", "Ar", "sc", "Meditron", "test"),
    ]

    stems_by_split = {"train": [], "test": []}
    for pid, rec, loc, mode, equip, split in recordings:
        stem = f"{pid}_{rec}_{loc}_{mode}_{equip}"
        _write_wav(root / f"{stem}.wav", sample_rate, audio)
        (root / f"{stem}.txt").write_text("0.0\t1.5\t0\t0\n1.5\t3.0\t1\t0\n")
        stems_by_split[split].append(stem)

    if write_diagnosis:
        (root / "ICBHI_challenge_diagnosis.txt").write_text(
            "101\tHealthy\n102\tURTI\n"
        )
    if write_demographics:
        (root / "ICBHI_Challenge_demographic_information.txt").write_text(
            "101\t45\tM\t22.5\tNA\tNA\n102\t5\tF\tNA\t18.2\t108.0\n"
        )
    if write_splits:
        # Real ICBHI ships a single combined <stem>\t<split> file.
        lines = [f"{s}\ttrain" for s in stems_by_split["train"]] + [
            f"{s}\ttest" for s in stems_by_split["test"]
        ]
        (root / "ICBHI_challenge_train_test.txt").write_text(
            "\n".join(lines) + "\n"
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
# Real-ICBHI edge cases — missing sidecar files. Each test builds a bare
# fixture that omits one of the optional metadata files and asserts the
# loader still produces a valid pair of split CSVs with sensible defaults.
# ---------------------------------------------------------------------------

class TestICBHIMissingSplitFileFallback(unittest.TestCase):
    """No split file → deterministic patient-level fallback split."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import ICBHIDataset

        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmpdir.name)
        _make_fixture(cls.root, write_splits=False)

        stub = ICBHIDataset.__new__(ICBHIDataset)
        stub.root = str(cls.root)
        stub.prepare_metadata()

        # Either the data dir or the user cache may hold the CSV (the
        # loader falls back to the cache on permission errors). Try
        # both locations so the test passes on read-only tmpdirs too.
        cls.train_df = cls._read_csv(cls.root, "train")
        cls.test_df = cls._read_csv(cls.root, "test")

    @staticmethod
    def _read_csv(root: Path, split: str) -> pd.DataFrame:
        shared = root / f"icbhi-{split}-pyhealth.csv"
        if shared.exists():
            return pd.read_csv(shared)
        cache = Path.home() / ".cache" / "pyhealth" / "icbhi"
        return pd.read_csv(cache / f"icbhi-{split}-pyhealth.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_both_csvs_written(self):
        self.assertGreater(len(self.train_df), 0)
        self.assertGreater(len(self.test_df), 0)

    def test_patient_disjoint_across_splits(self):
        """A patient must appear in exactly one split — no leakage."""
        train_patients = set(self.train_df["patient_id"].astype(str))
        test_patients = set(self.test_df["patient_id"].astype(str))
        self.assertEqual(train_patients & test_patients, set())

    def test_all_recordings_assigned(self):
        """Every WAV stem should land in exactly one split."""
        all_recs = set(self.train_df["recording_id"]) | set(
            self.test_df["recording_id"]
        )
        self.assertEqual(
            all_recs,
            {
                "101_1b1_Al_sc_Meditron",
                "101_2b1_Ar_sc_Meditron",
                "102_1b1_Ar_sc_Meditron",
            },
        )

    def test_deterministic_across_runs(self):
        """Same fixture + same seed → same split assignment."""
        from pyhealth.datasets import ICBHIDataset

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d2:
            root2 = Path(d2)
            _make_fixture(root2, write_splits=False)
            stub2 = ICBHIDataset.__new__(ICBHIDataset)
            stub2.root = str(root2)
            stub2.prepare_metadata()
            train2 = self._read_csv(root2, "train")
        self.assertEqual(
            sorted(set(self.train_df["recording_id"])),
            sorted(set(train2["recording_id"])),
        )


class TestICBHIMissingDiagnosisFile(unittest.TestCase):
    """Missing diagnosis file → diagnosis defaults to 'Unknown'."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import ICBHIDataset

        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmpdir.name)
        _make_fixture(cls.root, write_diagnosis=False)

        stub = ICBHIDataset.__new__(ICBHIDataset)
        stub.root = str(cls.root)
        stub.prepare_metadata()

        cls.train_df = pd.read_csv(cls.root / "icbhi-train-pyhealth.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_diagnosis_defaults_to_unknown(self):
        self.assertEqual(set(self.train_df["diagnosis"]), {"Unknown"})

    def test_metadata_text_omits_unknown_diagnosis(self):
        # _build_metadata_text skips diagnosis when it's "Unknown" —
        # never fabricate a placeholder string.
        for text in self.train_df["metadata_text"]:
            self.assertNotIn("Unknown", text)


class TestICBHIMissingDemographicsFile(unittest.TestCase):
    """Missing demographics file → age=NaN, sex='', BMI/weight/height NaN."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import ICBHIDataset

        cls.tmpdir = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.root = Path(cls.tmpdir.name)
        _make_fixture(cls.root, write_demographics=False)

        stub = ICBHIDataset.__new__(ICBHIDataset)
        stub.root = str(cls.root)
        stub.prepare_metadata()

        cls.train_df = pd.read_csv(cls.root / "icbhi-train-pyhealth.csv")

    @classmethod
    def tearDownClass(cls):
        cls.tmpdir.cleanup()

    def test_age_is_nan(self):
        self.assertTrue(self.train_df["age"].isna().all())

    def test_sex_is_empty(self):
        # pandas reads "" as NaN for object columns, which is fine.
        sex_values = self.train_df["sex"].fillna("").astype(str).unique()
        self.assertEqual(set(sex_values), {""})

    def test_bmi_weight_height_all_nan(self):
        for col in ("adult_bmi", "child_weight", "child_height"):
            self.assertTrue(self.train_df[col].isna().all(), col)


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


# ---------------------------------------------------------------------------
# Module constants — 4-class (crackle, wheeze) reference mapping.
# ---------------------------------------------------------------------------

class TestICBHILabelMap(unittest.TestCase):
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

class TestICBHISignalProcessing(unittest.TestCase):
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
