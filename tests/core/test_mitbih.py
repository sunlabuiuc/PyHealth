"""Unit tests for MIT-BIH dataset and ECGBoundaryDetection task.

Author: Anton Barchukov

Tests cover:
    - MITBIHDataset instantiation from real wfdb files
    - Metadata preparation (paced record exclusion)
    - Patient header parsing (age, sex, medications)
    - ECGBoundaryDetection task (windowing, downsampling, labels)
    - Full pipeline: dataset -> task -> samples
    - MedTsLLM model with real MIT-BIH data structure
"""

import os
import unittest

import numpy as np
import torch

TEST_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "test-resources", "core", "mitbih"
)
HAS_TEST_DATA = os.path.isdir(TEST_DATA_DIR) and any(
    f.endswith(".dat") for f in os.listdir(TEST_DATA_DIR)
)


def setUpModule():
    """Drop the committed CSV + cache so tests use the current schema."""
    if HAS_TEST_DATA:
        _force_regenerate_mitbih_metadata()


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHDataset(unittest.TestCase):
    """Tests for MITBIHDataset with real wfdb files."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset

        cls.dataset = MITBIHDataset(root=TEST_DATA_DIR, dev=True)

    def test_dataset_loads(self):
        """Dataset initializes without error."""
        self.assertIsNotNone(self.dataset)

    def test_metadata_csv_created(self):
        """prepare_metadata creates mitbih-pyhealth.csv."""
        csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
        self.assertTrue(os.path.exists(csv_path))

    def test_has_patients(self):
        """Dataset has at least 1 patient."""
        pids = self.dataset.unique_patient_ids
        self.assertGreater(len(pids), 0)

    def test_paced_records_excluded(self):
        """Paced records (102, 104, 107, 217) are not in dataset."""
        from pyhealth.datasets.mitbih import _PACED_RECORDS

        pids = set(self.dataset.unique_patient_ids)
        for paced in _PACED_RECORDS:
            self.assertNotIn(paced, pids)

    def test_patient_has_events(self):
        """Each patient has ECG events."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        events = patient.get_events()
        self.assertGreater(len(events), 0)

    def test_event_has_demographics(self):
        """Events contain age, sex, medications."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        event = patient.get_events()[0]
        self.assertTrue(hasattr(event, "age"))
        self.assertTrue(hasattr(event, "sex"))
        self.assertTrue(hasattr(event, "medications"))

    def test_event_has_abnormal_count(self):
        """Events track number of abnormal beats."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        event = patient.get_events()[0]
        self.assertTrue(hasattr(event, "n_abnormal"))


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHHeaderParsing(unittest.TestCase):
    """Tests for MIT-BIH patient metadata parsing."""

    def test_parse_header(self):
        """Header contains age and sex info."""
        import wfdb

        rec = wfdb.rdrecord(os.path.join(TEST_DATA_DIR, "100"))
        self.assertIsNotNone(rec.comments)
        self.assertGreater(len(rec.comments), 0)
        # First comment line should have age and sex
        first = rec.comments[0].strip()
        tokens = first.split()
        self.assertGreaterEqual(len(tokens), 2)


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestECGBoundaryDetectionTask(unittest.TestCase):
    """Tests for ECGBoundaryDetection with real MIT-BIH data."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset
        from pyhealth.tasks.ecg_boundary_detection import (
            ECGBoundaryDetection,
        )

        cls.dataset = MITBIHDataset(root=TEST_DATA_DIR, dev=True)
        cls.task = ECGBoundaryDetection(window_size=256, step_size=256)

    def test_task_attributes(self):
        """Task has correct name and schema."""
        self.assertEqual(self.task.task_name, "ECGBoundaryDetection")
        self.assertEqual(self.task.input_schema, {"signal": "tensor"})
        self.assertEqual(self.task.output_schema, {"label": "tensor"})

    def test_task_produces_samples(self):
        """Task generates samples from a patient."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        samples = self.task(patient)
        self.assertGreater(len(samples), 0)

    def test_signal_is_2_channel(self):
        """Signal has 2 channels (MLII, V1 or similar)."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        self.assertEqual(sample["signal"].shape, (256, 2))

    def test_label_is_binary(self):
        """Labels are 0 or 1 (R-peak or not)."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        unique = set(sample["label"].tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_sample_has_description(self):
        """Samples include patient description with medications."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        self.assertIn("description", sample)
        self.assertIn("age:", sample["description"])

    def test_set_task_produces_sample_dataset(self):
        """dataset.set_task() returns a usable SampleDataset."""
        sample_ds = self.dataset.set_task(self.task)
        self.assertGreater(len(sample_ds), 0)


# ------------------------------------------------------------------ #
# Phase 4: anomaly-detection task
# ------------------------------------------------------------------ #


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestECGAnomalyDetectionTask(unittest.TestCase):
    """Tests for ECGAnomalyDetection with real MIT-BIH data."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset
        from pyhealth.tasks.ecg_anomaly_detection import (
            ECGAnomalyDetection,
        )

        cls.dataset = MITBIHDataset(root=TEST_DATA_DIR, dev=True)
        cls.task = ECGAnomalyDetection(window_size=128, step_size=128)

    def test_task_attributes(self):
        """Task has correct name and schema."""
        self.assertEqual(self.task.task_name, "ECGAnomalyDetection")
        self.assertEqual(self.task.input_schema, {"signal": "tensor"})
        self.assertEqual(self.task.output_schema, {"label": "tensor"})

    def test_task_produces_samples(self):
        """Task generates samples from a patient."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        samples = self.task(patient)
        self.assertGreater(len(samples), 0)

    def test_signal_is_2_channel(self):
        """Signal has 2 channels (MLII, V1 or similar)."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        self.assertEqual(sample["signal"].shape, (128, 2))

    def test_label_is_binary_mask(self):
        """Labels are a per-timestep 0/1 anomaly mask."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        self.assertEqual(sample["label"].shape, (128,))
        unique = set(sample["label"].tolist())
        self.assertTrue(unique.issubset({0.0, 1.0}))

    def test_sample_has_description(self):
        """Samples include per-patient description."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        sample = self.task(patient)[0]
        self.assertIn("description", sample)

    def test_set_task_produces_sample_dataset(self):
        """dataset.set_task() works with anomaly detection task."""
        sample_ds = self.dataset.set_task(self.task)
        self.assertGreater(len(sample_ds), 0)


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestAnomalyTaskExportedFromPackage(unittest.TestCase):
    """ECGAnomalyDetection is importable from pyhealth.tasks."""

    def test_import_from_tasks(self):
        from pyhealth.tasks import ECGAnomalyDetection  # noqa: F401


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMedTsLLMAnomalyOnMITBIH(unittest.TestCase):
    """End-to-end anomaly-detection training step on MIT-BIH."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets import get_dataloader
        from pyhealth.datasets.mitbih import MITBIHDataset
        from pyhealth.models.medtsllm import MedTsLLM
        from pyhealth.tasks.ecg_anomaly_detection import (
            ECGAnomalyDetection,
        )

        dataset = MITBIHDataset(root=TEST_DATA_DIR, dev=True)
        task = ECGAnomalyDetection(window_size=128, step_size=128)
        cls.sample_ds = dataset.set_task(task)

        cls.model = MedTsLLM(
            dataset=cls.sample_ds,
            task="anomaly_detection",
            seq_len=128,
            n_features=2,
            backbone=None,
            word_embeddings=torch.randn(50, 32),
            d_model=16,
            d_ff=32,
            n_heads=4,
            num_tokens=50,
            covariate_mode="concat",
        )
        loader = get_dataloader(cls.sample_ds, batch_size=2, shuffle=False)
        cls.batch = next(iter(loader))

    def test_forward(self):
        """Anomaly-detection forward returns finite MSE loss."""
        out = self.model(**self.batch)
        self.assertIn("logit", out)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())

    def test_prediction_shape(self):
        """Reconstruction shape matches (bs, seq_len, n_features)."""
        out = self.model(**self.batch)
        bs = self.batch["signal"].shape[0]
        self.assertEqual(out["logit"].shape, (bs, 128, 2))

    def test_backward(self):
        """Backward pass works for anomaly detection on MIT-BIH."""
        out = self.model(**self.batch)
        out["loss"].backward()


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMedTsLLMWithMITBIH(unittest.TestCase):
    """Test MedTsLLM model with real MIT-BIH data structure."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset
        from pyhealth.tasks.ecg_boundary_detection import (
            ECGBoundaryDetection,
        )
        from pyhealth.datasets import get_dataloader
        from pyhealth.models.medtsllm import MedTsLLM

        dataset = MITBIHDataset(root=TEST_DATA_DIR, dev=True)
        task = ECGBoundaryDetection(window_size=128, step_size=64)
        cls.sample_ds = dataset.set_task(task)

        cls.model = MedTsLLM(
            dataset=cls.sample_ds,
            seq_len=128,
            n_features=2,
            n_classes=2,
            backbone=None,
            word_embeddings=torch.randn(50, 32),
            d_model=16,
            d_ff=32,
            n_heads=4,
            num_tokens=50,
            covariate_mode="concat",
        )
        loader = get_dataloader(cls.sample_ds, batch_size=2, shuffle=False)
        cls.batch = next(iter(loader))

    def test_forward(self):
        """Model forward pass works with real MIT-BIH samples."""
        out = self.model(**self.batch)
        self.assertIn("logit", out)
        self.assertIn("loss", out)
        self.assertTrue(out["loss"].isfinite())

    def test_backward(self):
        """Backward pass works with MIT-BIH data."""
        out = self.model(**self.batch)
        out["loss"].backward()


# ------------------------------------------------------------------ #
# Phase 6: preprocess=True -> .npz cache for MIT-BIH
# ------------------------------------------------------------------ #


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPreprocessCache(unittest.TestCase):
    """preprocess=True writes per-record .npz with signal + annotations."""

    @classmethod
    def setUpClass(cls):
        import shutil

        from pyhealth.datasets.mitbih import MITBIHDataset

        processed_dir = os.path.join(TEST_DATA_DIR, "processed")
        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir)
        _force_regenerate_mitbih_metadata()

        cls.dataset = MITBIHDataset(
            root=TEST_DATA_DIR,
            dev=True,
            preprocess=True,
            downsample_factor=3,
            trim=True,
        )

    def test_processed_dir_created(self):
        self.assertTrue(
            os.path.isdir(os.path.join(TEST_DATA_DIR, "processed"))
        )

    def test_event_has_processed_file_attr(self):
        pid = self.dataset.unique_patient_ids[0]
        event = self.dataset.get_patient(pid).get_events()[0]
        self.assertTrue(hasattr(event, "processed_file"))
        self.assertTrue(event.processed_file.endswith(".npz"))
        self.assertTrue(os.path.exists(event.processed_file))

    def test_cached_arrays_have_expected_keys(self):
        pid = self.dataset.unique_patient_ids[0]
        event = self.dataset.get_patient(pid).get_events()[0]
        with np.load(event.processed_file, allow_pickle=False) as npz:
            self.assertIn("signal", npz.files)
            self.assertIn("ann_sample", npz.files)
            self.assertIn("ann_symbol", npz.files)

    def test_trim_applied_to_cache(self):
        """After trim, first and last ann_sample are within [0, len-1]."""
        pid = self.dataset.unique_patient_ids[0]
        event = self.dataset.get_patient(pid).get_events()[0]
        with np.load(event.processed_file, allow_pickle=False) as npz:
            signal_len = int(npz["signal"].shape[0])
            ann_sample = np.asarray(npz["ann_sample"])
            self.assertGreaterEqual(int(ann_sample[0]), 0)
            self.assertLess(int(ann_sample[-1]), signal_len)
            # Trim: first annotation should be at or near sample 0.
            self.assertLessEqual(int(ann_sample[0]), 5)

    def test_boundary_task_uses_cache(self):
        from pyhealth.tasks.ecg_boundary_detection import (
            ECGBoundaryDetection,
        )

        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        samples = ECGBoundaryDetection(
            window_size=128, step_size=128
        )(patient)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0]["signal"].shape, (128, 2))

    def test_anomaly_task_uses_cache(self):
        from pyhealth.tasks.ecg_anomaly_detection import (
            ECGAnomalyDetection,
        )

        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        samples = ECGAnomalyDetection(
            window_size=128, step_size=128
        )(patient)
        self.assertGreater(len(samples), 0)
        self.assertEqual(samples[0]["signal"].shape, (128, 2))


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPreprocessTrimOff(unittest.TestCase):
    """trim=False caches the full downsampled signal."""

    def test_untrimmed_cache_is_longer(self):
        import shutil

        from pyhealth.datasets.mitbih import MITBIHDataset

        processed_dir = os.path.join(TEST_DATA_DIR, "processed")

        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir)
        _force_regenerate_mitbih_metadata()
        ds_trim = MITBIHDataset(
            root=TEST_DATA_DIR,
            dev=True,
            preprocess=True,
            trim=True,
        )
        ev_trim = ds_trim.get_patient(
            ds_trim.unique_patient_ids[0]
        ).get_events()[0]
        with np.load(ev_trim.processed_file, allow_pickle=False) as npz:
            trimmed_len = int(npz["signal"].shape[0])

        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir)
        _force_regenerate_mitbih_metadata()
        ds_full = MITBIHDataset(
            root=TEST_DATA_DIR,
            dev=True,
            preprocess=True,
            trim=False,
        )
        ev_full = ds_full.get_patient(
            ds_full.unique_patient_ids[0]
        ).get_events()[0]
        with np.load(ev_full.processed_file, allow_pickle=False) as npz:
            full_len = int(npz["signal"].shape[0])

        self.assertGreaterEqual(full_len, trimmed_len)


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPreprocessDisabled(unittest.TestCase):
    """preprocess=False leaves processed_file blank."""

    def test_processed_file_empty_when_disabled(self):
        import shutil

        from pyhealth.datasets.mitbih import MITBIHDataset

        processed_dir = os.path.join(TEST_DATA_DIR, "processed")
        if os.path.isdir(processed_dir):
            shutil.rmtree(processed_dir)
        _force_regenerate_mitbih_metadata()

        ds = MITBIHDataset(root=TEST_DATA_DIR, dev=True, preprocess=False)
        pid = ds.unique_patient_ids[0]
        event = ds.get_patient(pid).get_events()[0]
        value = getattr(event, "processed_file", "") or ""
        self.assertEqual(str(value).strip(), "")


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHInvalidDownsample(unittest.TestCase):
    """downsample_factor < 1 raises ValueError."""

    def test_invalid_downsample_raises(self):
        from pyhealth.datasets.mitbih import MITBIHDataset

        _force_regenerate_mitbih_metadata()
        with self.assertRaises(ValueError):
            MITBIHDataset(root=TEST_DATA_DIR, dev=True, downsample_factor=0)


# ------------------------------------------------------------------ #
# Phase 5: paper-match split column (MIT-BIH 80/20 seed=0)
# ------------------------------------------------------------------ #


def _reset_mitbih_csv():
    """Delete the committed metadata CSV so the next ``MITBIHDataset``
    rebuilds it with the current ``prepare_metadata`` implementation.

    Leaves the PyHealth cache warm — safe when the next dataset uses
    the same fingerprint as a prior class. For config changes (e.g.
    paper_split), use ``_force_regenerate_mitbih_metadata`` instead.
    """
    csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
    if os.path.exists(csv_path):
        os.remove(csv_path)


def _force_regenerate_mitbih_metadata():
    """Nuclear reset: CSV + entire PyHealth cache. Use in setUpModule
    once per file, or when a test wipes ``processed_dir`` / changes
    config and needs the event-dataframe cache to be rebuilt.
    """
    import shutil

    import platformdirs

    _reset_mitbih_csv()
    cache_root = platformdirs.user_cache_dir(appname="pyhealth")
    if os.path.isdir(cache_root):
        shutil.rmtree(cache_root)


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPaperSplitRandom(unittest.TestCase):
    """paper_split='random' writes an 80/20 seed=0 split column."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset

        _force_regenerate_mitbih_metadata()
        cls.dataset = MITBIHDataset(
            root=TEST_DATA_DIR, dev=True, paper_split="random"
        )

    def test_constructor_accepts_paper_split(self):
        """MITBIHDataset accepts a paper_split kwarg."""
        import inspect

        from pyhealth.datasets.mitbih import MITBIHDataset

        sig = inspect.signature(MITBIHDataset.__init__)
        self.assertIn("paper_split", sig.parameters)

    def test_csv_has_split_column(self):
        """Regenerated CSV contains a ``split`` column."""
        import pandas as pd

        csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("split", df.columns)

    def test_split_values_are_train_or_test(self):
        """Every split value is either 'train' or 'test'."""
        import pandas as pd

        csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
        df = pd.read_csv(csv_path)
        unique = set(df["split"].unique())
        self.assertTrue(unique.issubset({"train", "test"}))

    def test_event_has_split_attr(self):
        """Events expose a ``split`` attribute."""
        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        event = patient.get_events()[0]
        self.assertTrue(hasattr(event, "split"))
        self.assertIn(event.split, ("train", "test"))

    def test_boundary_task_emits_split_field(self):
        """ECGBoundaryDetection propagates event.split to each sample."""
        from pyhealth.tasks.ecg_boundary_detection import (
            ECGBoundaryDetection,
        )

        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        task = ECGBoundaryDetection(window_size=256, step_size=256)
        sample = task(patient)[0]
        self.assertIn("split", sample)
        self.assertIn(sample["split"], ("train", "test"))

    def test_anomaly_task_emits_split_field(self):
        """ECGAnomalyDetection propagates event.split to each sample."""
        from pyhealth.tasks.ecg_anomaly_detection import (
            ECGAnomalyDetection,
        )

        pid = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(pid)
        task = ECGAnomalyDetection(window_size=128, step_size=128)
        sample = task(patient)[0]
        self.assertIn("split", sample)
        self.assertIn(sample["split"], ("train", "test"))


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPaperSplitAbnormalSorted(unittest.TestCase):
    """paper_split='abnormal_sorted' orders patients by n_abnormal asc."""

    @classmethod
    def setUpClass(cls):
        from pyhealth.datasets.mitbih import MITBIHDataset

        _force_regenerate_mitbih_metadata()
        cls.dataset = MITBIHDataset(
            root=TEST_DATA_DIR, dev=True, paper_split="abnormal_sorted"
        )

    def test_csv_has_split_column(self):
        """Regenerated CSV contains a populated ``split`` column."""
        import pandas as pd

        csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
        df = pd.read_csv(csv_path)
        self.assertIn("split", df.columns)
        unique = set(df["split"].unique())
        self.assertTrue(unique.issubset({"train", "test"}))

    def test_train_has_lower_or_equal_n_abnormal(self):
        """All train patients have n_abnormal <= all test patients."""
        import pandas as pd

        csv_path = os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv")
        df = pd.read_csv(csv_path)
        train = df[df["split"] == "train"]["n_abnormal"]
        test = df[df["split"] == "test"]["n_abnormal"]
        if len(train) > 0 and len(test) > 0:
            self.assertLessEqual(train.max(), test.min())


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPaperSplitDisabled(unittest.TestCase):
    """paper_split=None leaves the split column blank."""

    def test_split_column_blank_when_disabled(self):
        """Without paper_split, split column is empty."""
        import pandas as pd
        from pyhealth.datasets.mitbih import MITBIHDataset

        _force_regenerate_mitbih_metadata()
        MITBIHDataset(root=TEST_DATA_DIR, dev=True, paper_split=None)
        df = pd.read_csv(os.path.join(TEST_DATA_DIR, "mitbih-pyhealth.csv"))
        self.assertIn("split", df.columns)
        cleaned = df["split"].fillna("").astype(str).str.strip()
        self.assertTrue((cleaned == "").all())


@unittest.skipUnless(HAS_TEST_DATA, "MIT-BIH test data not found")
class TestMITBIHPaperSplitInvalid(unittest.TestCase):
    """Unknown paper_split mode raises ValueError."""

    def test_invalid_mode_raises(self):
        from pyhealth.datasets.mitbih import MITBIHDataset

        _force_regenerate_mitbih_metadata()
        with self.assertRaises(ValueError):
            MITBIHDataset(
                root=TEST_DATA_DIR, dev=True, paper_split="nonsense"
            )


if __name__ == "__main__":
    unittest.main()
