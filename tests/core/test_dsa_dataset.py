"""Unit tests for DSADataset using fully synthetic data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyhealth.datasets.dsa import DSADataset
from pyhealth.tasks.dsa import DSAActivityClassification

# =====================================================================
# Synthetic data constants — kept small for fast test execution
# =====================================================================

_N_ACTIVITIES = 2
_N_SUBJECTS = 3
_N_SEGMENTS = 2
_N_TIMESTEPS = 125
_N_COLS = 45  # 5 domains × 9 channels


def _make_segment_file(path: Path, seed: int = 0) -> None:
    """Write a synthetic 125×45 segment file to ``path``."""
    rng = np.random.default_rng(seed)
    data = rng.uniform(-10.0, 10.0, size=(_N_TIMESTEPS, _N_COLS)).astype(
        np.float32
    )
    np.savetxt(path, data, delimiter=",", fmt="%.6f")


def _build_synthetic_dataset(root: Path) -> None:
    """Create the full activity/subject/segment folder structure."""
    for a in range(1, _N_ACTIVITIES + 1):
        for p in range(1, _N_SUBJECTS + 1):
            subject_dir = root / f"a{a:02d}" / f"p{p}"
            subject_dir.mkdir(parents=True, exist_ok=True)
            for s in range(1, _N_SEGMENTS + 1):
                seed = a * 1000 + p * 100 + s
                _make_segment_file(subject_dir / f"s{s:02d}.txt", seed=seed)


class TestDSADataset(unittest.TestCase):
    """Tests for DSADataset: loading, indexing, patient/event parsing."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        _build_synthetic_dataset(cls.root)
        cls.cache_dir = tempfile.TemporaryDirectory()

        with patch.multiple(
            DSADataset,
            _N_ACTIVITIES=_N_ACTIVITIES,
            _N_SUBJECTS=_N_SUBJECTS,
            _N_SEGMENTS=_N_SEGMENTS,
        ):
            cls.dataset = DSADataset(
                root=str(cls.root),
                cache_dir=cls.cache_dir.name,
                target_domain="LA",
                scale=True,
            )

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        cls.cache_dir.cleanup()

    # ------------------------------------------------------------------
    # Metadata index
    # ------------------------------------------------------------------

    def test_metadata_csv_created(self):
        """Index CSV must be written alongside the raw data."""
        csv_path = self.root / "dsa-metadata-pyhealth.csv"
        self.assertTrue(csv_path.exists())

    def test_metadata_row_count(self):
        """Row count must equal N_activities × N_subjects × N_segments."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        expected = _N_ACTIVITIES * _N_SUBJECTS * _N_SEGMENTS
        self.assertEqual(len(df), expected)

    def test_metadata_required_columns(self):
        """All columns required by dsa.yaml must be present."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        required = {
            "patient_id",
            "visit_id",
            "activity_id",
            "activity_name",
            "label",
            "segment_id",
            "filepath",
            "pair_id",
        }
        self.assertTrue(required.issubset(df.columns))

    def test_pair_id_format(self):
        """pair_id must be 'a{act:02d}_s{seg:02d}', shared across subjects."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        for _, row in df.iterrows():
            expected_pair = (
                f"a{int(row['activity_id']):02d}"
                f"_s{int(row['segment_id']):02d}"
            )
            self.assertEqual(row["pair_id"], expected_pair)

    def test_pair_id_shared_across_subjects(self):
        """All subjects must share the same pair_id for the same (act, seg)."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        groups = df.groupby("pair_id")["patient_id"].nunique()
        for pair_id, n_subjects in groups.items():
            self.assertEqual(
                n_subjects,
                _N_SUBJECTS,
                f"pair_id '{pair_id}' should have {_N_SUBJECTS} subjects, "
                f"got {n_subjects}",
            )

    def test_label_zero_indexed(self):
        """Label must be activity_id minus 1 (0-indexed)."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        for _, row in df.iterrows():
            self.assertEqual(int(row["label"]), int(row["activity_id"]) - 1)

    def test_all_filepaths_exist(self):
        """Every filepath stored in the index must point to a real file."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        for fp in df["filepath"]:
            self.assertTrue(
                Path(fp).exists(), f"Missing segment file: {fp}"
            )

    # ------------------------------------------------------------------
    # Patient structure
    # ------------------------------------------------------------------

    def test_num_patients(self):
        """One Patient object must exist per subject."""
        self.assertEqual(
            len(self.dataset.unique_patient_ids), _N_SUBJECTS
        )

    def test_patient_id_format(self):
        """Patient IDs must follow the 'p{n}' convention."""
        for pid in self.dataset.unique_patient_ids:
            self.assertTrue(
                pid.startswith("p"),
                f"Patient ID '{pid}' does not start with 'p'",
            )

    def test_events_per_patient(self):
        """Each patient must have N_activities × N_segments events."""
        expected = _N_ACTIVITIES * _N_SEGMENTS
        for pid in self.dataset.unique_patient_ids:
            events = self.dataset.get_patient(pid).get_events()
            self.assertEqual(
                len(events),
                expected,
                f"Patient '{pid}' has {len(events)} events, expected {expected}",
            )

    def test_event_attributes_present(self):
        """Every event must expose all declared attribute columns."""
        required = {
            "label",
            "activity_id",
            "activity_name",
            "filepath",
            "segment_id",
            "pair_id",
            "visit_id",
        }
        patient = self.dataset.get_patient(
            self.dataset.unique_patient_ids[0]
        )
        for event in patient.get_events():
            for attr in required:
                self.assertIn(
                    attr,
                    event.attr_dict,
                    f"Attribute '{attr}' missing from event",
                )

    def test_event_activity_names_valid(self):
        """Activity names must appear in DSADataset.activities."""
        patient = self.dataset.get_patient(
            self.dataset.unique_patient_ids[0]
        )
        for event in patient.get_events():
            self.assertIn(event.activity_name, DSADataset.activities)

    # ------------------------------------------------------------------
    # Time series loading
    # ------------------------------------------------------------------

    def test_load_time_series_shape(self):
        """Single-domain load must return (9, 125) array."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        fp = df.iloc[0]["filepath"]
        ts = self.dataset.load_time_series(fp, domain="LA")
        self.assertIn("LA", ts)
        self.assertEqual(ts["LA"].shape, (9, _N_TIMESTEPS))

    def test_load_time_series_all_domains(self):
        """No-domain load must return all five domain arrays."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        fp = df.iloc[0]["filepath"]
        ts = self.dataset.load_time_series(fp)
        self.assertEqual(set(ts.keys()), set(DSADataset.domains))
        for domain, arr in ts.items():
            self.assertEqual(
                arr.shape,
                (9, _N_TIMESTEPS),
                f"Domain '{domain}' has shape {arr.shape}",
            )

    def test_minmax_scale_range(self):
        """Scaled time series values must lie in [-1, 1]."""
        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        fp = df.iloc[0]["filepath"]
        ts = self.dataset.load_time_series(fp, domain="T")
        arr = ts["T"]
        self.assertGreaterEqual(arr.min(), -1.0 - 1e-6)
        self.assertLessEqual(arr.max(), 1.0 + 1e-6)

    def test_scale_false_preserves_raw_values(self):
        """With scale=False, raw values outside [-1, 1] must be present."""
        with patch.multiple(
            DSADataset,
            _N_ACTIVITIES=_N_ACTIVITIES,
            _N_SUBJECTS=_N_SUBJECTS,
            _N_SEGMENTS=_N_SEGMENTS,
        ):
            dataset_unscaled = DSADataset(
                root=str(self.root),
                cache_dir=self.cache_dir.name,
                target_domain="LA",
                scale=False,
            )

        import pandas as pd

        df = pd.read_csv(self.root / "dsa-metadata-pyhealth.csv")
        fp = df.iloc[0]["filepath"]
        ts = dataset_unscaled.load_time_series(fp, domain="T")
        arr = ts["T"]
        has_outside = (arr.min() < -1.0) or (arr.max() > 1.0)
        self.assertTrue(has_outside, "Unscaled values should exceed [-1, 1]")

    # ------------------------------------------------------------------
    # Subject split utilities
    # ------------------------------------------------------------------

    def test_get_subject_split_counts(self):
        """Train/test split must contain the correct number of rows."""
        train_df, test_df = self.dataset.get_subject_split(
            train_subjects=[1, 2], test_subjects=[3]
        )
        rows_per_subject = _N_ACTIVITIES * _N_SEGMENTS
        self.assertEqual(len(train_df), 2 * rows_per_subject)
        self.assertEqual(len(test_df), 1 * rows_per_subject)

    def test_get_subject_split_no_overlap(self):
        """No patient_id should appear in both train and test splits."""
        train_df, test_df = self.dataset.get_subject_split(
            train_subjects=[1, 2], test_subjects=[3]
        )
        train_ids = set(train_df["patient_id"])
        test_ids = set(test_df["patient_id"])
        self.assertTrue(train_ids.isdisjoint(test_ids))

    def test_get_subject_split_overlap_raises(self):
        """Overlapping train/test subjects must raise ValueError."""
        with self.assertRaises(ValueError):
            self.dataset.get_subject_split(
                train_subjects=[1, 2], test_subjects=[2, 3]
            )

    def test_random_subject_splits_count(self):
        """Generator must yield exactly n_repeats tuples."""
        results = list(
            self.dataset.random_subject_splits(
                n_repeats=3, n_train=2, random_seed=0
            )
        )
        self.assertEqual(len(results), 3)

    def test_random_subject_splits_reproducible(self):
        """Same seed must produce identical splits."""
        splits_a = [
            (tr, te)
            for _, tr, te, _, _ in self.dataset.random_subject_splits(
                n_repeats=3, n_train=2, random_seed=42
            )
        ]
        splits_b = [
            (tr, te)
            for _, tr, te, _, _ in self.dataset.random_subject_splits(
                n_repeats=3, n_train=2, random_seed=42
            )
        ]
        self.assertEqual(splits_a, splits_b)

    def test_random_subject_splits_no_overlap(self):
        """Every random split must have disjoint train/test sets."""
        for _, tr, te, _, _ in self.dataset.random_subject_splits(
            n_repeats=5, n_train=2, random_seed=0
        ):
            self.assertTrue(set(tr).isdisjoint(set(te)))

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    def test_default_task_type(self):
        """default_task must return a DSAActivityClassification instance."""
        self.assertIsInstance(
            self.dataset.default_task, DSAActivityClassification
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def test_invalid_target_domain_raises(self):
        """Unknown target_domain must raise ValueError at construction."""
        with self.assertRaises(ValueError):
            with patch.multiple(
                DSADataset,
                _N_ACTIVITIES=_N_ACTIVITIES,
                _N_SUBJECTS=_N_SUBJECTS,
                _N_SEGMENTS=_N_SEGMENTS,
            ):
                DSADataset(
                    root=str(self.root),
                    target_domain="WRIST",
                )

    def test_missing_root_raises(self):
        """Non-existent root directory must raise FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            DSADataset(root="/nonexistent/path/dsa")

    def test_missing_a01_raises(self):
        """Root without 'a01' subfolder must raise FileNotFoundError."""
        with tempfile.TemporaryDirectory() as bad_root:
            with self.assertRaises(FileNotFoundError):
                DSADataset(root=bad_root)


if __name__ == "__main__":
    unittest.main()
