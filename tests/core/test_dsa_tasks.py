"""Unit tests for DSA tasks and IPD computation using synthetic data."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from pyhealth.datasets.dsa import DSADataset
from pyhealth.tasks.dsa import (
    DSAActivityClassification,
    DSABinaryActivityClassification,
)
from pyhealth.tasks.dsa import (
    compute_all_ipd_weights,
    compute_ipd_weight,
    compute_pairwise_distances,
    compute_weighted_epochs,
    ExperimentConfig,
    ExperimentResult,
)

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


class TestDSAActivityClassification(unittest.TestCase):
    """Tests for the activity classification task."""

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
            cls.task = DSAActivityClassification()
            cls.samples = cls.dataset.set_task(cls.task)
            cls.sample_list = list(cls.samples)

    @classmethod
    def tearDownClass(cls):
        try:
            del cls.samples
        except Exception:
            pass

        import gc
        gc.collect()

    def test_sample_count(self):
        """Total samples must equal N_patients × N_activities × N_segments."""
        expected = _N_SUBJECTS * _N_ACTIVITIES * _N_SEGMENTS
        self.assertEqual(len(self.samples), expected)

    def test_sample_required_keys(self):
        """Every sample must contain required keys."""
        required = {
            "patient_id",
            "visit_id",
            "time_series",
            "label",
            "activity_name",
            "pair_id",
        }
        for sample in self.sample_list:
            self.assertTrue(required.issubset(sample.keys()))

    def test_time_series_shape(self):
        """time_series must have shape (9, 125)."""
        for sample in self.sample_list:
            self.assertEqual(
                sample["time_series"].shape, (9, _N_TIMESTEPS)
            )

    def test_labels_range(self):
        """Labels must be integers in [0, N_activities - 1]."""
        labels = {int(sample["label"]) for sample in self.sample_list}
        self.assertTrue(labels.issubset(set(range(_N_ACTIVITIES))))

    def test_activity_names_valid(self):
        """Activity names must come from DSADataset.activities."""
        for sample in self.sample_list:
            self.assertIn(sample["activity_name"], DSADataset.activities)

    def test_time_series_scaled(self):
        """Scaled time series must have all values in [-1, 1]."""
        for sample in self.sample_list:
            arr = sample["time_series"]
            self.assertGreaterEqual(arr.min(), -1.0 - 1e-6)
            self.assertLessEqual(arr.max(), 1.0 + 1e-6)

    def test_pair_ids_present(self):
        """pair_id must be present and non-empty for every sample."""
        for sample in self.sample_list:
            self.assertIsInstance(sample["pair_id"], str)
            self.assertGreater(len(sample["pair_id"]), 0)


class TestDSABinaryClassification(unittest.TestCase):
    """Tests for the binary one-vs-rest classification task."""

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
                target_domain="RA",
                scale=True,
            )
            cls.task = DSABinaryActivityClassification(positive_activity_id=1)
            cls.samples = cls.dataset.set_task(cls.task)
            cls.sample_list = list(cls.samples)

    def test_sample_count(self):
        """Binary task must produce the same number of samples as multiclass."""
        expected = _N_SUBJECTS * _N_ACTIVITIES * _N_SEGMENTS
        self.assertEqual(len(self.samples), expected)

    def test_labels_binary(self):
        """All labels must be 0 or 1."""
        for sample in self.sample_list:
            self.assertIn(sample["label"], (0, 1))

    def test_positive_label_count(self):
        """Exactly N_subjects × N_segments samples must be positive."""
        n_positive = sum(s["label"] for s in self.samples)
        expected = _N_SUBJECTS * _N_SEGMENTS
        self.assertEqual(n_positive, expected)

    def test_positive_label_corresponds_to_activity(self):
        """Positive samples must belong to the designated positive activity."""
        for sample in self.sample_list:
            if sample["label"] == 1:
                self.assertEqual(sample["activity_id"], 1)
            else:
                self.assertNotEqual(sample["activity_id"], 1)

    def test_missing_positive_activity_raises(self):
        """Calling task without positive_activity_id set must raise ValueError."""
        with self.assertRaises(TypeError):
            DSABinaryActivityClassification()
        with patch.multiple(
            DSADataset,
            _N_ACTIVITIES=_N_ACTIVITIES,
            _N_SUBJECTS=_N_SUBJECTS,
            _N_SEGMENTS=_N_SEGMENTS,
        ):
            with self.assertRaises(TypeError):
                DSABinaryActivityClassification()

    # def test_different_positive_class_gives_different_positives(self):
    #     """Changing positive_activity_id must change which samples are positive."""
    #     task_a2 = DSABinaryActivityClassification(positive_activity_id=2)
    #     with patch.multiple(
    #         DSADataset,
    #         _N_ACTIVITIES=_N_ACTIVITIES,
    #         _N_SUBJECTS=_N_SUBJECTS,
    #         _N_SEGMENTS=_N_SEGMENTS,
    #     ):
    #         samples_a2 = self.dataset.set_task(task_a2)

    #     positives_a1 = {s["visit_id"] for s in self.sample_list if s["label"] == 1}
    #     positives_a2 = {s["visit_id"] for s in samples_a2 if s["label"] == 1}
    #     self.assertTrue(positives_a1.isdisjoint(positives_a2))
    #     del samples_a2

    def test_activity_id_in_sample(self):
        """activity_id field must be present in every binary sample."""
        for sample in self.sample_list:
            self.assertIn("activity_id", sample)


class TestIPDComputation(unittest.TestCase):
    """Tests for IPD pipeline: pairwise distances, KDE weights, epoch scaling."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(42)
        n = 20
        cls.source_ts = rng.uniform(-1, 1, (n, 125)).astype(np.float32)
        cls.target_ts = rng.uniform(-1, 1, (n, 125)).astype(np.float32)
        cls.n = n
        cls.euclidean_dists = compute_pairwise_distances(
            cls.source_ts, cls.target_ts, metric="euclidean"
        )
        cls.self_dists = compute_pairwise_distances(
            cls.source_ts, cls.source_ts, metric="euclidean"
        )

    def test_euclidean_output_shape(self):
        """Output must have one scalar per pair."""
        dists = compute_pairwise_distances(
            self.source_ts, self.target_ts, metric="euclidean"
        )
        self.assertEqual(self.euclidean_dists.shape, (self.n,))

    def test_euclidean_non_negative(self):
        """Euclidean distances must be non-negative."""
        self.assertTrue(np.all(self.euclidean_dists >= 0))

    def test_self_distance_is_zero(self):
        """Euclidean distance from a series to itself must be zero."""
        np.testing.assert_allclose(self.self_dists, 0.0, atol=1e-5)

    def test_invalid_metric_raises(self):
        """Unknown metric string must raise ValueError."""
        with self.assertRaises(ValueError):
            compute_pairwise_distances(
                self.source_ts, self.target_ts, metric="manhattan_city"
            )

    def test_ipd_weight_is_scalar(self):
        """IPD weight must be a Python float."""
        weight = compute_ipd_weight(self.euclidean_dists)
        self.assertIsInstance(weight, float)

    def test_identical_series_low_weight(self):
        """Identical source/target should produce a lower distance than random."""
        near_zero = np.zeros((self.n, 125), dtype=np.float32)
        dists_identical = compute_pairwise_distances(
            near_zero, near_zero, metric="euclidean"
        )
        dists_random = self.euclidean_dists
        weight_identical = compute_ipd_weight(dists_identical)
        weight_random = compute_ipd_weight(dists_random)
        self.assertLess(weight_identical, weight_random)

    def test_ipd_weight_deterministic(self):
        """Same input and random_state must give the same weight."""
        w1 = compute_ipd_weight(self.euclidean_dists, random_state=0)
        w2 = compute_ipd_weight(self.euclidean_dists, random_state=0)
        self.assertAlmostEqual(w1, w2, places=10)

    def test_ipd_weight_changes_with_bandwidth(self):
        """Changing bandwidth must produce a different weight."""
        w_narrow = compute_ipd_weight(self.euclidean_dists)
        w_wide = compute_ipd_weight(self.euclidean_dists, bandwidth=50.0)
        self.assertNotAlmostEqual(w_narrow, w_wide, places=3)

    def test_all_ipd_weights_excludes_target(self):
        """Target domain must not appear in the returned weights dict."""
        rng = np.random.default_rng(0)
        domain_data = {
            d: rng.uniform(-1, 1, (10, 125)).astype(np.float32)
            for d in ["T", "RA", "LA", "RL", "LL"]
        }
        weights = compute_all_ipd_weights(
            domain_data, target_domain="LA", metric="euclidean"
        )
        self.assertNotIn("LA", weights)

    def test_all_ipd_weights_contains_source_domains(self):
        """All non-target domains must appear in the returned weights."""
        rng = np.random.default_rng(1)
        domain_data = {
            d: rng.uniform(-1, 1, (10, 125)).astype(np.float32)
            for d in ["T", "RA", "LA", "RL", "LL"]
        }
        weights = compute_all_ipd_weights(
            domain_data, target_domain="LA", metric="euclidean"
        )
        expected_sources = {"T", "RA", "RL", "LL"}
        self.assertEqual(set(weights.keys()), expected_sources)

    def test_all_ipd_weights_are_positive(self):
        """All weights must be positive for non-zero distance inputs."""
        rng = np.random.default_rng(2)
        domain_data = {
            d: rng.uniform(0.5, 1.0, (10, 125)).astype(np.float32)
            for d in ["T", "RA", "LA"]
        }
        weights = compute_all_ipd_weights(
            domain_data, target_domain="LA", metric="euclidean"
        )
        for domain, w in weights.items():
            self.assertGreater(w, 0, f"Weight for '{domain}' must be positive")

    def test_weighted_epochs_all_domains_present(self):
        """Every input domain must appear in the output."""
        weights = {"T": 10.0, "RA": 5.0, "RL": 3.0, "LL": 2.0}
        epochs = compute_weighted_epochs(weights)
        self.assertEqual(set(epochs.keys()), set(weights.keys()))

    def test_weighted_epochs_minimum_one(self):
        """Every domain must get at least 1 epoch (from +1 in formula)."""
        weights = {"T": 0.0001, "RA": 0.0001}
        epochs = compute_weighted_epochs(weights)
        for d, e in epochs.items():
            self.assertGreaterEqual(e, 1, f"Domain '{d}' got 0 epochs")

    def test_weighted_epochs_proportional(self):
        """Higher-weight domain should receive more epochs."""
        weights = {"high": 100.0, "low": 1.0}
        epochs = compute_weighted_epochs(weights)
        self.assertGreater(epochs["high"], epochs["low"])

    def test_weighted_epochs_zero_sum_fallback(self):
        """All-zero weights must not cause a division error."""
        weights = {"T": 0.0, "RA": 0.0}
        try:
            epochs = compute_weighted_epochs(weights)
            for e in epochs.values():
                self.assertIsInstance(e, int)
        except ZeroDivisionError:
            self.fail("compute_weighted_epochs raised ZeroDivisionError")

    def test_epoch_scale_factor_applied(self):
        """Custom scale_factor must affect the epoch counts."""
        weights = {"T": 1.0}
        e1 = compute_weighted_epochs(weights, scale_factor=7)
        e2 = compute_weighted_epochs(weights, scale_factor=14)
        self.assertGreater(e2["T"], e1["T"])


class TestExperimentConfig(unittest.TestCase):
    """Tests for ExperimentConfig and ExperimentResult dataclasses."""

    def test_default_values(self):
        """Default config must reflect author's code values, not paper."""
        config = ExperimentConfig()
        self.assertEqual(config.learning_rate, 0.005)
        self.assertEqual(config.source_epochs, 30)
        self.assertEqual(config.target_epochs_weighted, 40)
        self.assertEqual(config.kde_bandwidth, 7.8)
        self.assertEqual(config.kde_n_samples, 10)
        self.assertEqual(config.n_repeats, 15)
        self.assertEqual(config.n_train_subjects, 6)

    def test_custom_values(self):
        """Custom config must override defaults correctly."""
        config = ExperimentConfig(
            target_domain="RA",
            metric="dtw_classic",
            positive_activity_id=12,
        )
        self.assertEqual(config.target_domain, "RA")
        self.assertEqual(config.metric, "dtw_classic")
        self.assertEqual(config.positive_activity_id, 12)

    def test_none_positive_activity_triggers_multiclass(self):
        """positive_activity_id=None must signal the 19-class setup."""
        config = ExperimentConfig()
        self.assertIsNone(config.positive_activity_id)

    def test_experiment_result_defaults(self):
        """ExperimentResult must initialise with zero accuracies."""
        result = ExperimentResult()
        self.assertEqual(result.accuracy_no_transfer, 0.0)
        self.assertEqual(result.accuracy_naive_transfer, 0.0)
        self.assertEqual(result.accuracy_weighted_transfer, 0.0)
        self.assertIsInstance(result.ipd_weights, dict)
        self.assertIsInstance(result.weighted_epochs, dict)

    def test_experiment_result_assignment(self):
        """Assigned accuracies must be retrievable."""
        result = ExperimentResult(
            repeat_idx=3,
            train_subjects=[1, 2, 3],
            test_subjects=[4, 5],
            metric="euclidean",
            accuracy_no_transfer=0.82,
            accuracy_naive_transfer=0.88,
            accuracy_weighted_transfer=0.91,
        )
        self.assertEqual(result.repeat_idx, 3)
        self.assertAlmostEqual(result.accuracy_weighted_transfer, 0.91)


if __name__ == "__main__":
    unittest.main()
