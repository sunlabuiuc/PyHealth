"""Comprehensive tests for TimeseriesProcessor and TemporalTimeseriesProcessor.

This test suite covers:

1. **Backward compatibility** — all original behavior (resampling, imputation,
   metadata) works identically when normalize_strategy=None (the default).

2. **Standard (z-score) normalization** — fit() computes per-feature mean/std
   from training data; process() applies (x - mean) / std.

3. **Min-max normalization** — fit() computes per-feature min/max from training
   data; process() scales values to [0, 1].

4. **Edge cases** — zero-variance features, single-sample training, 1D input,
   empty timestamps, invalid strategy names.

5. **Data leakage prevention** — statistics come from training data only;
   test-set values can fall outside [0, 1] or ±3σ, which is correct.

6. **Save/load round-trip** — normalization statistics survive serialization.

7. **TemporalTimeseriesProcessor parity** — the temporal variant produces
   identical normalized values and additionally returns a time tensor.

8. **Clinical realism** — multi-scale synthetic data (heart rate, temperature,
   blood pressure) demonstrates that normalization fixes gradient imbalance.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.processors.timeseries_processor import TimeseriesProcessor
from pyhealth.processors.temporal_timeseries_processor import TemporalTimeseriesProcessor


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def make_timestamps(n, start=None, interval=timedelta(hours=1)):
    """Generate a list of evenly spaced timestamps."""
    start = start or datetime(2023, 1, 1, 0, 0)
    return [start + i * interval for i in range(n)]


def make_clinical_samples(n_patients=50, n_timesteps=24, seed=42):
    """Generate synthetic multi-scale clinical timeseries samples.

    Simulates 3 features at very different scales to test normalization:
      - Feature 0 (Heart Rate):     mean ≈ 80,  std ≈ 12
      - Feature 1 (Temperature):    mean ≈ 37,  std ≈ 0.4
      - Feature 2 (Systolic BP):    mean ≈ 120, std ≈ 20

    Returns:
        List of sample dicts with a "vitals" key holding (timestamps, values).
    """
    np.random.seed(seed)
    samples = []
    start = datetime(2023, 1, 1, 0)
    for _ in range(n_patients):
        ts = make_timestamps(n_timesteps, start=start)
        hr = np.random.normal(80, 12, n_timesteps)
        temp = np.random.normal(37.0, 0.4, n_timesteps)
        sbp = np.random.normal(120, 20, n_timesteps)
        values = np.stack([hr, temp, sbp], axis=1)
        samples.append({"vitals": (ts, values)})
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Backward Compatibility — no normalization (default)
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility(unittest.TestCase):
    """Ensure the default (normalize_strategy=None) behaves exactly as before."""

    def test_basic_processing_unchanged(self):
        """Raw values pass through without modification."""
        proc = TimeseriesProcessor(sampling_rate=timedelta(hours=1))
        ts = make_timestamps(3)
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = proc.process((ts, values))

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(result.numpy(), values, atol=1e-6)

    def test_fit_only_extracts_n_features(self):
        """With no normalization, fit() learns n_features and nothing else."""
        proc = TimeseriesProcessor()
        samples = [
            {"vitals": (make_timestamps(3), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))},
        ]
        proc.fit(samples, "vitals")

        self.assertEqual(proc.n_features, 3)
        self.assertIsNone(proc.mean_)
        self.assertIsNone(proc.std_)
        self.assertIsNone(proc.min_)
        self.assertIsNone(proc.max_)

    def test_forward_fill_imputation(self):
        """Gaps in the resampled grid are forward-filled."""
        proc = TimeseriesProcessor(sampling_rate=timedelta(hours=1))
        ts = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2)]
        values = np.array([[10.0, 20.0], [30.0, 40.0]])

        result = proc.process((ts, values))

        self.assertEqual(result.shape, (3, 2))
        np.testing.assert_allclose(result[0].numpy(), [10.0, 20.0])
        np.testing.assert_allclose(result[1].numpy(), [10.0, 20.0])  # filled
        np.testing.assert_allclose(result[2].numpy(), [30.0, 40.0])

    def test_zero_imputation(self):
        """Zero imputation fills NaN slots with 0."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1), impute_strategy="zero"
        )
        ts = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 2)]
        values = np.array([[10.0, 20.0], [30.0, 40.0]])

        result = proc.process((ts, values))
        np.testing.assert_allclose(result[1].numpy(), [0.0, 0.0])

    def test_empty_timestamps_raises(self):
        proc = TimeseriesProcessor()
        with self.assertRaises(ValueError):
            proc.process(([], np.array([])))

    def test_metadata_methods(self):
        proc = TimeseriesProcessor()
        self.assertFalse(proc.is_token())
        self.assertEqual(proc.schema(), ("value",))
        self.assertEqual(proc.dim(), (2,))
        self.assertEqual(proc.spatial(), (True, False))

    def test_resampling_rate_changes_output_length(self):
        ts = make_timestamps(5, interval=timedelta(hours=1))
        values = np.arange(10).reshape(5, 2).astype(float)

        result_1h = TimeseriesProcessor(sampling_rate=timedelta(hours=1)).process((ts, values))
        result_2h = TimeseriesProcessor(sampling_rate=timedelta(hours=2)).process((ts, values))

        self.assertEqual(result_1h.shape[0], 5)
        self.assertEqual(result_2h.shape[0], 3)

    def test_repr_includes_normalize_strategy(self):
        """repr should show the new parameter even when it's None."""
        proc = TimeseriesProcessor()
        r = repr(proc)
        self.assertIn("normalize_strategy=None", r)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Standard (z-score) Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestStandardNormalization(unittest.TestCase):
    """Test normalize_strategy='standard' (z-score: (x - mean) / std)."""

    def test_fit_computes_mean_and_std(self):
        """fit() should compute per-feature mean and std from training data."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=100)
        proc.fit(samples, "vitals")

        self.assertIsNotNone(proc.mean_)
        self.assertIsNotNone(proc.std_)
        self.assertEqual(proc.mean_.shape, (3,))
        self.assertEqual(proc.std_.shape, (3,))

        # Stats should approximate the generating distribution
        np.testing.assert_allclose(proc.mean_[0], 80.0, atol=2.0)   # HR
        np.testing.assert_allclose(proc.mean_[1], 37.0, atol=0.1)   # Temp
        np.testing.assert_allclose(proc.mean_[2], 120.0, atol=3.0)  # SBP

    def test_process_outputs_zero_mean_unit_variance(self):
        """After normalization, processed training data should have mean≈0, std≈1."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=100)
        proc.fit(samples, "vitals")

        # Process all training samples and check distributional properties
        all_vals = []
        for s in samples:
            result = proc.process(s["vitals"])
            all_vals.append(result.numpy())
        all_vals = np.concatenate(all_vals, axis=0)

        means = all_vals.mean(axis=0)
        stds = all_vals.std(axis=0)

        # All features should now be centered near 0 with std near 1
        for f in range(3):
            self.assertAlmostEqual(means[f], 0.0, places=1,
                msg=f"Feature {f} mean should be ~0 after standardization")
            self.assertAlmostEqual(stds[f], 1.0, places=1,
                msg=f"Feature {f} std should be ~1 after standardization")

    def test_scale_ratio_collapses(self):
        """The scale ratio between features should drop from ~50x to ~1x."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=100)
        proc.fit(samples, "vitals")

        all_vals = []
        for s in samples:
            all_vals.append(proc.process(s["vitals"]).numpy())
        all_vals = np.concatenate(all_vals, axis=0)

        stds = all_vals.std(axis=0)
        scale_ratio = stds.max() / stds.min()

        self.assertLess(scale_ratio, 1.5,
            "After z-score, all features should have similar variance")

    def test_training_stats_applied_to_test_data(self):
        """Test data should be normalized using training statistics, not its own."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        # Train on normal patients
        train_samples = make_clinical_samples(n_patients=50, seed=42)
        proc.fit(train_samples, "vitals")

        # Test with an outlier patient — very high heart rate
        ts = make_timestamps(5)
        outlier_values = np.array([
            [200.0, 37.0, 120.0],  # HR=200 is extreme
            [195.0, 37.0, 120.0],
            [210.0, 37.0, 120.0],
            [205.0, 37.0, 120.0],
            [198.0, 37.0, 120.0],
        ])
        result = proc.process((ts, outlier_values))

        # HR should be far above 0 (many standard deviations above mean)
        # This confirms we are using training stats, not test stats
        hr_normalized = result[:, 0].numpy()
        self.assertTrue(
            np.all(hr_normalized > 5.0),
            "Outlier HR values should be >5σ above training mean, "
            "proving training stats are used (not test stats)"
        )

        # Temperature and BP should be near 0 (normal values)
        temp_normalized = result[:, 1].numpy()
        np.testing.assert_allclose(temp_normalized, 0.0, atol=1.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Min-Max Normalization
# ═══════════════════════════════════════════════════════════════════════════════

class TestMinMaxNormalization(unittest.TestCase):
    """Test normalize_strategy='minmax' (scales to [0, 1] on training data)."""

    def test_fit_computes_min_and_max(self):
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        samples = make_clinical_samples(n_patients=50)
        proc.fit(samples, "vitals")

        self.assertIsNotNone(proc.min_)
        self.assertIsNotNone(proc.max_)
        self.assertEqual(proc.min_.shape, (3,))
        self.assertEqual(proc.max_.shape, (3,))
        # max should be greater than min for all features
        self.assertTrue(np.all(proc.max_ > proc.min_))

    def test_training_data_maps_to_zero_one(self):
        """Processed training data should fall within [0, 1]."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        samples = make_clinical_samples(n_patients=100)
        proc.fit(samples, "vitals")

        all_vals = []
        for s in samples:
            all_vals.append(proc.process(s["vitals"]).numpy())
        all_vals = np.concatenate(all_vals, axis=0)

        self.assertGreaterEqual(all_vals.min(), -1e-6,
            "Min-max normalized training data should be >= 0")
        self.assertLessEqual(all_vals.max(), 1.0 + 1e-6,
            "Min-max normalized training data should be <= 1")

    def test_test_data_can_exceed_zero_one(self):
        """Test data with values outside training range can go beyond [0, 1].
        This is correct behavior — clipping would lose clinical information."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        train_samples = make_clinical_samples(n_patients=50, seed=42)
        proc.fit(train_samples, "vitals")

        # Extreme outlier values
        ts = make_timestamps(3)
        outlier = np.array([
            [250.0, 40.0, 250.0],  # all features way above training max
            [250.0, 40.0, 250.0],
            [250.0, 40.0, 250.0],
        ])
        result = proc.process((ts, outlier))

        self.assertTrue(
            np.any(result.numpy() > 1.0),
            "Out-of-distribution test values should exceed 1.0 (no clipping)"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Edge Cases
# ═══════════════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):
    """Edge cases that could cause crashes or incorrect results."""

    def test_zero_variance_feature_standard(self):
        """A constant feature (std=0) should not cause division-by-zero.

        If a feature has the same value across the entire training set,
        its std is 0.  The processor should set std=1 for that feature,
        effectively leaving it unscaled.
        """
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        ts = make_timestamps(5)
        # Feature 0 varies, Feature 1 is constant at 42.0
        samples = [
            {"f": (ts, np.array([[10, 42], [20, 42], [30, 42], [40, 42], [50, 42]], dtype=float))},
            {"f": (ts, np.array([[15, 42], [25, 42], [35, 42], [45, 42], [55, 42]], dtype=float))},
        ]
        proc.fit(samples, "f")

        # std for constant feature should be set to 1.0 (not 0.0)
        self.assertEqual(proc.std_[1], 1.0)

        # Process should not crash
        result = proc.process(samples[0]["f"])
        self.assertFalse(torch.isnan(result).any(), "No NaNs from zero-variance feature")
        self.assertFalse(torch.isinf(result).any(), "No Infs from zero-variance feature")

    def test_zero_range_feature_minmax(self):
        """A constant feature (max==min) should not cause division-by-zero in minmax."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        ts = make_timestamps(3)
        samples = [
            {"f": (ts, np.array([[5, 99], [10, 99], [15, 99]], dtype=float))},
        ]
        proc.fit(samples, "f")

        result = proc.process(samples[0]["f"])
        self.assertFalse(torch.isnan(result).any())
        self.assertFalse(torch.isinf(result).any())

    def test_single_sample_training(self):
        """fit() should work with just one training sample."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        ts = make_timestamps(5)
        samples = [
            {"f": (ts, np.random.randn(5, 3))},
        ]
        proc.fit(samples, "f")

        self.assertIsNotNone(proc.mean_)
        result = proc.process(samples[0]["f"])
        self.assertEqual(result.shape, (5, 3))

    def test_1d_values_handled(self):
        """1D input values (single feature) should be promoted to 2D."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        ts = make_timestamps(3)
        samples = [
            {"f": (ts, np.array([10.0, 20.0, 30.0]))},
            {"f": (ts, np.array([15.0, 25.0, 35.0]))},
        ]
        proc.fit(samples, "f")
        result = proc.process(samples[0]["f"])

        self.assertEqual(result.shape, (3, 1))
        self.assertFalse(torch.isnan(result).any())

    def test_invalid_normalize_strategy_raises(self):
        """An unrecognized strategy should raise ValueError immediately."""
        with self.assertRaises(ValueError) as ctx:
            TimeseriesProcessor(normalize_strategy="l2_norm")
        self.assertIn("l2_norm", str(ctx.exception))

    def test_invalid_impute_strategy_raises(self):
        with self.assertRaises(ValueError):
            TimeseriesProcessor(impute_strategy="mean")

    def test_process_without_fit_no_normalization(self):
        """If fit() was never called, process() should still work (no-op normalization)."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        # Skip fit(), directly process
        ts = make_timestamps(3)
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = proc.process((ts, values))

        # Without fit(), mean_ is None, so normalization is skipped
        np.testing.assert_allclose(result.numpy(), values, atol=1e-6)

    def test_samples_with_missing_field_skipped(self):
        """Samples missing the target field should be silently skipped in fit()."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        ts = make_timestamps(3)
        samples = [
            {"other_field": "irrelevant"},  # missing "vitals"
            {"vitals": None},                # None value
            {"vitals": (ts, np.array([[1, 2], [3, 4], [5, 6]], dtype=float))},
        ]
        proc.fit(samples, "vitals")

        self.assertEqual(proc.n_features, 2)
        self.assertIsNotNone(proc.mean_)

    def test_imputation_happens_before_normalization_stats(self):
        """Statistics should reflect imputed values, not raw values with gaps.

        If we computed stats on raw values only, the mean/std would reflect
        the distribution of *observed* points, which differs from what the
        model actually sees (which includes imputed slots).
        """
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        # Sparse data: observations only at hours 0 and 4
        ts = [datetime(2023, 1, 1, 0), datetime(2023, 1, 1, 4)]
        values = np.array([[100.0], [200.0]])
        samples = [{"f": (ts, values)}]
        proc.fit(samples, "f")

        # After forward-fill, the 5 grid values are: [100, 100, 100, 100, 200]
        # mean = 120, not 150 (which is what we'd get from just raw [100, 200])
        self.assertAlmostEqual(proc.mean_[0], 120.0, places=1)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Save / Load Round-trip
# ═══════════════════════════════════════════════════════════════════════════════

class TestSaveLoad(unittest.TestCase):
    """Normalization statistics should survive serialization."""

    def test_standard_save_load_round_trip(self):
        """Save and reload a standard-normalized processor."""
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=20)
        proc.fit(samples, "vitals")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.npz")
            proc.save(path)

            # Create a new processor and load the stats
            proc2 = TimeseriesProcessor(
                sampling_rate=timedelta(hours=1),
                normalize_strategy="standard",
            )
            proc2.load(path)

            np.testing.assert_array_equal(proc.mean_, proc2.mean_)
            np.testing.assert_array_equal(proc.std_, proc2.std_)
            self.assertEqual(proc.n_features, proc2.n_features)

            # Both should produce identical output
            result1 = proc.process(samples[0]["vitals"])
            result2 = proc2.process(samples[0]["vitals"])
            np.testing.assert_allclose(result1.numpy(), result2.numpy())

    def test_minmax_save_load_round_trip(self):
        proc = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        samples = make_clinical_samples(n_patients=20)
        proc.fit(samples, "vitals")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.npz")
            proc.save(path)

            proc2 = TimeseriesProcessor(
                sampling_rate=timedelta(hours=1),
                normalize_strategy="minmax",
            )
            proc2.load(path)

            np.testing.assert_array_equal(proc.min_, proc2.min_)
            np.testing.assert_array_equal(proc.max_, proc2.max_)

    def test_no_normalization_save_load(self):
        """Saving a processor with no normalization should not crash."""
        proc = TimeseriesProcessor()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.npz")
            proc.save(path)
            # File might not exist (nothing to save) — that's fine


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TemporalTimeseriesProcessor Parity
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalProcessorParity(unittest.TestCase):
    """TemporalTimeseriesProcessor should mirror normalization behavior."""

    def test_temporal_standard_normalization(self):
        """Temporal processor should normalize the value tensor identically."""
        proc_ts = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        proc_temp = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )

        samples = make_clinical_samples(n_patients=50)
        proc_ts.fit(samples, "vitals")
        proc_temp.fit(samples, "vitals")

        # Both should compute the same statistics
        np.testing.assert_allclose(proc_ts.mean_, proc_temp.mean_, atol=1e-6)
        np.testing.assert_allclose(proc_ts.std_, proc_temp.std_, atol=1e-6)

        # Process the same sample
        sample = samples[0]["vitals"]
        result_ts = proc_ts.process(sample)
        result_temp = proc_temp.process(sample)

        # Value tensors should be identical
        np.testing.assert_allclose(
            result_ts.numpy(),
            result_temp["value"].numpy(),
            atol=1e-6,
        )

    def test_temporal_time_tensor_not_normalized(self):
        """The time tensor should contain raw hours, never normalized."""
        proc = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=2),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=10, n_timesteps=5)
        proc.fit(samples, "vitals")

        result = proc.process(samples[0]["vitals"])
        time = result["time"].numpy()

        # 5 hourly observations → 3 steps at 2h: [0, 2, 4] hours
        expected_time = np.array([0.0, 2.0, 4.0], dtype=np.float32)
        np.testing.assert_allclose(time, expected_time)

    def test_temporal_backward_compatible_without_normalization(self):
        """Default (no normalization) should produce the same output as before."""
        proc = TemporalTimeseriesProcessor(sampling_rate=timedelta(hours=1))
        ts = make_timestamps(3)
        values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        result = proc.process((ts, values))

        self.assertIn("value", result)
        self.assertIn("time", result)
        np.testing.assert_allclose(result["value"].numpy(), values, atol=1e-6)

    def test_temporal_minmax_normalization(self):
        """Min-max should also work on the temporal processor."""
        proc = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="minmax",
        )
        samples = make_clinical_samples(n_patients=50)
        proc.fit(samples, "vitals")

        all_vals = []
        for s in samples:
            out = proc.process(s["vitals"])
            all_vals.append(out["value"].numpy())
        all_vals = np.concatenate(all_vals, axis=0)

        self.assertGreaterEqual(all_vals.min(), -1e-6)
        self.assertLessEqual(all_vals.max(), 1.0 + 1e-6)

    def test_temporal_save_load(self):
        """Temporal processor save/load should preserve normalization."""
        proc = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        samples = make_clinical_samples(n_patients=20)
        proc.fit(samples, "vitals")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "temporal_stats.npz")
            proc.save(path)

            proc2 = TemporalTimeseriesProcessor(
                sampling_rate=timedelta(hours=1),
                normalize_strategy="standard",
            )
            proc2.load(path)

            result1 = proc.process(samples[0]["vitals"])
            result2 = proc2.process(samples[0]["vitals"])
            np.testing.assert_allclose(
                result1["value"].numpy(),
                result2["value"].numpy(),
            )

    def test_temporal_invalid_strategy_raises(self):
        with self.assertRaises(ValueError):
            TemporalTimeseriesProcessor(normalize_strategy="invalid")

    def test_temporal_repr(self):
        proc = TemporalTimeseriesProcessor(normalize_strategy="minmax")
        self.assertIn("normalize_strategy='minmax'", repr(proc))


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Clinical Realism — Gradient and Scale Analysis
# ═══════════════════════════════════════════════════════════════════════════════

class TestClinicalRealism(unittest.TestCase):
    """Demonstrate that normalization fixes real clinical ML problems."""

    def test_scale_ratio_before_and_after(self):
        """Without normalization: ~50x scale ratio. With: ~1x."""
        samples = make_clinical_samples(n_patients=100)

        # WITHOUT normalization
        proc_raw = TimeseriesProcessor(sampling_rate=timedelta(hours=1))
        raw_vals = []
        for s in samples:
            raw_vals.append(proc_raw.process(s["vitals"]).numpy())
        raw_vals = np.concatenate(raw_vals, axis=0)
        raw_ratio = raw_vals.std(axis=0).max() / raw_vals.std(axis=0).min()

        # WITH normalization
        proc_norm = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        proc_norm.fit(samples, "vitals")
        norm_vals = []
        for s in samples:
            norm_vals.append(proc_norm.process(s["vitals"]).numpy())
        norm_vals = np.concatenate(norm_vals, axis=0)
        norm_ratio = norm_vals.std(axis=0).max() / norm_vals.std(axis=0).min()

        self.assertGreater(raw_ratio, 30, "Raw data should have high scale ratio")
        self.assertLess(norm_ratio, 1.5, "Normalized data should have ~1x scale ratio")

    def test_gradient_balance_improves(self):
        """Gradient magnitudes across features should be more balanced after normalization."""
        np.random.seed(42)
        samples = make_clinical_samples(n_patients=50, n_timesteps=10)

        proc_raw = TimeseriesProcessor(sampling_rate=timedelta(hours=1))
        proc_norm = TimeseriesProcessor(
            sampling_rate=timedelta(hours=1),
            normalize_strategy="standard",
        )
        proc_norm.fit(samples, "vitals")

        # Build feature matrices (mean over time per patient)
        X_raw, X_norm = [], []
        for s in samples:
            X_raw.append(proc_raw.process(s["vitals"]).mean(dim=0).numpy())
            X_norm.append(proc_norm.process(s["vitals"]).mean(dim=0).numpy())

        X_raw = np.stack(X_raw)
        X_norm = np.stack(X_norm)
        y = np.random.randn(50)
        w = np.ones(3)

        # Compute gradient magnitudes for linear model
        grad_raw = np.abs(2 * X_raw.T @ (X_raw @ w - y) / len(y))
        grad_norm = np.abs(2 * X_norm.T @ (X_norm @ w - y) / len(y))

        raw_grad_ratio = grad_raw.max() / grad_raw.min()
        norm_grad_ratio = grad_norm.max() / grad_norm.min()

        self.assertLess(norm_grad_ratio, raw_grad_ratio,
            "Normalization should reduce gradient imbalance across features")


if __name__ == "__main__":
    unittest.main(verbosity=2)
