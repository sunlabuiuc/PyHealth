"""Training-only normalization contracts for both time-series processors."""

import copy
from datetime import datetime, timedelta
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

from pyhealth.datasets import create_sample_dataset
from pyhealth.datasets.sample_dataset import SampleBuilder
from pyhealth.processors import TemporalTimeseriesProcessor, TimeseriesProcessor


PROCESSORS = (TimeseriesProcessor, TemporalTimeseriesProcessor)


def series(values, hours=None):
    values = np.asarray(values, dtype=float)
    if hours is None:
        hours = range(len(values))
    timestamps = [datetime(2026, 1, 1) + timedelta(hours=h) for h in hours]
    return timestamps, values


def tensor(processor, value):
    result = processor.process(value)
    return result["value"] if isinstance(result, dict) else result


class TestTimeseriesNormalization(unittest.TestCase):
    def test_default_preserves_resampling_and_filling(self):
        value = series([[np.nan, 10], [4, np.nan]], hours=[0, 2])
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls()
                result = tensor(proc, value)
                self.assertEqual(result.dtype, torch.float32)
                np.testing.assert_array_equal(result, [[0, 10], [0, 10], [4, 10]])

    def test_standard_uses_all_resampled_steps_and_generator_once(self):
        # Filled training values are [10, 10, 30, 50]; each step has equal weight.
        samples = [
            {"signal": series([[10, 7], [30, 7]], hours=[0, 2])},
            {"signal": series([[50, 7]])},
        ]
        expected = (np.array([10, 10, 30, 50]) - 25) / np.sqrt(275)
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                proc.fit(iter(samples), "signal")
                result = torch.cat([tensor(proc, s["signal"]) for s in samples])
                np.testing.assert_allclose(result[:, 0], expected, rtol=1e-6)
                np.testing.assert_array_equal(result[:, 1], 0)
                self.assertEqual(proc.size(), 2)
                self.assertEqual(result.dtype, torch.float32)

    def test_zero_imputation_precedes_normalization(self):
        proc = TimeseriesProcessor(
            impute_strategy="zero", normalize_strategy="standard"
        )
        value = series([[10], [30]], hours=[0, 2])
        proc.fit([{"signal": value}], "signal")
        expected = np.array([[10], [0], [30]], dtype=float)
        expected = (expected - expected.mean()) / expected.std()
        np.testing.assert_allclose(tensor(proc, value), expected, rtol=1e-6)

    def test_temporal_time_is_unchanged_and_1d_is_supported(self):
        value = series([10, 30], hours=[0, 4])
        proc = TemporalTimeseriesProcessor(
            sampling_rate=timedelta(hours=2), normalize_strategy="standard"
        )
        proc.fit([{"signal": value}], "signal")
        result = proc.process_temporal(value)
        self.assertEqual(result["value"].shape, (3, 1))
        np.testing.assert_array_equal(result["time"], [0, 2, 4])
        self.assertEqual(result["time"].dtype, torch.float32)
        np.testing.assert_allclose(
            result["value"][:, 0],
            [-1 / np.sqrt(2), -1 / np.sqrt(2), np.sqrt(2)],
            rtol=1e-6,
        )

    def test_ordinary_processor_rejects_1d_values_cleanly(self):
        proc = TimeseriesProcessor(normalize_strategy="standard")
        with self.assertRaisesRegex(ValueError, "shape"):
            proc.fit([{"signal": series([10, 30])}], "signal")

    def test_constant_single_step_and_all_missing_features(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                proc.fit([{"signal": series([[7, np.nan]])}], "signal")
                np.testing.assert_array_equal(tensor(proc, series([[7, np.nan]])), 0)
                np.testing.assert_array_equal(tensor(proc, series([[9, 3]])), [[2, 3]])

    def test_processing_does_not_modify_inputs_or_fitted_state(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                proc.fit([{"signal": series([[10], [30]])}], "signal")
                before = copy.deepcopy(vars(proc))
                value = series([[1000], [np.nan]])
                original = value[1].copy()
                np.testing.assert_array_equal(tensor(proc, value), [[98], [98]])
                self.assertEqual(vars(proc), before)
                np.testing.assert_array_equal(value[1], original)

    def test_invalid_strategy(self):
        for cls in PROCESSORS:
            for strategy in ("minmax", "unknown", True):
                with self.subTest(processor=cls.__name__, strategy=strategy):
                    with self.assertRaisesRegex(ValueError, "normalize_strategy"):
                        cls(normalize_strategy=strategy)

    def test_fit_is_required_only_when_normalization_enabled(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                value = series([[10], [30]])
                np.testing.assert_array_equal(tensor(cls(), value), value[1])
                with self.assertRaisesRegex(RuntimeError, "fit"):
                    tensor(cls(normalize_strategy="standard"), value)

    def test_missing_fields_are_skipped_but_no_data_raises(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                proc.fit(
                    [{}, {"signal": None}, {"signal": series([[1], [3]])}], "signal"
                )
                np.testing.assert_array_equal(
                    tensor(proc, series([[1], [3]])), [[-1], [1]]
                )
                for samples in ([], [{}, {"signal": None}]):
                    with self.assertRaisesRegex(ValueError, "training"):
                        proc.fit(samples, "signal")
                    with self.assertRaises(RuntimeError):
                        tensor(proc, series([[1]]))

    def test_refit_replaces_statistics_and_failed_refit_clears_them(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                proc.fit([{"signal": series([[10], [30]])}], "signal")
                proc.fit([{"signal": series([[100], [300]])}], "signal")
                np.testing.assert_array_equal(
                    tensor(proc, series([[100], [300]])), [[-1], [1]]
                )
                with self.assertRaises(ValueError):
                    proc.fit(
                        [
                            {"signal": series([[1]])},
                            {"signal": series([[np.inf]])},
                        ],
                        "signal",
                    )
                with self.assertRaises(RuntimeError):
                    tensor(proc, series([[100]]))

    def test_invalid_present_samples_rejected_during_fit_and_process(self):
        bad_values = [
            series(np.empty((0, 2))),
            series([[1, 2]], hours=[0, 1]),
            series([[1, 2], [3, 4]], hours=[1, 0]),
            series([[[1, 2]]]),
            series(np.empty((1, 0))),
            series([[np.inf, 1]]),
            series([[-np.inf, 1]]),
        ]
        for cls in PROCESSORS:
            for value in bad_values:
                with self.subTest(processor=cls.__name__, shape=value[1].shape):
                    proc = cls(normalize_strategy="standard")
                    with self.assertRaises(ValueError):
                        proc.fit([{"signal": value}], "signal")
                    proc.fit([{"signal": series([[1, 2], [3, 4]])}], "signal")
                    with self.assertRaises(ValueError):
                        proc.process(value)

    def test_feature_count_must_match(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                with self.assertRaises(ValueError):
                    proc.fit(
                        [{"signal": series([[1, 2]])}, {"signal": series([[1]])}],
                        "signal",
                    )
                proc.fit([{"signal": series([[1, 2]])}], "signal")
                with self.assertRaises(ValueError):
                    tensor(proc, series([[1]]))

    def test_nonpositive_sampling_rate_rejected_when_enabled(self):
        for cls in PROCESSORS:
            for rate in (timedelta(0), timedelta(hours=-1)):
                with self.subTest(processor=cls.__name__, rate=rate):
                    with self.assertRaisesRegex(ValueError, "sampling_rate"):
                        cls(sampling_rate=rate, normalize_strategy="standard")

    def test_duplicate_timestamps_keep_existing_last_value_rule(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls(normalize_strategy="standard")
                value = series([[1], [10], [30]], hours=[0, 0, 1])
                proc.fit([{"signal": value}], "signal")
                np.testing.assert_array_equal(tensor(proc, value), [[-1], [1]])

    def test_legacy_pickle_without_normalization_fields(self):
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                proc = cls()
                proc.__dict__ = {
                    "sampling_rate": timedelta(hours=1),
                    "impute_strategy": "forward_fill",
                    "n_features": 1,
                }
                restored = pickle.loads(pickle.dumps(proc))
                np.testing.assert_array_equal(
                    tensor(restored, series([[10], [30]])), [[10], [30]]
                )
                restored.fit([{"signal": series([[20]])}], "signal")
                np.testing.assert_array_equal(tensor(restored, series([[20]])), [[20]])


class TestNormalizationIntegration(unittest.TestCase):
    def test_schema_instance_fingerprint_includes_normalization(self):
        # Task schemas containing instances use default=str in BaseDataset.
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                plain = json.dumps({"signal": cls()}, sort_keys=True, default=str)
                normalized = json.dumps(
                    {"signal": cls(normalize_strategy="standard")},
                    sort_keys=True,
                    default=str,
                )
                self.assertNotEqual(plain, normalized)
                old = cls()
                del old.normalize_strategy
                self.assertEqual(repr(old), repr(cls()))

    def test_training_processor_transfer_does_not_refit_on_held_out_data(self):
        for alias in ("timeseries", "temporal_timeseries"):
            with self.subTest(processor=alias):
                schema = {"signal": (alias, {"normalize_strategy": "standard"})}
                train = create_sample_dataset(
                    samples=[{"patient_id": "train", "signal": series([[10], [30]])}],
                    input_schema=schema,
                    output_schema={},
                )
                proc = train.input_processors["signal"]
                before = copy.deepcopy(vars(proc))
                for split in ("validation", "test"):
                    held_out = create_sample_dataset(
                        samples=[{"patient_id": split, "signal": series([[1000]])}],
                        input_schema=schema,
                        output_schema={},
                        input_processors=train.input_processors,
                    )
                    result = held_out[0]["signal"]
                    result = result["value"] if isinstance(result, dict) else result
                    np.testing.assert_array_equal(result, [[98]])
                    self.assertEqual(vars(proc), before)

    def test_sample_builder_save_load_preserves_normalization(self):
        for alias in ("timeseries", "temporal_timeseries"):
            with self.subTest(processor=alias), tempfile.TemporaryDirectory() as tmp:
                builder = SampleBuilder(
                    {"signal": (alias, {"normalize_strategy": "standard"})}, {}
                )
                builder.fit([{"signal": series([[10], [30]])}])
                path = str(Path(tmp) / "schema.pkl")
                builder.save(path)
                restored = SampleBuilder.load(path)
                value = {"sample": pickle.dumps({"signal": series([[40]])})}
                result = restored.transform(value)["signal"]
                result = result["value"] if isinstance(result, dict) else result
                np.testing.assert_array_equal(result, [[2]])

    def test_cache_fingerprint_contains_training_statistics(self):
        # BaseDataset.set_task serializes vars(processor) this way.
        for cls in PROCESSORS:
            with self.subTest(processor=cls.__name__):
                fingerprints = []
                for values in ([[10], [30]], [[100], [300]], [[10], [30]]):
                    proc = cls(normalize_strategy="standard")
                    proc.fit([{"signal": series(values)}], "signal")
                    fingerprints.append(
                        json.dumps(vars(proc), sort_keys=True, default=str)
                    )
                self.assertNotEqual(fingerprints[0], fingerprints[1])
                self.assertEqual(fingerprints[0], fingerprints[2])


if __name__ == "__main__":
    unittest.main()
