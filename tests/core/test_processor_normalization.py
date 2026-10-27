import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.processors import TensorProcessor, TimeseriesProcessor
from pyhealth.processors.utils import fit_normalization

T0 = datetime(2020, 1, 1)
NAN = float("nan")


def _series(values):
    return ([T0 + timedelta(hours=h) for h in range(len(values))], values)


class TestFitNormalization(unittest.TestCase):
    def test_zscore_matches_numpy_across_chunks(self):
        rng = np.random.default_rng(0)
        data = rng.normal(5.0, 2.0, size=(100, 3))
        data[rng.random(data.shape) < 0.3] = np.nan
        center, scale = fit_normalization(np.array_split(data, 7), "zscore")
        np.testing.assert_allclose(center, np.nanmean(data, axis=0))
        np.testing.assert_allclose(scale, np.nanstd(data, axis=0))

    def test_minmax(self):
        data = np.array([[1.0, NAN], [3.0, 2.0], [NAN, 6.0]])
        center, scale = fit_normalization([data[:1], data[1:]], "minmax")
        self.assertEqual(center, [1.0, 2.0])
        self.assertEqual(scale, [2.0, 4.0])

    def test_constant_and_unobserved_features_get_unit_scale(self):
        data = np.array([[4.0, NAN], [4.0, NAN]])
        center, scale = fit_normalization([data], "zscore")
        self.assertEqual(center, [4.0, 0.0])
        self.assertEqual(scale, [1.0, 1.0])

    def test_unknown_method(self):
        with self.assertRaises(ValueError):
            fit_normalization([np.zeros((1, 1))], "robust")


class TestTimeseriesProcessorOptions(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        p = TimeseriesProcessor()
        value = _series([[NAN, 1.0], [2.0, NAN]])
        p.fit([{"x": value}], "x")
        self.assertFalse(p.learns_statistics)
        self.assertEqual(p.size(), 2)
        np.testing.assert_array_equal(
            p.process(value).numpy(), [[0.0, 1.0], [2.0, 1.0]]
        )

    def test_zscore_normalization_with_indicators(self):
        train = [{"x": _series([[1.0], [NAN], [3.0]])}, {"x": _series([[5.0]])}]
        p = TimeseriesProcessor(normalization="zscore", missing_indicators=True)
        p.fit(train, "x")
        self.assertTrue(p.learns_statistics)
        self.assertEqual(p.size(), 2)
        mean, std = 3.0, np.std([1.0, 3.0, 5.0])

        out = p.process(_series([[NAN], [5.0], [NAN]])).numpy()
        # Leading gap -> 0 (the training mean), then forward-filled.
        np.testing.assert_allclose(out[:, 0], [0.0, (5.0 - mean) / std, (5.0 - mean) / std])
        np.testing.assert_array_equal(out[:, 1], [1.0, 0.0, 1.0])

    def test_invalid_normalization(self):
        with self.assertRaises(ValueError):
            TimeseriesProcessor(normalization="robust")


class TestTensorProcessorOptions(unittest.TestCase):
    def test_defaults_are_unchanged(self):
        p = TensorProcessor()
        p.fit([{"x": [1.0, NAN]}], "x")
        self.assertFalse(p.learns_statistics)
        out = p.process([1.0, NAN])
        self.assertTrue(torch.isnan(out[1]))

    def test_normalization_indicators_and_zero_fill(self):
        train = [{"x": [1.0, 10.0]}, {"x": [3.0, NAN]}]
        p = TensorProcessor(
            normalization="zscore", missing_indicators=True, impute_strategy="zero"
        )
        p.fit(train, "x")
        self.assertEqual(p.size(), 4)
        out = p.process([NAN, 10.0])
        np.testing.assert_allclose(out.numpy(), [0.0, 0.0, 1.0, 0.0])
        out = p.process([3.0, NAN])
        np.testing.assert_allclose(out.numpy(), [1.0, 0.0, 0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
