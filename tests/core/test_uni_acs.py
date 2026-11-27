"""Fast, deterministic unit tests for pyhealth.interpret.uni_acs."""

import importlib.util
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyhealth.interpret.uni_acs import (
    UniACS,
    build_score_table,
    find_novel_breakpoints,
    make_clinical_score_table,
    predict_proba_from_score,
    predict_score,
)

HAS_SHAP = importlib.util.find_spec("shap") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


class TestUniACS(unittest.TestCase):
    """Unit tests for UniACS and its functional helpers."""

    def setUp(self) -> None:
        self.X = pd.DataFrame(
            {
                "age": [20, 30, 40, 50, 60, 70, 80, 90],
                "heart_rate": [65, 72, 80, 95, 110, 85, 75, 100],
            }
        )
        self.shap_values = np.array(
            [
                [-0.4, -0.2],
                [-0.3, -0.1],
                [-0.2, 0.0],
                [-0.1, 0.1],
                [0.1, 0.3],
                [0.2, 0.1],
                [0.3, -0.1],
                [0.4, 0.2],
            ]
        )

    def test_build_score_table_rejects_invalid_inputs(self) -> None:
        for data, attributions, message in (
            (self.X.iloc[:0], self.shap_values[:0], "non-empty"),
            (self.X, self.shap_values[:, :1], "same shape"),
            (self.X, np.full(self.X.shape, np.nan), "finite values"),
            (self.X * np.nan, self.shap_values, "non-finite values"),
            (self.X, np.zeros(self.X.shape), "exceed the threshold"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                build_score_table(data, attributions)

    def test_build_score_table_rejects_invalid_parameters(self) -> None:
        for options, message in (
            ({"top_k": 0}, "top_k"),
            ({"threshold": 0}, "threshold"),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                build_score_table(self.X, self.shap_values, **options)

    def test_predict_score_rejects_non_dataframe_and_non_finite_values(self) -> None:
        table, _ = build_score_table(self.X, self.shap_values)
        with self.assertRaisesRegex(TypeError, "pandas DataFrame"):
            predict_score(self.X.to_numpy(), table)
        with self.assertRaisesRegex(ValueError, "non-finite values"):
            predict_score(self.X * np.nan, table)

    def test_calibration_rejects_mismatched_labels(self) -> None:
        scorer = UniACS().fit(self.X, self.shap_values, expected_value=0.0)
        with self.assertRaisesRegex(ValueError, "equal length"):
            scorer.calibrate_scores(self.X, np.array([0, 1]))

    def test_scipy_spline_fallbacks_find_a_zero_crossing(self) -> None:
        values = np.linspace(-2.0, 2.0, 101)
        with patch("pyhealth.interpret.uni_acs._csaps", None):
            for n_bins in (3, 10):
                with self.subTest(n_bins=n_bins):
                    breaks = find_novel_breakpoints(values, values, n_bins=n_bins)
                    self.assertTrue(np.any(np.abs(breaks) < 0.25), breaks)

    def test_fit_from_data_explains_missing_shap_dependency(self) -> None:
        with (
            patch.dict("sys.modules", {"shap": None}),
            self.assertRaisesRegex(ImportError, "required for UniACS.fit_from_data"),
        ):
            UniACS().fit_from_data(self.X, np.zeros(len(self.X)))

    def test_build_score_table(self) -> None:
        table, unit = build_score_table(
            self.X, self.shap_values, top_k=2, threshold=0.01
        )
        self.assertEqual(len(table["feature"].unique()), 2)
        self.assertGreater(unit, 0.0)
        self.assertTrue(np.issubdtype(table["points"].dtype, np.integer))
        self.assertTrue(np.isfinite(table["mean_shap"]).all())

    def test_top_k_selection(self) -> None:
        table, _ = build_score_table(
            self.X, self.shap_values, top_k=1, threshold=0.01
        )
        self.assertEqual(len(table["feature"].unique()), 1)
        self.assertEqual(table["feature"].iloc[0], "age")

    def test_predict_score(self) -> None:
        table, _ = build_score_table(
            self.X, self.shap_values, top_k=2, threshold=0.01
        )
        scores = predict_score(self.X, table)
        self.assertEqual(len(scores), len(self.X))
        self.assertTrue(np.issubdtype(scores.dtype, np.integer))

    def test_predict_clinical_score_non_negative(self) -> None:
        table, _ = build_score_table(
            self.X, self.shap_values, top_k=2, threshold=0.01
        )
        clinical_scores = predict_score(self.X, table, clinical=True)
        self.assertEqual(len(clinical_scores), len(self.X))
        self.assertTrue((clinical_scores >= 0).all())

    def test_score_boundaries(self) -> None:
        table = pd.DataFrame(
            {
                "feature": ["age", "age", "age"],
                "group": [0, 1, 2],
                "upper_bound": [40.0, 60.0, np.inf],
                "points": [0, 2, 5],
            }
        )
        X = pd.DataFrame({"age": [30, 40, 50, 60, 70]})
        scores = predict_score(X, table)
        np.testing.assert_array_equal(scores, [0, 0, 2, 2, 5])

    def test_binary_and_out_of_range_values_do_not_overflow(self) -> None:
        """Ensure values above training max do not trigger IndexError."""
        X_train = pd.DataFrame({"flag": [0, 0, 0, 1, 1, 1]})
        shap_train = np.array([[-0.2], [-0.2], [-0.1], [0.2], [0.2], [0.3]])
        table, _ = build_score_table(
            X_train, shap_train, top_k=1, threshold=0.01, method="quantile"
        )
        self.assertTrue(np.isinf(table["upper_bound"].iloc[-1]))

        X_test = pd.DataFrame({"flag": [-1, 0, 1, 5]})
        scores = predict_score(X_test, table)
        self.assertEqual(len(scores), 4)
        self.assertTrue(np.issubdtype(scores.dtype, np.integer))

    def test_clinical_score_shift(self) -> None:
        table, _ = build_score_table(
            self.X, self.shap_values, top_k=2, threshold=0.01
        )
        clinical_table = make_clinical_score_table(table)
        self.assertTrue((clinical_table["clinical_points"] >= 0).all())
        minimums = clinical_table.groupby("feature")["clinical_points"].min()
        self.assertTrue((minimums == 0).all())

    def test_probability_conversion(self) -> None:
        scores = np.array([-2, 0, 2])
        probs = predict_proba_from_score(scores, 0.5, 0.0)
        self.assertTrue(np.all((probs > 0) & (probs < 1)))
        self.assertTrue(np.all(np.diff(probs) > 0))
        self.assertAlmostEqual(float(probs[1]), 0.5, places=6)

    def test_novel_breakpoints(self) -> None:
        values = np.linspace(-3.0, 3.0, 100)
        shap_values = values.copy()
        breaks = find_novel_breakpoints(values, shap_values)
        self.assertGreaterEqual(len(breaks), 1)
        self.assertTrue(np.all(np.isfinite(breaks)))
        self.assertTrue(np.all(np.diff(breaks) > 0))
        self.assertTrue(np.all((breaks > values.min()) & (breaks < values.max())))

    def test_novel_breakpoints_few_unique_values(self) -> None:
        values = np.array([0.0, 0.0, 1.0, 1.0])
        shap_values = np.array([-0.1, -0.1, 0.1, 0.1])
        breaks = find_novel_breakpoints(values, shap_values)
        self.assertEqual(len(breaks), 0)

    def test_uni_acs_class_methods(self) -> None:
        for method in ("quantile", "novel"):
            with self.subTest(method=method):
                model = UniACS(top_k=2, threshold=0.01, method=method)
                model.fit(self.X, self.shap_values, expected_value=0.0)

                scores = model.predict_score(self.X)
                clinical = model.predict_score(self.X, clinical=True)
                probs = model.predict_proba(self.X)
                table = model.get_score_table()

                self.assertEqual(len(scores), len(self.X))
                self.assertEqual(len(clinical), len(self.X))
                self.assertTrue((clinical >= 0).all())
                self.assertEqual(len(probs), len(self.X))
                self.assertTrue(np.isfinite(probs).all())
                self.assertGreater(len(table), 0)

    def test_predict_before_fit(self) -> None:
        model = UniACS()
        with self.assertRaises(ValueError):
            model.predict_score(self.X)
        with self.assertRaises(ValueError):
            model.get_score_table()

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            build_score_table(self.X, self.shap_values, method="invalid")
        with self.assertRaises(ValueError):
            UniACS(top_k=0)
        with self.assertRaises(ValueError):
            UniACS(threshold=0.0)
        with self.assertRaises(ValueError):
            find_novel_breakpoints(
                np.array([1.0, 2.0, np.nan]),
                np.array([0.1, 0.2, 0.3]),
            )
        with self.assertRaises(ValueError):
            find_novel_breakpoints(
                np.array([1.0, 2.0, 3.0]),
                np.array([0.1, 0.2, 0.3]),
                n_bins=2,
            )

    def test_missing_feature_column_raises(self) -> None:
        model = UniACS(top_k=2, threshold=0.01, method="quantile")
        model.fit(self.X, self.shap_values, expected_value=0.0)
        with self.assertRaises(ValueError):
            model.predict_score(pd.DataFrame({"age": [30, 50]}))

    @unittest.skipUnless(
        HAS_SHAP and HAS_SKLEARN,
        "Optional 'shap' and 'scikit-learn' packages are required.",
    )
    def test_fit_from_data(self) -> None:
        from sklearn.datasets import make_classification

        X, y = make_classification(
            n_samples=100,
            n_features=5,
            n_informative=3,
            n_redundant=0,
            random_state=42,
        )
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

        model = UniACS(top_k=3, threshold=0.01, method="quantile")
        model.fit_from_data(X_df, y)

        scores = model.predict_score(X_df[:10])
        probs = model.predict_proba(X_df[:10])

        self.assertEqual(len(scores), 10)
        self.assertEqual(len(probs), 10)
        self.assertTrue(np.isfinite(probs).all())
        self.assertIsNotNone(model.model)
        self.assertGreater(len(model.get_score_table()), 0)


if __name__ == "__main__":
    unittest.main()
