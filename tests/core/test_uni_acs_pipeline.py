"""Synthetic pipeline tests for Uni-ACS score calibration and SHAP aggregation.

No protected MIMIC-IV files are needed for these tests.
"""

import importlib.util
import unittest

import numpy as np
import pandas as pd

from pyhealth.interpret.uni_acs import UniACS

HAS_SHAP = importlib.util.find_spec("shap") is not None
HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None


def _synthetic_data() -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(42)
    X = rng.normal(size=(180, 6))
    logits = 0.9 * X[:, 0] + 0.6 * X[:, 1] + 0.4 * X[:, 2]
    probs = 1.0 / (1.0 + np.exp(-logits))
    y = (probs >= 0.5).astype(int)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)])
    return df, y


def _fitted_score() -> tuple[UniACS, pd.DataFrame, np.ndarray, pd.DataFrame]:
    X, y = _synthetic_data()
    X_fit, X_cal, X_test = X.iloc[:100], X.iloc[100:140], X.iloc[140:]
    y_cal = y[100:140]
    # Fixed synthetic SHAP contributions; this tests UniACS, not the explainer.
    contributions = np.column_stack(
        [
            0.8 * X_fit.iloc[:, 0],
            0.5 * X_fit.iloc[:, 1],
            0.3 * X_fit.iloc[:, 2],
            np.zeros((len(X_fit), 3)),
        ]
    )
    score = UniACS(top_k=3, method="quantile", threshold=0.001)
    score.fit(X_fit, contributions, expected_value=0.0)
    return score, X_cal, y_cal, X_test


class TestUniACSPipeline(unittest.TestCase):
    """Pipeline tests for UniACS calibration and optional RF/SHAP helpers."""

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required.")
    def test_independent_score_calibration(self) -> None:
        score, X_cal, y_cal, X_test = _fitted_score()
        score.calibrate_scores(X_cal, y_cal)
        probs = score.predict_proba(X_test)
        self.assertEqual(probs.shape, (len(X_test),))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0.0) & (probs <= 1.0)).all())

    @unittest.skipUnless(HAS_SKLEARN, "scikit-learn is required.")
    def test_calibration_does_not_change_scores(self) -> None:
        score, X_cal, y_cal, X_test = _fitted_score()
        before = score.predict_score(X_test)
        score.calibrate_scores(X_cal, y_cal)
        after = score.predict_score(X_test)
        np.testing.assert_array_equal(before, after)

    @unittest.skipUnless(
        HAS_SHAP and HAS_SKLEARN,
        "Optional 'shap' and 'scikit-learn' packages are required.",
    )
    def test_calibrated_model_and_shap_aggregation(self) -> None:
        # Exercise the actual example workflow with synthetic data.
        from examples.uni_acs_utils import aggregate_shap, train_calibrated_rf

        X, y = _synthetic_data()
        model = train_calibrated_rf(X, y, n_estimators=10, cv=3)
        probs = model.predict_proba(X.iloc[:10])[:, 1]
        self.assertEqual(len(model.calibrated_classifiers_), 3)
        self.assertEqual(probs.shape, (10,))
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(((probs >= 0.0) & (probs <= 1.0)).all())

        values, expected = aggregate_shap(model, X.iloc[:8])
        self.assertEqual(values.shape, (8, 6))
        self.assertTrue(np.isfinite(values).all())
        self.assertTrue(np.isfinite(expected))


if __name__ == "__main__":
    unittest.main()
