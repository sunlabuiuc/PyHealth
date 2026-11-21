"""Synthetic tests for the Uni-ACS calibration and SHAP pipeline.

No MIMIC-IV files are needed for these tests.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from pyhealth.interpret.uni_acs import UniACS
from examples.uni_acs_utils import aggregate_shap, train_calibrated_rf


def synthetic_data():
    X, y = make_classification(
        n_samples=180, n_features=6, n_informative=4,
        n_redundant=0, random_state=42,
    )
    return pd.DataFrame(X, columns=[f"feature_{i}" for i in range(6)]), y


def fitted_score():
    X, y = synthetic_data()
    X_fit, X_cal, X_test = X.iloc[:100], X.iloc[100:140], X.iloc[140:]
    y_cal = y[100:140]
    # Fixed synthetic SHAP contributions; this tests UniACS, not the explainer.
    contributions = np.column_stack([
        0.8 * X_fit.iloc[:, 0],
        0.5 * X_fit.iloc[:, 1],
        0.3 * X_fit.iloc[:, 2],
        np.zeros((len(X_fit), 3)),
    ])
    score = UniACS(top_k=3, method="quantile", threshold=0.001)
    score.fit(X_fit, contributions, expected_value=0.0)
    return score, X_cal, y_cal, X_test


def test_calibrated_model():
    X, y = synthetic_data()
    model = train_calibrated_rf(X, y, n_estimators=10, cv=5)
    probs = model.predict_proba(X.iloc[:10])[:, 1]
    assert len(model.calibrated_classifiers_) == 5
    assert probs.shape == (10,)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_shap_aggregation():
    X, y = synthetic_data()
    model = train_calibrated_rf(X, y, n_estimators=10, cv=5)
    values, expected = aggregate_shap(model, X.iloc[:8])
    assert values.shape == (8, 6)
    assert np.isfinite(values).all()
    assert np.isfinite(expected)


def test_independent_score_calibration():
    score, X_cal, y_cal, X_test = fitted_score()
    score.calibrate_scores(X_cal, y_cal)
    probs = score.predict_proba(X_test)
    assert probs.shape == (len(X_test),)
    assert np.isfinite(probs).all()
    assert ((probs >= 0) & (probs <= 1)).all()


def test_calibration_does_not_change_scores():
    score, X_cal, y_cal, X_test = fitted_score()
    before = score.predict_score(X_test)
    score.calibrate_scores(X_cal, y_cal)
    after = score.predict_score(X_test)
    np.testing.assert_array_equal(before, after)
