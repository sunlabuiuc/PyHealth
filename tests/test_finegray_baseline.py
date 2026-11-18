"""Fast, synthetic unit tests for examples/neuralfg/finegray_baseline.py.

Run: python -m pytest tests/test_finegray_baseline.py -v
"""

import numpy as np
import pandas as pd
import pytest

from examples.neuralfg.finegray_baseline import (
    FineGrayBaseline,
    evaluate_finegray,
)


@pytest.fixture
def sample_data():
    rng = np.random.default_rng(45)
    n = 140
    x = rng.normal(size=(n, 3))
    t1 = rng.exponential(5.0 / np.exp(0.6 * x[:, 0]))
    t2 = rng.exponential(6.0 / np.exp(-0.5 * x[:, 1]))
    censor = rng.exponential(14, size=n)
    event = np.where(censor < np.minimum(t1, t2), 0, np.where(t1 < t2, 1, 2))
    time = np.floor(np.minimum(np.minimum(t1, t2), censor))
    df = pd.DataFrame(x, columns=["feature1", "feature2", "feature3"])
    df["time"] = time
    df["label"] = event
    return df.iloc[:100].copy(), df.iloc[100:].copy()


def test_fit_and_predict_shapes(sample_data):
    train, validation = sample_data
    model = FineGrayBaseline().fit(train)
    predictions = model.predict_cif(validation, [1, 3, 8])
    assert predictions.shape == (len(validation), 3, 2)
    assert np.isfinite(predictions).all()
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_cif_nondecreasing(sample_data):
    train, validation = sample_data
    model = FineGrayBaseline().fit(train)
    predictions = model.predict_cif(validation, [0, 1, 3, 8, 20])
    assert (np.diff(predictions, axis=1) >= -1e-12).all()


def test_risk_fit_and_coefficients(sample_data):
    train, _ = sample_data
    model = FineGrayBaseline().fit(train)
    assert model.coefficients().shape == (3, 3)
    assert len(model.fit_summary()) == 2
    assert np.isfinite(model.coefficients().iloc[:, 1:].to_numpy()).all()


@pytest.mark.filterwarnings("ignore:Independently fitted Fine-Gray CIFs sum to more than 1:RuntimeWarning")
def test_evaluation_matches_metric_schema(sample_data):
    train, validation = sample_data
    model = FineGrayBaseline().fit(train)
    result = evaluate_finegray(model, train, validation, evaluation_times=[1, 3, 8])
    assert len(result) == 6
    assert set(result.risk) == {1, 2}
    assert set(result.time) == {1.0, 3.0, 8.0}
    assert (result.n_test == len(validation)).all()
    assert set(["c_index", "brier_paper", "brier_competing"]).issubset(result.columns)


def test_overlap_rejected(sample_data):
    train, _ = sample_data
    model = FineGrayBaseline().fit(train)
    with pytest.raises(ValueError, match="overlap"):
        evaluate_finegray(model, train, train, [1, 3])


def test_invalid_times_rejected(sample_data):
    train, val = sample_data
    model = FineGrayBaseline().fit(train)
    with pytest.raises(ValueError, match="increasing"):
        model.predict_cif(val, [3, 1])


def test_prediction_before_fitting_rejected(sample_data):
    _, val = sample_data
    with pytest.raises(RuntimeError, match="fit"):
        FineGrayBaseline().predict_cif(val, [1, 3])


def test_outcome_leakage_avoided(sample_data):
    train, validation = sample_data
    model = FineGrayBaseline().fit(train)
    prediction_before = model.predict_cif(validation, [1, 3, 8])
    changed = validation.copy()
    changed["time"] = 999
    changed["label"] = 2
    prediction_after = model.predict_cif(changed, [1, 3, 8])
    np.testing.assert_allclose(prediction_before, prediction_after)
