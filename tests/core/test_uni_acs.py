import numpy as np
import pandas as pd
import pytest

from pyhealth.interpret.uni_acs import (
    UniACS,
    build_score_table,
    find_novel_breakpoints,
    make_clinical_score_table,
    predict_proba_from_score,
    predict_score,
)


@pytest.fixture
def sample_data():
    X = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 70, 80, 90],
        "heart_rate": [65, 72, 80, 95, 110, 85, 75, 100],
    })

    shap_values = np.array([
        [-0.4, -0.2],
        [-0.3, -0.1],
        [-0.2, 0.0],
        [-0.1, 0.1],
        [0.1, 0.3],
        [0.2, 0.1],
        [0.3, -0.1],
        [0.4, 0.2],
    ])

    return X, shap_values


def test_build_score_table(sample_data):
    X, shap_values = sample_data

    table, unit = build_score_table(
        X, shap_values, top_k=2, threshold=0.01
    )

    assert len(table["feature"].unique()) == 2
    assert unit > 0
    assert np.issubdtype(table["points"].dtype, np.integer)
    assert np.isfinite(table["mean_shap"]).all()


def test_top_k_selection(sample_data):
    X, shap_values = sample_data

    table, _ = build_score_table(
        X, shap_values, top_k=1, threshold=0.01
    )

    assert len(table["feature"].unique()) == 1
    assert table["feature"].iloc[0] == "age"


def test_predict_score(sample_data):
    X, shap_values = sample_data

    table, _ = build_score_table(
        X, shap_values, top_k=2, threshold=0.01
    )

    scores = predict_score(X, table)

    assert len(scores) == len(X)
    assert np.issubdtype(scores.dtype, np.integer)


def test_score_boundaries():
    table = pd.DataFrame({
        "feature": ["age", "age", "age"],
        "group": [0, 1, 2],
        "upper_bound": [40, 60, np.inf],
        "points": [0, 2, 5],
    })

    X = pd.DataFrame({"age": [30, 40, 50, 60, 70]})

    scores = predict_score(X, table)

    np.testing.assert_array_equal(scores, [0, 0, 2, 2, 5])


def test_clinical_score_shift(sample_data):
    X, shap_values = sample_data

    table, _ = build_score_table(
        X, shap_values, top_k=2, threshold=0.01
    )

    clinical_table = make_clinical_score_table(table)

    assert (clinical_table["clinical_points"] >= 0).all()

    minimums = clinical_table.groupby("feature")[
        "clinical_points"
    ].min()

    assert (minimums == 0).all()


def test_probability_conversion():
    scores = np.array([-2, 0, 2])
    probs = predict_proba_from_score(scores, 0.5, 0.0)

    assert np.all((probs > 0) & (probs < 1))
    assert np.all(np.diff(probs) > 0)
    assert probs[1] == pytest.approx(0.5)


def test_novel_breakpoints():
    values = np.linspace(-3, 3, 100)
    shap_values = values.copy()

    breaks = find_novel_breakpoints(values, shap_values)

    assert len(breaks) >= 1
    assert np.all(np.isfinite(breaks))
    assert np.all(np.diff(breaks) > 0)
    assert np.all((breaks > values.min()) & (breaks < values.max()))


@pytest.mark.parametrize("method", ["quantile", "novel"])
def test_uni_acs_class(sample_data, method):
    X, shap_values = sample_data

    model = UniACS(top_k=2, threshold=0.01, method=method)
    model.fit(X, shap_values, expected_value=0.0)

    scores = model.predict_score(X)
    probs = model.predict_proba(X)
    table = model.get_score_table()

    assert len(scores) == len(X)
    assert len(probs) == len(X)
    assert np.isfinite(probs).all()
    assert len(table) > 0


def test_predict_before_fit(sample_data):
    X, _ = sample_data
    model = UniACS()

    with pytest.raises(ValueError):
        model.predict_score(X)


def test_invalid_method(sample_data):
    X, shap_values = sample_data

    with pytest.raises(ValueError):
        build_score_table(
            X, shap_values, method="invalid"
        )


def test_invalid_novel_input():
    values = np.array([1.0, 2.0, np.nan])
    shap_values = np.array([0.1, 0.2, 0.3])

    with pytest.raises(ValueError):
        find_novel_breakpoints(values, shap_values)


def test_fit_from_data():
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=0,
        random_state=42,
    )
    X = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])

    model = UniACS(top_k=3, threshold=0.01, method="quantile")
    model.fit_from_data(X, y)

    scores = model.predict_score(X[:10])
    probs = model.predict_proba(X[:10])

    assert len(scores) == 10
    assert len(probs) == 10
    assert np.isfinite(probs).all()
    assert model.model is not None
    assert len(model.get_score_table()) > 0
