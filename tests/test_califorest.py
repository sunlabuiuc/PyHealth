# Authors: Cesar Jesus Giglio Badoino (cesarjg2@illinois.edu)
#          Arjun Tangella (avtange2@illinois.edu)
#          Tony Nguyen (tonyln2@illinois.edu)
# Paper: CaliForest: Calibrated Random Forest for Health Data
# Paper link: https://doi.org/10.1145/3368555.3384461
# Description: Tests for CaliForest model
"""Tests for the CaliForest model."""

import numpy as np
import torch
import pytest

from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.califorest import CaliForest


def _make_dataset(n=40, d=10):
    """Create a small synthetic binary classification dataset."""
    np.random.seed(42)
    samples = []
    for i in range(n):
        label = int(i >= n // 2)
        feats = np.random.randn(d).tolist()
        if label == 1:
            feats = [f + 1.0 for f in feats]
        samples.append(
            {
                "patient_id": f"p{i}",
                "visit_id": f"v{i}",
                "features": feats,
                "label": label,
            }
        )
    return create_sample_dataset(
        samples=samples,
        input_schema={"features": "tensor"},
        output_schema={"label": "binary"},
        dataset_name="synth",
    )


class TestCaliForestInstantiation:
    def test_default_params(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=10,
        )
        assert model.n_estimators == 10
        assert model.is_fitted_ is False

    def test_invalid_ctype_still_creates(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            ctype="isotonic",
        )
        assert model.ctype == "isotonic"


class TestCaliForestForward:
    def test_unfitted_forward(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=5,
        )
        loader = get_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader))
        out = model(**batch)
        assert "loss" in out
        assert "y_prob" in out
        assert "y_true" in out
        assert "logit" in out

    def test_fitted_forward_isotonic(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=10,
            max_depth=3,
            ctype="isotonic",
        )
        loader = get_dataloader(ds, batch_size=40, shuffle=False)
        model.fit(loader)
        assert model.is_fitted_

        loader2 = get_dataloader(ds, batch_size=8, shuffle=False)
        batch = next(iter(loader2))
        out = model(**batch)
        assert out["y_prob"].shape[0] == 8
        assert (out["y_prob"] >= 0).all()
        assert (out["y_prob"] <= 1).all()

    def test_fitted_forward_logistic(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=10,
            max_depth=3,
            ctype="logistic",
        )
        loader = get_dataloader(ds, batch_size=40, shuffle=False)
        model.fit(loader)
        assert model.is_fitted_

        batch = next(iter(loader))
        out = model(**batch)
        assert (out["y_prob"] >= 0).all()
        assert (out["y_prob"] <= 1).all()

    def test_fitted_forward_beta(self):
        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=10,
            max_depth=3,
            ctype="beta",
        )
        loader = get_dataloader(ds, batch_size=40, shuffle=False)
        model.fit(loader)
        assert model.is_fitted_

        batch = next(iter(loader))
        out = model(**batch)
        assert (out["y_prob"] >= 0).all()
        assert (out["y_prob"] <= 1).all()


class TestCaliForestNumpyFit:
    def test_fit_and_predict(self):
        np.random.seed(0)
        X = np.random.randn(60, 5)
        y = (X[:, 0] > 0).astype(int)

        ds = _make_dataset()
        model = CaliForest(
            dataset=ds,
            feature_keys=["features"],
            label_key="label",
            n_estimators=20,
            max_depth=3,
        )
        model._fit_numpy(X, y)
        proba = model._predict_proba_numpy(X)
        assert proba.shape == (60,)
        assert np.all(proba >= 0) and np.all(proba <= 1)
