# Contributor: Xingyao Wang (xingyao6)
# Paper: Deep Cox Mixtures for Survival Regression
# https://proceedings.mlr.press/v149/nagpal21a.html
"""Checks of the study driver's preprocessing, masking and IPCW metrics.

Run from the repository root with ``pytest examples/support2_dcm``; pytest puts
this directory on ``sys.path`` so ``experiment`` imports directly. Skips when
scikit-survival is not installed.
"""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("sksurv", reason="optional dependency of the study driver")

from experiment.data import (  # noqa: E402
    CATEGORICAL,
    LABS,
    NUMERIC,
    features,
    mask_labs,
    preprocessing,
)
from experiment.evaluation import metrics  # noqa: E402
from sksurv.metrics import brier_score, concordance_index_ipcw  # noqa: E402
from sksurv.util import Surv  # noqa: E402


def test_train_only_preprocessing_and_paired_masks() -> None:
    """Preprocessing is fit on train rows; masks are seeded and paired."""
    rng = np.random.default_rng(9)
    frame = pd.DataFrame({key: rng.normal(size=12) for key in NUMERIC})
    for key in CATEGORICAL:
        frame[key] = ["a", "b"] * 6
    frame.loc[0, LABS] = np.nan
    prep = preprocessing(True).fit(features(frame.iloc[:8], True))
    imputer = prep.named_transformers_["numeric"].named_steps["simpleimputer"]
    medians = imputer.statistics_.copy()
    test = frame.iloc[8:].copy()
    a, info = mask_labs(test, 0.3, 71)
    b, _ = mask_labs(test, 0.3, 71)
    pd.testing.assert_frame_equal(a, b)
    assert (a[LABS].isna().sum() - test[LABS].isna().sum()).sum() == info[
        "induced_missing"
    ]
    x = prep.transform(features(a, True))
    assert np.isfinite(x).all()
    np.testing.assert_array_equal(x[:, -8:], a[LABS].isna().to_numpy())
    np.testing.assert_array_equal(medians, imputer.statistics_)
    unseen = a.copy()
    unseen["sex"] = "unseen"
    assert np.isfinite(prep.transform(features(unseen, True))).all()
    original, _ = mask_labs(frame, 0, 71)
    pd.testing.assert_frame_equal(original, frame)
    with pytest.raises(ValueError):
        mask_labs(frame, 1.1, 71)


def test_ipcw_matches_reference_and_handles_late_followup() -> None:
    """metrics() equals sksurv and ignores follow-up past the last horizon."""
    train = Surv.from_arrays([1, 0, 1, 0, 1, 0, 1, 0], [1, 2, 3, 4, 5, 6, 8, 10])
    test = Surv.from_arrays([1, 0, 1, 0], [1, 3, 5, 7])
    horizons = np.array([2.0, 4.0])
    curves = np.array([[0.3, 0.1], [0.8, 0.6], [0.9, 0.7], [0.95, 0.8]])
    result = metrics(train, test, curves, horizons)
    expected = brier_score(train, test, curves, horizons)[1]
    np.testing.assert_allclose(result["brier"], expected)
    for j, t in enumerate(horizons):
        expected_c = concordance_index_ipcw(train, test, 1 - curves[:, j], tau=t)[0]
        np.testing.assert_allclose(result["ipcw_concordance"][j], expected_c)
    late = test.copy()
    late["time"][-1] = 100
    late["event"][-1] = True
    np.testing.assert_allclose(
        metrics(train, late, curves, horizons)["brier"], expected
    )
