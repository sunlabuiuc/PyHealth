from __future__ import annotations

import numpy as np

from pyhealth.datasets.wesad_nonstationary import apply_nonstationarity


def test_apply_nonstationarity_none_mode():
    x = np.linspace(0.0, 1.0, 20)
    y = apply_nonstationarity(x, mode="none")
    assert y.shape == x.shape
    assert np.allclose(x, y)


def test_apply_nonstationarity_mean_changes_values():
    x = np.linspace(0.0, 1.0, 20)
    y = apply_nonstationarity(
        x,
        mode="learned",
        change_type="mean",
        magnitude=0.5,
        duration_ratio=0.3,
        random_state=42,
    )
    assert y.shape == x.shape
    assert not np.allclose(x, y)


def test_apply_nonstationarity_std_changes_values():
    x = np.linspace(0.0, 1.0, 20)
    y = apply_nonstationarity(
        x,
        mode="learned",
        change_type="std",
        magnitude=0.5,
        duration_ratio=0.3,
        random_state=42,
    )
    assert y.shape == x.shape
    assert not np.allclose(x, y)