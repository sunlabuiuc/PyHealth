"""Regression tests for the Uni-ACS examples using synthetic inputs."""

import numpy as np
import pandas as pd
import pytest

from examples.mimic4_mortality_uni_acs import _average_precision
from examples.uni_acs_mimic31_temporal_validation import assign_admission_eras


def test_average_precision_is_invariant_to_tie_order():
    scores = np.array([2, 2, 1, 0])
    for labels in (np.array([1, 0, 1, 0]), np.array([0, 1, 1, 0])):
        assert _average_precision(labels, scores) == pytest.approx(7 / 12)


def test_assign_admission_eras_applies_year_offsets_and_reports_exclusions(capsys):
    cohort = pd.DataFrame({
        "subject_id": [1, 2, 3, 4, 5],
        "anchor_year": [2150] * 5,
        "anchor_year_group": [
            "2017 - 2019",
            "2020 - 2022",
            "2017 - 2019",
            "2020 - 2022",
            None,
        ],
        "admittime": pd.to_datetime([
            "2150-01-01",
            "2150-01-01",
            "2151-01-01",
            "2151-01-01",
            "2150-01-01",
        ]),
    })

    included = assign_admission_eras(cohort)

    assert included["subject_id"].tolist() == [1, 2]
    assert included["recent"].tolist() == [False, True]
    output = capsys.readouterr().out
    assert "Historical (2008-2019): 1" in output
    assert "Recent (2020-2022): 1" in output
    assert "Estimated year range extends outside 2008-2022: 1" in output
    assert "Estimated year range crosses the 2020 boundary: 1" in output
    assert "Missing or invalid year range: 1" in output
    assert "Total excluded: 3" in output
