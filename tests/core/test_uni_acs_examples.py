"""Test the same metric, era grouping, and feature extraction functions used by the examples.

Run from the repository root. Synthetic inputs keep these checks independent
of MIMIC-IV files.
"""

import numpy as np
import pandas as pd
import pytest

from examples.mimic4_mortality_uni_acs import _average_precision
from examples.uni_acs_mimic31_features import extract_labs
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


def test_extract_labs_keeps_only_results_available_by_cutoff(tmp_path):
    cohort = pd.DataFrame({
        "hadm_id": [1, 2, 3, 4],
        "admittime": pd.Timestamp("2024-01-01 08:00:00"),
        "cutoff": pd.Timestamp("2024-01-02 08:00:00"),
    })
    labs = pd.DataFrame(
        [
            (1, 50813, "2024-01-02 07:00:00", "2024-01-02 07:30:00", 2.0),
            (2, 50813, "2024-01-02 07:00:00", "2024-01-02 08:00:00", 3.0),
            (3, 50813, "2024-01-02 07:00:00", "2024-01-02 08:01:00", 6.0),
            (4, 50813, "2024-01-02 07:00:00", None, 4.0),
        ],
        columns=["hadm_id", "itemid", "charttime", "storetime", "valuenum"],
    )
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    labs.to_csv(hosp / "labevents.csv.gz", index=False)

    result = extract_labs(
        cohort, {50813: "lactate"}, data_path=tmp_path, require_available=True
    )

    assert result.set_index("hadm_id")["lactate"].to_dict() == {1: 2.0, 2: 3.0}
