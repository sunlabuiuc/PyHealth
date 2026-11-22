# Contributor: Xiao Luo, UIUC NetID xiaoluo6.
# Paper: Temporal Pointwise Convolutional Networks for Length of Stay
# Prediction in the Intensive Care Unit. https://arxiv.org/abs/2007.09483
"""Synthetic cohort and hourly-label regression tests.

Reference: https://arxiv.org/abs/2007.09483. No clinical records used.
"""

import numpy as np
import pytest

from pyhealth.tasks import TPCLengthOfStayEICU
from tests.core.tpc_fixtures import _patient, _labels, _lab


@pytest.mark.parametrize(
    "hours,age,eligible",
    [
        (4.99, "60", False),
        (5, "18", True),
        (5.5, "60", True),
        (12, "17", False),
        (12, None, False),
        (12, "> 89", True),
        (12, "invalid", False),
        (-1, "60", False),
    ],
)
def test_cohort(hours, age, eligible):
    samples = TPCLengthOfStayEICU()(_patient(hours=hours, age=age, hr_hours=[]))
    assert bool(samples) == eligible


def test_static_features():
    sample = TPCLengthOfStayEICU()(_patient(hours=5, age="> 89", gender="Female"))[0]
    assert sample["static"]["values"] == [[90, 0, 180, 80]]
    missing = TPCLengthOfStayEICU()(
        _patient(hours=5, gender="Unknown", height=-1, weight=999)
    )[0]
    assert np.isnan(missing["static"]["values"][0][1:]).all()


def test_labels_and_horizon():
    task = TPCLengthOfStayEICU()
    sample = task(_patient(hours=400.5, hr_hours=[0, 1]))[0]
    assert sample["los"]["time"] == list(range(5, 337))
    assert _labels(sample["los"])[0] == 395.5
    assert _labels(sample["los"])[-1] == 64.5
    assert sample["dynamic"]["values"].shape == (336, 48)
    assert _labels(task(_patient(hours=5, hr_hours=[]))[0]["los"]) == [0]


def test_fractional_remaining_hours():
    sample = TPCLengthOfStayEICU()(_patient(hours=6.5, hr_hours=[]))[0]
    assert sample["los"]["time"] == [5, 6]
    assert _labels(sample["los"]) == [1.5, 0.5]


@pytest.mark.skip(reason="TODO(B): enable after masking/fill/decay integration")
def test_missingness_count_repeatability_no_mutation():
    values = np.arange(20, dtype=float).reshape(5, 4)
    values[0, 0] = np.nan
    original = values.copy()
    for rate in [0, 0.1, 0.2, 1]:
        task = TPCLengthOfStayEICU(rate)
        result = task._apply_missingness(values, 10)
        assert np.isnan(result).sum() == 1 + round(19 * rate)
        np.testing.assert_equal(result, task._apply_missingness(values, 10))
        np.testing.assert_equal(values, original)
    np.testing.assert_equal(
        TPCLengthOfStayEICU()._apply_missingness(values, 10), values
    )


@pytest.mark.parametrize("rate", [-0.1, 1.1, float("nan")])
def test_invalid_rate(rate):
    with pytest.raises(ValueError):
        TPCLengthOfStayEICU(rate)


@pytest.mark.skip(reason="TODO(B): enable after masking/fill/decay integration")
def test_hourly_measurements_and_missing_data():
    task = TPCLengthOfStayEICU()
    p = _patient(
        hours=6, hr_hours=[], extra=[_lab(60, "potassium", 4), _lab(90, "potassium", 6)]
    )
    sample = task(p)[0]
    k = task.DYNAMIC_FEATURES.index("k")
    assert sample["dynamic"]["values"][1, k] == 5
    assert np.isnan(sample["dynamic"]["values"][0, k])
    assert np.isnan(sample["dynamic"]["values"][2, k])
    masked = TPCLengthOfStayEICU(1)(p)[0]
    assert np.isnan(masked["dynamic"]["values"]).all()
    assert masked["los"] == sample["los"]
    assert masked["static"] == sample["static"]


@pytest.mark.skip(reason="TODO(B): enable after masking/fill/decay integration")
def test_forward_fill_after_removal():
    from pyhealth.processors import JaggedFeatureProcessor

    values = {
        "time": [0.0, 1.0, 2.0, 3.0],
        "values": [[80.0], [np.nan], [100.0], [np.nan]],
    }
    proc = JaggedFeatureProcessor(
        levels=[
            {
                "ordering": "series",
                "values": [1],
                "impute": [{"method": "forward_fill"}, {"method": "zero"}],
            }
        ]
    )
    proc.fit([{"x": values}], "x")
    result = proc.process(values)
    np.testing.assert_allclose(result[0].reshape(-1).numpy(), [80, 80, 100, 100])
    removed = {"time": values["time"], "values": [[80.0], [np.nan], [np.nan], [np.nan]]}
    np.testing.assert_allclose(proc.process(removed)[0].reshape(-1).numpy(), [80] * 4)


@pytest.mark.skip(reason="TODO(B): enable after masking/fill/decay integration")
def test_decay_oracle_and_full_removal():
    raw = np.array(
        [[np.nan, 80], [4, np.nan], [np.nan, 100], [np.nan, np.nan], [3, np.nan]],
        dtype=float,
    )
    expected = [[0, 1], [1, 0.75], [0.75, 1], [0.375, 0.75], [1, 0.375]]
    np.testing.assert_allclose(TPCLengthOfStayEICU._decay(raw), expected)
    removed = TPCLengthOfStayEICU(1)._apply_missingness(raw, 10)
    assert not TPCLengthOfStayEICU._decay(removed).any()


def test_decay_optional_schema_does_not_mutate_other_instances():
    assert "decay" in TPCLengthOfStayEICU(include_decay=True).input_schema
    assert "decay" not in TPCLengthOfStayEICU().input_schema


def test_future_observations_do_not_change_earlier_hours():
    task = TPCLengthOfStayEICU()
    first = task(_patient(hours=12, hr_hours=[0, 2]))[0]
    future = task(_patient(hours=12, hr_hours=[0, 2, 10]))[0]
    for field in ["dynamic"]:
        np.testing.assert_equal(
            first[field]["values"][:10], future[field]["values"][:10]
        )


def test_stays_are_isolated():
    from pyhealth.data import Patient
    from tests.core.tpc_fixtures import _frame, _stay_rows

    task = TPCLengthOfStayEICU()
    rows = _stay_rows(10, 8, hr_hours=[0]) + _stay_rows(11, 8, hr_hours=[])
    samples = {
        s["stay_id"]: s
        for s in task(Patient(patient_id="same-patient", data_source=_frame(rows)))
    }
    assert np.isfinite(samples[10]["dynamic"]["values"]).any()
    assert np.isnan(samples[11]["dynamic"]["values"]).all()


def test_empty_patient():
    from pyhealth.data import Patient
    from tests.core.tpc_fixtures import _frame, _event

    # Patient with no patient-table admission record.
    patient = Patient(patient_id="empty", data_source=_frame([_event("lab")]))
    assert TPCLengthOfStayEICU()(patient) == []


@pytest.mark.parametrize("rate", [0, 0.1, 0.2, 1])
@pytest.mark.skip(reason="TODO(B): enable after masking/fill/decay integration")
def test_combined_mask_fill_and_decay(rate):
    from pyhealth.processors import JaggedFeatureProcessor

    patient = _patient(hours=8, hr_hours=[0, 2, 4, 6])
    task = TPCLengthOfStayEICU(missing_rate=rate, include_decay=True)
    sample = task(patient)[0]
    original = TPCLengthOfStayEICU()(patient)[0]["dynamic"]["values"]
    raw = sample["dynamic"]["values"]
    assert np.isfinite(original).sum() - np.isfinite(raw).sum() == round(4 * rate)
    np.testing.assert_equal(raw, task(patient)[0]["dynamic"]["values"])
    proc = JaggedFeatureProcessor(
        levels=[
            {
                "ordering": "series",
                "values": [48],
                "impute": [{"method": "forward_fill"}, {"method": "zero"}],
            }
        ]
    )
    proc.fit([{"x": sample["dynamic"]}], "x")
    filled = proc.process(sample["dynamic"])[0].numpy()
    hr = task.DYNAMIC_FEATURES.index("hr")
    # A retained value must carry forward until the next retained observation;
    # no measurements before the first retained observation may be invented.
    retained = np.flatnonzero(np.isfinite(raw[:, hr]))
    if len(retained):
        assert not filled[: retained[0], hr].any()
        assert not sample["decay"]["values"][: retained[0], hr].any()
        for start, end in zip(retained, [*retained[1:], len(raw)]):
            np.testing.assert_allclose(filled[start:end, hr], raw[start, hr])
            assert sample["decay"]["values"][start, hr] == 1
    else:
        assert not filled.any()
        assert not sample["decay"]["values"].any()


def test_incomplete_hour_cannot_enter_prediction():
    task = TPCLengthOfStayEICU()
    baseline = task(_patient(hours=8, hr_hours=[]))[0]
    changed = task(_patient(hours=8, hr_hours=[], extra=[_lab(359, "potassium", 5)]))[0]
    for sample in [baseline, changed]:
        assert sample["dynamic"]["time"] == list(range(1, 9))
    for field in ["dynamic"]:
        times = np.asarray(changed[field]["time"])
        np.testing.assert_equal(
            baseline[field]["values"][times <= 5], changed[field]["values"][times <= 5]
        )
    k = task.DYNAMIC_FEATURES.index("k")
    assert changed["dynamic"]["values"][5, k] == 5
    assert changed["dynamic"]["time"][5] == 6


def test_zero_removal_baseline_and_unfinished_extensions():
    raw = np.array([[80.0, np.nan]])
    result = TPCLengthOfStayEICU()._apply_missingness(raw, 10)
    np.testing.assert_equal(raw, result)
    assert not np.shares_memory(raw, result)
    with pytest.raises(NotImplementedError, match="TODO"):
        TPCLengthOfStayEICU(0.1)._apply_missingness(raw, 10)
    with pytest.raises(NotImplementedError, match="TODO"):
        TPCLengthOfStayEICU._decay(raw)
