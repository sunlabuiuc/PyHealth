"""
Contributors: Manling Tan, Yining Jin, Xiao Luo
NetID: manling4, yining34, xiaoluo6

Paper Title:
Temporal Pointwise Convolutional Networks for
Length of Stay Prediction in the Intensive Care Unit

Paper Authors:
Emma Rocheteau, Pietro Liò, and Stephanie Hyland (2021)

Paper URL:
https://arxiv.org/abs/2007.09483
"""

"""Synthetic cohort, hourly-label, and preprocessing regression tests.

Tests for the TPCLengthOfStayEICU task using synthetic patient data.
No real clinical records are used.

Coverage includes cohort eligibility, hourly LOS labels, truncation,
missingness, decay, temporal leakage prevention, and stay isolation.
"""

import numpy as np
import pytest

from pyhealth.tasks import TPCLengthOfStayEICU
from tests.core.tpc_fixtures import _lab, _labels, _patient


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
    """Check adult cohort and minimum ICU stay requirements."""
    samples = TPCLengthOfStayEICU()(_patient(hours=hours, age=age, hr_hours=[]))
    assert bool(samples) == eligible


def test_static_features():
    """Verify static features and invalid-value handling."""
    sample = TPCLengthOfStayEICU()(_patient(hours=5, age="> 89", gender="Female"))[0]

    assert sample["static"]["values"] == [[90, 0, 180, 80]]

    missing = TPCLengthOfStayEICU()(
        _patient(
            hours=5,
            gender="Unknown",
            height=-1,
            weight=999,
        )
    )[0]

    assert np.isnan(missing["static"]["values"][0][1:]).all()


def test_labels_and_horizon():
    """Verify hourly remaining LOS and 14-day truncation."""
    task = TPCLengthOfStayEICU()

    sample = task(_patient(hours=400.5, hr_hours=[0, 1]))[0]

    assert sample["los"]["time"] == list(range(5, 337))
    assert _labels(sample["los"])[0] == 395.5
    assert _labels(sample["los"])[-1] == 64.5
    assert sample["dynamic"]["values"].shape == (336, 48)

    short_sample = task(_patient(hours=5, hr_hours=[]))[0]

    assert _labels(short_sample["los"]) == [0]


def test_fractional_remaining_hours():
    """Check remaining LOS for fractional stay durations."""
    sample = TPCLengthOfStayEICU()(_patient(hours=6.5, hr_hours=[]))[0]

    assert sample["los"]["time"] == [5, 6]
    assert _labels(sample["los"]) == [1.5, 0.5]


def test_missingness_count_repeatability_no_mutation():
    """Verify masking fraction, reproducibility, and input safety."""
    values = np.arange(20, dtype=float).reshape(5, 4)
    values[0, 0] = np.nan

    original = values.copy()

    for rate in [0, 0.1, 0.2, 1]:
        task = TPCLengthOfStayEICU(missing_rate=rate)

        result = task._apply_missingness(values, 10)

        assert np.isnan(result).sum() == (1 + round(19 * rate))

        np.testing.assert_equal(
            result,
            task._apply_missingness(values, 10),
        )

        np.testing.assert_equal(values, original)

        assert not np.shares_memory(values, result)

    np.testing.assert_equal(
        TPCLengthOfStayEICU()._apply_missingness(values, 10),
        values,
    )


@pytest.mark.parametrize(
    "rate",
    [-0.1, 1.1, float("nan")],
)
def test_invalid_rate(rate):
    """Reject missingness rates outside the valid interval."""
    with pytest.raises(ValueError):
        TPCLengthOfStayEICU(missing_rate=rate)


def test_hourly_measurements_and_missing_data():
    """Check hourly lab aggregation and masking."""
    task = TPCLengthOfStayEICU()

    patient = _patient(
        hours=6,
        hr_hours=[],
        extra=[
            _lab(60, "potassium", 4),
            _lab(90, "potassium", 6),
        ],
    )

    sample = task(patient)[0]
    k = task.DYNAMIC_FEATURES.index("k")

    assert sample["dynamic"]["values"][1, k] == 5
    assert np.isnan(sample["dynamic"]["values"][0, k])
    assert np.isnan(sample["dynamic"]["values"][2, k])

    masked = TPCLengthOfStayEICU(missing_rate=1)(patient)[0]

    assert np.isnan(masked["dynamic"]["values"]).all()

    assert masked["los"] == sample["los"]
    assert masked["static"] == sample["static"]


def test_forward_fill_after_removal():
    """Check forward-fill behavior after missingness removal."""
    from pyhealth.processors import JaggedFeatureProcessor

    values = {
        "time": [0.0, 1.0, 2.0, 3.0],
        "values": [
            [80.0],
            [np.nan],
            [100.0],
            [np.nan],
        ],
    }

    proc = JaggedFeatureProcessor(
        levels=[
            {
                "ordering": "series",
                "values": [1],
                "impute": [
                    {"method": "forward_fill"},
                    {"method": "zero"},
                ],
            }
        ]
    )

    proc.fit([{"x": values}], "x")

    result = proc.process(values)

    np.testing.assert_allclose(
        result[0].reshape(-1).numpy(),
        [80, 80, 100, 100],
    )

    removed = {
        "time": values["time"],
        "values": [
            [80.0],
            [np.nan],
            [np.nan],
            [np.nan],
        ],
    }

    np.testing.assert_allclose(
        proc.process(removed)[0].reshape(-1).numpy(),
        [80] * 4,
    )


def test_decay_oracle_and_full_removal():
    """Verify exponential decay and fully missing features."""
    raw = np.array(
        [
            [np.nan, 80],
            [4, np.nan],
            [np.nan, 100],
            [np.nan, np.nan],
            [3, np.nan],
        ],
        dtype=float,
    )

    expected = [
        [0, 1],
        [1, 0.75],
        [0.75, 1],
        [0.5625, 0.75],
        [1, 0.5625],
    ]

    np.testing.assert_allclose(
        TPCLengthOfStayEICU._decay(raw),
        expected,
    )

    removed = TPCLengthOfStayEICU(missing_rate=1)._apply_missingness(raw, 10)

    assert not TPCLengthOfStayEICU._decay(removed).any()


def test_decay_optional_schema_does_not_mutate_other_instances():
    """Ensure optional decay does not modify the class schema."""
    assert "decay" in TPCLengthOfStayEICU(include_decay=True).input_schema

    assert "decay" not in TPCLengthOfStayEICU().input_schema


def test_future_observations_do_not_change_earlier_hours():
    """Ensure future observations do not leak into earlier bins."""
    task = TPCLengthOfStayEICU()

    first = task(_patient(hours=12, hr_hours=[0, 2]))[0]

    future = task(_patient(hours=12, hr_hours=[0, 2, 10]))[0]

    np.testing.assert_equal(
        first["dynamic"]["values"][:10],
        future["dynamic"]["values"][:10],
    )


def test_stays_are_isolated():
    """Ensure measurements cannot leak between ICU stays."""
    from pyhealth.data import Patient
    from tests.core.tpc_fixtures import _frame, _stay_rows

    task = TPCLengthOfStayEICU()

    rows = _stay_rows(10, 8, hr_hours=[0]) + _stay_rows(11, 8, hr_hours=[])

    samples = {
        sample["stay_id"]: sample
        for sample in task(
            Patient(
                patient_id="same-patient",
                data_source=_frame(rows),
            )
        )
    }

    assert np.isfinite(samples[10]["dynamic"]["values"]).any()

    assert np.isnan(samples[11]["dynamic"]["values"]).all()


def test_empty_patient():
    """Return no samples for a patient without admission records."""
    from pyhealth.data import Patient
    from tests.core.tpc_fixtures import _event, _frame

    patient = Patient(
        patient_id="empty",
        data_source=_frame([_event("lab")]),
    )

    assert TPCLengthOfStayEICU()(patient) == []


@pytest.mark.parametrize(
    "rate",
    [0, 0.1, 0.2, 1],
)
def test_combined_mask_fill_and_decay(rate):
    """Check masking, forward-fill, and decay together."""
    from pyhealth.processors import JaggedFeatureProcessor

    patient = _patient(
        hours=8,
        hr_hours=[0, 2, 4, 6],
    )

    task = TPCLengthOfStayEICU(
        missing_rate=rate,
        include_decay=True,
    )

    sample = task(patient)[0]

    original = TPCLengthOfStayEICU()(patient)[0]["dynamic"]["values"]

    raw = sample["dynamic"]["values"]

    assert np.isfinite(original).sum() - np.isfinite(raw).sum() == round(4 * rate)

    np.testing.assert_equal(
        raw,
        task(patient)[0]["dynamic"]["values"],
    )

    proc = JaggedFeatureProcessor(
        levels=[
            {
                "ordering": "series",
                "values": [48],
                "impute": [
                    {"method": "forward_fill"},
                    {"method": "zero"},
                ],
            }
        ]
    )

    proc.fit(
        [{"x": sample["dynamic"]}],
        "x",
    )

    filled = proc.process(sample["dynamic"])[0].numpy()

    hr = task.DYNAMIC_FEATURES.index("hr")

    retained = np.flatnonzero(np.isfinite(raw[:, hr]))

    if len(retained):
        assert not filled[: retained[0], hr].any()

        assert not sample["decay"]["values"][: retained[0], hr].any()

        for start, end in zip(
            retained,
            [*retained[1:], len(raw)],
        ):
            np.testing.assert_allclose(
                filled[start:end, hr],
                raw[start, hr],
            )

            assert sample["decay"]["values"][start, hr] == 1

    else:
        assert not filled.any()
        assert not sample["decay"]["values"].any()


def test_incomplete_hour_cannot_enter_prediction():
    """Prevent measurements from unfinished bins entering predictions."""
    task = TPCLengthOfStayEICU()

    baseline = task(_patient(hours=8, hr_hours=[]))[0]

    changed = task(
        _patient(
            hours=8,
            hr_hours=[],
            extra=[_lab(359, "potassium", 5)],
        )
    )[0]

    for sample in [baseline, changed]:
        assert sample["dynamic"]["time"] == list(range(1, 9))

    times = np.asarray(changed["dynamic"]["time"])

    np.testing.assert_equal(
        baseline["dynamic"]["values"][times <= 5],
        changed["dynamic"]["values"][times <= 5],
    )

    k = task.DYNAMIC_FEATURES.index("k")

    assert changed["dynamic"]["values"][5, k] == 5

    assert changed["dynamic"]["time"][5] == 6


def test_zero_removal_baseline_and_completed_extensions():
    """Verify implemented missingness and decay functionality."""
    raw = np.array([[80.0, np.nan]])

    result = TPCLengthOfStayEICU(missing_rate=0)._apply_missingness(raw, 10)

    np.testing.assert_equal(
        raw,
        result,
    )

    assert not np.shares_memory(
        raw,
        result,
    )

    masked = TPCLengthOfStayEICU(missing_rate=1.0)._apply_missingness(raw, 10)

    assert np.isnan(masked).all()

    decay = TPCLengthOfStayEICU._decay(raw)

    np.testing.assert_allclose(
        decay,
        [[1.0, 0.0]],
    )


def test_output_timeline_with_optional_decay():
    """Ensure temporal features use the correct prediction timeline."""
    with_decay = TPCLengthOfStayEICU(include_decay=True)
    without_decay = TPCLengthOfStayEICU(include_decay=False)

    assert with_decay.output_schema["los"][1]["timeline"] == [
        "dynamic",
        "decay",
    ]
    assert without_decay.output_schema["los"][1]["timeline"] == ["dynamic"]
