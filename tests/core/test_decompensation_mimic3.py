"""Tests for the MIMIC-III decompensation task (synthetic data only)."""

import math
from datetime import datetime, timedelta

import numpy as np

from pyhealth.tasks.decompensation_mimic3 import (
    _REQUIRED_COLUMNS,
    FEATURES,
    DecompensationMIMIC3,
    age_years,
    apply_range,
    build_sample,
    clean_events,
    clean_value,
    decompensation_steps,
    hourly_matrix,
    in_hospital_mortality,
    select_stays,
    time_to_event_hours,
)


def test_bp_split_and_plain():
    assert clean_value("Systolic blood pressure", "120/80") == 120.0
    assert clean_value("Diastolic blood pressure", "120/80") == 80.0
    assert clean_value("Systolic blood pressure", "bad/") is None
    assert clean_value("Diastolic blood pressure", "70") == 70.0


def test_crr_gcs_text():
    assert clean_value("Capillary refill rate", "Brisk") == 0.0
    assert clean_value("Capillary refill rate", "Delayed") == 1.0
    assert clean_value("Capillary refill rate", "x") is None
    assert clean_value("Glascow coma scale eye opening", "Spontaneously") == 4.0
    assert clean_value("Glascow coma scale verbal response", "zzz") is None


def test_lab_strict_and_o2sat():
    assert clean_value("Glucose", "ERROR") is None
    assert clean_value("Glucose", "<5") is None
    assert clean_value("pH", "7.4") == 7.4
    assert clean_value("Oxygen saturation", "0.95") == 95.0
    assert clean_value("Oxygen saturation", "97") == 97.0


def test_fio2_temperature_weight_height():
    assert clean_value("Fraction inspired oxygen", "40") == 0.4
    assert clean_value("Fraction inspired oxygen", "0.4") == 0.4
    assert clean_value("Fraction inspired oxygen", "40", "torr") == 40.0
    assert math.isclose(clean_value("Temperature", "98.6"), 37.0)
    assert clean_value("Temperature", "37") == 37.0
    # benchmark quirk: low Fahrenheit-labelled values are not converted
    assert clean_value("Temperature", "70", "", "Temperature F") == 70.0
    assert math.isclose(clean_value("Weight", "160", "", "x (oz)"), 160 / 16 * 0.453592)
    assert math.isclose(clean_value("Weight", "100", "lb"), 45.3592)
    assert clean_value("Weight", "70", "kg") == 70.0
    assert clean_value("Height", "70", "", "Height Inches") == 178.0
    assert clean_value("Height", "178", "", "Height (cm)") == 178.0


def test_ranges():
    assert apply_range("Temperature", 20.0) == 26
    assert apply_range("Temperature", 50.0) is None
    assert apply_range("Heart Rate", 100.0) == 100.0


def test_clean_events_maps_and_drops():
    rows = [
        (1.0, "chartevents", 211, "80", ""),
        (1.0, "labevents", 50809, "120", ""),
        (1.0, "chartevents", 999, "1", ""),
        (2.0, "chartevents", 211, "abc", ""),
    ]
    ev = clean_events(rows)
    assert ev == [(1.0, "Heart Rate", 80.0), (1.0, "Glucose", 120.0)]
    assert clean_events([(1.0, "chartevents", 678, "20", "")], True)[0][2] == 26


def test_hourly_matrix_binning():
    hr = FEATURES.index("Heart Rate")
    ev = [(0.0, "Heart Rate", 70.0), (0.5, "Heart Rate", 80.0),
          (0.9, "Heart Rate", 85.0), (1.0, "Heart Rate", 82.0),
          (1.5, "Heart Rate", 90.0), (3.0, "Heart Rate", 99.0)]
    m = hourly_matrix(ev, 2)
    assert m.shape == (3, len(FEATURES))
    assert m[0, hr] == 70.0
    assert m[1, hr] == 82.0  # 0.5, 0.9, 1.0 -> hour 1; latest wins
    assert m[2, hr] == 90.0
    # same instant: max wins
    m2 = hourly_matrix([(2.0, "Heart Rate", 60.0), (2.0, "Heart Rate", 65.0)], 2)
    assert m2[2, hr] == 65.0
    assert np.isnan(m2[0, hr])


def test_mortality_age_stays():
    a, d = datetime(2100, 1, 1), datetime(2100, 1, 10)
    assert in_hospital_mortality(a, d, datetime(2100, 1, 5), None) == 1
    assert in_hospital_mortality(a, d, None, datetime(2100, 1, 9)) == 1
    assert in_hospital_mortality(a, d, None, datetime(2101, 1, 1)) == 0
    assert in_hospital_mortality(a, d, None, None) == 0
    assert age_years(datetime(2100, 1, 1), datetime(2200, 1, 1)) == 90.0
    base = {
        "first_wardid": 1,
        "last_wardid": 1,
        "first_careunit": "M",
        "last_careunit": "M",
        "age": 50.0,
    }
    stays = [
        dict(base, hadm_id=1),
        dict(base, hadm_id=2), dict(base, hadm_id=2),
        dict(base, hadm_id=3, last_wardid=2),
        dict(base, hadm_id=4, age=17.0),
        dict(base, hadm_id=5, last_careunit="S"),
    ]
    assert [s["hadm_id"] for s in select_stays(stays)] == [1]


def test_steps_death_inside():
    s = decompensation_steps(30.0, 40.0, 1, 0.5)
    ts = [x["t"] for x in s]
    assert ts[0] == 5.0 and ts[-1] == 30.0
    assert sum(x["label"] for x in s) == 14  # t=17..30 -> lived-t<24
    assert all(x["event"] == 1 for x in s)
    assert s[0]["duration"] == 35.0


def test_steps_death_during_stay_cuts_at_death():
    s = decompensation_steps(50.0, 20.0, 1, 0.0)
    assert s[-1]["t"] == 20.0
    assert [x["label"] for x in s][-1] == 1
    assert s[0]["label"] == 1  # 20-5=15<24


def test_steps_survivor_and_late_death():
    s = decompensation_steps(30.0, None, 0, 0.5)
    assert sum(x["label"] for x in s) == 0
    assert not any(x["event"] for x in s)
    assert s[0]["duration"] == 25.0
    s = decompensation_steps(30.0, 100.0, 1, 0.5)  # died after ICU discharge
    assert sum(x["label"] for x in s) == 0
    assert all(x["event"] == 1 for x in s)


def test_steps_first_event_delay_and_short():
    s = decompensation_steps(30.0, None, 0, 10.5)
    assert s[0]["t"] == 11.0
    assert decompensation_steps(4.0, None, 0, 0.0) == []


def test_horizon_param():
    s = decompensation_steps(30.0, 40.0, 1, 0.5, horizon=12.0)
    # lived - t < 12 -> t > 28 -> t in {29, 30}
    assert sum(x["label"] for x in s) == 2


def test_build_sample():
    ev = [(0.5, "Heart Rate", 80.0), (9.2, "Heart Rate", 90.0),
          (99.0, "Heart Rate", 1.0)]
    smp = build_sample("p1", ev, 12.0, None, 0)
    assert smp["dynamic"]["values"].shape == (13, len(FEATURES))
    assert smp["label"]["time"][0] == 5.0 and smp["label"]["time"][-1] == 12.0
    assert smp["label"]["values"][0] == [0.0]
    assert smp["dynamic"]["time"] == [float(h) for h in range(13)]
    assert build_sample("p2", [], 12.0, None, 0) is None
    assert build_sample("p3", ev, 3.0, None, 0) is None
    # death at 10h cuts series
    d = build_sample("p4", ev, 12.0, 10.0, 1)
    assert d["dynamic"]["values"].shape[0] == 11
    assert d["label"]["values"][-1] == [1.0]


T0 = datetime(2100, 1, 1, 0, 0)


class Frame:
    def __init__(self, rows):
        self.rows = rows

    def to_dicts(self):
        return self.rows


class FakePatient:
    def __init__(self, tables, patient_id="p1"):
        self.patient_id = patient_id
        self.tables = tables

    def get_events(self, event_type, return_df=False):
        return Frame(self.tables.get(event_type, []))


def at(hours):
    return T0 + timedelta(hours=hours)


def patient(deathtime=None, dod=None, los_days=2.0, extra_stay=False):
    stays = [{
        "timestamp": T0, "icustays/hadm_id": "10", "icustays/icustay_id": "100",
        "icustays/outtime": T0 + timedelta(days=los_days),
        "icustays/first_wardid": 7, "icustays/last_wardid": 7,
        "icustays/first_careunit": "MICU", "icustays/last_careunit": "MICU",
        "icustays/los": los_days,
    }]
    if extra_stay:
        stays.append(dict(stays[0], **{"icustays/icustay_id": "101"}))
    chart = [
        {"timestamp": at(0.5), "chartevents/hadm_id": "10",
         "chartevents/icustay_id": "100", "chartevents/itemid": "211",
         "chartevents/value": "80", "chartevents/valueuom": "bpm"},
        {"timestamp": at(9.5), "chartevents/hadm_id": "10",
         "chartevents/icustay_id": "100", "chartevents/itemid": "211",
         "chartevents/value": "90", "chartevents/valueuom": "bpm"},
        {"timestamp": at(1.0), "chartevents/hadm_id": "10",
         "chartevents/icustay_id": "100", "chartevents/itemid": "999",
         "chartevents/value": "1", "chartevents/valueuom": ""},
    ]
    labs = [
        {"timestamp": at(2.0), "labevents/hadm_id": "10",
         "labevents/itemid": "50809", "labevents/value": "120",
         "labevents/valueuom": "mg/dL"},
        {"timestamp": at(500.0), "labevents/hadm_id": "10",
         "labevents/itemid": "50809", "labevents/value": "300",
         "labevents/valueuom": "mg/dL"},
    ]
    return FakePatient({
        "patients": [{"patients/dob": "2040-01-01 00:00:00",
                      "patients/dod": dod}],
        "admissions": [{
            "timestamp": T0 - timedelta(hours=2), "admissions/hadm_id": "10",
            "admissions/dischtime": T0 + timedelta(days=5),
            "admissions/deathtime": deathtime,
        }],
        "icustays": stays, "chartevents": chart, "labevents": labs,
    })


def test_survivor_sample():
    samples = DecompensationMIMIC3()(patient(los_days=0.5))
    assert len(samples) == 1
    s = samples[0]
    assert s["stay_id"] == "100" and s["patient_id"] == "p1"
    values = s["dynamic"]["values"]
    assert values.shape == (13, len(FEATURES))
    assert values[1, FEATURES.index("Heart Rate")] == 80.0
    assert values[10, FEATURES.index("Heart Rate")] == 90.0
    assert values[2, FEATURES.index("Glucose")] == 120.0
    assert not np.isnan(values[:, FEATURES.index("Heart Rate")]).all()
    assert s["decompensation"]["time"][0] == 5.0
    assert all(v == [0.0] for v in s["decompensation"]["values"])


def test_death_labels_and_cut():
    samples = DecompensationMIMIC3()(patient(deathtime=at(20.0)))
    s = samples[0]
    assert s["dynamic"]["values"].shape[0] == 21  # cut at hour 20
    assert s["decompensation"]["time"][-1] == 20.0
    assert s["decompensation"]["values"][0] == [1.0]  # 20 - 5 < 24
    assert s["decompensation"]["values"][-1] == [1.0]


def test_horizon_changes_labels_and_name():
    task = DecompensationMIMIC3(horizon=6)
    assert task.task_name == "DecompensationMIMIC3_h6"
    s = task(patient(deathtime=at(20.0)))[0]
    labels = [v[0] for v in s["decompensation"]["values"]]
    assert labels[0] == 0.0 and labels[-1] == 1.0
    assert sum(labels) == 6  # t = 15..20, lived - t < 6 -> t > 14


def test_exclusions():
    assert DecompensationMIMIC3()(patient(extra_stay=True)) == []
    p = patient()
    p.tables["patients"][0]["patients/dob"] = "2095-01-01 00:00:00"
    assert DecompensationMIMIC3()(p) == []
    p = patient()
    p.tables["icustays"][0]["icustays/last_wardid"] = 8
    assert DecompensationMIMIC3()(p) == []


def test_late_death_and_dod_only():
    s = DecompensationMIMIC3()(patient(deathtime=at(300.0)))[0]
    assert all(v == [0.0] for v in s["decompensation"]["values"])
    s = DecompensationMIMIC3()(patient(dod=datetime(2100, 1, 3)))[0]
    assert all(v == [0.0] for v in s["decompensation"]["values"])


def test_missing_columns():
    task = DecompensationMIMIC3()
    assert task._missing(["event_type"])
    cols = [f"{t}/{c}" for t, cs in _REQUIRED_COLUMNS.items() for c in cs]
    assert task._missing(cols) == []


# --- survival label mode ---------------------------------------------------


def test_time_to_event_hours():
    assert time_to_event_hours(5.5, 1) == 6  # death in (5, 6]
    assert time_to_event_hours(5.0, 1) == 6
    assert time_to_event_hours(0.0, 1) == 1  # death at the step's own hour
    assert time_to_event_hours(5.5, 0) == 5  # censored: whole hours observed
    assert time_to_event_hours(0.0, 0) == 0


def test_time_to_event_matches_binary_label():
    # k <= horizon  <=>  death within the horizon, for whole horizons
    for lived in (12.0, 29.5, 30.0, 47.9, 48.0, 100.0):
        for horizon in (6.0, 12.0, 24.0, 48.0):
            for step in decompensation_steps(40.0, lived, 1, 0.5, horizon=horizon):
                k = time_to_event_hours(step["duration"], step["event"])
                assert (step["event"] == 1 and k <= horizon) == bool(step["label"])


def test_build_sample_survival_keys_and_alignment():
    ev = [(0.5, "Heart Rate", 80.0), (9.2, "Heart Rate", 90.0)]
    plain = build_sample("p", ev, 12.0, 10.0, 1)
    assert "time_to_event" not in plain and "event" not in plain
    smp = build_sample("p", ev, 12.0, 10.0, 1, label_mode="survival")
    times = smp["label"]["time"]
    assert smp["time_to_event"]["time"] == times
    assert smp["event"]["time"] == times
    # steps at hours 5..10, death at 10: remaining 5, 4, ..., 0 -> hour 6..1
    assert [v[0] for v in smp["time_to_event"]["values"]] == [6, 5, 4, 3, 2, 1]
    assert all(v == [1.0] for v in smp["event"]["values"])
    assert smp["label"] == plain["label"]  # the binary label is unchanged


def test_build_sample_survival_censored():
    ev = [(0.5, "Heart Rate", 80.0)]
    smp = build_sample("p", ev, 12.0, None, 0, label_mode="survival")
    assert [v[0] for v in smp["time_to_event"]["values"]] == [7, 6, 5, 4, 3, 2, 1, 0]
    assert all(v == [0.0] for v in smp["event"]["values"])
    assert all(v == [0.0] for v in smp["label"]["values"])


def test_build_sample_rejects_unknown_mode():
    try:
        build_sample("p", [(0.5, "Heart Rate", 80.0)], 12.0, None, 0, label_mode="x")
    except ValueError as err:
        assert "label_mode" in str(err)
    else:
        raise AssertionError("expected ValueError")


def test_task_default_mode_unchanged():
    task = DecompensationMIMIC3()
    assert task.label_mode == "binary"
    assert list(task.output_schema) == ["decompensation"]
    assert task.task_name == "DecompensationMIMIC3"


def test_task_survival_schema_and_name():
    task = DecompensationMIMIC3(label_mode="survival")
    assert task.output_schema == {
        "decompensation": ("timeseries_binary", {"timeline": ["dynamic"]}),
        "time_to_event": ("timeseries_regression", {"timeline": ["dynamic"]}),
        "event": ("timeseries_binary", {"timeline": ["dynamic"]}),
    }
    assert task.task_name == "DecompensationMIMIC3_survival"
    assert DecompensationMIMIC3.output_schema == {
        "decompensation": ("timeseries_binary", {"timeline": ["dynamic"]})
    }  # the class (and other instances) keep the binary schema
    assert DecompensationMIMIC3(12, label_mode="survival").task_name == (
        "DecompensationMIMIC3_h12_survival"
    )


def test_task_survival_rejects_bad_arguments():
    for kwargs in ({"label_mode": "x"}, {"horizon": 12.5, "label_mode": "survival"}):
        try:
            DecompensationMIMIC3(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")
    DecompensationMIMIC3(horizon=12.5)  # fine in binary mode


def test_task_survival_samples():
    task = DecompensationMIMIC3(label_mode="survival")
    s = task(patient(deathtime=at(20.0)))[0]
    assert s["time_to_event"]["time"] == s["decompensation"]["time"]
    assert s["time_to_event"]["values"][0] == [16.0]  # hour 5 -> death in (15, 16]
    assert s["time_to_event"]["values"][-1] == [1.0]
    assert all(v == [1.0] for v in s["event"]["values"])
    survivor = task(patient(los_days=0.5))[0]
    assert all(v == [0.0] for v in survivor["event"]["values"])
    assert survivor["time_to_event"]["values"][-1] == [0.0]
    binary = DecompensationMIMIC3()(patient(deathtime=at(20.0)))[0]
    assert "time_to_event" not in binary
    assert binary["decompensation"] == s["decompensation"]
