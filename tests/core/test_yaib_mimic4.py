import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import (
    YAIBAKIMIMIC4,
    YAIBKidneyFunctionMIMIC4,
    YAIBMortalityMIMIC4,
    YAIBSepsisMIMIC4,
)

T0 = datetime(2150, 1, 1, 8, 0, 0)
MICU = "Medical Intensive Care Unit (MICU)"
HR = YAIBMortalityMIMIC4.DYNAMIC_FEATURES.index("hr")
TEMP = YAIBMortalityMIMIC4.DYNAMIC_FEATURES.index("temp")
URINE = YAIBMortalityMIMIC4.DYNAMIC_FEATURES.index("urine")
CREA = YAIBMortalityMIMIC4.DYNAMIC_FEATURES.index("crea")


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _fmt(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _patient(*, age=60, gender="M", stays, admissions, extra=()):
    """Builds a patient from (stay_id, hadm_id, intime, hours) stays.

    Each stay gets one heart rate measurement per hour, so it passes the
    YAIB base cohort unless a test removes measurements.
    """
    rows = [_event("patients", None, anchor_age=age, gender=gender)]
    for hadm_id, admittime, dischtime, expire, deathtime, last_unit in admissions:
        rows.append(
            _event(
                "admissions",
                admittime,
                hadm_id=hadm_id,
                dischtime=_fmt(dischtime),
                deathtime=None if deathtime is None else _fmt(deathtime),
                hospital_expire_flag=expire,
            )
        )
        rows.append(
            _event("transfers", admittime, hadm_id=hadm_id, eventtype="admit",
                   careunit=last_unit)
        )
        rows.append(
            _event("transfers", dischtime, hadm_id=hadm_id, eventtype="discharge",
                   careunit=None)
        )
    for stay_id, hadm_id, intime, hours, hr_hours in stays:
        outtime = intime + timedelta(hours=hours)
        rows.append(
            _event("icustays", intime, stay_id=stay_id, hadm_id=hadm_id,
                   first_careunit=MICU, outtime=_fmt(outtime))
        )
        for hour in hr_hours:
            rows.append(
                _event("chartevents", intime + timedelta(hours=hour, minutes=5),
                       stay_id=stay_id, hadm_id=hadm_id, itemid=220045,
                       value=80 + hour, valuenum=80 + hour, valueuom="bpm")
            )
    rows.extend(extra)
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


def _admission(hadm_id=1, hours=60, expire=0, deathtime=None, last_unit=MICU):
    return (hadm_id, T0 - timedelta(hours=10), T0 + timedelta(hours=hours), expire,
            deathtime, last_unit)


class TestYAIBMortalityMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = YAIBMortalityMIMIC4()

    def test_survivor_features(self):
        intime = T0
        extra = [
            # Out of range heart rate, dropped.
            _event("chartevents", intime + timedelta(hours=2, minutes=30),
                   stay_id=10, hadm_id=1, itemid=220045, valuenum=400,
                   valueuom="bpm"),
            # Fahrenheit temperature, converted to Celsius.
            _event("chartevents", intime + timedelta(hours=3, minutes=1),
                   stay_id=10, hadm_id=1, itemid=223761, valuenum=98.6,
                   valueuom="°F"),
            # Height in inches, converted to cm.
            _event("chartevents", intime + timedelta(hours=1), stay_id=10,
                   hadm_id=1, itemid=226707, valuenum=70, valueuom="Inch"),
            _event("chartevents", intime + timedelta(hours=1), stay_id=10,
                   hadm_id=1, itemid=226512, valuenum=80, valueuom="kg"),
            # Urine output is summed within an hour.
            _event("outputevents", intime + timedelta(hours=4, minutes=10),
                   stay_id=10, hadm_id=1, itemid=226559, value=100,
                   valueuom="mL"),
            _event("outputevents", intime + timedelta(hours=4, minutes=50),
                   stay_id=10, hadm_id=1, itemid=226559, value=50,
                   valueuom="mL"),
            # Lab events carry only the admission ID; median within an hour.
            _event("labevents", intime + timedelta(hours=5, minutes=0),
                   hadm_id=1, itemid=50912, valuenum=1.0, valueuom="mg/dL"),
            _event("labevents", intime + timedelta(hours=5, minutes=20),
                   hadm_id=1, itemid=50912, valuenum=2.0, valueuom="mg/dL"),
            # Lab before ICU admission: outside the observation grid.
            _event("labevents", intime - timedelta(hours=3), hadm_id=1,
                   itemid=50912, valuenum=5.0, valueuom="mg/dL"),
        ]
        patient = _patient(
            stays=[(10, 1, intime, 48, range(48))],
            admissions=[_admission()],
            extra=extra,
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["stay_id"], 10)
        self.assertEqual(sample["mortality"], 0)

        timestamps, values = sample["dynamic"]
        self.assertEqual(values.shape, (25, len(self.task.DYNAMIC_FEATURES)))
        self.assertEqual(timestamps[0], intime)
        self.assertEqual(timestamps[-1], intime + timedelta(hours=24))
        np.testing.assert_allclose(values[:, HR], 80 + np.arange(25))
        self.assertAlmostEqual(values[3, TEMP], 37.0)
        self.assertEqual(values[4, URINE], 150)
        self.assertEqual(values[5, CREA], 1.5)
        self.assertEqual(np.isfinite(values[:, CREA]).sum(), 1)

        age, sex, height, weight = sample["static"]
        self.assertEqual((age, sex, weight), (60.0, 1.0, 80.0))
        self.assertAlmostEqual(height, 177.8)

    def test_icu_death_label(self):
        death = T0 + timedelta(hours=40)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death)],
        )
        self.assertEqual([s["mortality"] for s in self.task(patient)], [1])

    def test_hospital_death_outside_icu_is_negative(self):
        death = T0 + timedelta(hours=58)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death, last_unit="Med/Surg")],
        )
        self.assertEqual([s["mortality"] for s in self.task(patient)], [0])

    def test_death_within_30_hours_is_excluded(self):
        death = T0 + timedelta(hours=29)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death)],
        )
        self.assertEqual(self.task(patient), [])

    def test_death_is_attributed_to_the_right_stay(self):
        second = T0 + timedelta(hours=60)
        death = second + timedelta(hours=40)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48)), (11, 1, second, 48, range(48))],
            admissions=[_admission(hours=120, expire=1, deathtime=death)],
        )
        labels = {s["stay_id"]: s["mortality"] for s in self.task(patient)}
        self.assertEqual(labels, {10: 0, 11: 1})

    def test_short_stay_is_excluded(self):
        patient = _patient(
            stays=[(10, 1, T0, 29.5, range(29))],
            admissions=[_admission()],
        )
        self.assertEqual(self.task(patient), [])

    def test_measurement_gap_is_excluded(self):
        hours = [h for h in range(48) if not 10 <= h <= 22]  # 13 empty bins
        patient = _patient(
            stays=[(10, 1, T0, 48, hours)],
            admissions=[_admission()],
        )
        self.assertEqual(self.task(patient), [])

        hours = [h for h in range(48) if not 10 <= h <= 21]  # 12 empty bins
        patient = _patient(
            stays=[(10, 1, T0, 48, hours)],
            admissions=[_admission()],
        )
        self.assertEqual(len(self.task(patient)), 1)

    def test_minor_is_excluded(self):
        patient = _patient(
            age=17,
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
        )
        self.assertEqual(self.task(patient), [])

    def test_labs_are_attributed_to_stays_of_the_admission(self):
        second = T0 + timedelta(hours=60)
        extra = [
            _event("labevents", T0 + timedelta(hours=2), hadm_id=1,
                   itemid=50912, valuenum=1.0, valueuom="mg/dL"),
            _event("labevents", second + timedelta(hours=3), hadm_id=1,
                   itemid=50912, valuenum=2.0, valueuom="mg/dL"),
        ]
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48)), (11, 1, second, 48, range(48))],
            admissions=[_admission(hours=120)],
            extra=extra,
        )
        crea = {
            s["stay_id"]: np.nonzero(np.isfinite(s["dynamic"][1][:, CREA]))[0]
            for s in self.task(patient)
        }
        np.testing.assert_array_equal(crea[10], [2])
        np.testing.assert_array_equal(crea[11], [3])

    def test_o2sat_includes_arterial_saturation(self):
        extra = [
            _event("chartevents", T0 + timedelta(hours=1, minutes=10),
                   stay_id=10, hadm_id=1, itemid=220227, valuenum=90,
                   valueuom="%"),
            _event("labevents", T0 + timedelta(hours=2, minutes=10),
                   hadm_id=1, itemid=50817, valuenum=94, valueuom="%"),
        ]
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
            extra=extra,
        )
        _, values = self.task(patient)[0]["dynamic"]
        o2sat = values[:, YAIBMortalityMIMIC4.DYNAMIC_FEATURES.index("o2sat")]
        self.assertEqual(o2sat[1], 90)
        self.assertEqual(o2sat[2], 94)

    def test_pre_filter_requires_measurement_tables(self):
        df = pl.LazyFrame(
            {"patient_id": ["p1"], "event_type": ["icustays"],
             "timestamp": [T0], "labevents/itemid": ["50912"]}
        )
        with self.assertRaises(ValueError):
            self.task.pre_filter(df)


def _crea(time, value, hadm_id=1):
    return _event("labevents", time, hadm_id=hadm_id, itemid=50912,
                  valuenum=value, valueuom="mg/dL")


class TestYAIBKidneyFunctionMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = YAIBKidneyFunctionMIMIC4()

    def test_label_is_median_of_hourly_medians_in_hours_25_to_48(self):
        extra = [
            _crea(T0 + timedelta(hours=10), 9.0),  # observation window, not label
            _crea(T0 + timedelta(hours=24, minutes=30), 9.0),  # bin 24, excluded
            _crea(T0 + timedelta(hours=30, minutes=0), 1.0),
            _crea(T0 + timedelta(hours=30, minutes=30), 3.0),  # bin 30 -> 2.0
            _crea(T0 + timedelta(hours=40), 4.0),
            _crea(T0 + timedelta(hours=48, minutes=30), 5.0),  # bin 48, included
            _crea(T0 + timedelta(hours=49), 9.0),  # after the window
        ]
        patient = _patient(
            stays=[(10, 1, T0, 60, range(60))],
            admissions=[_admission(hours=70)],
            extra=extra,
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["creatinine"], 4.0)
        _, values = samples[0]["dynamic"]
        self.assertEqual(values.shape[0], 25)
        self.assertEqual(values[10, CREA], 9.0)

    def test_short_stay_is_excluded(self):
        patient = _patient(
            stays=[(10, 1, T0, 47.5, range(47))],
            admissions=[_admission()],
            extra=[_crea(T0 + timedelta(hours=30), 1.0)],
        )
        self.assertEqual(self.task(patient), [])

    def test_stay_without_day_two_creatinine_is_excluded(self):
        patient = _patient(
            stays=[(10, 1, T0, 60, range(60))],
            admissions=[_admission(hours=70)],
            extra=[_crea(T0 + timedelta(hours=24, minutes=59), 1.0)],
        )
        self.assertEqual(self.task(patient), [])


def _urine(time, value, stay_id=10):
    return _event("outputevents", time, stay_id=stay_id, hadm_id=1,
                  itemid=226559, value=value, valueuom="mL")


def _weight(value, stay_id=10):
    return _event("chartevents", T0 + timedelta(hours=1), stay_id=stay_id,
                  hadm_id=1, itemid=226512, valuenum=value, valueuom="kg")


class TestYAIBAKIMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = YAIBAKIMIMIC4()

    def _run(self, extra, hours=60):
        patient = _patient(
            stays=[(10, 1, T0, hours, range(int(hours)))],
            admissions=[_admission(hours=hours + 10)],
            extra=extra,
        )
        return self.task(patient)

    def test_creatinine_rise_labels_six_hours_around_onset(self):
        extra = [
            _crea(T0 + timedelta(hours=2), 1.0),
            _crea(T0 + timedelta(hours=20), 1.3),  # rise of 0.3 within 48h
        ]
        samples = self._run(extra)
        self.assertEqual(len(samples), 1)
        timestamps, labels = samples[0]["aki"]
        # Observed until 6 hours after the onset at hour 20.
        self.assertEqual(len(timestamps), 27)
        self.assertEqual(samples[0]["dynamic"][1].shape[0], 27)
        self.assertEqual(labels, [0.0] * 14 + [1.0] * 13)

    def test_stay_without_aki_is_negative_for_seven_days(self):
        samples = self._run([_crea(T0 + timedelta(hours=2), 1.0)], hours=200)
        timestamps, labels = samples[0]["aki"]
        self.assertEqual(len(timestamps), 169)
        self.assertEqual(sum(labels), 0)

    def test_onset_within_six_hours_is_excluded(self):
        extra = [
            _crea(T0 - timedelta(hours=10), 1.0),
            _crea(T0 + timedelta(hours=3), 1.5),
        ]
        self.assertEqual(self._run(extra), [])

    def test_high_baseline_creatinine_is_excluded(self):
        extra = [
            _crea(T0 - timedelta(hours=5), 4.5),
            _crea(T0 + timedelta(hours=2), 4.6),
        ]
        self.assertEqual(self._run(extra), [])

    def test_urine_stage_divides_by_weight(self):
        # 20 mL every hour: 6-hour sum of rates 6 * 20 / 2 = 60 mL/h, below
        # 0.5 per kg only when divided by a weight above 120 kg.
        urine = [_urine(T0 + timedelta(hours=h, minutes=30), 20) for h in range(40)]
        _, labels = self._run(urine + [_weight(130)])[0]["aki"]
        self.assertGreater(sum(labels), 0)
        _, labels = self._run(urine + [_weight(100)])[0]["aki"]
        self.assertEqual(sum(labels), 0)
        # Without any weight the sums are not divided, as in YAIB.
        _, labels = self._run(urine)[0]["aki"]
        self.assertEqual(sum(labels), 0)


def _antibiotic(start, hours=1, stay_id=10):
    return _event("inputevents", start, stay_id=stay_id, hadm_id=1, itemid=225798,
                  endtime=_fmt(start + timedelta(hours=hours)), rate=None,
                  linkorderid=1)


def _culture(time):
    return _event("microbiologyevents", time.replace(hour=0, minute=0), hadm_id=1,
                  charttime=_fmt(time))


def _platelets(time, value):
    return _event("labevents", time, hadm_id=1, itemid=51265, valuenum=value,
                  valueuom="K/uL")


class TestYAIBSepsisMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = YAIBSepsisMIMIC4()

    def test_sepsis_onset_labels_six_hours_around_onset(self):
        extra = [_platelets(T0 + timedelta(hours=h), 200) for h in range(0, 22, 3)]
        extra.append(_platelets(T0 + timedelta(hours=22, minutes=10), 15))
        # Antibiotics every 12 hours for 3 days: continuous.
        extra += [_antibiotic(T0 + timedelta(hours=20 + 12 * k)) for k in range(7)]
        extra.append(_culture(T0 + timedelta(hours=21)))
        patient = _patient(
            stays=[(10, 1, T0, 120, range(120))],
            admissions=[_admission(hours=130)],
            extra=extra,
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        timestamps, labels = samples[0]["sepsis"]
        self.assertEqual(len(timestamps), 29)  # until 6 hours after hour 22
        self.assertEqual(labels, [0.0] * 16 + [1.0] * 13)

    def test_intermittent_antibiotics_are_not_suspected_infection(self):
        extra = [_platelets(T0 + timedelta(hours=h), 200) for h in range(0, 22, 3)]
        extra.append(_platelets(T0 + timedelta(hours=22, minutes=10), 15))
        extra.append(_antibiotic(T0 + timedelta(hours=20)))  # single dose
        extra.append(_culture(T0 + timedelta(hours=21)))
        patient = _patient(
            stays=[(10, 1, T0, 120, range(120))],
            admissions=[_admission(hours=130)],
            extra=extra,
        )
        _, labels = self.task(patient)[0]["sepsis"]
        self.assertEqual(sum(labels), 0)

    def test_suspected_infection_both_directions(self):
        infection = YAIBSepsisMIMIC4._suspected_infection
        self.assertEqual(infection([10], [30]), 10)  # sample within 24h after
        self.assertIsNone(infection([10], [35]))
        self.assertEqual(infection([50], [0]), 0)  # antibiotic within 72h after
        self.assertIsNone(infection([80], [0]))

    def test_onset_is_relative_to_running_minimum_in_window(self):
        hours = np.arange(0, 10)
        sofa = np.array([5, 5, 3, 3, 4, 5, 5, 5, 5, 5], dtype=float)
        # Window from hour 0: rise of 2 over the minimum 3 at hour 5.
        self.assertEqual(YAIBSepsisMIMIC4._sepsis_onset(hours, sofa, 48), 5)
        self.assertIsNone(YAIBSepsisMIMIC4._sepsis_onset(hours, sofa, None))

    def test_merge_ranges_groups_by_first_overlap(self):
        merged = YAIBSepsisMIMIC4._merge_ranges(
            np.array([0, 8, 18]), np.array([10, 20, 30]), gap=0
        )
        # As ricu: [8, 20] joins [0, 10]; [18, 30] joins [8, 20], its own group.
        self.assertEqual(sorted(merged), [(0, 20), (8, 30)])

    def test_pafi_pairs_values_up_to_two_hours_apart(self):
        pafi = YAIBSepsisMIMIC4._pafi({5: 80.0, 20: 90.0}, {3: 40.0})
        self.assertEqual(pafi[5], 200.0)  # FiO2 from 2 hours earlier
        self.assertEqual(pafi[20], 100 * 90.0 / 21.0)  # no FiO2: 21%
        self.assertNotIn(3, pafi)  # no PaO2 at or before hour 3

    def test_gcs_carries_forward_and_counts_intubation_as_normal(self):
        inputs = {"egcs": {0: 1.0}, "vgcs": {0: 1.0}, "mgcs": {0: 1.0, 5: 3.0, 9: 3.0}}
        gcs = YAIBSepsisMIMIC4._gcs(inputs, intubated={9})
        self.assertEqual(gcs[0], 3.0)
        self.assertEqual(gcs[5], 5.0)  # eye and verbal carried forward
        self.assertEqual(gcs[9], 15.0)

    def test_urine24_needs_twelve_hours(self):
        urine = YAIBSepsisMIMIC4._urine24({h: 50.0 for h in range(30)})
        self.assertTrue(np.isnan(urine[10]))
        self.assertEqual(urine[11], 50.0 * 24)
        self.assertEqual(urine[29], 50.0 * 25 * 24 / 25)


if __name__ == "__main__":
    unittest.main()
