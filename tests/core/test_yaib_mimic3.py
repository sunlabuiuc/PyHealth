import unittest
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN_V2, LevelEmbedding
from pyhealth.tasks import (
    YAIBAKIMIMIC3,
    YAIBKidneyFunctionMIMIC3,
    YAIBLengthOfStayMIMIC3,
    YAIBMortalityMIMIC3,
    YAIBSepsisMIMIC3,
)

T0 = datetime(2150, 1, 1, 8, 0, 0)
FEATURES = YAIBMortalityMIMIC3.DYNAMIC_FEATURES
HR = FEATURES.index("hr")
TEMP = FEATURES.index("temp")
URINE = FEATURES.index("urine")
CREA = FEATURES.index("crea")
O2SAT = FEATURES.index("o2sat")


def _labels(records):
    """The label values of a ``timeseries_*`` label series."""
    return [value[0] for value in records["values"]]


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _fmt(time):
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _patient(*, dob=T0 - timedelta(days=60 * 365 + 10), gender="M", stays,
             admissions, extra=(), hr_itemid=211):
    """Builds a patient from (stay_id, hadm_id, intime, hours, hr_hours) stays.

    Each stay gets a heart rate measurement in each of ``hr_hours``, so it
    passes the YAIB base cohort unless a test removes measurements.
    ``admissions`` are (hadm_id, admittime, dischtime, expire, deathtime,
    last_unit) tuples; each gets an admit and a discharge transfer.
    """
    rows = [_event("patients", None, dob=_fmt(dob), gender=gender)]
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
                   curr_careunit=last_unit)
        )
        rows.append(
            _event("transfers", dischtime, hadm_id=hadm_id, eventtype="discharge",
                   curr_careunit=None)
        )
    for stay_id, hadm_id, intime, hours, hr_hours in stays:
        outtime = intime + timedelta(hours=hours)
        rows.append(
            _event("icustays", intime, icustay_id=stay_id, hadm_id=hadm_id,
                   first_careunit="MICU", outtime=_fmt(outtime))
        )
        for hour in hr_hours:
            rows.append(
                _event("chartevents", intime + timedelta(hours=hour, minutes=5),
                       icustay_id=stay_id, hadm_id=hadm_id, itemid=hr_itemid,
                       value=80 + hour, valuenum=80 + hour, valueuom="BPM")
            )
    rows.extend(extra)
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


def _admission(hadm_id=1, hours=60, expire=0, deathtime=None, last_unit="MICU"):
    return (hadm_id, T0 - timedelta(hours=10), T0 + timedelta(hours=hours), expire,
            deathtime, last_unit)


def _chart(time, itemid, value, unit=None, stay_id=10, text=None):
    return _event("chartevents", time, icustay_id=stay_id, hadm_id=1, itemid=itemid,
                  value=text if text is not None else value, valuenum=value,
                  valueuom=unit)


def _crea(time, value, hadm_id=1):
    return _event("labevents", time, hadm_id=hadm_id, itemid=50912,
                  valuenum=value, valueuom="mg/dL")


class TestYAIBMortalityMIMIC3(unittest.TestCase):
    def setUp(self):
        self.task = YAIBMortalityMIMIC3()

    def test_carevue_and_metavision_features(self):
        intime = T0
        extra = [
            # Fahrenheit temperature (CareVue), converted to Celsius.
            _chart(intime + timedelta(hours=3, minutes=1), 678, 98.6, "Deg. F"),
            # Height in inches (CareVue), converted to cm.
            _chart(intime + timedelta(hours=1), 920, 70, "Inch"),
            _chart(intime + timedelta(hours=1), 762, 80, "kg"),
            # CareVue and MetaVision urine, summed within an hour.
            _event("outputevents", intime + timedelta(hours=4, minutes=10),
                   icustay_id=10, hadm_id=1, itemid=40055, value=100,
                   valueuom="ml"),
            _event("outputevents", intime + timedelta(hours=4, minutes=50),
                   icustay_id=10, hadm_id=1, itemid=226559, value=50,
                   valueuom="mL"),
            # Lab events carry only the admission ID; median within an hour.
            _crea(intime + timedelta(hours=5), 1.0),
            _crea(intime + timedelta(hours=5, minutes=20), 2.0),
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

        values = sample["dynamic"]["values"]
        self.assertEqual(values.shape, (25, len(FEATURES)))
        np.testing.assert_allclose(values[:, HR], 80 + np.arange(25))
        self.assertAlmostEqual(values[3, TEMP], 37.0)
        self.assertEqual(values[4, URINE], 150)
        self.assertEqual(values[5, CREA], 1.5)

        ((age, sex, height, weight),) = sample["static"]["values"]
        self.assertAlmostEqual(age, (60 * 365 + 10) / 365)
        self.assertEqual((sex, weight), (1.0, 80.0))
        self.assertAlmostEqual(height, 177.8)

    def test_metavision_heart_rate(self):
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
            hr_itemid=220045,
        )
        values = self.task(patient)[0]["dynamic"]["values"]
        np.testing.assert_allclose(values[:, HR], 80 + np.arange(25))

    def test_age_from_date_of_birth(self):
        # Shifted date of birth of a patient older than 89: capped at 90.
        old = _patient(
            dob=T0 - timedelta(days=300 * 365),
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
        )
        self.assertEqual(self.task(old)[0]["static"]["values"][0][0], 90.0)
        # Age is taken at each ICU admission.
        minor = _patient(
            dob=T0 - timedelta(days=17 * 365 + 360),
            stays=[(10, 1, T0, 48, range(48)),
                   (11, 1, T0 + timedelta(days=20), 48, range(48))],
            admissions=[_admission(hours=24 * 23)],
        )
        self.assertEqual([s["stay_id"] for s in self.task(minor)], [11])

    def test_icu_death_label(self):
        death = T0 + timedelta(hours=40)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death)],
        )
        self.assertEqual([s["mortality"] for s in self.task(patient)], [1])

    def test_hospital_death_on_a_general_ward_is_negative(self):
        # General wards have no care unit in MIMIC-III.
        death = T0 + timedelta(hours=58)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death, last_unit=None)],
        )
        self.assertEqual([s["mortality"] for s in self.task(patient)], [0])

    def test_hospital_death_without_discharge_transfer_is_positive(self):
        death = T0 + timedelta(hours=40)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death, last_unit=None)],
        )
        without_discharge = patient.data_source.filter(
            pl.col("transfers/eventtype").ne("discharge").fill_null(True)
        )
        patient = Patient(patient_id="p1", data_source=without_discharge)
        self.assertEqual([s["mortality"] for s in self.task(patient)], [1])

    def test_last_ward_is_found_per_icu_stay(self):
        # As YAIB: transfers are mapped onto ICU stays first. The death is
        # attributed to the second stay, whose last ward before discharge is
        # an ICU; the general ward between the stays does not count.
        second = T0 + timedelta(hours=60)
        death = second + timedelta(hours=40)
        extra = [
            _event("transfers", T0 + timedelta(hours=50), hadm_id=1,
                   eventtype="transfer", curr_careunit=None),
            _event("transfers", second, hadm_id=1, eventtype="transfer",
                   curr_careunit="MICU"),
        ]
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48)), (11, 1, second, 48, range(48))],
            admissions=[_admission(hours=120, expire=1, deathtime=death)],
            extra=extra,
        )
        labels = {s["stay_id"]: s["mortality"] for s in self.task(patient)}
        self.assertEqual(labels, {10: 0, 11: 1})

    def test_death_within_30_hours_is_excluded(self):
        death = T0 + timedelta(hours=29)
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission(expire=1, deathtime=death)],
        )
        self.assertEqual(self.task(patient), [])

    def test_measurement_gap_is_excluded(self):
        hours = [h for h in range(48) if not 10 <= h <= 22]  # 13 empty bins
        patient = _patient(
            stays=[(10, 1, T0, 48, hours)],
            admissions=[_admission()],
        )
        self.assertEqual(self.task(patient), [])

    def test_o2sat_includes_arterial_saturation(self):
        extra = [
            _chart(T0 + timedelta(hours=1, minutes=10), 834, 90, "%"),
            _chart(T0 + timedelta(hours=2, minutes=10), 220227, 92, "%"),
            _event("labevents", T0 + timedelta(hours=3, minutes=10),
                   hadm_id=1, itemid=50817, valuenum=94, valueuom="%"),
            _chart(T0 + timedelta(hours=4, minutes=10), 646, 96, "%"),
        ]
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
            extra=extra,
        )
        o2sat = self.task(patient)[0]["dynamic"]["values"][:, O2SAT]
        self.assertEqual(o2sat[1:5].tolist(), [90, 92, 94, 96])

    def test_pre_filter_requires_transfers(self):
        columns = {
            "patient_id": ["p1"],
            "event_type": ["icustays"],
            "timestamp": [T0],
            "labevents/itemid": ["50912"],
            "chartevents/itemid": ["211"],
            "outputevents/itemid": ["40055"],
        }
        with self.assertRaisesRegex(ValueError, "transfers"):
            self.task.pre_filter(pl.LazyFrame(columns))
        columns["transfers/curr_careunit"] = [None]
        self.task.pre_filter(pl.LazyFrame(columns)).collect()


class TestYAIBKidneyFunctionMIMIC3(unittest.TestCase):
    def test_label_is_median_creatinine_of_day_two(self):
        extra = [
            _crea(T0 + timedelta(hours=10), 9.0),  # observation window
            _crea(T0 + timedelta(hours=30), 1.0),
            _crea(T0 + timedelta(hours=30, minutes=30), 3.0),  # bin 30 -> 2.0
            _crea(T0 + timedelta(hours=40), 4.0),
            _crea(T0 + timedelta(hours=48, minutes=30), 5.0),
        ]
        patient = _patient(
            stays=[(10, 1, T0, 60, range(60))],
            admissions=[_admission(hours=70)],
            extra=extra,
        )
        samples = YAIBKidneyFunctionMIMIC3()(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["creatinine"], 4.0)
        self.assertEqual(samples[0]["dynamic"]["values"].shape[0], 25)


class TestYAIBLengthOfStayMIMIC3(unittest.TestCase):
    def test_remaining_hours_at_every_hour(self):
        patient = _patient(
            stays=[(10, 1, T0, 20.5, range(20))], admissions=[_admission()]
        )
        sample = YAIBLengthOfStayMIMIC3()(patient)[0]
        self.assertEqual(sample["los"]["time"], sample["dynamic"]["time"])
        self.assertEqual(_labels(sample["los"]), [float(20 - h) for h in range(21)])


class TestYAIBAKIMIMIC3(unittest.TestCase):
    def _run(self, extra, hours=60):
        patient = _patient(
            stays=[(10, 1, T0, hours, range(int(hours)))],
            admissions=[_admission(hours=hours + 10)],
            extra=extra,
        )
        return YAIBAKIMIMIC3()(patient)

    def test_creatinine_rise_labels_six_hours_around_onset(self):
        samples = self._run([
            _crea(T0 + timedelta(hours=2), 1.0),
            _crea(T0 + timedelta(hours=20), 1.3),
        ])
        aki = samples[0]["aki"]
        self.assertEqual(len(aki["time"]), 27)
        self.assertEqual(_labels(aki), [0.0] * 14 + [1.0] * 13)

    def test_carevue_urine_and_weight(self):
        # 20 mL every hour, below 0.5 mL/kg/h only for a weight above 120 kg.
        urine = [
            _event("outputevents", T0 + timedelta(hours=h, minutes=30),
                   icustay_id=10, hadm_id=1, itemid=40055, value=20,
                   valueuom="ml")
            for h in range(40)
        ]
        heavy = _chart(T0 + timedelta(hours=1), 762, 130, "kg")
        light = _chart(T0 + timedelta(hours=1), 4183, 100, "kg")
        self.assertGreater(sum(_labels(self._run(urine + [heavy])[0]["aki"])), 0)
        self.assertEqual(sum(_labels(self._run(urine + [light])[0]["aki"])), 0)


def _antibiotic(start, hours=1, stay_id=10):
    return _event("inputevents_mv", start, icustay_id=stay_id, hadm_id=1,
                  itemid=225798, endtime=_fmt(start + timedelta(hours=hours)),
                  rate=None, linkorderid=1)


def _culture(time):
    return _event("microbiologyevents", time.replace(hour=0, minute=0), hadm_id=1,
                  charttime=_fmt(time))


def _platelets(time, value):
    return _event("labevents", time, hadm_id=1, itemid=51265, valuenum=value,
                  valueuom="K/uL")


def _carevue_rate(time, rate, itemid=30120, link=1, stay_id=10):
    return _event("inputevents_cv", time, icustay_id=stay_id, hadm_id=1,
                  itemid=itemid, rate=rate, linkorderid=link)


def _metavision_infusion(start, minutes, rate, itemid=221906, link=1, stay_id=10):
    return _event("inputevents_mv", start, icustay_id=stay_id, hadm_id=1,
                  itemid=itemid, endtime=_fmt(start + timedelta(minutes=minutes)),
                  rate=rate, linkorderid=link)


def _extubation(time, stay_id=10):
    return _event("procedureevents_mv", time, icustay_id=stay_id, hadm_id=1,
                  itemid=227194, endtime=_fmt(time + timedelta(minutes=1)))


class TestYAIBSepsisMIMIC3(unittest.TestCase):
    def setUp(self):
        self.task = YAIBSepsisMIMIC3()

    def _sofa_patient(self, extra):
        patient = _patient(
            stays=[(10, 1, T0, 48, range(48))],
            admissions=[_admission()],
            extra=extra,
        )
        return patient, self.task._stays(patient)

    def test_sepsis_onset_labels_six_hours_around_onset(self):
        extra = [_platelets(T0 + timedelta(hours=h), 200) for h in range(0, 22, 3)]
        extra.append(_platelets(T0 + timedelta(hours=22, minutes=10), 15))
        extra += [_antibiotic(T0 + timedelta(hours=20 + 12 * k)) for k in range(7)]
        extra.append(_culture(T0 + timedelta(hours=21)))
        patient = _patient(
            stays=[(10, 1, T0, 120, range(120))],
            admissions=[_admission(hours=130)],
            extra=extra,
        )
        samples = self.task(patient)
        self.assertEqual(len(samples), 1)
        sepsis = samples[0]["sepsis"]
        self.assertEqual(len(sepsis["time"]), 29)
        self.assertEqual(_labels(sepsis), [0.0] * 16 + [1.0] * 13)

    def test_stay_without_metavision_antibiotics_is_negative(self):
        # The same course with no antibiotic infusions (as in CareVue).
        extra = [_platelets(T0 + timedelta(hours=h), 200) for h in range(0, 22, 3)]
        extra.append(_platelets(T0 + timedelta(hours=22, minutes=10), 15))
        extra.append(_culture(T0 + timedelta(hours=21)))
        patient = _patient(
            stays=[(10, 1, T0, 120, range(120))],
            admissions=[_admission(hours=130)],
            extra=extra,
        )
        self.assertEqual(sum(_labels(self.task(patient)[0]["sepsis"])), 0)

    def test_carevue_rates_last_until_the_next_rate_of_the_order(self):
        patient, stays = self._sofa_patient([
            # Order 1: rates at 5:00 and 6:30. The order runs from its first
            # to its last rate, so the last rate's hour past 6:30 is cut.
            _carevue_rate(T0 + timedelta(hours=5), 0.05, link=1),
            _carevue_rate(T0 + timedelta(hours=6, minutes=30), 0.2, link=1),
            # Order 2: a single rate, an order of no duration.
            _carevue_rate(T0 + timedelta(hours=20), 0.5, link=2),
        ])
        norepi = self.task._vasopressors(patient, stays)[10]["norepi"]
        self.assertEqual(sorted(norepi), [5, 6])
        self.assertAlmostEqual(norepi[5], 0.05)
        self.assertAlmostEqual(norepi[6], 0.2)
        hours, sofa = self.task._sofa({}, {"norepi": norepi}, set(), set())
        scores = dict(zip(hours.tolist(), sofa.tolist()))
        self.assertEqual(scores[5], 3.0)  # norepinephrine <= 0.1
        self.assertEqual(scores[6], 4.0)  # norepinephrine > 0.1

    def test_carevue_rates_last_at_most_six_hours(self):
        patient, stays = self._sofa_patient([
            _carevue_rate(T0 + timedelta(hours=2), 0.2, link=1),
            _carevue_rate(T0 + timedelta(hours=12), 0.2, link=1),
        ])
        norepi = self.task._vasopressors(patient, stays)[10]["norepi"]
        self.assertEqual(sorted(norepi), [2, 3, 4, 5, 6, 7, 12])

    def test_metavision_infusions(self):
        patient, stays = self._sofa_patient([
            _metavision_infusion(T0 + timedelta(hours=5), minutes=120, rate=0.2),
            _metavision_infusion(T0 + timedelta(hours=20), minutes=30, rate=0.5,
                                 link=2),
        ])
        norepi = self.task._vasopressors(patient, stays)[10]["norepi"]
        self.assertEqual(sorted(norepi), [5, 6, 7])

    def test_ventilation_is_inferred_from_charted_settings(self):
        patient, stays = self._sofa_patient([
            # Ventilator setting at 9:10, extubation at 11:40: hours 9 to 11.
            _chart(T0 + timedelta(hours=9, minutes=10), 223849, 1),
            _extubation(T0 + timedelta(hours=11, minutes=40)),
            # A setting without an extubation within 6 hours: 6 hours.
            _chart(T0 + timedelta(hours=20), 720, 1),
            _chart(T0 + timedelta(hours=30, minutes=5), 445, 1),
            # A setting 10 minutes before a charted extubation: dropped.
            _chart(T0 + timedelta(hours=40), 445, 1),
            _chart(T0 + timedelta(hours=40, minutes=10), 467, 1),
        ])
        ventilated = self.task._ventilated_hours(patient, stays)[10]
        self.assertEqual(ventilated, {9, 10, 11, *range(30, 37)})

    def test_ventilation_lifts_the_pafi_cap(self):
        inputs = {"po2": {10: 75.0}, "fio2": {10: 50.0}}  # PaO2/FiO2 = 150
        _, unventilated = self.task._sofa(inputs, {}, set(), set())
        _, ventilated = self.task._sofa(inputs, {}, {10}, set())
        self.assertEqual(unventilated[-1], 2.0)  # capped at 200
        self.assertEqual(ventilated[-1], 3.0)

    def test_carevue_intubation_counts_as_normal_gcs(self):
        hour = T0 + timedelta(hours=8, minutes=15)
        patient, stays = self._sofa_patient([
            _chart(hour, 184, 1.0),
            _chart(hour, 454, 1.0),
            _chart(hour, 723, 1.0, text="1.0 ET/Trach"),
        ])
        intubated = self.task._intubated_hours(patient, stays)[10]
        self.assertEqual(intubated, {8})
        inputs = self.task._sofa_inputs(patient, stays)[10]
        _, not_intubated = self.task._sofa(inputs, {}, set(), set())
        _, with_intubation = self.task._sofa(inputs, {}, set(), intubated)
        self.assertEqual(not_intubated[-1], 4.0)  # GCS 3
        self.assertEqual(with_intubation[-1], 0.0)  # GCS 15

    def test_charted_gcs_total_takes_precedence(self):
        inputs = {
            "egcs": {0: 1.0}, "vgcs": {0: 1.0}, "mgcs": {0: 1.0, 4: 1.0},
            "tgcs": {2: 12.0},
        }
        gcs = YAIBSepsisMIMIC3._gcs(inputs, intubated=set())
        self.assertEqual(gcs[0], 3.0)  # sum of the components
        self.assertEqual(gcs[2], 12.0)  # charted total
        self.assertEqual(gcs[4], 12.0)  # total carried forward over the sum


class TestYAIBMIMIC3Training(unittest.TestCase):
    def test_tasks_train_with_rnn_v2(self):
        patients = [
            _patient(
                stays=[(10 + i, 1, T0, 40 + 5 * i, range(40 + 5 * i))],
                admissions=[_admission(hours=60 + 5 * i)],
            )
            for i in range(2)
        ]
        dims: dict[str, list[LevelEmbedding]] = {
            "dynamic": [{"values": 8}],
            "static": [{"values": 4}],
        }
        for task, label in (
            (YAIBMortalityMIMIC3(), "mortality"),
            (YAIBLengthOfStayMIMIC3(), "los"),
            (YAIBAKIMIMIC3(), "aki"),
        ):
            with self.subTest(task=task.task_name):
                samples = [sample for patient in patients for sample in task(patient)]
                if label == "mortality":  # a binary label needs both classes
                    samples[0]["mortality"], samples[1]["mortality"] = 0, 1
                dataset = create_sample_dataset(
                    samples, task.input_schema, task.output_schema
                )
                model = RNN_V2(dataset, embedding_dim=dims, hidden_dim=8, mask="concat")
                out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
                self.assertTrue(torch.isfinite(out["loss"]))


if __name__ == "__main__":
    unittest.main()
