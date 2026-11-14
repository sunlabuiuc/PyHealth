import unittest
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
import torch
import yaml

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN_V2, LevelEmbedding
from pyhealth.processors import LabelProcessor
from pyhealth.tasks import (
    YAIBAKIEICU,
    YAIBKidneyFunctionEICU,
    YAIBLengthOfStayEICU,
    YAIBMortalityEICU,
    YAIBSepsisEICU,
)

FEATURES = YAIBMortalityEICU.DYNAMIC_FEATURES
HR = FEATURES.index("hr")
TEMP = FEATURES.index("temp")
URINE = FEATURES.index("urine")
CREA = FEATURES.index("crea")
MG = FEATURES.index("mg")
CA = FEATURES.index("ca")
CRP = FEATURES.index("crp")
FIO2 = FEATURES.index("fio2")
O2SAT = FEATURES.index("o2sat")


# Every column of the eICU config, as in the dataset's event dataframe.
_CONFIG = yaml.safe_load(
    (
        Path(__file__).parents[2] / "pyhealth" / "datasets" / "configs" / "eicu.yaml"
    ).read_text()
)
COLUMNS = [
    f"{table}/{column}"
    for table, config in _CONFIG["tables"].items()
    for column in config["attributes"]
]


def _labels(records):
    """The label values of a ``timeseries_*`` label series."""
    return [value[0] for value in records["values"]]


def _event(event_type, **attrs):
    # Attributes are strings in the dataset's event dataframe.
    row = {"event_type": event_type, "timestamp": None}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _frame(rows, patient_id="p1"):
    df = pl.from_dicts(rows, infer_schema_length=None)
    return df.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(
            (pl.col(c) if c in df.columns else pl.lit(None)).cast(pl.String).alias(c)
            for c in COLUMNS
        ),
    )


def _stay_rows(stay_id, hours, age="60", gender="Male", height=180, weight=80,
               status="Alive", hospital_id=1, hr_hours=None):
    """A stay and one heart rate per hour, so it passes the base cohort."""
    rows = [
        _event(
            "patient",
            patientunitstayid=stay_id,
            unitdischargeoffset=int(hours * 60),
            gender=gender,
            age=age,
            admissionheight=height,
            admissionweight=weight,
            unitdischargestatus=status,
            hospitalid=hospital_id,
        )
    ]
    for hour in range(int(hours)) if hr_hours is None else hr_hours:
        rows.append(
            _event("vitalperiodic", patientunitstayid=stay_id,
                   observationoffset=hour * 60 + 5, heartrate=80 + hour)
        )
    return rows


def _patient(stay_id=10, hours: float = 48, extra=(), patient_id="p1", **stay):
    rows = _stay_rows(stay_id, hours, **stay) + list(extra)
    return Patient(patient_id=patient_id, data_source=_frame(rows, patient_id))


def _lab(minute, name, value, unit="mg/dL", stay_id=10):
    return _event("lab", patientunitstayid=stay_id, labresultoffset=minute,
                  labname=name, labresult=value, labmeasurenameinterface=unit)


def _urine(minute, value, label="Urine", stay_id=10):
    return _event("intakeoutput", patientunitstayid=stay_id,
                  intakeoutputoffset=minute, celllabel=label,
                  cellvaluenumeric=value)


def _nurse(minute, label, value, stay_id=10):
    return _event("nursecharting", patientunitstayid=stay_id,
                  nursingchartoffset=minute,
                  nursingchartcelltypevalname=label, nursingchartvalue=value)


def _resp(minute, typecat="respFlowSettings", label="FiO2", value="40%",
          stay_id=10):
    return _event("respiratorycharting", patientunitstayid=stay_id,
                  respchartoffset=minute, respcharttypecat=typecat,
                  respchartvaluelabel=label, respchartvalue=value)


def _care(minute=0, start=0, end=0, prior_start=0, prior_end=0, stay_id=10):
    return _event("respiratorycare", patientunitstayid=stay_id,
                  respcarestatusoffset=minute, ventstartoffset=start,
                  ventendoffset=end, priorventstartoffset=prior_start,
                  priorventendoffset=prior_end)


def _infusion(minute, drug, rate, weight=80, stay_id=10):
    return _event("infusiondrug", patientunitstayid=stay_id,
                  infusionoffset=minute, drugname=drug, drugrate=rate,
                  patientweight=weight)


def _dataset_frame(*patients):
    """The event dataframe of a dataset holding the patients."""
    return pl.concat(
        [p.data_source for p in patients], how="diagonal_relaxed"
    ).lazy()


def _all_tables(df):
    """Adds an event of every table, as in a dataset that loads them all."""
    rows = [
        {"event_type": table, "timestamp": None}
        for table in _CONFIG["tables"]
        if table != "patient"
    ]
    return pl.concat(
        [df, _frame(rows, patient_id="none").lazy()], how="diagonal_relaxed"
    )


class TestYAIBMortalityEICU(unittest.TestCase):
    def setUp(self):
        self.task = YAIBMortalityEICU()

    def test_features_and_unit_conversions(self):
        extra = [
            # Fahrenheit temperature, converted to Celsius.
            _nurse(3 * 60 + 1, "Temperature (F)", 98.6),
            # Both urine items are summed within an hour.
            _urine(4 * 60 + 10, 100),
            _urine(4 * 60 + 50, 50, label="Output (mL)-Urethral Catheter"),
            # Median creatinine within an hour; a value before ICU admission
            # is outside the grid.
            _lab(5 * 60, "creatinine", 1.0),
            _lab(5 * 60 + 20, "creatinine", 2.0),
            _lab(-3 * 60, "creatinine", 5.0),
            # Magnesium in mEq/L is divided by 1.215, calcium in mmol/L is
            # multiplied by 4, CRP in mg/dL is multiplied by 10.
            _lab(6 * 60, "magnesium", 1.215, unit="mEq/L"),
            _lab(7 * 60, "calcium", 2.5, unit="mmol/L"),
            _lab(8 * 60, "CRP", 6.1, unit="mg/dl"),
            # FiO2 from respiratory charting is a percentage string.
            _resp(9 * 60),
            # Arterial saturation from the lab counts as o2sat.
            _lab(10 * 60, "O2 Sat (%)", 95, unit="%"),
        ]
        samples = self.task(_patient(extra=extra))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual((sample["stay_id"], sample["mortality"]), (10, 0))
        values = sample["dynamic"]["values"]
        self.assertEqual(values.shape, (25, len(FEATURES)))
        self.assertEqual(sample["dynamic"]["time"], [float(h) for h in range(25)])
        np.testing.assert_allclose(values[:, HR], 80 + np.arange(25))
        self.assertAlmostEqual(values[3, TEMP], 37.0)
        self.assertEqual(values[4, URINE], 150)
        self.assertEqual(values[5, CREA], 1.5)
        self.assertAlmostEqual(values[6, MG], 1.0)
        self.assertAlmostEqual(values[7, CA], 10.0)
        self.assertAlmostEqual(values[8, CRP], 61.0)
        self.assertAlmostEqual(values[9, FIO2], 40.0)
        self.assertEqual(values[10, O2SAT], 95.0)
        self.assertEqual(sample["static"]["time"], [0.0])
        self.assertEqual(sample["static"]["values"], [[60.0, 1.0, 180.0, 80.0]])

    def test_implausible_values_are_dropped(self):
        extra = [_lab(5 * 60, "creatinine", 20.0), _nurse(6 * 60, "Temperature (C)", 50)]
        values = self.task(_patient(extra=extra))[0]["dynamic"]["values"]
        self.assertTrue(np.isnan(values[5, CREA]))
        self.assertTrue(np.isnan(values[6, TEMP]))

    def test_icu_death_label(self):
        patient = _patient(status="Expired")
        self.assertEqual([s["mortality"] for s in self.task(patient)], [1])

    def test_short_stay_is_excluded(self):
        # Also excludes deaths within 30 hours, whose stay ends at death.
        self.assertEqual(self.task(_patient(hours=29, status="Expired")), [])
        self.assertEqual(self.task(_patient(hours=29)), [])

    def test_measurement_gap_is_excluded(self):
        hours = [h for h in range(48) if not 10 <= h < 23]
        self.assertEqual(self.task(_patient(hr_hours=hours)), [])

    def test_minor_is_excluded(self):
        self.assertEqual(self.task(_patient(age="17")), [])

    def test_age_over_89_and_missing_static(self):
        sample = self.task(_patient(age="> 89", gender="Female", height=None,
                                    weight=600))[0]
        age, sex, height, weight = sample["static"]["values"][0]
        self.assertEqual((age, sex), (90.0, 0.0))
        self.assertTrue(np.isnan(height) and np.isnan(weight))

    def test_pre_filter_requires_tables(self):
        # Without nursecharting, the dataset has none of its columns.
        df = _dataset_frame(_patient()).select(
            pl.exclude("^nursecharting/.*$")
        )
        with self.assertRaisesRegex(ValueError, "requires the eICU tables"):
            self.task.pre_filter(df)

    def test_pre_filter_drops_unused_rows(self):
        patient = _patient(extra=[
            _lab(60, "creatinine", 1.0),
            _lab(60, "troponin - I", 1.0),  # not a YAIB feature
            _nurse(60, "Temperature (C)", 37),
            _nurse(60, "Pain Score", 3),
            _resp(60),
            _resp(60, label="PEEP", value="5"),
        ])
        kept = self.task.pre_filter(_all_tables(_dataset_frame(patient))).collect()
        kept = kept.filter(pl.col("patient_id") == "p1")
        self.assertEqual(
            kept.filter(pl.col("event_type") == "lab")["lab/labname"].to_list(),
            ["creatinine"],
        )
        self.assertEqual(
            kept.filter(pl.col("event_type") == "nursecharting")[
                "nursecharting/nursingchartcelltypevalname"
            ].to_list(),
            ["Temperature (C)"],
        )
        self.assertEqual(
            kept.filter(pl.col("event_type") == "respiratorycharting")[
                "respiratorycharting/respchartvaluelabel"
            ].to_list(),
            ["FiO2"],
        )
        # Vital signs are read whole.
        self.assertEqual((kept["event_type"] == "vitalperiodic").sum(), 48)


class TestYAIBKidneyFunctionEICU(unittest.TestCase):
    def test_label_is_median_creatinine_of_day_two(self):
        extra = [_lab(h * 60, "creatinine", 1.0 if h <= 24 else 2.0 + (h == 48))
                 for h in range(49)]
        samples = YAIBKidneyFunctionEICU()(_patient(hours=60, extra=extra))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["creatinine"], 2.0)
        self.assertEqual(samples[0]["dynamic"]["values"].shape[0], 25)

    def test_stay_without_day_two_creatinine_is_excluded(self):
        extra = [_lab(h * 60, "creatinine", 1.0) for h in range(25)]
        self.assertEqual(YAIBKidneyFunctionEICU()(_patient(hours=60, extra=extra)), [])
        self.assertEqual(YAIBKidneyFunctionEICU()(_patient(hours=47)), [])


class TestYAIBLengthOfStayEICU(unittest.TestCase):
    def test_remaining_hours_at_every_hour(self):
        sample = YAIBLengthOfStayEICU()(_patient(hours=48.5))[0]
        self.assertEqual(sample["los"]["time"], [float(h) for h in range(49)])
        np.testing.assert_allclose(_labels(sample["los"]), 48 - np.arange(49))

    def test_remaining_hours_are_capped_at_seven_days(self):
        sample = YAIBLengthOfStayEICU()(_patient(hours=200))[0]
        labels = _labels(sample["los"])
        self.assertEqual(len(labels), 169)
        self.assertEqual((labels[0], labels[-1]), (168.0, 32.0))


class TestYAIBAKIEICU(unittest.TestCase):
    def setUp(self):
        self.task = YAIBAKIEICU()

    def test_creatinine_rise_labels_six_hours_around_onset(self):
        extra = [_lab(h * 60, "creatinine", 1.0 if h < 12 else 2.0) for h in range(72)]
        extra += [_urine(h * 60, 100) for h in range(72)]
        sample = self.task(_patient(hours=72, extra=extra))[0]
        labels = _labels(sample["aki"])
        self.assertEqual(labels, [0.0] * 6 + [1.0] * 13)  # onset 12, until 18

    def test_stay_without_aki_is_negative_for_seven_days(self):
        extra = [_urine(h * 60, 100) for h in range(200)]
        labels = _labels(self.task(_patient(hours=200, extra=extra))[0]["aki"])
        self.assertEqual(labels, [0.0] * 169)

    def test_onset_within_six_hours_is_excluded(self):
        extra = [_lab(h * 60, "creatinine", 1.0 if h < 3 else 2.0) for h in range(72)]
        self.assertEqual(self.task(_patient(hours=72, extra=extra)), [])

    def test_high_baseline_creatinine_is_excluded(self):
        extra = [_lab(h * 60, "creatinine", 5.0) for h in range(72)]
        self.assertEqual(self.task(_patient(hours=72, extra=extra)), [])

    def test_urine_rate_follows_ricu_row_order(self):
        # Hour 9 has a labeled row of 0 and a catheter row of 100 after a
        # gap since hour 3. As in ricu, the labeled item comes first and
        # takes the gap (0 / 7), so the catheter row is a rate of 100.
        extra = [
            _urine(60, 40), _urine(2 * 60, 20), _urine(3 * 60, 25),
            _urine(9 * 60 + 47, 100, label="Output (mL)-Urethral Catheter"),
            _urine(9 * 60 + 47, 0),
        ]
        rates = self.task._urine_rates(_patient(extra=extra), pl.DataFrame({"stay_id": [10]}))
        self.assertEqual(rates.filter(pl.col("hour") == 9)["rate"].to_list(), [100.0])

    def test_kdigo_urine_stages(self):
        hours = np.arange(0, 25)
        rate = np.full(len(hours), 0.4 * 80)  # 0.4 ml/kg/h for 80 kg
        _, stage = YAIBAKIEICU._kdigo_urine(hours, rate, 80.0)
        self.assertEqual(stage[5], 0)  # before 6 hours
        self.assertEqual(stage[6], 0)  # 6-hour sum 2.8 ml/kg (7 bins) >= 0.5
        _, low = YAIBAKIEICU._kdigo_urine(hours, np.zeros(len(hours)), 80.0)
        self.assertEqual((low[6], low[12], low[24]), (1, 3, 3))

    def test_hospital_without_aki_is_excluded(self):
        def stay(stay_id, hospital_id, patient_id, aki):
            extra = [_lab(h * 60, "creatinine", 2.0 if aki and h >= 12 else 1.0,
                          stay_id=stay_id) for h in range(72)]
            extra += [_urine(h * 60, 100, stay_id=stay_id) for h in range(72)]
            return _patient(stay_id, 72, extra, patient_id, hospital_id=hospital_id)

        df = _all_tables(_dataset_frame(
            stay(10, 1, "p1", aki=True),
            stay(11, 1, "p2", aki=False),
            stay(20, 2, "p3", aki=False),
        ))
        class SmallBatches(YAIBAKIEICU):
            _PREVALENCE_BATCH_SIZE = 2  # several batches

        task = SmallBatches()
        kept = task.pre_filter(df).collect()
        stays = kept.filter(pl.col("event_type") == "patient")[
            "patient/patientunitstayid"
        ].to_list()
        self.assertEqual(sorted(stays), ["10", "11"])
        self.assertFalse((kept["lab/patientunitstayid"] == "20").any())


class TestYAIBSepsisEICU(unittest.TestCase):
    def setUp(self):
        self.task = YAIBSepsisEICU()

    @staticmethod
    def _sepsis_patient(antibiotic=True, hospital_id=1, stay_id=10, patient_id="p1"):
        # Platelets and bilirubin worsen at hour 12, with antibiotics from
        # hour 12: SOFA rises by 4 at hour 12.
        extra = []
        for h in range(72):
            extra += [
                _lab(h * 60, "platelets x 1000", 200 if h < 12 else 40, "K/uL", stay_id),
                _lab(h * 60, "total bilirubin", 0.5 if h < 12 else 3.0, stay_id=stay_id),
                _lab(h * 60, "creatinine", 1.0, stay_id=stay_id),
            ]
            if antibiotic and h >= 12:
                extra.append(_infusion(h * 60, "Vancomycin (mg/hr)", 100, stay_id=stay_id))
        return _patient(stay_id, 72, extra, patient_id, hospital_id=hospital_id)

    def test_sepsis_onset_labels_six_hours_around_onset(self):
        samples = self.task(self._sepsis_patient())
        self.assertEqual(len(samples), 1)
        self.assertEqual(_labels(samples[0]["sepsis"]), [0.0] * 6 + [1.0] * 13)

    def test_stay_without_antibiotics_is_negative(self):
        samples = self.task(self._sepsis_patient(antibiotic=False))
        self.assertEqual(_labels(samples[0]["sepsis"]), [0.0] * 73)

    def test_ventilation_windows(self):
        patient = _patient(extra=[
            # Charted setting at 9:10, charted "off" at 11:40: hours 9 to 11.
            _resp(9 * 60 + 10, typecat="respFlowPtVentData", label="Total RR", value="14"),
            _resp(11 * 60 + 40, typecat="respFlowCareData", label="Vent", value="off"),
            # A start without an end: 6 hours, even late in the stay.
            _care(start=30 * 60 + 5),
            # A start 10 minutes before an end: dropped.
            _care(start=40 * 60, end=40 * 60 + 10),
            # A zero offset is no start; a window that ends before ICU
            # admission is extended to hour 0.
            _care(start=0, prior_start=-10 * 60),
        ])
        ventilated = self.task._ventilated_hours(patient)[10]
        self.assertEqual(ventilated, {*range(-10, 1), 9, 10, 11, *range(30, 37)})

    def test_ventilation_lifts_the_pafi_cap(self):
        inputs = {"po2": {10: 75.0}, "fio2": {10: 50.0}}  # PaO2/FiO2 = 150
        _, unventilated = self.task._sofa(inputs, {}, set(), set())
        _, ventilated = self.task._sofa(inputs, {}, {10}, set())
        self.assertEqual(unventilated[-1], 2.0)  # capped at 200
        self.assertEqual(ventilated[-1], 3.0)

    def test_sedation_counts_as_normal_gcs(self):
        patient = _patient(extra=[
            _nurse(8 * 60 + 15, "GCS Total", 3),
            _nurse(8 * 60 + 15, "Sedation Score", -4),
        ])
        intubated = self.task._intubated_hours(patient)[10]
        self.assertEqual(intubated, set(range(8, 15)))  # carried forward 6 h
        stays = self.task._stays(patient)
        inputs = self.task._sofa_inputs(patient, stays)[10]
        _, awake = self.task._sofa(inputs, {}, set(), set())
        _, sedated = self.task._sofa(inputs, {}, set(), intubated)
        self.assertEqual((awake[-1], sedated[-1]), (4.0, 0.0))  # GCS 3 vs 15

    def test_vasopressor_rate_per_kg_and_minute(self):
        # 8 mcg/min for 80 kg is 0.1 mcg/kg/min, over two hours.
        patient = _patient(extra=[
            _infusion(5 * 60, "Norepinephrine (mcg/min)", 8),
            _infusion(6 * 60, "Norepinephrine (mcg/min)", 8),
            _infusion(7 * 60, "Norepinephrine (mcg/min)", 8),
        ])
        static = self.task._static(patient, self.task._stays(patient))
        norepi = self.task._vasopressors(patient, static)[10]["norepi"]
        self.assertEqual(sorted(norepi), [5, 6, 7])
        self.assertAlmostEqual(norepi[6], 0.1)

    def test_hospital_without_sepsis_is_excluded(self):
        df = _all_tables(_dataset_frame(
            self._sepsis_patient(hospital_id=1, stay_id=10, patient_id="p1"),
            self._sepsis_patient(antibiotic=False, hospital_id=2, stay_id=20,
                                 patient_id="p2"),
        ))
        kept = self.task.pre_filter(df).collect()
        stays = kept.filter(pl.col("event_type") == "patient")[
            "patient/patientunitstayid"
        ].to_list()
        self.assertEqual(stays, ["10"])


class TestYAIBProcessingEICU(unittest.TestCase):
    """The tasks' schemas: processors, and a model, on task samples."""

    TASKS = (
        (YAIBMortalityEICU(), "mortality"),
        (YAIBKidneyFunctionEICU(), "creatinine"),
        (YAIBLengthOfStayEICU(), "los"),
        (YAIBAKIEICU(), "aki"),
        (YAIBSepsisEICU(), "sepsis"),
    )

    @staticmethod
    def _samples(task, label):
        samples = []
        for i in range(2):
            extra = [_lab(30 * 60, "creatinine", 1.0 + i, stay_id=10 + i)]
            extra += [_nurse(h * 60 + 10, "Temperature (C)", 36.5 + i + 0.1 * h, 10 + i)
                      for h in range(5, 40, 3)]
            patient = _patient(10 + i, 50 + 5 * i, extra, f"p{i}", weight=None)
            samples += task(patient)
        if label == "mortality":  # a binary label needs both classes
            samples[0]["mortality"], samples[1]["mortality"] = 0, 1
        return samples

    def test_task_samples_are_processed_by_the_schemas(self):
        for task, label in self.TASKS:
            with self.subTest(task=task.task_name):
                samples = self._samples(task, label)
                self.assertEqual(len(samples), 2)
                dataset = create_sample_dataset(samples, task.input_schema, task.output_schema)
                for i, sample in enumerate(samples):
                    raw = np.asarray(sample["dynamic"]["values"], dtype=float)
                    values, mask, time = dataset[i]["dynamic"]
                    self.assertEqual(tuple(values.shape), raw.shape)
                    self.assertTrue(torch.isfinite(values).all())
                    self.assertEqual(mask.bool().tolist(), np.isnan(raw).tolist())
                    self.assertEqual(time.tolist(), sample["dynamic"]["time"])
                    # Training mean (0) before the first temperature, then
                    # forward-filled.
                    self.assertTrue((values[:5, TEMP] == 0).all())
                    filled = values[5:8, TEMP]
                    self.assertTrue((filled == filled[0]).all())
                    static, static_mask, _ = dataset[i]["static"]
                    self.assertTrue(torch.isfinite(static).all())
                    self.assertEqual(static_mask[0].bool().tolist(),
                                     [False, False, False, True])

    def test_regression_label_is_minmax_scaled(self):
        task = YAIBKidneyFunctionEICU()
        samples = self._samples(task, "creatinine")
        dataset = create_sample_dataset(samples, task.input_schema, task.output_schema)
        processor = cast(LabelProcessor, dataset.output_processors["creatinine"])
        ys = sorted(float(dataset[i]["creatinine"]) for i in range(2))
        self.assertEqual(ys, [0.0, 1.0])
        y = processor.process(3.0)  # 3 on 1..2
        self.assertAlmostEqual(float(y[0]), 2.0)

    def test_tasks_train_with_rnn_v2(self):
        dims: dict[str, list[LevelEmbedding]] = {
            "dynamic": [{"values": 8}],
            "static": [{"values": 4}],
        }
        for task, label in self.TASKS:
            with self.subTest(task=task.task_name):
                samples = self._samples(task, label)
                dataset = create_sample_dataset(samples, task.input_schema, task.output_schema)
                model = RNN_V2(dataset, embedding_dim=dims, hidden_dim=8, mask="concat")
                out = model(**next(iter(get_dataloader(dataset, batch_size=2))))
                self.assertTrue(torch.isfinite(out["loss"]))
                per_stay = label in ("mortality", "creatinine")
                expected = 2 if per_stay else sum(len(s[label]["time"]) for s in samples)
                self.assertEqual(out["logit"].shape[0], expected)


if __name__ == "__main__":
    unittest.main()
