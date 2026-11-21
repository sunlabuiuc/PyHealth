import unittest
from datetime import datetime, timedelta

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import MAESTROMortalityMIMIC4
from pyhealth.tasks.ehr_cxr_mimic4 import _MEASURED_NAMES
from pyhealth.tasks.maestro_mortality_mimic4 import (
    MODALITY_GROUPS,
    _modality_columns,
)

T0 = datetime(2150, 1, 1, 8, 0, 0)
MICU = "Medical Intensive Care Unit (MICU)"
HR_ITEMID = 220045  # heart rate, a "vitals" variable


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _hr(hours, value=80, stay_id=1):
    return _event(
        "chartevents",
        T0 + timedelta(hours=hours),
        stay_id=stay_id,
        hadm_id=10,
        itemid=HR_ITEMID,
        value=value,
        valuenum=value,
        valueuom="bpm",
    )


def _stay(hours=72.0, stay_id=1, hadm_id=10):
    outtime = T0 + timedelta(hours=hours)
    return _event(
        "icustays",
        T0,
        stay_id=stay_id,
        hadm_id=hadm_id,
        first_careunit=MICU,
        last_careunit=MICU,
        outtime=outtime.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _patient(*events, stays=None, expire=0, hadm_id=10):
    rows = [
        _event("patients", None, anchor_age=60, anchor_year=2150),
        _event("admissions", T0, hadm_id=hadm_id, hospital_expire_flag=expire),
        # Every table exists in every patient, as in the global event frame.
        _event(
            "chartevents",
            None,
            stay_id=None,
            hadm_id=None,
            itemid=None,
            value=None,
            valuenum=None,
            valueuom=None,
        ),
        _event(
            "labevents",
            None,
            hadm_id=None,
            itemid=None,
            value=None,
            valuenum=None,
            valueuom=None,
        ),
        *(stays if stays is not None else [_stay()]),
        *events,
    ]
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


class TestMAESTROMortalityMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = MAESTROMortalityMIMIC4()

    def test_input_schema_has_one_key_per_modality(self):
        self.assertEqual(sorted(self.task.input_schema), sorted(MODALITY_GROUPS))
        self.assertEqual(self.task.output_schema, {"mortality": "binary"})

    def test_modalities_cover_the_sixteen_measured_variables(self):
        covered = [name for names in MODALITY_GROUPS.values() for name in names]
        self.assertEqual(sorted(covered), sorted(_MEASURED_NAMES))

    def test_sample_shapes_match_modality_widths(self):
        samples = self.task(_patient(_hr(1), _hr(30)))
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["mortality"], 0)
        for name, names in MODALITY_GROUPS.items():
            # 24 two-hour bins; each variable has a value and a mask column.
            self.assertEqual(sample[name].shape, (24, 2 * len(names)))

    def test_heart_rate_lands_in_vitals_value_and_mask_columns(self):
        sample = self.task(_patient(_hr(1, value=95)))[0]
        vitals = MODALITY_GROUPS["vitals"]
        hr_col = vitals.index("hr")
        # The HR measurement at hour 1 falls in bin 0; its mask is 1 there.
        self.assertAlmostEqual(float(sample["vitals"][0, hr_col]), 95.0, places=4)
        self.assertEqual(float(sample["vitals"][0, len(vitals) + hr_col]), 1.0)
        self.assertEqual(float(sample["vitals"][1, len(vitals) + hr_col]), 0.0)

    def test_modality_columns_point_to_values_then_masks(self):
        columns = _modality_columns("neuro")
        names = MODALITY_GROUPS["neuro"]
        self.assertEqual(len(columns), 2 * len(names))
        self.assertEqual(
            columns[: len(names)], [_MEASURED_NAMES.index(n) for n in names]
        )

    def test_stay_shorter_than_48_hours_is_excluded(self):
        self.assertEqual(self.task(_patient(_hr(1), stays=[_stay(hours=24)])), [])

    def test_stay_without_measurements_is_excluded(self):
        self.assertEqual(self.task(_patient(stays=[_stay()])), [])

    def test_mortality_label_comes_from_admission(self):
        sample = self.task(_patient(_hr(1), expire=1))[0]
        self.assertEqual(sample["mortality"], 1)


if __name__ == "__main__":
    unittest.main()
