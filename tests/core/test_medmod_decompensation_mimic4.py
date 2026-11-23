# Authors: Christopher Perez, Kai Cheung (kaicc2), Sarah Lee
# Paper: MedMod: Multimodal Benchmark for Medical Prediction Tasks with Electronic
#   Health Records and Chest X-Ray Scans (Elsharief et al., CHIL 2025)
# Paper link: https://proceedings.mlr.press/v287/elsharief25a.html
# Description: Tests for MedModDecompensationMIMIC4, synthetic patients only.

import unittest
from datetime import datetime, timedelta

import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import DAFT
from pyhealth.tasks import MedModDecompensationMIMIC4

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001 - MIMIC times are naive
FMT = "%Y-%m-%d %H:%M:%S"
HR = MedModDecompensationMIMIC4.FEATURES.index("hr")
N_FEATURES = len(MedModDecompensationMIMIC4.FEATURES)


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
        itemid=220045,
        value=value,
        valuenum=value,
        valueuom="bpm",
    )


def _lab_glucose(hours, value=120):
    return _event(
        "labevents",
        T0 + timedelta(hours=hours),
        hadm_id=10,
        itemid=50931,
        value=value,
        valuenum=value,
        valueuom="mg/dL",
    )


def _cxr(hours, path, view="AP", dicom_id=None):
    return _event(
        "metadata",
        T0 + timedelta(hours=hours),
        image_path=path,
        viewposition=view,
        dicom_id=dicom_id or path,
    )


def _stay(hours=24.0, stay_id=1, hadm_id=10, first="MICU", last="MICU"):
    return _event(
        "icustays",
        T0,
        stay_id=stay_id,
        hadm_id=hadm_id,
        first_careunit=first,
        last_careunit=last,
        outtime=(T0 + timedelta(hours=hours)).strftime(FMT),
    )


def _patient(
    *events, stay_hours=24.0, death_hours=None, expire=None, age=60, stays=None
):
    if expire is None:
        expire = int(death_hours is not None)
    deathtime = None
    if death_hours is not None:
        deathtime = (T0 + timedelta(hours=death_hours)).strftime(FMT)
    rows = [
        _event("patients", None, anchor_age=age, anchor_year=2150),
        _event(
            "admissions",
            T0,
            hadm_id=10,
            deathtime=deathtime,
            hospital_expire_flag=expire,
        ),
        # Every table exists in every patient, as in the global event frame.
        _event("metadata", None, image_path=None, viewposition=None, dicom_id=None),
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
        *(stays if stays is not None else [_stay(stay_hours)]),
        *events,
    ]
    df = pl.from_dicts(rows, infer_schema_length=None)
    df = df.with_columns(
        pl.lit("p1").alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )
    return Patient(patient_id="p1", data_source=df)


class TestSampling(unittest.TestCase):
    def setUp(self):
        self.task = MedModDecompensationMIMIC4(sample_every_hours=4)

    def test_prediction_hours(self):
        self.assertEqual(self.task.prediction_hours(24.0), [8, 12, 16, 20, 24])
        self.assertEqual(self.task.prediction_hours(23.9), [8, 12, 16, 20])
        self.assertEqual(self.task.prediction_hours(4.0), [])
        hourly = MedModDecompensationMIMIC4(sample_every_hours=1)
        self.assertEqual(hourly.prediction_hours(7.5), [5, 6, 7])

    def test_one_sample_per_prediction_time(self):
        samples = self.task(_patient(_hr(1)))
        self.assertEqual([s["hour"] for s in samples], [8, 12, 16, 20, 24])
        sample = samples[0]
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["visit_id"], "10")
        self.assertEqual(sample["stay_id"], 1)

    def test_no_sample_before_first_measurement(self):
        samples = self.task(_patient(_hr(10)))
        self.assertEqual([s["hour"] for s in samples], [12, 16, 20, 24])

    def test_sampling_stops_at_death(self):
        samples = self.task(_patient(_hr(1), stay_hours=24, death_hours=13))
        self.assertEqual([s["hour"] for s in samples], [8, 12])

    def test_excluded_stays_give_no_samples(self):
        self.assertEqual(self.task(_patient(_hr(1), age=16)), [])
        moved = _stay(first="MICU", last="SICU")
        self.assertEqual(self.task(_patient(_hr(1), stays=[moved])), [])

    def test_invalid_arguments(self):
        for bad in (
            dict(bin_hours=0),
            dict(window_hours=5, bin_hours=2),
            dict(sample_every_hours=0),
            dict(horizon_hours=0),
        ):
            with self.subTest(**bad), self.assertRaises(ValueError):
                MedModDecompensationMIMIC4(**bad)


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.task = MedModDecompensationMIMIC4(sample_every_hours=4)

    def _labels(self, **kwargs):
        return {
            s["hour"]: s["decompensation"]
            for s in self.task(_patient(_hr(1), **kwargs))
        }

    def test_survivor_is_always_negative(self):
        self.assertEqual(set(self._labels(stay_hours=48).values()), {0})

    def test_death_within_horizon_is_positive(self):
        labels = self._labels(stay_hours=48, death_hours=40)
        # 40 - t < 24  <=>  t > 16
        self.assertEqual(
            labels, {8: 0, 12: 0, 16: 0, 20: 1, 24: 1, 28: 1, 32: 1, 36: 1, 40: 1}
        )

    def test_horizon_is_configurable(self):
        task = MedModDecompensationMIMIC4(sample_every_hours=4, horizon_hours=6)
        samples = task(_patient(_hr(1), stay_hours=24, death_hours=20))
        labels = {s["hour"]: s["decompensation"] for s in samples}
        self.assertEqual(labels, {8: 0, 12: 0, 16: 1, 20: 1})

    def test_death_without_deathtime_is_negative(self):
        samples = self.task(_patient(_hr(1), stay_hours=24, expire=1))
        self.assertEqual([s["hour"] for s in samples], [8, 12, 16, 20, 24])
        self.assertEqual({s["decompensation"] for s in samples}, {0})


class TestInputs(unittest.TestCase):
    def setUp(self):
        self.task = MedModDecompensationMIMIC4(sample_every_hours=4)

    def _by_hour(self, *events, **kwargs):
        return {s["hour"]: s for s in self.task(_patient(*events, **kwargs))}

    def test_ehr_shape_is_fixed(self):
        for sample in self.task(_patient(_hr(1), stay_hours=72)):
            self.assertEqual(sample["ehr"].shape, (24, 2 * N_FEATURES))

    def test_window_ends_at_prediction_time(self):
        samples = self._by_hour(
            _hr(1, value=70), _hr(9, value=90), _hr(12.5, value=110)
        )
        # At t=8 the 48 h window is [-40, 8): the 1 h value sits in bin 20.
        self.assertEqual(samples[8]["ehr"][20, HR], 70)
        self.assertEqual(samples[8]["ehr"][-1, HR], 70)  # forward-filled
        self.assertEqual(samples[8]["ehr"][-1, N_FEATURES + HR], 0)
        # At t=12 the 9 h value is the latest; the 12.5 h value is after t.
        self.assertEqual(samples[12]["ehr"][-1, HR], 90)

    def test_window_sliding_past_admission(self):
        glucose = MedModDecompensationMIMIC4.FEATURES.index("glu")
        samples = self._by_hour(
            _hr(1, value=70),
            _lab_glucose(2, value=120),
            _hr(51, value=95),
            stay_hours=72,
        )
        # t=48: the window is [0, 48), both values are measured in it.
        ehr = samples[48]["ehr"]
        self.assertEqual(ehr[0, N_FEATURES + HR], 1)
        self.assertEqual(ehr[1, glucose], 120)
        self.assertEqual(ehr[1, N_FEATURES + glucose], 1)
        # t=52: the window is [4, 52). Values from before it are carried in as
        # unmeasured, labs and charts alike.
        ehr = samples[52]["ehr"]
        self.assertEqual(ehr[0, HR], 70)
        self.assertEqual(ehr[0, glucose], 120)
        self.assertEqual(ehr[-1, glucose], 120)
        self.assertEqual(ehr[:, N_FEATURES + glucose].sum(), 0)
        self.assertEqual(ehr[:-1, N_FEATURES + HR].sum(), 0)
        self.assertEqual(ehr[-1, HR], 95)
        self.assertEqual(ehr[-1, N_FEATURES + HR], 1)

    def test_pre_icu_labs_are_not_used(self):
        sample = self._by_hour(_hr(1), _lab_glucose(-3))[8]
        glucose = MedModDecompensationMIMIC4.FEATURES.index("glu")
        self.assertEqual(sample["ehr"][:, N_FEATURES + glucose].sum(), 0)

    def test_chart_value_just_before_admission(self):
        samples = self._by_hour(_hr(-1, value=60), stay_hours=52)
        # t=8: the window is [-40, 8), so the -1 h value sits in its own bin,
        # the last one before admission, and counts as measured.
        ehr = samples[8]["ehr"]
        self.assertEqual(ehr[19, HR], 60)
        self.assertEqual(ehr[19, N_FEATURES + HR], 1)
        self.assertEqual(ehr[:, N_FEATURES:].sum(), 1)
        # t=48: the window starts at admission; the value goes to the first bin.
        ehr = samples[48]["ehr"]
        self.assertEqual(ehr[0, HR], 60)
        self.assertEqual(ehr[0, N_FEATURES + HR], 1)
        # t=52: the window starts after admission; it is carried in unmeasured.
        ehr = samples[52]["ehr"]
        self.assertEqual(ehr[0, HR], 60)
        self.assertEqual(ehr[:, N_FEATURES:].sum(), 0)

    def test_bin_spanning_admission(self):
        task = MedModDecompensationMIMIC4(sample_every_hours=1)
        samples = {
            s["hour"]: s
            for s in task(_patient(_hr(-0.5, value=60), _hr(0.5, value=80)))
        }
        # t=5 is not a multiple of the 2 h bins: the window is [-43, 5) and bin 21,
        # [-1, 1), spans admission, so it holds the -0.5 h and the 0.5 h value.
        ehr = samples[5]["ehr"]
        self.assertEqual(ehr[21, HR], 80)
        self.assertEqual(ehr[21, N_FEATURES + HR], 1)
        self.assertEqual(ehr[:, N_FEATURES + HR].sum(), 1)

    def test_latest_image_up_to_prediction_time(self):
        samples = self._by_hour(
            _hr(1), _cxr(6, "a.jpg"), _cxr(11, "b.jpg"), stay_hours=16
        )
        self.assertEqual(samples[8]["cxr"], "a.jpg")
        self.assertEqual(samples[12]["cxr"], "b.jpg")
        self.assertEqual(samples[12]["cxr_mask"], 1.0)

    def test_image_after_prediction_time_is_never_used(self):
        samples = self._by_hour(_hr(1), _cxr(9, "late.jpg"), stay_hours=12)
        self.assertIsNone(samples[8]["cxr"])
        self.assertEqual(samples[8]["cxr_mask"], 0.0)
        self.assertEqual(samples[12]["cxr"], "late.jpg")

    def test_image_rules(self):
        samples = self._by_hour(
            _hr(1),
            _cxr(-2, "before_icu.jpg"),
            _cxr(5, "pa.jpg", view="PA"),
            _cxr(6, "b.jpg", dicom_id="b"),
            _cxr(6, "a.jpg", dicom_id="a"),
            stay_hours=8,
        )
        self.assertEqual(samples[8]["cxr"], "b.jpg")

    def test_paired_only_drops_times_without_image(self):
        task = MedModDecompensationMIMIC4(sample_every_hours=4, paired_only=True)
        samples = task(_patient(_hr(1), _cxr(10, "a.jpg"), stay_hours=16))
        self.assertEqual([s["hour"] for s in samples], [12, 16])


class TestWithModel(unittest.TestCase):
    def test_samples_feed_daft(self):
        task = MedModDecompensationMIMIC4(sample_every_hours=4, image_size=32)
        patient = _patient(_hr(1), _hr(9, value=95), stay_hours=32, death_hours=32)
        samples = task(patient)
        labels = [s["decompensation"] for s in samples]
        self.assertEqual(labels, [0, 1, 1, 1, 1, 1, 1])
        dataset = create_sample_dataset(
            samples=samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name="decompensation_daft_test",
        )
        model = DAFT(
            dataset, ehr_hidden_dim=8, cxr_backbone="resnet18", cxr_pretrained=False
        )
        batch = next(iter(get_dataloader(dataset, batch_size=len(samples))))
        out = model(**batch)
        self.assertEqual(out["y_prob"].shape, (len(samples), 1))
        self.assertEqual(out["y_true"].view(-1).tolist(), [float(y) for y in labels])
        self.assertFalse(torch.isnan(out["y_prob"]).any())


if __name__ == "__main__":
    unittest.main()
