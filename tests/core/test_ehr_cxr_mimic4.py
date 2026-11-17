import pickle
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import numpy as np
import polars as pl
import torch
from PIL import Image

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import MedFuse
from pyhealth.processors import ImageProcessor, MultiLabelProcessor
from pyhealth.tasks import EHRCXRMortalityMIMIC4, EHRCXRPhenotypingMIMIC4
from pyhealth.tasks.ehr_cxr_mimic4 import _OptionalImageProcessor

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001 - MIMIC times are naive
FEATURES = EHRCXRMortalityMIMIC4.FEATURES
N_FEATURES = len(FEATURES)
HR = FEATURES.index("hr")
CRR = FEATURES.index("crr")
GCS_TOTAL = FEATURES.index("gcs_total")
MICU = "Medical Intensive Care Unit (MICU)"
SICU = "Surgical Intensive Care Unit (SICU)"
SEPSIS = "Septicemia (except in labor)"
SHOCK = "Shock"


def _event(event_type, timestamp, **attrs):
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = None if value is None else str(value)
    return row


def _hr(hours, value=80, stay_id=1):
    return _event(
        "chartevents", T0 + timedelta(hours=hours), stay_id=stay_id, hadm_id=10,
        itemid=220045, value=value, valuenum=value, valueuom="bpm",
    )


def _chart(hours, itemid, value=None, valuenum=None):
    return _event(
        "chartevents", T0 + timedelta(hours=hours), stay_id=1, hadm_id=10,
        itemid=itemid, value=value, valuenum=valuenum, valueuom=None,
    )


def _cxr(hours, path, view="AP", dicom_id=None):
    return _event(
        "metadata", T0 + timedelta(hours=hours), image_path=path,
        viewposition=view, dicom_id=dicom_id or path,
    )


def _diagnosis(version, code, hadm_id=10):
    return _event(
        "diagnoses_icd", T0, hadm_id=hadm_id, icd_version=version, icd_code=code
    )


def _stay(
    hours=72.0, stay_id=1, hadm_id=10, first=MICU, last=MICU, intime=T0
):
    outtime = intime + timedelta(hours=hours)
    return _event(
        "icustays", intime, stay_id=stay_id, hadm_id=hadm_id,
        first_careunit=first, last_careunit=last,
        outtime=outtime.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _patient(*events, stays=None, age=60, expire=0, hadm_id=10):
    rows = [
        _event("patients", None, anchor_age=age, anchor_year=2150),
        _event("admissions", T0, hadm_id=hadm_id, hospital_expire_flag=expire),
        # Every table exists in every patient, as in the global event frame.
        _event("metadata", None, image_path=None, viewposition=None, dicom_id=None),
        _event("chartevents", None, stay_id=None, hadm_id=None, itemid=None,
               value=None, valuenum=None, valueuom=None),
        _event("labevents", None, hadm_id=None, itemid=None, value=None,
               valuenum=None, valueuom=None),
        _event("diagnoses_icd", None, hadm_id=None, icd_version=None, icd_code=None),
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


class TestCXRPairing(unittest.TestCase):
    def setUp(self):
        self.task = EHRCXRMortalityMIMIC4()

    def _sample(self, *events, task=None):
        samples = (task or self.task)(_patient(_hr(1), *events))
        self.assertEqual(len(samples), 1)
        return samples[0]

    def test_paired_stay(self):
        sample = self._sample(_cxr(10, "a.jpg"))
        self.assertEqual(sample["cxr"], "a.jpg")
        self.assertEqual(sample["cxr_mask"], 1.0)
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["stay_id"], 1)

    def test_missing_image_keeps_stay_with_zero_mask(self):
        sample = self._sample()
        self.assertIsNone(sample["cxr"])
        self.assertEqual(sample["cxr_mask"], 0.0)

    def test_last_image_in_window_is_used(self):
        sample = self._sample(
            _cxr(30, "late.jpg"), _cxr(5, "early.jpg"), _cxr(20, "middle.jpg")
        )
        self.assertEqual(sample["cxr"], "late.jpg")

    def test_image_after_prediction_time_is_never_used(self):
        sample = self._sample(_cxr(10, "ok.jpg"), _cxr(48.1, "after.jpg"))
        self.assertEqual(sample["cxr"], "ok.jpg")
        # Still inside the stay, but after the prediction time.
        sample = self._sample(_cxr(48.1, "after.jpg"))
        self.assertIsNone(sample["cxr"])

    def test_image_at_prediction_time_is_used(self):
        self.assertEqual(self._sample(_cxr(48, "edge.jpg"))["cxr"], "edge.jpg")

    def test_image_before_admission_is_not_used(self):
        self.assertIsNone(self._sample(_cxr(-1, "before.jpg"))["cxr"])

    def test_other_views_are_not_used(self):
        sample = self._sample(_cxr(5, "ap.jpg"), _cxr(9, "pa.jpg", view="PA"))
        self.assertEqual(sample["cxr"], "ap.jpg")
        self.assertIsNone(self._sample(_cxr(9, "pa.jpg", view="PA"))["cxr"])
        task = EHRCXRMortalityMIMIC4(view_positions=("AP", "PA"))
        self.assertEqual(
            self._sample(_cxr(9, "pa.jpg", view="PA"), task=task)["cxr"], "pa.jpg"
        )

    def test_same_time_images_are_chosen_deterministically(self):
        sample = self._sample(
            _cxr(9, "b.jpg", dicom_id="b"), _cxr(9, "a.jpg", dicom_id="a")
        )
        self.assertEqual(sample["cxr"], "b.jpg")

    def test_paired_only_drops_stays_without_image(self):
        task = EHRCXRMortalityMIMIC4(paired_only=True)
        self.assertEqual(task(_patient(_hr(1))), [])
        self.assertEqual(len(task(_patient(_hr(1), _cxr(9, "a.jpg")))), 1)

    def test_images_go_to_the_right_stay(self):
        later = T0 + timedelta(days=30)
        patient = _patient(
            _hr(1),
            _hr(24 * 30 + 1, stay_id=2),
            _cxr(24 * 30 + 2, "second.jpg"),
            _event("admissions", later, hadm_id=11, hospital_expire_flag=1),
            stays=[_stay(), _stay(stay_id=2, hadm_id=11, intime=later)],
        )
        samples = self.task(patient)
        self.assertEqual([s["stay_id"] for s in samples], [1, 2])
        self.assertEqual([s["cxr"] for s in samples], [None, "second.jpg"])
        self.assertEqual([s["mortality"] for s in samples], [0, 1])


class TestCohort(unittest.TestCase):
    def setUp(self):
        self.task = EHRCXRMortalityMIMIC4()

    def test_mortality_labels(self):
        self.assertEqual(self.task(_patient(_hr(1)))[0]["mortality"], 0)
        self.assertEqual(self.task(_patient(_hr(1), expire=1))[0]["mortality"], 1)

    def test_stay_shorter_than_48_hours_is_excluded(self):
        self.assertEqual(self.task(_patient(_hr(1), stays=[_stay(47.9)])), [])
        self.assertEqual(len(self.task(_patient(_hr(1), stays=[_stay(48)]))), 1)

    def test_minors_are_excluded(self):
        self.assertEqual(self.task(_patient(_hr(1), age=17)), [])

    def test_age_is_taken_at_admission(self):
        patient = _patient(_hr(1), age=17)
        rows = patient.data_source.with_columns(
            pl.when(pl.col("event_type") == "patients")
            .then(pl.lit("2149"))
            .otherwise(pl.col("patients/anchor_year"))
            .alias("patients/anchor_year")
        )
        self.assertEqual(len(self.task(Patient("p1", rows))), 1)

    def test_care_unit_transfer_is_excluded(self):
        patient = _patient(_hr(1), stays=[_stay(first=MICU, last=SICU)])
        self.assertEqual(self.task(patient), [])

    def test_admission_with_two_stays_is_excluded(self):
        second = _stay(stay_id=2, intime=T0 + timedelta(days=5))
        self.assertEqual(self.task(_patient(_hr(1), stays=[_stay(), second])), [])

    def test_stay_without_measurements_is_excluded(self):
        self.assertEqual(self.task(_patient(_cxr(1, "a.jpg"))), [])
        # A measurement after the window does not rescue it.
        self.assertEqual(self.task(_patient(_hr(50))), [])

    def test_patient_without_stay_gives_no_samples(self):
        self.assertEqual(self.task(_patient(stays=[])), [])


class TestTimeSeries(unittest.TestCase):
    def setUp(self):
        self.task = EHRCXRMortalityMIMIC4()

    def _ehr(self, *events):
        return self.task(_patient(*events))[0]["ehr"]

    def test_shape_covers_first_48_hours(self):
        self.assertEqual(self._ehr(_hr(1)).shape, (24, 2 * N_FEATURES))

    def test_bin_hours(self):
        task = EHRCXRMortalityMIMIC4(bin_hours=1)
        self.assertEqual(task(_patient(_hr(1)))[0]["ehr"].shape[0], 48)

    def test_last_value_of_a_bin_is_kept(self):
        ehr = self._ehr(_hr(0.5, value=70), _hr(1.5, value=90))
        self.assertEqual(ehr[0, HR], 90)

    def test_forward_fill_and_mask(self):
        ehr = self._ehr(_hr(3, value=75))
        self.assertTrue(np.isnan(ehr[0, HR]))  # before the first measurement
        self.assertEqual(ehr[1, HR], 75)
        self.assertEqual(ehr[23, HR], 75)
        mask = ehr[:, N_FEATURES + HR]
        self.assertEqual(mask.tolist(), [0, 1] + [0] * 22)

    def test_measurements_after_48_hours_are_ignored(self):
        ehr = self._ehr(_hr(1, value=70), _hr(48, value=200), _hr(60, value=200))
        self.assertTrue(np.all(ehr[:, HR] == 70))

    def test_chart_rows_of_the_stay_before_admission_go_to_the_first_bin(self):
        weight = FEATURES.index("weight")
        event = _chart(-0.01, 226512, valuenum=77)
        ehr = self._ehr(_hr(1), event)
        self.assertEqual(ehr[0, weight], 77)
        self.assertEqual(ehr[0, N_FEATURES + weight], 1)
        self.assertEqual(ehr[23, weight], 77)

    def test_chart_rows_of_another_stay_before_admission_are_ignored(self):
        ehr = self._ehr(_hr(1, value=80), _hr(-1, value=120, stay_id=9))
        self.assertEqual(ehr[0, HR], 80)
        self.assertEqual(ehr[0, N_FEATURES + HR], 1)

    def test_out_of_range_values_are_dropped(self):
        ehr = self._ehr(_hr(1, value=70), _hr(5, value=900))
        self.assertTrue(np.all(ehr[:, HR] == 70))

    def test_capillary_refill_text(self):
        ehr = self._ehr(
            _hr(1),
            _chart(1, 223951, value="Abnormal >3 Seconds"),
            _chart(3, 224308, value="Normal <3 Seconds"),
        )
        self.assertEqual(ehr[0, CRR], 1)
        self.assertEqual(ehr[1, CRR], 0)

    def test_gcs_total_needs_all_three_scales(self):
        ehr = self._ehr(
            _chart(1, 220739, valuenum=4),
            _chart(1, 223901, valuenum=6),
            _chart(1, 223900, valuenum=5),
            _chart(3, 220739, valuenum=3),
        )
        self.assertEqual(ehr[0, GCS_TOTAL], 15)
        self.assertEqual(ehr[1, N_FEATURES + GCS_TOTAL], 0)

    def test_temperature_units_are_converted(self):
        event = _event(
            "chartevents", T0, stay_id=1, hadm_id=10, itemid=223761,
            value=98.6, valuenum=98.6, valueuom="°F",
        )
        self.assertAlmostEqual(
            float(self._ehr(event)[0, FEATURES.index("temp")]), 37.0, places=3
        )

    def test_glucose_from_labevents(self):
        event = _event(
            "labevents", T0, hadm_id=10, itemid=50931, value=120, valuenum=120,
            valueuom="mg/dL",
        )
        self.assertEqual(self._ehr(event)[0, FEATURES.index("glu")], 120)


class TestPhenotyping(unittest.TestCase):
    def setUp(self):
        icd9 = mock.Mock()
        icd9.map.side_effect = lambda code: {"99591": ["2"], "V1582": ["663"]}.get(
            code, []
        )
        icd10 = mock.Mock()
        icd10.map.side_effect = lambda code: {"R570": ["249"]}.get(code, [])
        patcher = mock.patch(
            "pyhealth.tasks.ehr_cxr_mimic4._ccs_maps",
            return_value={"9": icd9, "10": icd10},
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.task = EHRCXRPhenotypingMIMIC4()

    def test_labels_from_icd9_and_icd10(self):
        sample = self.task(
            _patient(_hr(1), _diagnosis(9, "99591"), _diagnosis(10, "R570"))
        )[0]
        self.assertEqual(sample["phenotypes"], sorted([SEPSIS, SHOCK]))

    def test_categories_outside_the_benchmark_are_ignored(self):
        sample = self.task(_patient(_hr(1), _diagnosis(9, "V1582")))[0]
        self.assertEqual(sample["phenotypes"], [])

    def test_other_admissions_do_not_count(self):
        sample = self.task(_patient(_hr(1), _diagnosis(9, "99591", hadm_id=99)))[0]
        self.assertEqual(sample["phenotypes"], [])

    def test_whole_stay_and_any_length(self):
        sample = self.task(_patient(_hr(1), _hr(70), stays=[_stay(71)]))[0]
        self.assertEqual(sample["ehr"].shape, (36, 2 * N_FEATURES))
        self.assertEqual(sample["ehr"][35, N_FEATURES + HR], 1)
        short = self.task(_patient(_hr(1), stays=[_stay(5)]))[0]
        self.assertEqual(short["ehr"].shape[0], 3)

    def test_last_image_of_the_stay_is_used(self):
        sample = self.task(
            _patient(_hr(1), _cxr(5, "a.jpg"), _cxr(60, "b.jpg"), _cxr(80, "c.jpg"))
        )[0]
        self.assertEqual(sample["cxr"], "b.jpg")
        self.assertEqual(sample["cxr_mask"], 1.0)

    def test_image_after_stay_is_missing(self):
        sample = self.task(_patient(_hr(1), _cxr(80, "c.jpg")))[0]
        self.assertIsNone(sample["cxr"])
        self.assertEqual(sample["cxr_mask"], 0.0)

    def test_output_has_all_25_labels_whatever_the_data(self):
        processor = MultiLabelProcessor(
            **EHRCXRPhenotypingMIMIC4.output_schema["phenotypes"][1]
        )
        processor.fit([{"phenotypes": [SHOCK]}], "phenotypes")
        target = processor.process([SEPSIS])
        self.assertEqual(target.shape, (25,))
        self.assertEqual(target.sum(), 1)
        self.assertEqual(target[0], 1)


class TestPreFilter(unittest.TestCase):
    def _frame(self, tables=("chartevents", "labevents", "metadata")):
        rows = {
            "event_type": ["chartevents", "chartevents", "labevents", "patients"],
            **{f"{table}/itemid": ["220045", "1", "1", None] for table in tables},
        }
        return pl.LazyFrame(rows)

    def test_drops_unused_items_only(self):
        kept = EHRCXRMortalityMIMIC4().pre_filter(self._frame()).collect()
        self.assertEqual(kept["event_type"].to_list(), ["chartevents", "patients"])

    def test_missing_table_is_reported(self):
        with self.assertRaisesRegex(ValueError, "metadata"):
            EHRCXRMortalityMIMIC4().pre_filter(
                self._frame(("chartevents", "labevents"))
            )
        with self.assertRaisesRegex(ValueError, "diagnoses_icd"):
            EHRCXRPhenotypingMIMIC4().pre_filter(self._frame())


class TestMedFuseIntegration(unittest.TestCase):
    def test_forward_gradients_and_missing_image(self):
        task = EHRCXRMortalityMIMIC4(image_size=32)
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "cxr.png")
            Image.fromarray(np.full((12, 12), 128, dtype=np.uint8)).save(path)
            samples = [
                *task(_patient(_hr(1, value=70), _cxr(10, path), expire=1)),
                *task(_patient(_hr(1, value=80), _cxr(20, path))),
                *task(_patient(_hr(1, value=90), _chart(1, 220739, valuenum=4))),
            ]
            for index, sample in enumerate(samples):
                sample["patient_id"] = f"p{index}"
            dataset = create_sample_dataset(
                samples=samples,
                input_schema=task.input_schema,
                output_schema=task.output_schema,
                dataset_name="ehr_cxr_test",
                in_memory=True,
            )
            batch = next(iter(get_dataloader(dataset, batch_size=3)))
            self.assertEqual(batch["ehr"].shape, (3, 24, 34))
            self.assertEqual(batch["cxr"].shape, (3, 3, 32, 32))
            self.assertEqual(sorted(batch["cxr_mask"].tolist()), [0.0, 1.0, 1.0])
            self.assertFalse(torch.isnan(batch["ehr"]).any())
            missing = batch["cxr_mask"] == 0
            self.assertEqual(batch["cxr"][missing].abs().sum(), 0)

            model = MedFuse(
                dataset=dataset,
                ehr_hidden_dim=4,
                ehr_num_layers=1,
                cxr_backbone="resnet18",
                cxr_pretrained=False,
                fusion_hidden_dim=4,
                projection_dim=4,
                dropout=0.0,
            )
            output = model(**batch)
            output["loss"].backward()
            self.assertEqual(output["y_prob"].shape, (3, 1))
            self.assertIsNotNone(model.layer.projection.weight.grad)


class TestOptionalImageProcessor(unittest.TestCase):
    def test_missing_image_is_a_zero_image(self):
        processor = _OptionalImageProcessor(image_size=4, mode="RGB")
        self.assertEqual(processor.process(None).shape, (3, 4, 4))
        self.assertEqual(processor.process(None).sum(), 0)
        self.assertEqual(
            _OptionalImageProcessor(image_size=4, mode="L").process(None).shape,
            (1, 4, 4),
        )

    def test_present_image_matches_image_processor(self):
        with tempfile.TemporaryDirectory() as root:
            path = str(Path(root) / "cxr.png")
            Image.fromarray(np.full((6, 6), 200, dtype=np.uint8)).save(path)
            ours = _OptionalImageProcessor(image_size=4, mode="RGB").process(path)
            reference = ImageProcessor(image_size=4, mode="RGB").process(path)
            self.assertTrue(torch.equal(ours, reference))

    def test_shape_must_be_known(self):
        with self.assertRaises(ValueError):
            _OptionalImageProcessor(image_size=4, mode="P")
        with self.assertRaises(ValueError):
            _OptionalImageProcessor(image_size=4, mode="RGB", to_tensor=False)

    def test_pickles(self):
        processor = pickle.loads(
            pickle.dumps(_OptionalImageProcessor(image_size=4, mode="RGB"))
        )
        self.assertEqual(processor.process(None).shape, (3, 4, 4))


if __name__ == "__main__":
    unittest.main()
