import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

import dask.dataframe as dd
import numpy as np
import pandas as pd
import polars as pl
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

from pyhealth.data import Patient
from pyhealth.datasets import (
    GoldTestSplit,
    MIMIC4ImaGenomeDataset,
    create_sample_dataset,
    get_dataloader,
)
from pyhealth.processors import MultiLabelProcessor
from pyhealth.tasks import DiseaseProgressionMIMIC4

T0 = datetime(2150, 1, 1, 8, 0, 0)  # noqa: DTZ001
TASK = DiseaseProgressionMIMIC4
HR = TASK.FEATURES.index("hr")
N_FEATURES = len(TASK.FEATURES)
LUNG = "left lung"
OPACITY = "lung opacity"
EDEMA = "pulmonary edema/hazy opacity"
MICU = "Medical Intensive Care Unit (MICU)"
NOT_MENTIONED = 3


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


def _image(hours, dicom_id, study_id=None, view="AP", scene_graph=True):
    """A metadata row and, by default, the image's scene graph."""
    rows = [
        _event(
            "metadata", T0 + timedelta(hours=hours), dicom_id=dicom_id,
            study_id=study_id or f"s-{dicom_id}", viewposition=view,
            image_path=f"{dicom_id}.jpg",
        )
    ]
    if scene_graph:
        rows.append(_event("silver_images", T0, dicom_id=dicom_id))
    return rows


def _silver(first, second, findings=OPACITY, status="worsened", region=LUNG):
    return _event(
        "silver_comparisons", T0, dicom_id=second, previous_dicom_id=first,
        bbox_name=region, findings=findings, comparisons=status,
    )


def _gold(first, second, finding=OPACITY, status="improved", region=LUNG):
    return _event(
        "gold_comparisons", T0, dicom_id=second, previous_dicom_id=first,
        bbox_name=region, finding=finding, comparison=status,
    )


def _stay(hours=120.0, stay_id=1, hadm_id=10, intime=T0):
    outtime = intime + timedelta(hours=hours)
    return _event(
        "icustays", intime, stay_id=stay_id, hadm_id=hadm_id,
        first_careunit=MICU, last_careunit=MICU,
        outtime=outtime.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _rows(*events, stays=None, patient_id="p1", age=60):
    flat = []
    for event in events:
        flat.extend(event if isinstance(event, list) else [event])
    # Every table exists in every patient, as in the global event frame.
    return [
        _event("patients", None, gender="M", anchor_age=age, anchor_year=2150),
        _event("metadata", None, dicom_id=None, study_id=None, viewposition=None,
               image_path=None),
        _event("silver_images", None, dicom_id=None),
        _event("silver_comparisons", None, dicom_id=None, previous_dicom_id=None,
               bbox_name=None, findings=None, comparisons=None),
        _event("gold_comparisons", None, dicom_id=None, previous_dicom_id=None,
               bbox_name=None, finding=None, comparison=None),
        _event("chartevents", None, stay_id=None, hadm_id=None, itemid=None,
               value=None, valuenum=None, valueuom=None),
        _event("labevents", None, hadm_id=None, itemid=None, value=None,
               valuenum=None, valueuom=None),
        *(stays if stays is not None else [_stay()]),
        *flat,
    ]


def _frame(rows, patient_id="p1"):
    df = pl.from_dicts(rows, infer_schema_length=None)
    return df.with_columns(
        pl.lit(patient_id).alias("patient_id"),
        pl.col("timestamp").cast(pl.Datetime("ms")),
        *(pl.col(c).cast(pl.String) for c in df.columns if "/" in c),
    )


def _patient(*events, stays=None):
    return Patient(patient_id="p1", data_source=_frame(_rows(*events, stays=stays)))


def _labels(*diseases_and_statuses):
    return [f"{d}: {s}" for d, s in diseases_and_statuses]


class TestImagePairs(unittest.TestCase):
    def setUp(self):
        self.task = TASK()

    def _pairs(self, *events, stays=None):
        samples = self.task(_patient(_hr(1), *events, stays=stays))
        return [tuple(s["cxr"][0]) for s in samples]

    def test_paired_studies(self):
        samples = self.task(
            _patient(_hr(1), _image(10, "a"), _image(20, "b"), _silver("a", "b"))
        )
        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["cxr"], (["a.jpg", "b.jpg"], [10.0, 20.0]))
        self.assertEqual(sample["progression"], _labels((OPACITY, "worsened")))
        self.assertEqual(sample["patient_id"], "p1")
        self.assertEqual(sample["visit_id"], "10")
        self.assertEqual(sample["stay_id"], 1)
        self.assertEqual(sample["from_gold"], 0)

    def test_three_images_give_two_pairs(self):
        pairs = self._pairs(
            _image(10, "a"), _image(20, "b"), _image(30, "c"),
            _silver("a", "b"), _silver("b", "c"),
        )
        self.assertEqual(pairs, [("a.jpg", "b.jpg"), ("b.jpg", "c.jpg")])

    def test_unpaired_study_gives_no_sample(self):
        self.assertEqual(self._pairs(_image(10, "a"), _silver("x", "a")), [])

    def test_pair_without_comparison_gives_no_sample(self):
        self.assertEqual(self._pairs(_image(10, "a"), _image(20, "b")), [])

    def test_comparison_must_match_neighbouring_images(self):
        # a -> c skips b, so Chest ImaGenome's pair is not a neighbour pair.
        pairs = self._pairs(
            _image(10, "a"), _image(20, "b"), _image(30, "c"), _silver("a", "c"),
        )
        self.assertEqual(pairs, [])

    def test_reversed_comparison_is_not_used(self):
        self.assertEqual(
            self._pairs(_image(10, "a"), _image(20, "b"), _silver("b", "a")), []
        )

    def test_image_without_scene_graph_is_removed(self):
        pairs = self._pairs(
            _image(10, "a"), _image(20, "b", scene_graph=False), _image(30, "c"),
            _silver("a", "c"),
        )
        self.assertEqual(pairs, [("a.jpg", "c.jpg")])

    def test_other_views_are_removed(self):
        pairs = self._pairs(
            _image(10, "a"), _image(20, "b", view="PA"), _image(30, "c"),
            _silver("a", "c"),
        )
        self.assertEqual(pairs, [("a.jpg", "c.jpg")])

    def test_last_image_of_a_study_is_used(self):
        pairs = self._pairs(
            _image(10, "a", study_id="s1"), _image(10.5, "b", study_id="s1"),
            _image(20, "c"), _silver("b", "c"),
        )
        self.assertEqual(pairs, [("b.jpg", "c.jpg")])

    def test_images_outside_the_stay_are_removed(self):
        pairs = self._pairs(
            _image(-1, "before"), _image(10, "a"), _image(20, "b"),
            _image(125, "after"), _silver("before", "a"), _silver("a", "b"),
            _silver("b", "after"),
        )
        self.assertEqual(pairs, [("a.jpg", "b.jpg")])

    def test_stay_of_48_hours_or_less_is_excluded(self):
        events = (_image(10, "a"), _image(20, "b"), _silver("a", "b"))
        self.assertEqual(self._pairs(*events, stays=[_stay(hours=48)]), [])
        self.assertEqual(len(self._pairs(*events, stays=[_stay(hours=48.5)])), 1)

    def test_image_under_8_hours_after_previous_is_removed(self):
        pairs = self._pairs(
            _image(10, "a"), _image(15, "close"), _image(30, "b"),
            _silver("a", "b"),
        )
        self.assertEqual(pairs, [("a.jpg", "b.jpg")])

    def test_gaps_are_not_recomputed_after_a_removal(self):
        # 15 is removed, 19 is 4 hours after it, so 19 is removed too although
        # it is 9 hours after the image that stays, as in DiPro.
        pairs = self._pairs(
            _image(10, "a"), _image(15, "b"), _image(19, "c"), _silver("a", "c"),
        )
        self.assertEqual(pairs, [])

    def test_gap_of_exactly_8_hours_is_kept(self):
        pairs = self._pairs(_image(10, "a"), _image(18, "b"), _silver("a", "b"))
        self.assertEqual(pairs, [("a.jpg", "b.jpg")])

    def test_gap_of_96_hours_excludes_the_stay(self):
        stay = [_stay(hours=200)]
        pairs = self._pairs(
            _image(10, "a"), _image(20, "b"), _image(116, "c"),
            _silver("a", "b"), _silver("b", "c"), stays=stay,
        )
        self.assertEqual(pairs, [])

    def test_more_than_14_images_excludes_the_stay(self):
        def images(n):
            return [_image(5 + 8 * i, f"i{i}") for i in range(n)] + [
                _silver(f"i{i}", f"i{i + 1}") for i in range(n - 1)
            ]

        self.assertEqual(len(self._pairs(*images(14))), 13)
        self.assertEqual(self._pairs(*images(15), stays=[_stay(hours=125)]), [])

    def test_images_pair_within_a_stay_only(self):
        second = T0 + timedelta(hours=200)
        stays = [_stay(), _stay(stay_id=2, hadm_id=11, intime=second)]
        pairs = self._pairs(
            _image(10, "a"), _image(210, "b"), _silver("a", "b"), stays=stays,
        )
        self.assertEqual(pairs, [])

    def test_no_stay_or_no_images_gives_no_sample(self):
        self.assertEqual(self.task(_patient(_hr(1), stays=[])), [])
        self.assertEqual(self.task(_patient(_hr(1))), [])

    def test_no_measurement_gives_no_sample(self):
        samples = self.task(_patient(_image(10, "a"), _image(20, "b"), _silver("a", "b")))
        self.assertEqual(samples, [])


class TestLabels(unittest.TestCase):
    def _sample(self, *events):
        samples = TASK()(_patient(_hr(1), _image(10, "a"), _image(20, "b"), *events))
        self.assertEqual(len(samples), 1)
        return samples[0]

    def _region(self, sample, region, disease):
        row = TASK.REGIONS.index(region)
        return sample["region_progression"][row][TASK.DISEASES.index(disease)]

    def test_every_status(self):
        for status in TASK.STATUSES:
            sample = self._sample(_silver("a", "b", status=status))
            self.assertEqual(sample["progression"], _labels((OPACITY, status)))

    def test_findings_outside_the_seven_diseases_give_no_sample(self):
        self.assertEqual(
            TASK()(
                _patient(_hr(1), _image(10, "a"), _image(20, "b"),
                         _silver("a", "b", findings="scoliosis"))
            ),
            [],
        )

    def test_edema_is_named_without_hazy_opacity(self):
        sample = self._sample(_silver("a", "b", findings=EDEMA))
        self.assertEqual(sample["progression"], _labels(("pulmonary edema", "worsened")))
        self.assertEqual(self._region(sample, LUNG, "pulmonary edema"), 0)

    def test_region_with_several_comparisons_is_ambiguous(self):
        sample = self._sample(
            _silver("a", "b", status="worsened;improved"),
            _silver("a", "b", findings="atelectasis", region="right lung"),
        )
        self.assertEqual(sample["progression"], _labels(("atelectasis", "worsened")))

    def test_regions_that_agree(self):
        sample = self._sample(
            _silver("a", "b"), _silver("a", "b", region="right lung")
        )
        self.assertEqual(sample["progression"], _labels((OPACITY, "worsened")))

    def test_regions_that_disagree_drop_only_that_disease(self):
        sample = self._sample(
            _silver("a", "b"),
            _silver("a", "b", status="improved", region="right lung"),
            _silver("a", "b", findings="atelectasis", region="right lung"),
        )
        self.assertEqual(sample["progression"], _labels(("atelectasis", "worsened")))
        self.assertEqual(self._region(sample, LUNG, OPACITY), 0)
        self.assertEqual(self._region(sample, "right lung", OPACITY), 1)

    def test_region_with_several_findings(self):
        sample = self._sample(_silver("a", "b", findings=f"{OPACITY};atelectasis"))
        self.assertCountEqual(
            sample["progression"],
            _labels((OPACITY, "worsened"), ("atelectasis", "worsened")),
        )

    def test_negated_findings_count_as_in_dipro(self):
        row = _silver("a", "b")
        row["silver_comparisons/finding_polarities"] = "no"
        sample = self._sample(row)
        self.assertEqual(sample["progression"], _labels((OPACITY, "worsened")))

    def test_gold_replaces_silver(self):
        sample = self._sample(_silver("a", "b"), _gold("a", "b"))
        self.assertEqual(sample["progression"], _labels((OPACITY, "improved")))
        self.assertEqual(sample["from_gold"], 1)

    def test_gold_labels_each_disease_separately(self):
        # DiPro's _get_golden_labels would give both diseases the first row's.
        sample = self._sample(
            _silver("a", "b", findings=f"{OPACITY};atelectasis"),
            _gold("a", "b", finding=OPACITY, status="improved"),
            _gold("a", "b", finding="atelectasis", status="no change"),
        )
        self.assertCountEqual(
            sample["progression"],
            _labels((OPACITY, "improved"), ("atelectasis", "no change")),
        )

    def test_silver_disease_absent_from_gold_is_dropped(self):
        sample = self._sample(
            _silver("a", "b", findings=f"{OPACITY};atelectasis"), _gold("a", "b"),
        )
        self.assertEqual(sample["progression"], _labels((OPACITY, "improved")))

    def test_gold_disease_absent_from_silver_is_kept(self):
        sample = self._sample(_gold("a", "b", finding="atelectasis"))
        self.assertEqual(sample["progression"], _labels(("atelectasis", "improved")))

    def test_gold_regions_that_disagree_drop_that_disease(self):
        sample = self._sample(
            _gold("a", "b", finding="atelectasis", status="worsened"),
            _gold("a", "b"),
            _gold("a", "b", status="worsened", region="right lung"),
        )
        self.assertEqual(sample["progression"], _labels(("atelectasis", "worsened")))

    def test_region_labels_gold_overwrites_silver_where_it_has_a_row(self):
        sample = self._sample(
            _silver("a", "b"),
            _silver("a", "b", region="right lung"),
            _gold("a", "b"),
        )
        self.assertEqual(self._region(sample, LUNG, OPACITY), 1)
        self.assertEqual(self._region(sample, "right lung", OPACITY), 0)

    def test_conflicting_gold_rows_leave_the_region_not_mentioned(self):
        # silver says worsened; gold has two rows for the same region and
        # disease that disagree, so the region is not mentioned, not silver's
        # (another gold disease keeps the pair, which would otherwise have
        # no label and be dropped)
        sample = self._sample(
            _silver("a", "b"),
            _gold("a", "b", status="improved"),
            _gold("a", "b", status="no change"),
            _gold("a", "b", finding="atelectasis", region="right lung"),
        )
        self.assertEqual(self._region(sample, LUNG, OPACITY), NOT_MENTIONED)
        self.assertEqual(sample["progression"], _labels(("atelectasis", "improved")))

    def test_regions_outside_the_ten_are_not_in_region_labels(self):
        sample = self._sample(_silver("a", "b", region="trachea"))
        self.assertEqual(sample["progression"], _labels((OPACITY, "worsened")))
        self.assertEqual(
            {value for row in sample["region_progression"] for value in row},
            {NOT_MENTIONED},
        )

    def test_region_labels_shape(self):
        sample = self._sample(_silver("a", "b"))
        grid = np.array(sample["region_progression"])
        self.assertEqual(grid.shape, (10, 7))


class TestEHRWindow(unittest.TestCase):
    def _sample(self, *events, task=None, first=10.5, second=20.25):
        samples = (task or TASK())(
            _patient(
                *events, _image(first, "a"), _image(second, "b"), _silver("a", "b")
            )
        )
        self.assertEqual(len(samples), 1)
        return samples[0]

    def test_bins_run_from_the_first_to_the_second_image(self):
        sample = self._sample(_hr(1))
        self.assertEqual(sample["first_ehr_hour"], 10)
        self.assertEqual(sample["ehr"].shape, (11, 2 * N_FEATURES))

    def test_values_before_the_first_image_are_carried_forward(self):
        ehr = self._sample(_hr(1, value=70))["ehr"]
        np.testing.assert_array_equal(ehr[:, HR], 70)
        self.assertEqual(ehr[:, N_FEATURES + HR].sum(), 0)

    def test_measurements_in_the_window(self):
        ehr = self._sample(_hr(1), _hr(15.5, value=90))["ehr"]
        self.assertEqual(ehr[4, HR], 80)
        self.assertEqual(ehr[5, HR], 90)
        self.assertEqual(ehr[5, N_FEATURES + HR], 1)

    def test_nothing_after_the_second_image_is_used(self):
        ehr = self._sample(_hr(1), _hr(20.1, value=90), _hr(20.5, value=100))["ehr"]
        self.assertEqual(ehr[-1, HR], 90)
        ehr = self._sample(_hr(1), _hr(21, value=100))["ehr"]
        self.assertEqual(ehr[-1, HR], 80)

    def test_image_on_the_hour_starts_one_bin_earlier_as_in_dipro(self):
        self.assertEqual(self._sample(_hr(1), first=10.0)["first_ehr_hour"], 9)

    def test_first_bin_is_never_negative(self):
        self.assertEqual(self._sample(_hr(0.5), first=0.0)["first_ehr_hour"], 0)

    def test_option_adds_dipro_bin_after_second_image(self):
        task = TASK(include_bin_after_cxr2=True)
        sample = self._sample(_hr(1), _hr(21.5, value=100), task=task)
        self.assertEqual(sample["ehr"].shape[0], 12)
        self.assertEqual(sample["ehr"][-1, HR], 100)

    def test_option_stops_at_the_end_of_the_stay(self):
        task = TASK(include_bin_after_cxr2=True)
        samples = task(
            _patient(
                _hr(1), _hr(50.2, value=100), _image(40, "a"), _image(49.9, "b"),
                _silver("a", "b"), stays=[_stay(hours=50)],
            )
        )
        self.assertEqual(samples[0]["ehr"][-1, HR], 80)


class TestPreFilter(unittest.TestCase):
    def _frame(self, los_hours):
        frames = []
        for index, hours in enumerate(los_hours):
            rows = _rows(stays=[_stay(hours=hours, stay_id=index, hadm_id=index)])
            frames.append(_frame(rows, patient_id=f"p{index}"))
        return pl.concat(frames, how="diagonal").lazy()

    def _kept(self, frame, task=None):
        return sorted(
            (task or TASK())
            .pre_filter(frame)
            .filter(pl.col("event_type") == "icustays")
            .collect()["icustays/stay_id"]
            .to_list()
        )

    def test_iqr_limits_remove_stays_and_ignore_short_ones(self):
        # 49..54 has Q1 50.25 and Q3 52.75, so the limits are 46.5 and 56.5.
        # The 10-hour stay is under 48 hours, so it is not in the quartiles.
        kept = self._kept(self._frame([49, 50, 51, 52, 53, 54, 500, 10]))
        self.assertNotIn("6", kept)
        self.assertEqual(kept, ["0", "1", "2", "3", "4", "5", "7"])

    def test_option_keeps_outliers(self):
        kept = self._kept(
            self._frame([49, 50, 51, 500]), TASK(exclude_los_outliers=False)
        )
        self.assertEqual(kept, ["0", "1", "2", "3"])

    def test_missing_table_is_reported(self):
        frame = _frame(_rows()).drop("gold_comparisons/dicom_id").lazy()
        frame = frame.drop(
            [c for c in frame.collect_schema().names() if c.startswith("gold_")]
        )
        with self.assertRaisesRegex(ValueError, "gold_comparisons"):
            TASK().pre_filter(frame)

    def test_measurement_rows_with_unused_items_are_dropped(self):
        frame = _frame(
            _rows(
                _hr(1),
                _event("chartevents", T0, stay_id=1, hadm_id=10, itemid=999,
                       value=1, valuenum=1, valueuom=None),
            )
        ).lazy()
        rows = TASK(exclude_los_outliers=False).pre_filter(frame).collect()
        self.assertEqual(rows["chartevents/itemid"].drop_nulls().to_list(), ["220045"])


class TestGoldTestSplit(unittest.TestCase):
    def _parts(self, split):
        subjects = {f"p{i:02d}": [i] for i in range(20)}
        return [part.tolist() for part in split.split_indices(subjects)]

    def test_gold_subjects_are_in_the_test_part(self):
        train, val, test = self._parts(GoldTestSplit(["p03", "p07"]))
        self.assertIn(3, test)
        self.assertIn(7, test)
        self.assertEqual(sorted(train + val + test), list(range(20)))
        self.assertEqual(len(test), 4)

    def test_same_as_dipro_arithmetic(self):
        rest = [f"p{i:02d}" for i in range(20) if i not in (3, 7)]
        train_val, test = train_test_split(rest, test_size=2 / 20, random_state=0)
        train, val = train_test_split(train_val, test_size=1 / 8, random_state=0)
        parts = self._parts(GoldTestSplit(["p03", "p07"]))
        self.assertEqual(parts[0], sorted(int(s[1:]) for s in train))
        self.assertEqual(parts[1], sorted(int(s[1:]) for s in val))
        self.assertEqual(parts[2], sorted([3, 7, *(int(s[1:]) for s in test)]))

    def test_more_gold_than_the_test_share(self):
        gold = [f"p{i:02d}" for i in range(6)]
        self.assertEqual(self._parts(GoldTestSplit(gold))[2], list(range(6)))

    def test_gold_subject_without_samples_is_ignored(self):
        train, val, test = self._parts(GoldTestSplit(["other"]))
        self.assertEqual(sorted(train + val + test), list(range(20)))

    def test_seed_changes_the_parts_and_the_key(self):
        one, two = GoldTestSplit(["p03"], seed=0), GoldTestSplit(["p03"], seed=1)
        self.assertNotEqual(self._parts(one), self._parts(two))
        self.assertNotEqual(one.to_dict(), two.to_dict())

    def test_key_ignores_order_of_gold_subjects(self):
        self.assertEqual(
            GoldTestSplit(["b", "a"]).to_dict(), GoldTestSplit(["a", "b"]).to_dict()
        )

    def test_tiny_cohorts(self):
        parts = GoldTestSplit(["a"]).split_indices({"a": [0], "b": [1]})
        self.assertEqual([part.tolist() for part in parts], [[1], [], [0]])


class TestProcessors(unittest.TestCase):
    def test_labels_keep_their_size_and_order(self):
        processor = MultiLabelProcessor(labels=TASK.LABELS)
        processor.fit([{"y": []}], "y")
        vector = processor.process(_labels((OPACITY, "improved")))
        self.assertEqual(vector.shape, (21,))
        disease, status = TASK.DISEASES.index(OPACITY), TASK.STATUSES.index("improved")
        self.assertEqual(vector.reshape(7, 3)[disease, status], 1)
        self.assertEqual(vector.sum(), 1)

    def test_samples_batch_through_the_schema(self):
        task = TASK(image_size=8)
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("a", "b", "c"):
                Image.new("L", (16, 16)).save(Path(tmp) / f"{name}.jpg")
            samples = []
            pairs = (("a", "b", 10.5, 20.25), ("b", "c", 20.25, 41.5))
            for index, (first, second, start, end) in enumerate(pairs):
                events = [
                    _hr(1), _image(start, first), _image(end, second),
                    _silver(first, second),
                ]
                sample = task(_patient(*events))[0]
                sample["patient_id"] = f"p{index}"
                sample["cxr"] = (
                    [str(Path(tmp) / path) for path in sample["cxr"][0]],
                    sample["cxr"][1],
                )
                samples.append(sample)
            dataset = create_sample_dataset(
                samples, task.input_schema, task.output_schema, in_memory=True
            )
            batch = next(iter(get_dataloader(dataset, batch_size=2)))
        self.assertEqual(batch["ehr"].shape, (2, 22, 34))
        self.assertEqual(batch["progression"].shape, (2, 21))
        images, hours, _ = batch["cxr"][0]
        self.assertEqual(images.shape, (2, 3, 8, 8))
        self.assertEqual(hours.tolist(), [10.5, 20.25])
        self.assertFalse(torch.isnan(batch["ehr"]).any())


class TestModelOptions(unittest.TestCase):
    PAIR = (_hr(1), _image(10, "a"), _image(20, "b"), _silver("a", "b"))

    @staticmethod
    def _box(dicom_id, region=LUNG, box=(1, 2, 30, 40)):
        return _event(
            "silver_objects", T0, dicom_id=dicom_id, bbox_name=region,
            original_x1=box[0], original_y1=box[1],
            original_x2=box[2], original_y2=box[3],
        )

    @staticmethod
    def _admission(race="WHITE - RUSSIAN", language="ENGLISH", marital="MARRIED"):
        return _event(
            "admissions", T0, hadm_id=10, race=race, language=language,
            marital_status=marital,
        )

    @staticmethod
    def _chart(hours, itemid, value):
        return _event(
            "chartevents", T0 + timedelta(hours=hours), stay_id=1, hadm_id=10,
            itemid=itemid, value=value, valuenum=value, valueuom="kg",
        )

    def test_defaults_are_unchanged(self):
        task = TASK()
        self.assertEqual(sorted(task.input_schema), ["cxr", "ehr"])
        self.assertEqual(task.input_schema["cxr"][0], "time_image")
        sample = task(_patient(*self.PAIR))[0]
        self.assertEqual(len(sample["cxr"]), 2)
        self.assertNotIn("demographics", sample)

    def test_region_boxes_in_region_order(self):
        task = TASK(region_crops=True)
        events = (*self.PAIR, self._box("a"), self._box("b", "mediastinum"))
        _, _, boxes = task(_patient(*events))[0]["cxr"]
        self.assertEqual(np.array(boxes).shape, (2, 10, 4))
        self.assertEqual(boxes[0][0], [1.0, 2.0, 30.0, 40.0])
        self.assertEqual(boxes[0][1], [0.0] * 4)
        self.assertEqual(boxes[1][0], [0.0] * 4)
        self.assertEqual(boxes[1][TASK.REGIONS.index("mediastinum")][3], 40.0)

    def test_image_without_boxes(self):
        task = TASK(region_crops=True)
        _, _, boxes = task(_patient(*self.PAIR, self._box("a")))[0]["cxr"]
        self.assertEqual(boxes[1], [[0.0] * 4] * 10)

    def test_other_boxes_are_ignored(self):
        task = TASK(region_crops=True)
        events = (*self.PAIR, self._box("a", "aortic arch"))
        _, _, boxes = task(_patient(*events))[0]["cxr"]
        self.assertEqual(boxes[0], [[0.0] * 4] * 10)

    def test_pre_filter_requires_boxes_only_with_crops(self):
        frame = _frame(_rows(*self.PAIR)).lazy()
        TASK(exclude_los_outliers=False).pre_filter(frame)
        with self.assertRaisesRegex(ValueError, "silver_objects"):
            TASK(region_crops=True, exclude_los_outliers=False).pre_filter(frame)

    def _demographics(self, *events):
        task = TASK(include_demographics=True)
        sample = task(_patient(*self.PAIR, *events))[0]["demographics"]
        self.assertEqual(len(sample), len(TASK.DEMOGRAPHICS))
        return dict(zip(TASK.DEMOGRAPHICS, sample))

    def test_demographics_names_match_dipro(self):
        self.assertEqual(len(TASK.DEMOGRAPHICS), 17)
        self.assertEqual(
            TASK.DEMOGRAPHICS[:6],
            ["age", "height", "weight", "gender_F", "gender_M",
             "race_group_ASIAN"],
        )
        self.assertEqual(TASK.DEMOGRAPHICS[-1], "marital_status_WIDOWED")

    def test_demographics_values(self):
        values = self._demographics(self._admission(), self._chart(2, 226512, 70))
        self.assertEqual(values["age"], 60.0)
        self.assertEqual(values["weight"], 70.0)
        self.assertTrue(np.isnan(values["height"]))
        self.assertEqual(values["gender_M"], 1.0)
        self.assertEqual(values["gender_F"], 0.0)
        self.assertEqual(values["race_group_WHITE"], 1.0)
        self.assertEqual(values["language_ENGLISH"], 1.0)
        self.assertEqual(values["marital_status_MARRIED"], 1.0)

    def test_weight_after_the_second_image_stays_out(self):
        values = self._demographics(self._admission(), self._chart(25, 226512, 70))
        self.assertTrue(np.isnan(values["weight"]))

    def test_first_weight_wins(self):
        values = self._demographics(
            self._admission(), self._chart(5, 226512, 70), self._chart(2, 226512, 80)
        )
        self.assertEqual(values["weight"], 80.0)

    def test_unknown_categories_are_all_zero(self):
        values = self._demographics(self._admission("NEW", "SPANISH", None))
        categorical = {
            k: v for k, v in values.items()
            if k.startswith(("race", "language", "marital"))
        }
        self.assertEqual(set(categorical.values()), {0.0})

    def test_no_admission_row(self):
        self.assertEqual(self._demographics()["race_group_WHITE"], 0.0)


class TestComposite(unittest.TestCase):
    def _dataset(self, **kwargs):
        with (
            mock.patch("pyhealth.datasets.mimic4_imagenome.MIMIC4Dataset") as mimic4,
            mock.patch(
                "pyhealth.datasets.mimic4_imagenome.ChestImaGenomeDataset"
            ) as imagenome,
            tempfile.TemporaryDirectory() as tmp,
        ):
            dataset = MIMIC4ImaGenomeDataset(cache_dir=tmp, **kwargs)
        return dataset, mimic4, imagenome

    def test_both_datasets_are_sub_datasets(self):
        dataset, mimic4, imagenome = self._dataset(
            imagenome_root="genome",
            imagenome_tables=["silver_images"],
            ehr_root="ehr",
            ehr_tables=["chartevents"],
        )
        self.assertEqual(sorted(dataset.sub_datasets), ["imagenome", "mimic4"])
        self.assertEqual(mimic4.call_args.kwargs["ehr_root"], "ehr")
        self.assertEqual(mimic4.call_args.kwargs["ehr_tables"], ["chartevents"])
        self.assertEqual(imagenome.call_args.kwargs["root"], "genome")
        self.assertEqual(imagenome.call_args.kwargs["tables"], ["silver_images"])
        self.assertEqual(sorted(dataset.tables), ["chartevents", "silver_images"])
        self.assertEqual(dataset.root, "ehr|None|None|None|genome")
        for sub in (mimic4, imagenome):
            self.assertEqual(sub.call_args.kwargs["cache_dir"], str(dataset.cache_dir))

    def test_default_imagenome_tables_are_in_tables_and_passed_on(self):
        # The cache key (tables) and the sub-dataset must agree on the
        # default Chest ImaGenome tables.
        dataset, _, imagenome = self._dataset(imagenome_root="genome")
        defaults = ["silver_images", "silver_objects", "silver_comparisons",
                    "gold_comparisons"]
        self.assertEqual(imagenome.call_args.kwargs["tables"], defaults)
        self.assertEqual(sorted(dataset.tables), sorted(defaults))

    def test_load_data_combines_both(self):
        dataset, mimic4, imagenome = self._dataset(imagenome_root="genome")
        frames = [
            pd.DataFrame({"patient_id": ["p1"], "event_type": ["metadata"]}),
            pd.DataFrame({"patient_id": ["p1"], "event_type": ["silver_images"]}),
        ]
        mimic4.return_value.load_data.return_value = dd.from_pandas(frames[0])
        imagenome.return_value.load_data.return_value = dd.from_pandas(frames[1])
        combined = dataset.load_data().compute()
        self.assertEqual(
            sorted(combined["event_type"]), ["metadata", "silver_images"]
        )

    def test_gold_subjects(self):
        rows = [
            _event("gold_comparisons", T0, dicom_id="b"),
            _event("gold_comparisons", T0, dicom_id="c"),
            _event("metadata", T0, dicom_id="z"),
        ]
        frames = [_frame(rows[:2], "p2"), _frame(rows[2:], "p3"), _frame(rows[:1], "p1")]
        dataset = mock.Mock(global_event_df=pl.concat(frames, how="diagonal").lazy())
        self.assertEqual(TASK.gold_subjects(dataset), ["p1", "p2"])


if __name__ == "__main__":
    unittest.main()
