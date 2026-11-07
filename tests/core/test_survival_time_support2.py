"""SUPPORT2 task tests use at most four fabricated patients and no downloads."""

from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from pyhealth.data import Event
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.tasks import BaseTask, SurvivalTimeSupport2


def _patient(**changes):
    row = {
        "age": 50.0,
        "sex": "female",
        "num.co": 2.0,
        "race": None,
        "diabetes": 1.0,
        "dementia": 0.0,
        "ca": "no",
        "meanbp": 80.0,
        "hrt": 70.0,
        "resp": 16.0,
        "temp": 37.0,
        "sod": 140.0,
        "wblc": 6.0,
        "crea": 1.0,
        "d.time": 12.0,
        "death": 0,
    }
    row.update(changes)
    event = Event("support2", **row)
    return SimpleNamespace(
        patient_id="fabricated", get_events=Mock(return_value=[event])
    )


class TestSurvivalTimeSupport2(unittest.TestCase):
    def test_schema_and_distinct_encoding_cache_names(self):
        onehot, benchmark = SurvivalTimeSupport2(), SurvivalTimeSupport2("benchmark14")
        self.assertIsInstance(onehot, BaseTask)
        self.assertEqual(onehot.input_schema, {"features": "tensor"})
        self.assertEqual(
            onehot.output_schema, {"duration": "tensor", "event": "tensor"}
        )
        self.assertNotEqual(onehot.task_name, benchmark.task_name)
        with self.assertRaises(ValueError):
            SurvivalTimeSupport2("other")

    def test_known_benchmark_order_and_category_codes(self):
        patient = _patient()
        sample = SurvivalTimeSupport2("benchmark14")(patient)[0]
        self.assertEqual(
            sample["features"],
            [
                50.0,
                1.0,
                2.0,
                5.0,
                1.0,
                0.0,
                1.0,
                80.0,
                70.0,
                16.0,
                37.0,
                140.0,
                6.0,
                1.0,
            ],
        )
        self.assertEqual(sample["duration"], 12.0)
        self.assertEqual(sample["event"], 0.0)
        patient.get_events.assert_called_once_with(event_type="support2")

    def test_known_onehot_layout_and_missing_race(self):
        task = SurvivalTimeSupport2()
        sample = task(_patient())[0]
        self.assertEqual(len(sample["features"]), 22)
        named = dict(zip(task.feature_names, sample["features"]))
        self.assertEqual(named["sex=female"], 1.0)
        self.assertEqual(named["sex=male"], 0.0)
        self.assertEqual(named["race=<missing>"], 1.0)
        self.assertEqual(
            sum(named[f"race={name}"] for name in task.categories["race"]), 1.0
        )
        self.assertEqual(named["ca=no"], 1.0)
        for missing in (None, float("nan")):
            self.assertEqual(
                SurvivalTimeSupport2("benchmark14")(_patient(race=missing))[0][
                    "features"
                ][3],
                5.0,
            )

    def test_observed_and_censored_labels_are_preserved(self):
        for event in (0, 1):
            sample = SurvivalTimeSupport2()(_patient(death=event))[0]
            self.assertEqual(sample["event"], event)
            self.assertEqual(sample["duration"], 12.0)

    def test_invalid_followup_or_missing_numeric_excludes_record(self):
        for duration in (None, 0, -1, float("nan"), float("inf")):
            self.assertEqual(
                SurvivalTimeSupport2()(_patient(**{"d.time": duration})), []
            )
        for missing in (None, float("nan"), float("inf"), "not-numeric"):
            self.assertEqual(SurvivalTimeSupport2()(_patient(age=missing)), [])

    def test_bad_event_and_unknown_categories_raise(self):
        for event in (None, -1, 2, float("nan"), "unknown"):
            with self.subTest(event=event), self.assertRaises(ValueError):
                SurvivalTimeSupport2()(_patient(death=event))
        for field, value in (
            ("sex", "unknown"),
            ("race", "unknown"),
            ("ca", "unknown"),
        ):
            with self.subTest(field=field), self.assertRaises(ValueError):
                SurvivalTimeSupport2()(_patient(**{field: value}))

    def test_single_row_contract(self):
        for rows in ([], [Event("support2"), Event("support2")]):
            patient = SimpleNamespace(
                patient_id="p", get_events=Mock(return_value=rows)
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                SurvivalTimeSupport2()(patient)

    def test_no_outcome_or_prognosis_field_enters_features(self):
        task = SurvivalTimeSupport2()
        original = task(_patient())[0]
        changed = task(
            _patient(
                death=1, surv2m=0.99, surv6m=0.01, hospdead=1, **{"d.time": 1000.0}
            )
        )[0]
        self.assertEqual(original["features"], changed["features"])
        self.assertNotEqual(original["duration"], changed["duration"])

    def test_four_patient_pyhealth_numeric_tensor_path(self):
        task = SurvivalTimeSupport2()
        samples = []
        for i in range(4):
            sample = task(_patient(age=40.0 + i, death=i % 2, **{"d.time": 10.0 + i}))[
                0
            ]
            sample["patient_id"] = f"p{i}"
            samples.append(sample)
        dataset = create_sample_dataset(samples, task.input_schema, task.output_schema)
        batch = next(iter(get_dataloader(dataset, batch_size=4, shuffle=False)))
        self.assertEqual(batch["features"].shape, (4, 22))
        torch.testing.assert_close(
            batch["duration"].flatten(), torch.arange(10.0, 14.0)
        )
        np.testing.assert_array_equal(batch["event"].numpy().flatten(), [0, 1, 0, 1])
        self.assertTrue(bool(torch.isfinite(batch["features"]).all()))


if __name__ == "__main__":
    unittest.main()
