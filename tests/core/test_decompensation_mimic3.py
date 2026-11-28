import math
import unittest
from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from pyhealth.tasks.decompensation_mimic3 import DecompensationMIMIC3

T0 = datetime(2020, 1, 1, 0, 0, 0)


class FakePatient:
    def __init__(self, events):
        self.patient_id = "p1"
        self.events = events

    def get_events(self, event_type):
        return self.events.get(event_type, [])


def _patient(death_hours=29.5, stay_hours=30.0):
    stay = SimpleNamespace(
        hadm_id="100",
        icustay_id="200",
        timestamp=T0,
        outtime=T0 + timedelta(hours=stay_hours),
        first_wardid="1",
        last_wardid="1",
        first_careunit="MICU",
        last_careunit="MICU",
    )
    admission = SimpleNamespace(
        hadm_id="100",
        timestamp=T0 - timedelta(hours=2),
        dischtime=T0 + timedelta(hours=40),
        deathtime=(
            None if death_hours is None else T0 + timedelta(hours=death_hours)
        ),
    )
    patient_row = SimpleNamespace(
        dob=datetime(1980, 1, 1),
        dod=None,
    )
    chart = [
        SimpleNamespace(
            icustay_id="200",
            hadm_id="100",
            timestamp=T0 + timedelta(minutes=5),
            itemid=220045,
            value="80",
            valueuom="bpm",
            label="Heart Rate",
        )
    ]
    return FakePatient(
        {
            "icustays": [stay],
            "admissions": [admission],
            "patients": [patient_row],
            "chartevents": chart,
            "labevents": [],
        }
    )


class TestDecompensationMIMIC3(unittest.TestCase):
    def test_dynamic_shapes_and_normalization(self):
        task = DecompensationMIMIC3()
        hourly = {
            0: {
                "Diastolic blood pressure": 100.0,
                "Heart Rate": 80.0,
            }
        }
        continuous, categorical, mask = task._build_dynamic(hourly, 2)

        self.assertEqual(continuous["values"].shape, (2, 12))
        self.assertEqual(categorical["values"].shape, (2, 47))
        self.assertEqual(mask["values"].shape, (2, 17))

        dbp = [
            c
            for c in task.CHANNELS
            if c not in task.CATEGORICAL_VALUES
        ].index("Diastolic blood pressure")
        expected = (
            100.0 - task.CONTINUOUS_MEANS[dbp]
        ) / task.CONTINUOUS_STDS[dbp]

        self.assertAlmostEqual(
            continuous["values"][0, dbp],
            expected,
        )
        self.assertAlmostEqual(
            continuous["values"][1, dbp],
            expected,
        )

        dbp_mask = task.CHANNELS.index("Diastolic blood pressure")
        self.assertEqual(mask["values"][0, dbp_mask], 1.0)
        self.assertEqual(mask["values"][1, dbp_mask], 0.0)

    def test_normal_value_imputation(self):
        task = DecompensationMIMIC3()
        continuous, categorical, mask = task._build_dynamic({}, 1)

        continuous_channels = [
            c
            for c in task.CHANNELS
            if c not in task.CATEGORICAL_VALUES
        ]

        for i, channel in enumerate(continuous_channels):
            expected = (
                float(task.NORMAL_VALUES[channel])
                - task.CONTINUOUS_MEANS[i]
            ) / task.CONTINUOUS_STDS[i]
            self.assertAlmostEqual(
                continuous["values"][0, i],
                expected,
            )

        self.assertTrue(np.all(mask["values"] == 0.0))
        self.assertEqual(
            categorical["values"].sum(),
            len(task.CATEGORICAL_VALUES),
        )

    def test_decompensation_labels_and_hazard(self):
        task = DecompensationMIMIC3()
        samples = task(_patient())

        self.assertEqual(len(samples), 25)
        self.assertEqual(samples[0]["prediction_time"], 5.0)
        self.assertEqual(samples[-1]["prediction_time"], 29.0)

        by_time = {
            int(sample["prediction_time"]): sample
            for sample in samples
        }

        self.assertEqual(by_time[5]["y_true"], 0)
        self.assertEqual(sum(by_time[5]["hazard_label"]), 0.0)

        self.assertEqual(by_time[6]["y_true"], 1)
        self.assertEqual(
            by_time[6]["hazard_label"][23],
            1.0,
        )
        self.assertEqual(
            sum(by_time[6]["hazard_mask"]),
            24.0,
        )

        self.assertEqual(by_time[29]["y_true"], 1)
        self.assertEqual(
            by_time[29]["hazard_label"][0],
            1.0,
        )
        self.assertEqual(
            sum(by_time[29]["hazard_mask"]),
            1.0,
        )

    def test_survtls_invalid_lengthscale(self):
        with self.assertRaises(ValueError):
            DecompensationMIMIC3(survtls_lengthscale=0.0)

        with self.assertRaises(ValueError):
            DecompensationMIMIC3(survtls_lengthscale=-1.0)

    def test_survtls_smoothed_label(self):
        task = DecompensationMIMIC3(survtls_lengthscale=8.0)

        event_bin = 12
        label = task._smooth_survival_label(event_bin)

        self.assertEqual(len(label), 24)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in label))
        self.assertAlmostEqual(sum(label), 1.0, places=6)

        mean = float(event_bin)
        std = mean / 8.0

        def cdf(x):
            z = (x - mean) / (std * math.sqrt(2.0))
            return 0.5 * (1.0 + math.erf(z))

        expected = [
            cdf(k + 0.5) - cdf(k - 0.5)
            for k in range(1, 25)
        ]
        expected[0] = cdf(1.5)
        expected[-1] = 1.0 - cdf(23.5)

        np.testing.assert_allclose(
            label,
            expected,
            rtol=1e-7,
            atol=1e-7,
        )

    def test_survtls_task_generation(self):
        task = DecompensationMIMIC3(survtls_lengthscale=8.0)
        samples = task(_patient())

        by_time = {
            int(sample["prediction_time"]): sample
            for sample in samples
        }

        hard_task = DecompensationMIMIC3()
        hard_samples = hard_task(_patient())
        hard_by_time = {
            int(sample["prediction_time"]): sample
            for sample in hard_samples
        }

        # At t=6, death is in the 24th future hourly bin.
        smooth = by_time[6]["hazard_label"]
        hard = hard_by_time[6]["hazard_label"]

        self.assertAlmostEqual(sum(smooth), 1.0, places=6)
        self.assertGreater(sum(value > 0.0 for value in smooth), 1)
        self.assertEqual(hard[23], 1.0)
        self.assertEqual(sum(hard), 1.0)

        # survTLS changes training labels, not the binary EEP target.
        self.assertEqual(by_time[6]["y_true"], hard_by_time[6]["y_true"])
        self.assertEqual(
            by_time[6]["hazard_mask"],
            hard_by_time[6]["hazard_mask"],
        )

    def test_survivor_has_no_positive_labels(self):
        task = DecompensationMIMIC3()
        samples = task(_patient(death_hours=None))

        self.assertGreater(len(samples), 0)
        self.assertTrue(
            all(sample["y_true"] == 0 for sample in samples)
        )
        self.assertTrue(
            all(
                sum(sample["hazard_label"]) == 0.0
                for sample in samples
            )
        )


    def test_same_ward_ids_retains_stay(self):
        task = DecompensationMIMIC3()
        patient = _patient()
        self.assertEqual(len(task(patient)), 25)

    def test_different_ward_ids_excludes_stay(self):
        task = DecompensationMIMIC3()
        patient = _patient()
        stay = patient.get_events("icustays")[0]
        stay.last_wardid = "2"

        self.assertEqual(task(patient), [])

    def test_missing_both_ward_ids_retains_stay(self):
        task = DecompensationMIMIC3()
        patient = _patient()
        stay = patient.get_events("icustays")[0]

        del stay.first_wardid
        del stay.last_wardid

        self.assertEqual(len(task(patient)), 25)

    def test_missing_one_ward_id_excludes_stay(self):
        task = DecompensationMIMIC3()

        for field in ("first_wardid", "last_wardid"):
            with self.subTest(field=field):
                patient = _patient()
                stay = patient.get_events("icustays")[0]
                delattr(stay, field)

                self.assertEqual(task(patient), [])

    def test_different_careunits_excludes_stay(self):
        task = DecompensationMIMIC3()
        patient = _patient()
        stay = patient.get_events("icustays")[0]
        stay.last_careunit = "SICU"

        self.assertEqual(task(patient), [])

if __name__ == "__main__":
    unittest.main()
