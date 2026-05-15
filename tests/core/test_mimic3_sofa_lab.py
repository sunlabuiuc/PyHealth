from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from typing import List, Sequence, Tuple

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks import SofaLabForecastingMIMIC3

# MIMIC-III item IDs from the paper's Appendix A, Table 4.
BILI = "50885"
CREAT = "50912"
PLT = "51265"
T0 = datetime(2026, 1, 1, 0, 0, 0)

Stay = Tuple[datetime, datetime, Sequence[Tuple[datetime, str, float]]]

def normalized(itemid: str, value: float) -> float:
    mean, std = SofaLabForecastingMIMIC3.LAB_NORMALIZATION_STATS[itemid]
    return (value - mean) / std


def _lab_row(ts: datetime, itemid: str, valuenum: float) -> dict:
    return {
        "event_type": "labevents",
        "timestamp": ts,
        "labevents/itemid": str(itemid),
        "labevents/valuenum": float(valuenum),
    }


def _icu_row(intime: datetime, outtime: datetime, icustay_id: str) -> dict:
    return {
        "event_type": "icustays",
        "timestamp": intime,
        "icustays/icustay_id": icustay_id,
        "icustays/outtime": outtime.strftime("%Y-%m-%d %H:%M:%S"),
    }


def make_patient(patient_id: str, stays: Sequence[Stay]) -> Patient:
    """Build a synthetic Patient from one or more ICU stays."""
    rows: List[dict] = []
    for idx, (intime, outtime, lab_events) in enumerate(stays):
        rows.append(_icu_row(intime, outtime, f"{100000 + idx}"))
        for ts, itemid, valuenum in lab_events:
            rows.append(_lab_row(ts, itemid, valuenum))
    df = pl.DataFrame(rows).with_columns(pl.col("timestamp").cast(pl.Datetime))
    return Patient(patient_id=patient_id, data_source=df)


def single_stay_patient(
    patient_id: str,
    lab_events: Sequence[Tuple[datetime, str, float]],
    hours: int = 49,
) -> Patient:
    return make_patient(patient_id, [(T0, T0 + timedelta(hours=hours), lab_events)])


class TestTaskSchemaAndInit(unittest.TestCase):
    """Test the schema and init simple behavior."""

    def test_task_contract_and_windows(self) -> None:
        default = SofaLabForecastingMIMIC3()
        custom = SofaLabForecastingMIMIC3(lookback_hours=12, prediction_hours=6)
        self.assertEqual(
            SofaLabForecastingMIMIC3.task_name, "SofaLabForecastingMIMIC3"
        )
        self.assertEqual(
            SofaLabForecastingMIMIC3.input_schema,
            {"observation_values": "tensor", "observation_masks": "tensor"},
        )
        self.assertEqual(
            SofaLabForecastingMIMIC3.output_schema,
            {
                "target_values": "tensor",
                "target_masks": "tensor",
                "sofa_label": "binary",
            },
        )
        self.assertEqual((default.lookback_hours, default.prediction_hours), (24, 24))
        self.assertEqual((custom.lookback_hours, custom.prediction_hours), (12, 6))


class TestSofaScoring(unittest.TestCase):
    """Test direct SOFA subscore helpers follow the paper's Appendix B thresholds."""

    def setUp(self) -> None:
        self.task = SofaLabForecastingMIMIC3()

    def test_threshold_boundaries(self) -> None:
        self.assertEqual(
            [self.task._sofa_bilirubin(v) for v in (1.1, 1.2, 2.0, 6.0, 12.0)],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [self.task._sofa_creatinine(v) for v in (1.1, 1.2, 2.0, 3.5, 5.0)],
            [0, 1, 2, 3, 4],
        )
        self.assertEqual(
            [self.task._sofa_platelets(v) for v in (151.0, 149.0, 99.0, 49.0, 19.0)],
            [0, 1, 2, 3, 4],
        )

    def test_compute_lab_sofa_worst_values(self) -> None:
        patient = single_stay_patient(
            "abnormal",
            [
                (T0 + timedelta(hours=1), BILI, 6.0),
                (T0 + timedelta(hours=2), CREAT, 2.2),
                (T0 + timedelta(hours=3), PLT, 140.0),
                (T0 + timedelta(hours=4), PLT, 40.0),
            ],
        )
        labs_df = patient.get_events(event_type="labevents", return_df=True)
        self.assertEqual(self.task._compute_lab_sofa(labs_df), 8)


class TestFeatureExtraction(unittest.TestCase):
    """Tests the hourly binning behavior for the forecasting."""

    def setUp(self) -> None:
        self.task = SofaLabForecastingMIMIC3()

    def _labs_df(self, patient: Patient) -> pl.DataFrame:
        return patient.get_events(
            event_type="labevents",
            start=T0,
            end=T0 + timedelta(hours=24),
            return_df=True,
        )

    def test_hourly_binning_and_masks(self) -> None:
        patient = single_stay_patient(
            "binning",
            [
                (T0 + timedelta(hours=2, minutes=30), BILI, 1.5),
                (T0 + timedelta(hours=0, minutes=10), PLT, 180.0),
            ],
        )
        values, masks = self.task._bin_hourly(self._labs_df(patient), T0, 24)
        self.assertAlmostEqual(values[6], normalized(BILI, 1.5), places=6)
        self.assertEqual(masks[6], 1.0)
        self.assertAlmostEqual(values[2], normalized(PLT, 180.0), places=6)
        self.assertEqual(masks[2], 1.0)
        self.assertEqual(values[5], 0.0)
        self.assertEqual(masks[5], 0.0)


class TestSampleGeneration(unittest.TestCase):
    """Sample-level contract and SOFA label derivation."""

    def setUp(self) -> None:
        self.task = SofaLabForecastingMIMIC3()

    def _baseline_events(self) -> List[Tuple[datetime, str, float]]:
        return [
            (T0 + timedelta(hours=1), BILI, 1.0),
            (T0 + timedelta(hours=2), CREAT, 1.0),
            (T0 + timedelta(hours=3), PLT, 180.0),
        ]

    def _make_forecasting_patient(
        self,
        patient_id: str,
        future_bili: float = 1.0,
        future_creat: float = 1.0,
        future_plt: float = 180.0,
    ) -> Patient:
        return single_stay_patient(
            patient_id,
            self._baseline_events() + [
                (T0 + timedelta(hours=25), BILI, future_bili),
                (T0 + timedelta(hours=26), CREAT, future_creat),
                (T0 + timedelta(hours=27), PLT, future_plt),
            ],
        )

    def test_sample_shapes(self) -> None:
        patient = self._make_forecasting_patient("shape")
        sample = self.task(patient)[0]
        self.assertEqual(len(sample["observation_values"]), 72)
        self.assertEqual(len(sample["observation_masks"]), 72)
        self.assertEqual(len(sample["target_values"]), 72)
        self.assertEqual(len(sample["target_masks"]), 72)
        self.assertIn("sofa_label", sample)

    def test_custom_window_shapes(self) -> None:
        task = SofaLabForecastingMIMIC3(lookback_hours=12, prediction_hours=12)
        patient = single_stay_patient(
            "custom",
            [
                (T0 + timedelta(hours=1), BILI, 1.0),
                (T0 + timedelta(hours=13), BILI, 2.0),
            ],
            hours=25,
        )
        sample = task(patient)[0]
        for key in ("observation_values", "observation_masks",
                    "target_values", "target_masks"):
            self.assertEqual(len(sample[key]), 36)

    def test_sofa_labels(self) -> None:
        """Negative, borderline (delta=1), boundary (delta=2), and spike cases."""
        cases = [
            # (future_bili, expected_label)
            (1.0, 0),  # no change
            (1.5, 0),  # delta 1
            (2.0, 1),  # delta 2 -> positive at boundary
            (6.0, 1),  # severe spike
        ]
        for future_bili, expected in cases:
            patient = self._make_forecasting_patient(
                f"label-{future_bili}",
                future_bili=future_bili,
            )
            sample = self.task(patient)[0]
            self.assertEqual(
                sample["sofa_label"], expected,
                f"future_bili={future_bili}",
            )


class TestEdgeCases(unittest.TestCase):
    """Stays that cannot produce a valid forecasting sample should be skipped."""

    def setUp(self) -> None:
        self.task = SofaLabForecastingMIMIC3()

    def test_invalid_stays_are_skipped(self) -> None:
        patients = [
            single_stay_patient("no-labs", []),
            single_stay_patient(
                "short",
                [(T0 + timedelta(hours=1), BILI, 1.0)],
                hours=40,
            ),
            single_stay_patient(
                "obs-only",
                [
                    (T0 + timedelta(hours=1), BILI, 1.0),
                    (T0 + timedelta(hours=2), CREAT, 1.0),
                    (T0 + timedelta(hours=3), PLT, 180.0),
                ],
            ),
            single_stay_patient(
                "pred-only",
                [
                    (T0 + timedelta(hours=25), BILI, 6.0),
                    (T0 + timedelta(hours=26), CREAT, 2.0),
                    (T0 + timedelta(hours=27), PLT, 80.0),
                ],
            ),
        ]
        for patient in patients:
            self.assertEqual(self.task(patient), [])

    def test_too_many_stays_are_skipped(self) -> None:
        t1 = T0 + timedelta(days=4)
        stay = lambda start, itemid: (
            start, start + timedelta(hours=49),
            [
                (start + timedelta(hours=1), itemid, 1.0),
                (start + timedelta(hours=25), itemid, 5.0),
            ],
        )
        patient = make_patient("multi", [stay(T0, BILI), stay(t1, CREAT)])
        self.assertEqual(len(self.task(patient)), 2)


if __name__ == "__main__":
    unittest.main()
