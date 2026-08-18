# Author: Anish Gupta
# NetID: anishg8
# Paper Title: N/A (original task contribution, not a paper reproduction)
# Paper Link: N/A
# Description: Synthetic-data test suite for SepsisPredictionMIMIC4. All
#     fixtures are hand-built, in-memory Patient objects (2-8 fake patients,
#     a handful of events each) -- no real MIMIC-IV files, no network, no
#     downloads. The full suite runs in well under a second.
"""Tests for SepsisPredictionMIMIC4.

The task is applied directly to hand-built ``Patient`` objects (bypassing
``BaseDataset``/YAML loading entirely), matching the layout PyHealth's
``global_event_df`` uses internally: one wide, sparse polars DataFrame with
an ``event_type``/``timestamp`` pair per row and ``{event_type}/{attr}``
prefixed attribute columns. This keeps the fixtures small and avoids needing
real MIMIC-IV files (including the ``d_items.csv.gz``/``d_labitems.csv.gz``
lookup joins) just to exercise the task's own logic.

Manual test cases
------------------
The automated tests below double as the manually-specified test cases
required by the contribution guide. Each one is stated here as
input -> expected output for quick review, independent of reading the code:

1. ``test_qsofa_and_antibiotic_together_trigger_sepsis``
   Input: RR=25 + SBP=90 at hour 6 (qSOFA=2), antibiotic order at hour 5
   (within the 24h window), plus normal vitals at hour 1 and a lab at
   hour 3 and hour 8.
   Expected: 1 sample, ``sepsis=1``, only pre-hour-6 observations present
   (the hour-8 reading must not appear -- leakage guard).

2. ``test_normal_vitals_never_trigger_sepsis``
   Input: RR/SBP always normal across two vitals timestamps + one lab.
   Expected: 1 sample, ``sepsis=0``, all 3 observation timestamps retained
   (no censoring for a negative label).

3. ``test_altered_mentation_contributes_to_qsofa``
   Input: RR=24 (1 point) + GCS eye/verbal/motor summing to 8 < 15 (1
   point), SBP unset, antibiotic nearby.
   Expected: 1 sample, ``sepsis=1`` -- proves the GCS-based criterion
   alone can supply the second qSOFA point.

4. ``test_sbp_uses_minimum_across_available_itemids``
   Input: RR=24 (1 point), NIBP SBP=120 (normal) and arterial SBP=85 (low)
   at the same timestamp, antibiotic nearby.
   Expected: 1 sample, ``sepsis=1`` -- proves the lower of the two BP
   readings is the one used, not whichever happens to be first.

5. ``test_multiple_admissions_labeled_independently``
   Input: one patient with two admissions 30 days apart -- the first
   qualifies for sepsis, the second has normal vitals.
   Expected: 2 samples, one per admission, each labeled independently
   (``sepsis=1`` and ``sepsis=0`` respectively).

6. ``test_onset_with_no_prior_observations_yields_no_sample``
   Input: the only vitals for the admission are the exact qSOFA-triggering
   reading itself.
   Expected: 0 samples -- after censoring strictly before onset, there is
   nothing left to predict from, so the admission is dropped rather than
   returned with an empty feature matrix.

7. ``test_organ_dysfunction_without_infection_is_not_sepsis``
   Input: qSOFA=2 at hour 4, but the only antibiotic order is 5 days
   later (outside the admission and the window).
   Expected: 1 sample, ``sepsis=0`` -- proves organ dysfunction alone,
   without a nearby infection signal, is not labeled sepsis.

8. ``test_missing_dischtime_admission_is_skipped``
   Input: an admission whose ``dischtime`` is ``None``.
   Expected: 0 samples -- malformed data is skipped, not raised.

9. ``test_samples_flow_through_the_real_processors``
   Input: one positive and one negative sample, run through
   ``create_sample_dataset`` with the task's real ``input_schema``/
   ``output_schema``.
   Expected: both samples processed successfully; the positive sample's
   ``observations`` tensor has width ``len(OBSERVATION_ITEMIDS)`` and its
   ``sepsis`` value is 1.0.

10. ``test_tiny_model_trains_one_step``
    Input: the same two samples as above, fed through an ``RNN``
    (``hidden_dim=4``) and one ``Trainer.train()`` epoch
    (``batch_size=2``).
    Expected: training completes without error and produces a finite,
    non-NaN loss -- confirms the task's output is actually consumable by
    a real model, not just schema-shaped.
"""

import unittest
from datetime import datetime

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models import RNN
from pyhealth.tasks.sepsis_prediction_mimic4 import SepsisPredictionMIMIC4
from pyhealth.trainer import Trainer

T0 = datetime(2024, 1, 1, 0, 0, 0)  # naive, matching MIMIC-IV's own timestamps  # noqa: DTZ001


def _row(event_type: str, timestamp: datetime, **attrs) -> dict:
    row = {"event_type": event_type, "timestamp": timestamp}
    for key, value in attrs.items():
        row[f"{event_type}/{key}"] = value
    return row


# A real PyHealth dataset guarantees every configured table's attribute
# columns exist in the global event frame, regardless of whether any given
# admission has rows for that table -- get_events() filters against
# "{event_type}/{attr}" columns unconditionally. These fixtures must
# guarantee the same columns exist even when a scenario has zero rows for
# a given event type (e.g. no prescriptions at all).
_REQUIRED_COLUMNS = {
    "admissions/hadm_id": pl.Int64,
    "admissions/dischtime": pl.Utf8,
    "prescriptions/hadm_id": pl.Int64,
    "prescriptions/drug": pl.Utf8,
    "labevents/hadm_id": pl.Int64,
    "labevents/itemid": pl.Utf8,
    "labevents/valuenum": pl.Float64,
    "chartevents/hadm_id": pl.Int64,
    "chartevents/itemid": pl.Utf8,
    "chartevents/valuenum": pl.Float64,
}


def _build_patient(patient_id: str, rows: list) -> Patient:
    df = pl.DataFrame(rows)
    for col, dtype in _REQUIRED_COLUMNS.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))
    df = df.cast(_REQUIRED_COLUMNS)
    return Patient(patient_id=patient_id, data_source=df)


def _admission(hadm_id: int, admit: datetime, dischtime: datetime) -> dict:
    return _row(
        "admissions",
        admit,
        hadm_id=hadm_id,
        dischtime=dischtime.strftime("%Y-%m-%d %H:%M:%S"),
    )


def _vital(hadm_id: int, ts: datetime, itemid: str, valuenum: float) -> dict:
    return _row(
        "chartevents", ts, hadm_id=hadm_id, itemid=itemid, valuenum=valuenum
    )


def _lab(hadm_id: int, ts: datetime, itemid: str, valuenum: float) -> dict:
    return _row(
        "labevents", ts, hadm_id=hadm_id, itemid=itemid, valuenum=valuenum
    )


def _antibiotic(hadm_id: int, ts: datetime, drug: str = "Vancomycin") -> dict:
    return _row("prescriptions", ts, hadm_id=hadm_id, drug=drug)


class TestSepsisPredictionMIMIC4(unittest.TestCase):
    def setUp(self):
        self.task = SepsisPredictionMIMIC4()

    def test_qsofa_and_antibiotic_together_trigger_sepsis(self):
        from datetime import timedelta

        rows = [
            _admission(1001, T0, T0 + timedelta(hours=10)),
            # Normal vitals early in the stay.
            _vital(1001, T0 + timedelta(hours=1), "220210", 16.0),  # RR normal
            _vital(1001, T0 + timedelta(hours=1), "220179", 120.0),  # SBP normal
            # A relevant lab before onset.
            _lab(1001, T0 + timedelta(hours=3), "50813", 4.5),  # Lactate
            # qSOFA >= 2 at hour 6: RR high + SBP low.
            _vital(1001, T0 + timedelta(hours=6), "220210", 25.0),
            _vital(1001, T0 + timedelta(hours=6), "220179", 90.0),
            # Antibiotic within the 24h window of onset.
            _antibiotic(1001, T0 + timedelta(hours=5)),
            # A later vital that must NOT leak into the returned features.
            _vital(1001, T0 + timedelta(hours=8), "220210", 30.0),
        ]
        patient = _build_patient("p1", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample["sepsis"], 1)

        timestamps, values = sample["observations"]
        self.assertTrue(all(t < T0 + timedelta(hours=6) for t in timestamps))
        # The hour-8 RR=30.0 value must not appear anywhere in the matrix.
        self.assertNotIn(30.0, values.flatten().tolist())

    def test_normal_vitals_never_trigger_sepsis(self):
        from datetime import timedelta

        rows = [
            _admission(1002, T0, T0 + timedelta(hours=8)),
            _vital(1002, T0 + timedelta(hours=1), "220210", 14.0),
            _vital(1002, T0 + timedelta(hours=1), "220179", 118.0),
            _vital(1002, T0 + timedelta(hours=5), "220210", 15.0),
            _vital(1002, T0 + timedelta(hours=5), "220179", 122.0),
            _lab(1002, T0 + timedelta(hours=2), "50912", 0.9),  # Creatinine
        ]
        patient = _build_patient("p2", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sepsis"], 0)
        timestamps, _ = samples[0]["observations"]
        # 3 distinct timestamps: hour-1 vitals, hour-2 lab, hour-5 vitals
        # (the RR+SBP pair at each vitals timestamp collapse into one row).
        self.assertEqual(len(timestamps), 3)

    def test_altered_mentation_contributes_to_qsofa(self):
        """The GCS/altered-mentation criterion must itself be able to
        contribute a qSOFA point -- every other test only exercises RR and
        SBP, leaving the GCS-scoring branch entirely untested."""
        from datetime import timedelta

        rows = [
            _admission(1004, T0, T0 + timedelta(hours=10)),
            # A baseline reading, so censoring at onset leaves something.
            _vital(1004, T0 + timedelta(hours=1), "220210", 16.0),
            # RR alone (1 point) + GCS sum=8 < 15 (1 point) = 2 points.
            # SBP is left unset (no reading), so it cannot contribute.
            _vital(1004, T0 + timedelta(hours=4), "220210", 24.0),
            _vital(1004, T0 + timedelta(hours=4), "220739", 2.0),  # eye
            _vital(1004, T0 + timedelta(hours=4), "223900", 2.0),  # verbal
            _vital(1004, T0 + timedelta(hours=4), "223901", 4.0),  # motor
            _antibiotic(1004, T0 + timedelta(hours=4)),
        ]
        patient = _build_patient("p4", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sepsis"], 1)

    def test_sbp_uses_minimum_across_available_itemids(self):
        """When both an NIBP and an arterial-line SBP reading exist at the
        same timestamp, the lower of the two must be used -- proving the
        min() selection, not just picking whichever itemid happens first."""
        from datetime import timedelta

        rows = [
            _admission(1005, T0, T0 + timedelta(hours=10)),
            # A baseline reading, so censoring at onset leaves something.
            _vital(1005, T0 + timedelta(hours=1), "220179", 122.0),
            # NIBP reads normal (120); arterial line reads low (85).
            # Only the minimum (85 <= 100) should count as a qSOFA point.
            _vital(1005, T0 + timedelta(hours=3), "220210", 24.0),  # RR point
            _vital(1005, T0 + timedelta(hours=3), "220179", 120.0),  # NIBP
            _vital(1005, T0 + timedelta(hours=3), "220050", 85.0),  # arterial
            _antibiotic(1005, T0 + timedelta(hours=3)),
        ]
        patient = _build_patient("p5", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sepsis"], 1)

    def test_multiple_admissions_labeled_independently(self):
        """A patient with two admissions must get one sample per admission,
        each labeled from its own vitals/antibiotics -- not pooled."""
        from datetime import timedelta

        first_admit = T0
        second_admit = T0 + timedelta(days=30)
        rows = [
            _admission(2001, first_admit, first_admit + timedelta(hours=10)),
            _vital(2001, first_admit + timedelta(hours=1), "220210", 16.0),
            _vital(2001, first_admit + timedelta(hours=4), "220210", 25.0),
            _vital(2001, first_admit + timedelta(hours=4), "220179", 90.0),
            _antibiotic(2001, first_admit + timedelta(hours=4)),
            _admission(2002, second_admit, second_admit + timedelta(hours=8)),
            _vital(2002, second_admit + timedelta(hours=1), "220210", 14.0),
            _vital(2002, second_admit + timedelta(hours=1), "220179", 118.0),
        ]
        patient = _build_patient("p6", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 2)
        by_admission = {s["admission_id"]: s for s in samples}
        self.assertEqual(by_admission[2001]["sepsis"], 1)
        self.assertEqual(by_admission[2002]["sepsis"], 0)

    def test_onset_with_no_prior_observations_yields_no_sample(self):
        """If the only vitals for an admission are the exact qSOFA-
        triggering reading itself, strict pre-onset censoring leaves no
        observations at all -- the sample must be dropped, not returned
        with an empty feature matrix."""
        from datetime import timedelta

        rows = [
            _admission(1007, T0, T0 + timedelta(hours=10)),
            _vital(1007, T0 + timedelta(hours=4), "220210", 25.0),
            _vital(1007, T0 + timedelta(hours=4), "220179", 90.0),
            _antibiotic(1007, T0 + timedelta(hours=4)),
        ]
        patient = _build_patient("p8", rows)
        samples = self.task(patient)

        self.assertEqual(samples, [])

    def test_missing_dischtime_admission_is_skipped(self):
        """An admission with a null/unparseable discharge time must be
        skipped rather than raising."""
        rows = [
            _row(
                "admissions",
                T0,
                hadm_id=1006,
                dischtime=None,
            ),
        ]
        patient = _build_patient("p7", rows)
        samples = self.task(patient)

        self.assertEqual(samples, [])

    def test_organ_dysfunction_without_infection_is_not_sepsis(self):
        """qSOFA >= 2 alone, with no antibiotic order anywhere nearby, must
        not be labeled sepsis -- this is what proves the task enforces both
        signals rather than just a vitals threshold."""
        from datetime import timedelta

        rows = [
            _admission(1003, T0, T0 + timedelta(hours=10)),
            _vital(1003, T0 + timedelta(hours=4), "220210", 25.0),
            _vital(1003, T0 + timedelta(hours=4), "220179", 90.0),
            # An antibiotic order, but far outside the 24h onset window.
            _antibiotic(1003, T0 + timedelta(days=5)),
        ]
        patient = _build_patient("p3", rows)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["sepsis"], 0)

    def test_samples_flow_through_the_real_processors(self):
        """End-to-end: raw task output must be accepted by the actual
        'timeseries'/'binary' processors used by set_task(), not just be a
        plausible-looking dict."""
        from datetime import timedelta

        positive_rows = [
            _admission(1001, T0, T0 + timedelta(hours=10)),
            _vital(1001, T0 + timedelta(hours=1), "220210", 16.0),
            _vital(1001, T0 + timedelta(hours=1), "220179", 120.0),
            _vital(1001, T0 + timedelta(hours=6), "220210", 25.0),
            _vital(1001, T0 + timedelta(hours=6), "220179", 90.0),
            _antibiotic(1001, T0 + timedelta(hours=5)),
        ]
        negative_rows = [
            _admission(1002, T0, T0 + timedelta(hours=8)),
            _vital(1002, T0 + timedelta(hours=1), "220210", 14.0),
            _vital(1002, T0 + timedelta(hours=1), "220179", 118.0),
        ]
        samples = self.task(_build_patient("p1", positive_rows))
        samples += self.task(_build_patient("p2", negative_rows))
        # BinaryLabelProcessor.fit requires seeing both classes; fit on both
        # samples, then assert on the positive one.
        self.assertEqual(len(samples), 2)

        sample_dataset = create_sample_dataset(
            samples=samples,
            input_schema=self.task.input_schema,
            output_schema=self.task.output_schema,
            dataset_name="sepsis_test",
        )
        self.assertEqual(len(sample_dataset), 2)
        processed = next(s for s in sample_dataset if s["admission_id"] == 1001)
        # (timesteps, num_observation_itemids)
        self.assertEqual(
            processed["observations"].shape[1],
            len(SepsisPredictionMIMIC4.OBSERVATION_ITEMIDS),
        )
        self.assertEqual(float(processed["sepsis"]), 1.0)

    def test_tiny_model_trains_one_step(self):
        """Smoke test: task output must be trainable by a real model, not
        just schema-shaped. Uses a tiny RNN (hidden_dim=4) and a single
        epoch/batch so this stays a millisecond-scale unit test."""
        from datetime import timedelta

        torch.manual_seed(42)
        np.random.seed(42)

        positive_rows = [
            _admission(1001, T0, T0 + timedelta(hours=10)),
            _vital(1001, T0 + timedelta(hours=1), "220210", 16.0),
            _vital(1001, T0 + timedelta(hours=6), "220210", 25.0),
            _vital(1001, T0 + timedelta(hours=6), "220179", 90.0),
            _antibiotic(1001, T0 + timedelta(hours=5)),
        ]
        negative_rows = [
            _admission(1002, T0, T0 + timedelta(hours=8)),
            _vital(1002, T0 + timedelta(hours=1), "220210", 14.0),
            _vital(1002, T0 + timedelta(hours=1), "220179", 118.0),
        ]
        samples = self.task(_build_patient("p1", positive_rows))
        samples += self.task(_build_patient("p2", negative_rows))
        self.assertEqual(len(samples), 2)

        sample_dataset = create_sample_dataset(
            samples=samples,
            input_schema=self.task.input_schema,
            output_schema=self.task.output_schema,
            dataset_name="sepsis_tiny_model_test",
        )
        dataloader = get_dataloader(sample_dataset, batch_size=2, shuffle=False)

        model = RNN(dataset=sample_dataset, embedding_dim=4, hidden_dim=4)
        trainer = Trainer(model=model, enable_logging=False)
        trainer.train(train_dataloader=dataloader, epochs=1)

        batch = next(iter(dataloader))
        output = model(**batch)
        loss = float(output["loss"])
        self.assertTrue(np.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
