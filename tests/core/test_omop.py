import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

from pyhealth.datasets import OMOPDataset
from pyhealth.tasks import MortalityPredictionOMOP, ReadmissionPredictionOMOP


class TestOMOPDataset(unittest.TestCase):
    """Test OMOP dataset with local test data."""

    def setUp(self):
        """Set up test data and dataset."""
        self.test_data_path = (
            Path(__file__).parent.parent.parent / "test-resources" / "omop"
        )

        # Load dataset with all available tables
        self.tables = [
            "person",
            "visit_occurrence",
            "death",
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
            "observation_period",
            "visit_detail",
            "observation",
            "device_exposure",
        ]
        self.dataset = OMOPDataset(root=str(self.test_data_path), tables=self.tables)

    def test_dataset_loading(self):
        """Test that dataset loads successfully."""
        self.assertIsInstance(self.dataset, OMOPDataset)
        self.assertIsNotNone(self.dataset.unique_patient_ids)

    def test_patient_count(self):
        """Test that we have the expected number of patients."""
        # Should have multiple patients from our test subset
        self.assertGreater(len(self.dataset.unique_patient_ids), 0)

    def test_patient_attributes(self):
        """Test that patients have expected attributes."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Check basic patient attributes
        self.assertIsNotNone(patient.patient_id)
        self.assertEqual(patient.patient_id, patient_id)

        # Check that patient has data_source
        self.assertIsNotNone(patient.data_source)
        self.assertGreater(len(patient.data_source), 0)

    def test_visit_events(self):
        """Test that visit events are loaded correctly."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get visit events
        visits = patient.get_events(event_type="visit_occurrence")
        self.assertIsNotNone(visits)

        # Check that visits have required attributes
        if len(visits) > 0:
            visit = visits[0]
            self.assertIsNotNone(visit.event_type)
            self.assertIsNotNone(visit.timestamp)
            self.assertEqual(visit.event_type, "visit_occurrence")

    def test_condition_events(self):
        """Test that condition events are loaded correctly."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get condition events
        conditions = patient.get_events(event_type="condition_occurrence")
        self.assertIsNotNone(conditions)

        # Check that conditions have required attributes
        if len(conditions) > 0:
            condition = conditions[0]
            self.assertIsNotNone(condition.event_type)
            self.assertIsNotNone(condition.timestamp)
            self.assertEqual(condition.event_type, "condition_occurrence")
            self.assertIn("condition_concept_id", condition.attr_dict)

    def test_procedure_events(self):
        """Test that procedure events are loaded correctly."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get procedure events
        procedures = patient.get_events(event_type="procedure_occurrence")
        self.assertIsNotNone(procedures)

        # Check that procedures have required attributes
        if len(procedures) > 0:
            procedure = procedures[0]
            self.assertIsNotNone(procedure.event_type)
            self.assertIsNotNone(procedure.timestamp)
            self.assertEqual(procedure.event_type, "procedure_occurrence")
            self.assertIn("procedure_concept_id", procedure.attr_dict)

    def test_drug_events(self):
        """Test that drug events are loaded correctly."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get drug events
        drugs = patient.get_events(event_type="drug_exposure")
        self.assertIsNotNone(drugs)

        # Check that drugs have required attributes
        if len(drugs) > 0:
            drug = drugs[0]
            self.assertIsNotNone(drug.event_type)
            self.assertIsNotNone(drug.timestamp)
            self.assertEqual(drug.event_type, "drug_exposure")
            self.assertIn("drug_concept_id", drug.attr_dict)

    def test_measurement_events(self):
        """Test that measurement events are loaded correctly."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get measurement events
        measurements = patient.get_events(event_type="measurement")
        self.assertIsNotNone(measurements)

        # Check that measurements have required attributes
        if len(measurements) > 0:
            measurement = measurements[0]
            self.assertIsNotNone(measurement.event_type)
            self.assertIsNotNone(measurement.timestamp)
            self.assertEqual(measurement.event_type, "measurement")
            self.assertIn("measurement_concept_id", measurement.attr_dict)

    def test_added_tables(self):
        """Test that observation_period, visit_detail, observation and device_exposure load."""
        patient = self.dataset.get_patient("-4873075614181207858")
        for table in [
            "observation_period",
            "visit_detail",
            "observation",
            "device_exposure",
        ]:
            events = patient.get_events(event_type=table)
            self.assertGreater(len(events), 0, table)
            self.assertIsNotNone(events[0].timestamp)

    def test_event_types(self):
        """Test that all expected event types are present."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get all event types for the patient
        # event_type_partitions.keys() returns tuples like
        # ('visit_occurrence',)
        event_types = [key[0] for key in patient.event_type_partitions.keys()]

        # Should have multiple event types (at least visits)
        self.assertGreater(len(event_types), 0)
        self.assertIn("visit_occurrence", event_types)

    def test_temporal_ordering(self):
        """Test that events are ordered chronologically."""
        patient_id = self.dataset.unique_patient_ids[0]
        patient = self.dataset.get_patient(patient_id)

        # Get all visits (should have multiple)
        visits = patient.get_events(event_type="visit_occurrence")

        if len(visits) > 1:
            # Check that events are ordered
            for i in range(len(visits) - 1):
                self.assertLessEqual(
                    visits[i].timestamp,
                    visits[i + 1].timestamp,
                    "Events should be chronologically ordered",
                )


class _Events:
    """Minimal stand-in for ``Patient.get_events``."""

    def __init__(self, events, patient_id="p1"):
        self.patient_id = patient_id
        self.events = events

    def get_events(self, event_type=None, start=None, end=None, filters=None):
        events = list(self.events.get(event_type, []))
        for attr, _, value in filters or []:
            events = [e for e in events if getattr(e, attr) == value]
        return events


def _event(timestamp=None, **attributes):
    return SimpleNamespace(timestamp=timestamp, **attributes)


def _coded_visit(timestamp, visit_id, visit_end):
    return _event(
        timestamp, visit_occurrence_id=visit_id, visit_end_datetime=visit_end
    )


def _clinical_events(visit_id):
    return {
        "condition_occurrence": [
            _event(visit_occurrence_id=visit_id, condition_concept_id=201826)
        ],
        "procedure_occurrence": [
            _event(visit_occurrence_id=visit_id, procedure_concept_id=4163872)
        ],
        "drug_exposure": [
            _event(visit_occurrence_id=visit_id, drug_concept_id=1503297)
        ],
    }


def _mortality_patient(
    visit_start,
    visit_end,
    next_start,
    periods,
    death=None,
    visit_id="v1",
):
    events = {
        "visit_occurrence": [
            _coded_visit(visit_start, visit_id, visit_end),
            _coded_visit(next_start, "v-next", "2151-01-01 00:00:00"),
        ],
        "death": [] if death is None else [_event(death)],
        "observation_period": periods,
    }
    events.update(_clinical_events(visit_id))
    return _Events(events)


def _period(start, end):
    return _event(
        observation_period_start_date=start,
        observation_period_end_date=end,
    )


def _readmission_patient(visit_end, next_start, periods):
    visit_start = datetime(2020, 1, 1, 8, 0, 0)
    events = {
        "person": [_event(year_of_birth=1980, month_of_birth=1, day_of_birth=1)],
        "visit_occurrence": [
            _coded_visit(visit_start, 1, visit_end),
            _coded_visit(next_start, 2, "2020-06-01 00:00:00"),
        ],
        "observation_period": periods,
    }
    events.update(_clinical_events(1))
    return _Events(events)


class TestObservationPeriodCensoring(unittest.TestCase):
    """Right-censoring when a visit leaves OMOP observation early.

    A wrong end-of-day parse, a default that starts dropping visits, or a
    follow-up check that ignores an observed death makes these fail.
    """

    def test_mortality_prediction_omop_default_unchanged(self):
        start = datetime(2150, 1, 1, 8, 0, 0)
        patient = _mortality_patient(
            start,
            "2150-01-03 12:00:00",
            datetime(2150, 2, 1, 8, 0, 0),
            [_period("2150-01-01", "2150-01-08")],
            death=datetime(2150, 1, 20, 0, 0, 0),
        )
        default = MortalityPredictionOMOP()
        explicit = MortalityPredictionOMOP(check_observation_period=False)
        samples = default(patient)
        self.assertEqual(samples, explicit(patient))
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["visit_id"], "v1")
        self.assertEqual(samples[0]["mortality"], 1)
        self.assertEqual(samples[0]["conditions"], ["201826"])

        mapped = MortalityPredictionOMOP(
            code_mapping={"conditions": ("ICD9CM", "CCSCM")}
        )
        self.assertEqual(
            mapped.input_schema["conditions"],
            ("sequence", {"code_mapping": ("ICD9CM", "CCSCM")}),
        )
        # code_mapping stays the first positional argument. A dict here must
        # not be read as check_observation_period.
        positional = MortalityPredictionOMOP({"conditions": ("ICD9CM", "CCSCM")})
        self.assertEqual(
            positional.input_schema["conditions"],
            ("sequence", {"code_mapping": ("ICD9CM", "CCSCM")}),
        )
        self.assertFalse(positional.check_observation_period)
        with self.assertRaises(TypeError):
            MortalityPredictionOMOP({"conditions": ("ICD9CM", "CCSCM")}, True)

    def test_mortality_prediction_omop_observation_period_censoring(self):
        start = datetime(2150, 1, 1, 8, 0, 0)
        discharge = "2150-01-03 12:00:00"
        next_start = datetime(2150, 6, 1, 8, 0, 0)
        task = MortalityPredictionOMOP(
            check_observation_period=True,
            min_followup=timedelta(days=30),
        )

        short_followup = _mortality_patient(
            start,
            discharge,
            next_start,
            [_period("2150-01-01", "2150-01-08")],
        )
        self.assertEqual(task(short_followup), [])

        # min_followup turns the check on even when the flag is passed False.
        forced = MortalityPredictionOMOP(
            check_observation_period=False,
            min_followup=timedelta(days=30),
        )
        self.assertEqual(forced(short_followup), [])

        long_followup = _mortality_patient(
            start,
            discharge,
            next_start,
            [
                _period("not-a-date", "still-not"),
                _period(date(2150, 1, 1), date(2150, 3, 4)),
            ],
        )
        long_samples = task(long_followup)
        self.assertEqual(len(long_samples), 1)
        self.assertEqual(long_samples[0]["mortality"], 0)

        # Death on day 90 is outside the 30-day horizon, so the label stays 0
        # even though a death record exists.
        late_death = _mortality_patient(
            start,
            discharge,
            next_start,
            [_period(date(2150, 1, 1), date(2150, 3, 4))],
            death=datetime(2150, 4, 3, 12, 0, 0),
        )
        self.assertEqual(task(late_death)[0]["mortality"], 0)

        # Death 17 days after admission, inside a period that does not reach
        # the full 30-day horizon, is an observed outcome and is kept.
        observed_death = _mortality_patient(
            start,
            discharge,
            next_start,
            [
                _period(
                    datetime(2150, 1, 1, 0, 0, 0),
                    datetime(2150, 1, 25, 0, 0, 0),
                )
            ],
            death=datetime(2150, 1, 18, 9, 0, 0),
        )
        death_samples = task(observed_death)
        self.assertEqual(len(death_samples), 1)
        self.assertEqual(death_samples[0]["mortality"], 1)

        # Death inside the horizon but after the observation period ends is
        # not evidence: the visit is right-censored.
        unobserved_death = _mortality_patient(
            start,
            discharge,
            next_start,
            [_period("2150-01-01", "2150-01-04")],
            death=datetime(2150, 1, 10, 0, 0, 0),
        )
        self.assertEqual(task(unobserved_death), [])

        outside = _mortality_patient(
            start,
            discharge,
            next_start,
            [_period("2150-02-01", "2150-06-01")],
        )
        self.assertEqual(task(outside), [])

        # test-resources/omop visit ending 12:18 on its period's last day.
        midday = _mortality_patient(
            datetime(2148, 9, 10, 9, 23, 0),
            "2148-09-23 12:18:00",
            datetime(2148, 10, 1, 9, 0, 0),
            [_period("2148-09-10", "2148-09-23")],
        )
        zero_horizon = MortalityPredictionOMOP(
            check_observation_period=True,
            min_followup=timedelta(0),
        )
        midday_samples = zero_horizon(midday)
        self.assertEqual(len(midday_samples), 1)
        self.assertEqual(midday_samples[0]["mortality"], 0)

        with self.assertRaises(ValueError):
            MortalityPredictionOMOP(min_followup=timedelta(days=-1))

        # No min_followup: the horizon is the next visit. A death after that
        # visit is 0 when the period still covers the horizon, same as a
        # death past min_followup.
        until_next = MortalityPredictionOMOP(check_observation_period=True)
        death_after_horizon = _mortality_patient(
            start,
            discharge,
            datetime(2150, 1, 20, 8, 0, 0),
            [_period("2150-01-01", "2150-01-25")],
            death=datetime(2150, 1, 22, 0, 0, 0),
        )
        self.assertEqual(until_next(death_after_horizon)[0]["mortality"], 0)
        death_before_next = _mortality_patient(
            start,
            discharge,
            datetime(2150, 1, 20, 8, 0, 0),
            [_period("2150-01-01", "2150-01-25")],
            death=datetime(2150, 1, 15, 0, 0, 0),
        )
        self.assertEqual(until_next(death_before_next)[0]["mortality"], 1)
        # Death before the next visit counts even when the period ends
        # before that visit.
        observed_before_next = _mortality_patient(
            start,
            discharge,
            datetime(2150, 2, 1, 8, 0, 0),
            [_period("2150-01-01", "2150-01-20")],
            death=datetime(2150, 1, 18, 0, 0, 0),
        )
        self.assertEqual(until_next(observed_before_next)[0]["mortality"], 1)
        survivor_until_next = _mortality_patient(
            start,
            discharge,
            datetime(2150, 1, 20, 8, 0, 0),
            [_period("2150-01-01", "2150-01-25")],
        )
        self.assertEqual(until_next(survivor_until_next)[0]["mortality"], 0)
        death_after_period = _mortality_patient(
            start,
            discharge,
            datetime(2150, 1, 20, 8, 0, 0),
            [_period("2150-01-01", "2150-01-25")],
            death=datetime(2150, 2, 1, 0, 0, 0),
        )
        self.assertEqual(until_next(death_after_period)[0]["mortality"], 0)
        censored_before_next = _mortality_patient(
            start,
            discharge,
            datetime(2150, 1, 20, 8, 0, 0),
            [_period("2150-01-01", "2150-01-10")],
        )
        self.assertEqual(until_next(censored_before_next), [])

    def test_mortality_warns_once_when_observation_period_is_missing(self):
        start = datetime(2150, 1, 1, 8, 0, 0)
        patient = _mortality_patient(
            start,
            "2150-01-03 12:00:00",
            datetime(2150, 2, 1, 8, 0, 0),
            [],
        )
        task = MortalityPredictionOMOP(check_observation_period=True)
        with self.assertLogs(
            "pyhealth.tasks.mortality_prediction", level="WARNING"
        ) as logs:
            self.assertEqual(task(patient), [])
        self.assertIn("observation_period", logs.output[0])
        with self.assertNoLogs("pyhealth.tasks.mortality_prediction", level="WARNING"):
            self.assertEqual(task(patient), [])

    def test_readmission_prediction_omop_observation_period_censoring(self):
        task = ReadmissionPredictionOMOP(
            window=timedelta(days=15),
            check_observation_period=True,
        )
        discharge = "2020-01-05 12:00:00"
        later = datetime(2020, 3, 1, 8, 0, 0)
        soon = datetime(2020, 1, 10, 8, 0, 0)
        short_observation = _readmission_patient(
            discharge, later, [_period("2020-01-01", "2020-01-10")]
        )
        self.assertEqual(task(short_observation), [])
        self.assertEqual(len(ReadmissionPredictionOMOP()(short_observation)), 1)

        readmitted = _readmission_patient(
            discharge, soon, [_period("2020-01-01", "2020-01-06")]
        )
        readmitted_samples = task(readmitted)
        self.assertEqual(len(readmitted_samples), 1)
        self.assertEqual(readmitted_samples[0]["readmission"], 1)
        self.assertEqual(readmitted_samples[0]["visit_id"], 1)

        observed_negative = _readmission_patient(
            discharge,
            later,
            [_period("2020-01-01 00:00:00", "2020-02-01 00:00:00")],
        )
        negative_samples = task(observed_negative)
        self.assertEqual(len(negative_samples), 1)
        self.assertEqual(negative_samples[0]["readmission"], 0)

        # date and datetime bounds, with one unparseable period ignored.
        mixed_bounds = _readmission_patient(
            discharge,
            later,
            [
                _period("bad", "nope"),
                _period(date(2020, 1, 1), datetime(2020, 2, 1, 0, 0, 0)),
            ],
        )
        mixed_samples = task(mixed_bounds)
        self.assertEqual(len(mixed_samples), 1)
        self.assertEqual(mixed_samples[0]["readmission"], 0)

        outside_visit = _readmission_patient(
            discharge, soon, [_period("2020-02-01", "2020-03-01")]
        )
        self.assertEqual(task(outside_visit), [])

        mapped = ReadmissionPredictionOMOP(
            code_mapping={"drugs": ("NDC", "ATC")}
        )
        self.assertEqual(
            mapped.input_schema["drugs"],
            ("sequence", {"code_mapping": ("NDC", "ATC")}),
        )


if __name__ == "__main__":
    unittest.main()
