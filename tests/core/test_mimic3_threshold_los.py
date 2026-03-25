"""
Unit tests for LengthOfStayThresholdPredictionMIMIC3 task.

Tests:
- Sample generation
- Label generation
- Feature extraction (conditions, procedures, drugs)
- Edge cases (empty visits, minimal samples, minor patients)
"""

import unittest
from datetime import datetime, timedelta
from typing import Any, List, Optional, Tuple

from pyhealth.tasks import LengthOfStayThresholdPredictionMIMIC3


class MockEvent:
    """A simple container for event attributes for testing purposes. This is
    useful to avoid using real MIMIC 3 data for testing.

    Example:
        >>> e = MockEvent(hadm_id="v1", icd9_code="c1")
        >>> e.hadm_id
        'v1'
    """

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a mock event.

        Args:
            **kwargs: event attributes (hadm_id, icd9_code, ...).

        Example:
            >>> e = MockEvent(hadm_id="v1")
            >>> e.hadm_id
            'v1'
        """
        self.__dict__.update(kwargs)


class MockPatient:
    """A mock patient object for testing purposes.

    Example:
        >>> patient = MockPatient()
        >>> admissions = patient.get_events("admissions")
        >>> len(admissions) > 0
        True
    """

    def __init__(
            self,
            los_days: float = 4.0,
            include_features: bool = True,
            minor: bool = False) -> None:
        """Initializes a patient for testing purposes.

        Args:
            los_days (float): Length of stay in days. Defaults to 4.0 days.
            include_features (bool): Whether to include conditions/procedures/drugs.
            Defaults to True.
            minor (bool): True if the patient's age is below 18,
            False otherwise. Defaults to False.
        """
        self.patient_id = "p0"

        admission_time = datetime(2020, 1, 1)
        discharge_time = datetime(2020, 1, 1) + timedelta(days = los_days)

        date_of_birth_year = 2009 if minor else 2000

        self.events = {
            "admissions": [
                MockEvent(
                    hadm_id = "v0",
                    timestamp = admission_time,
                    dischtime = discharge_time.strftime("%Y-%m-%d %H:%M:%S"),
                )
            ],
            "patients": [
                MockEvent(dob = f"{date_of_birth_year}-01-01 00:00:00")
            ],
            "diagnoses_icd": (
                [MockEvent(hadm_id = "v0",
                           icd9_code = "c1")] if include_features else []
            ),
            "procedures_icd": (
                [MockEvent(hadm_id = "v0",
                           icd9_code = "p1")] if include_features else []
            ),
            "prescriptions": (
                [MockEvent(hadm_id = "v0", ndc = "d1")] if include_features else []
            ),
        }

    def get_events(self,
                   event_type: str,
                   filters: Optional[List[Tuple[str, str, Any]]] = None
                   ) -> List[MockEvent]:
        """Return events by type with filters if defined.

        Args:
            event_type (str): Type of event.
            filters (list, optional): Filtering rules (will only return events with
            matching these attributes). Defaults to None.

        Returns:
            list: events

        Example:
            >>> patient = MockPatient()
            >>> patient.get_events("diagnoses_icd")
            [MockEvent]
        """
        events = self.events.get(event_type, [])

        # Filter, given filters have been provided (defaults to None)
        if filters:
            key, _, value = filters[0]
            return [e for e in events if getattr(e, key) == value]

        return events


class TestLengthOfStayThresholdPrediction(unittest.TestCase):
    """Unit tests for LengthOfStayThresholdPredictionMIMIC3 task.

    Tests:
    - Sample generation
    - Label generation
    - Feature extraction (conditions, procedures, drugs)
    - Edge cases (empty visits, minimal samples, minor patients)
    """

    def setUp(self):
        """Sets up task for testing."""
        self.task = LengthOfStayThresholdPredictionMIMIC3(days=3)

    def test_generates_samples(self):
        """Tests sample generation.

        Example:
            >>> samples = task(patient)
            >>> len(samples) > 0
            True
        """
        patient = MockPatient(los_days=5)
        samples = self.task(patient)

        self.assertGreater(len(samples), 0)

    def test_label_thresholding(self):
        """Test binary LOS label generation."""
        patient_below_threshold = MockPatient(los_days=2)
        patient_beyond_threshold = MockPatient(los_days=5)

        below_threshold_label = self.task(patient_below_threshold)[0]["los"]
        beyond_threshold_label = self.task(patient_beyond_threshold)[0]["los"]

        # Verify labels reflect whether the patients stay was beyond three days
        self.assertEqual(below_threshold_label, 0)
        self.assertEqual(beyond_threshold_label, 1)

    def test_feature_integrity(self):
        """Tests feature extraction."""
        patient = MockPatient(los_days=4)
        sample = self.task(patient)[0]

        self.assertIn("conditions", sample)
        self.assertIn("procedures", sample)
        self.assertIn("drugs", sample)

        self.assertGreater(len(sample["conditions"]), 0)
        self.assertGreater(len(sample["procedures"]), 0)
        self.assertGreater(len(sample["drugs"]), 0)

        self.assertEqual(sample["los"], 1)

    def test_empty_features(self):
        """Test samples with missing features aren't excluded."""
        patient = MockPatient(include_features=False)
        samples = self.task(patient)

        self.assertEqual(len(samples), 0)

    def test_exclude_minors(self):
        """Test that minor patients are excluded if exclude minors flag is enabled."""
        task = LengthOfStayThresholdPredictionMIMIC3(days=3, exclude_minors=True)

        patient = MockPatient(minor=True)
        samples = task(patient)

        # Verify there are no samples since the only sample was from a minor patient
        self.assertEqual(len(samples), 0)

    def test_single_sample(self):
        """Test task behavior with a single valid admission."""
        patient = MockPatient(los_days=10)
        samples = self.task(patient)

        self.assertEqual(len(samples), 1)
        self.assertIn("los", samples[0])
        self.assertEqual(samples[0]["los"], 1)


if __name__ == "__main__":
    unittest.main()
