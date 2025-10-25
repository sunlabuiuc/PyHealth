import unittest
from pathlib import Path

from pyhealth.datasets import OMOPDataset


class TestOMOPDataset(unittest.TestCase):
    """Test OMOP dataset with local test data."""

    def setUp(self):
        """Set up test data and dataset."""
        self.test_data_path = (
            Path(__file__).parent.parent.parent / "test-resources" / "omop"
        )

        # Check if test data exists
        if not self.test_data_path.exists():
            self.skipTest("OMOP test data not found in test-resources/omop/")

        # Load dataset with all available tables
        self.tables = [
            "person",
            "visit_occurrence",
            "death",
            "condition_occurrence",
            "procedure_occurrence",
            "drug_exposure",
            "measurement",
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


if __name__ == "__main__":
    unittest.main()
