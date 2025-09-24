from typing import List
from pyhealth.datasets import BaseEHRDataset

class EHRDatasetStatAssertion:
    
    def __init__(self, dataset: BaseEHRDataset, eps: float):
        self.dataset = dataset
        self.eps = eps
        # return self # builder
    
    def assertEHRStats(
        self,
        expected_num_patients: int,
        expected_num_visits: int,
        expected_num_visits_per_patient: float,
        expected_events_per_visit_per_table: List[float],
    ):
        self.assertNumPatients(expected_num_patients)
        self.assertNumVisits(expected_num_visits)
        self.assertMeanVisitsPerPatient(expected_num_visits_per_patient)
        self.assertTableMeans(expected_events_per_visit_per_table)
        
    def assertNumPatients(self, expected: int):
        actual = len(self.dataset.patients)
        if expected != actual:
            raise AssertionError(f"Expected {expected} patients got {actual}")

    def assertNumVisits(self, expected: int):
        actual = sum([len(patient) for patient in self.dataset.patients.values()])
        if expected != actual:
            raise AssertionError(f"Expected {expected} num visits got {actual}")
        
    def assertMeanVisitsPerPatient(self, expected: int):
        actual_visits = [len(patient) for patient in self.dataset.patients.values()]
        actual = sum(actual_visits) / len(actual_visits)
        if abs(expected - actual) > self.eps:
            raise AssertionError(f"Expected {expected} mean num visits got {actual}")
        
        
    # expected list must be ordered by tables
    def assertTableMeans(self, expected_per_table: List[float]):
        for expected_value, table in zip(expected_per_table, self.dataset.tables):
            actual_num_events = [
                len(v.get_event_list(table))
                for p in self.dataset.patients.values()
                for v in p
            ]

            actual_value = sum(actual_num_events) / len(actual_num_events)
            
            if abs(expected_value - actual_value) > self.eps:
                raise AssertionError(f"Expected {expected_value} mean for events in {table} got {actual_value}")