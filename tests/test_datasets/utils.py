import polars as pl

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
        expected_events_per_visit_per_table: dict[str, float],
    ):
        self.assertNumPatients(expected_num_patients)
        self.assertNumVisits(expected_num_visits)
        self.assertMeanVisitsPerPatient(expected_num_visits_per_patient)
        self.assertTableMeans(expected_events_per_visit_per_table)

    def assertNumPatients(self, expected: int):
        actual = len(self.dataset.unique_patient_ids)
        if expected != actual:
            raise AssertionError(f"Expected {expected} patients got {actual}")

    def assertNumVisits(self, expected: int):
        actual = sum(len(patient.get_events("admissions")) for patient in self.dataset.iter_patients())
        if expected != actual:
            raise AssertionError(f"Expected {expected} num visits got {actual}")

    def assertMeanVisitsPerPatient(self, expected: int):
        actual_visits = [len(patient.get_events("admissions")) for patient in self.dataset.iter_patients()]
        actual = sum(actual_visits) / len(actual_visits)
        if abs(expected - actual) > self.eps:
            raise AssertionError(f"Expected {expected} mean num visits got {actual}")


    # expected list must be ordered by tables
    def assertTableMeans(self, expected_per_table: dict[str, float]):
        for table, expected_value in expected_per_table.items():
            group_by_col = pl.col(f"{table}/hadm_id")
            actual_value = (
                self.dataset.collected_global_event_df
                .filter(pl.col("event_type") == table)
                .drop_nulls(group_by_col)
                .group_by(group_by_col)
                .len()
                .mean()
                .item(row=0, column="len")
            )

            if abs(expected_value - actual_value) > self.eps:
                raise AssertionError(f"Expected {expected_value} mean for events in {table} got {actual_value}")