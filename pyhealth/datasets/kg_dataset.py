import os
import pandas as pd
from pyhealth.datasets import BaseDataset


class CustomRadiologyDataset(BaseDataset):
    """A simple CSV-based dataset for radiology reports.

    Expected CSV format:
        report_id: unique identifier
        report: free-text radiology report

    Example usage:
        dataset = CustomRadiologyDataset(
            root="path/to/csv/",
            csv_name="mimic_re ports.csv"
        )
    """

    def __init__(self, root: str, csv_name: str = "mimic_reports.csv"):
        self.csv_path = os.path.join(root, csv_name)

        df = pd.read_csv(self.csv_path)

        if "report_id" not in df.columns or "report" not in df.columns:
            raise ValueError("CSV must contain columns: report_id, report")

        samples = []
        for _, row in df.iterrows():
            samples.append({
                "patient_id": str(row["report_id"]),
                "visit_id": str(row["report_id"]),
                "text": row["report"]
            })

        super().__init__(samples=samples, dataset_name="custom_radiology")

    def get_all_labels(self):
        """Radiology dataset has no ground-truth labels."""
        return None
