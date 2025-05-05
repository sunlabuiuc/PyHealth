import pandas as pd
from pyhealth.datasets import SampleEHRDataset  # EHR-style sample dataset

class NFGDataset(SampleEHRDataset):
    """A custom flat CSV dataset for PyHealth, with raw inputs and a binary label."""

    def __init__(self, csv_path: str, dataset_name: str = "NFGDataset", task_name: str = ""):
        df = pd.read_csv(csv_path)

        samples = []
        for idx, row in df.iterrows():
            sample = {
                "patient_id": str(idx),
                "visit_id": str(idx),
                "age": row["age"],
                "sex": row["sex"],
                "cr_type": int(row["cr_type"]),
            }
            for col in ["time", "event"]:
                if col in row:
                    sample[col] = row[col]
            samples.append(sample)

        self.samples = samples
        super().__init__(samples=samples, dataset_name=dataset_name, task_name=task_name)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]
