import os
import pandas as pd
from typing import List, Optional

class SeizureDataset:
    """Dataset loader for Seizure Patient Information.

    This class loads and filters clinical information from the CSV file
    available at https://zenodo.org/records/2547147.

    Patients are identified as having seizures if the 'Primary Localisation'
    column is non-empty.

    Args:
        root: Directory where `clinical_information.csv` is stored.
        filename: Name of the CSV file (default: 'clinical_information.csv').

    Attributes:
        all_data: Full DataFrame loaded from CSV.
        seizure_patients: Filtered DataFrame of patients with seizure events.
        seizure_patient_ids: List of patient IDs with seizures.

    Example:
        >>> dataset = SeizureDataset(root="/path/to/downloaded/data")
        >>> dataset.stat()
        >>> print(dataset.seizure_patient_ids)
    """

    def __init__(self, root: str, filename: str = "clinical_information.csv"):
        self.root = root
        self.filepath = os.path.join(root, filename)
        self.all_data = None
        self.seizure_patients = None
        self.seizure_patient_ids: Optional[List[str]] = None

        self.load_and_filter()

    def load_and_filter(self):
        # Load the CSV
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"File not found: {self.filepath}")

        self.all_data = pd.read_csv(self.filepath)

        # Filter for seizure patients (Primary Localisation is not blank)
        self.seizure_patients = self.all_data[
            self.all_data['Primary Localisation'].notna() &
            (self.all_data['Primary Localisation'].str.strip() != '')
        ]

        # Store Patient IDs
        self.seizure_patient_ids = self.seizure_patients['Patient ID'].tolist()

    def stat(self):
        print(f"Total patients in dataset: {len(self.all_data)}")
        print(f"Patients with seizure events: {len(self.seizure_patient_ids)}")

    def info(self):
        print("Sample seizure patients:")
        print(self.seizure_patients[['Patient ID', 'Primary Localisation']].head())


if __name__ == "__main__":
    dataset = SeizureDataset(root="/path/to/downloaded/data")
    dataset.stat()
    dataset.info()