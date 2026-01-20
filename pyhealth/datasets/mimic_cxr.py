import pandas as pd
from pyhealth.datasets import BaseDataset


class MIMICCXRReportDataset(BaseDataset):
    """
    Dataset loader for MIMIC-CXR radiology reports.

    Each report is treated as a single visit with:
    - report_id (patient identifier surrogate)
    - report_text (free-text radiology report)
    - doc_label (binary abnormal / normal label)
    """

    def __init__(self, root, csv_path, id_col="report_id", text_col="report_text", label_col="doc_label"):
        super().__init__(root=root)
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        self.id_col = id_col
        self.text_col = text_col
        self.label_col = label_col

        self._create_patients()

    def _create_patients(self):
        for _, row in self.df.iterrows():
            patient_id = str(row[self.id_col])
            visit = {
                "report_text": str(row[self.text_col]),
                "label": int(row[self.label_col]),
            }
            self.add_patient(patient_id=patient_id, visits=[visit])
