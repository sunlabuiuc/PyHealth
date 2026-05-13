import os
import pandas as pd
from typing import Dict, List, Optional
from pyhealth.datasets import BaseDataset

class MIMICCXRLongitudinalDataset(BaseDataset):
    """MIMIC-CXR dataset registry for longitudinal multi-modal analysis."""

    def __init__(
        self,
        root: str,
        tables: List[str] = ["metadata", "chexpert"],
        dataset_name: Optional[str] = "MIMIC-CXR-Longitudinal",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            # Fixing the __file__ issue for Jupyter/VS Code environments
            try:
                base_path = os.path.dirname(os.path.abspath(__file__))
            except NameError:
                base_path = os.getcwd()
            config_path = os.path.join(base_path, "configs", "mimic_cxr_longitudinal.yaml")
            
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            config_path=config_path,
            **kwargs,
        )

    def parse_tables(self) -> Dict[int, List[Dict]]:
        """
        The actual implementation logic: Merges tables and extracts the 14 labels.
        """
        # 1. Load files
        df_meta = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-metadata.csv.gz"))
        df_labels = pd.read_csv(os.path.join(self.root, "mimic-cxr-2.0.0-chexpert.csv.gz"))

        # 2. Define the 14 categories specifically required for the assignment
        label_cols = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 
            'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 
            'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]

        # 3. Align metadata with labels
        combined_df = pd.merge(df_meta, df_labels, on=["subject_id", "study_id"])

        # 4. Group by patient and sort chronologically
        patients = {}
        for subject_id, group in combined_df.groupby("subject_id"):
            group = group.sort_values("study_id")
            
            visits = []
            for _, row in group.iterrows():
                visits.append({
                    "study_id": int(row["study_id"]),
                    "image_path": os.path.join(str(subject_id), f"{row['study_id']}.jpg"),
                    # This line is where the 14 labels are extracted into a vector
                    "label": row[label_cols].values.astype(int).tolist()
                })
            
            patients[int(subject_id)] = visits

        return patients