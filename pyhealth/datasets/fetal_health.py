import os
import pandas as pd
from typing import Optional
from pyhealth.datasets import BaseDataset


class FetalHealthDataset(BaseDataset):
    """Fetal Health Classification Dataset from Kaggle.

    Reduction of child mortality is reflected in several of the United Nations' Sustainable Development Goals 
    and is a key indicator of human progress.
    The UN expects that by 2030, countries end preventable deaths of newborns and children under 5 years of age,
    with all countries aiming to reduce under‑5 mortality to at least as low as 25 per 1,000 live births.

    Parallel to notion of child mortality is of course maternal mortality, which accounts for 295 000 deaths during and following pregnancy and childbirth (as of 2017).
    The vast majority of these deaths (94%) occurred in low-resource settings, and most could have been prevented.
    In light of what was mentioned above, Cardiotocograms (CTGs) are a simple and cost accessible option to assess fetal health, allowing healthcare professionals to take action in order to prevent child and maternal mortality.
    The equipment itself works by sending ultrasound pulses and reading its response, thus shedding light on fetal heart rate (FHR), fetal movements, uterine contractions and more.

    Data
    This dataset contains 2126 records of features extracted from Cardiotocogram exams, which were then classified by 
    three expert obstetritians into 3 classes:

    1) Normal
    2) Suspect
    3) Pathological

    https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification
    """

    def __init__(self, root: str, dev: bool = False):
        super().__init__(dataset_name="fetal_health", root=root, dev=dev)
        self.data = self.load_data()
        self.parse_samples()

    def load_data(self):
        file_path = os.path.join(self.root, "/fetal_health.csv")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        return pd.read_csv(file_path)

    def parse_samples(self):
        for idx, row in self.data.iterrows():
            features = row.drop("fetal_health").to_dict()
            label = int(row["fetal_health"])
            self.add_sample(
                sample_id=str(idx),
                patient_id=str(idx),
                admission_id=str(idx),
                visit_id=str(idx),
                events=features,
                label=label,
            )
