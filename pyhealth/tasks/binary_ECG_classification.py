from pyhealth.tasks.base_task import BaseTask
from typing import List, Dict
import polars as pl
import torch


class BinaryECGClassification(BaseTask):
    task_name = "binary_ecg_classification"
    input_schema = {"signal": "float32"}  # could be more specific
    output_schema = {"label": "int64"}

    def __init__(self, dataset):
        self.dataset = dataset
        self.samples = []

    def preprocess(self, limit: int = None):
        """Build model-ready (x, y) pairs."""
        self.samples = []
        for i, sample in enumerate(self.dataset.samples):
            if limit is not None and i >= limit:
                break
            signal = torch.tensor(sample["signal"], dtype=torch.float32)
            label = torch.tensor(sample["label"], dtype=torch.long)
            self.samples.append({"signal": signal, "label": label})

    def __call__(self, patient) -> List[Dict]:
        """
        Required by BaseTask — PyHealth uses this for patient-wise event generation.
        For PTBXL, each 'patient' is really a single ECG recording.
        """
        signal = torch.tensor(patient["signal"], dtype=torch.float32)
        label = torch.tensor(patient["label"], dtype=torch.long)
        return [{"signal": signal, "label": label}]
    def __getitem__(self, index):
        return self.samples[index]
    def __len__(self):
        return len(self.samples)