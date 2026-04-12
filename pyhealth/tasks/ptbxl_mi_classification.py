import ast
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import wfdb

from pyhealth.tasks import BaseTask


class PTBXLMIClassificationTask(BaseTask):
    task_name = "ptbxl_mi_classification"
    input_schema = {
        "signal": "tensor",
    }
    output_schema = {
        "label": "binary",
    }

    def __init__(
        self,
        root: str,
        signal_length: int = 1000,
        normalize: bool = True,
    ):
        self.root = root
        self.signal_length = signal_length
        self.normalize = normalize

        scp_path = os.path.join(self.root, "scp_statements.csv")
        scp_df = pd.read_csv(scp_path, index_col=0)
        self.mi_codes = set(
            scp_df[scp_df["diagnostic_class"] == "MI"].index.astype(str).tolist()
        )

    def _load_ecg_signal(self, record_rel_path: str) -> np.ndarray:
        """Loads a PTB-XL WFDB record and returns shape (12, signal_length)."""
        record_path = os.path.join(self.root, record_rel_path)

        # WFDB expects the record path without file extension.
        signal, _ = wfdb.rdsamp(record_path)

        # rdsamp returns shape (num_samples, num_channels)
        signal = signal.T.astype(np.float32)  # -> (channels, time)

        if self.normalize:
            mean = signal.mean(axis=1, keepdims=True)
            std = signal.std(axis=1, keepdims=True)
            std = np.where(std < 1e-6, 1.0, std)
            signal = (signal - mean) / std

        current_len = signal.shape[1]
        if current_len >= self.signal_length:
            signal = signal[:, : self.signal_length]
        else:
            pad_width = self.signal_length - current_len
            signal = np.pad(signal, ((0, 0), (0, pad_width)), mode="constant")

        return signal

    def __call__(self, patient) -> List[Dict]:
        samples = []

        rows = patient.data_source.to_dicts()

        for idx, row in enumerate(rows):
            raw_label = row["ptbxl/scp_codes"]
            record_rel_path = row["ptbxl/record_path"]

            try:
                scp_codes = (
                    ast.literal_eval(raw_label)
                    if isinstance(raw_label, str)
                    else raw_label
                )
            except (ValueError, SyntaxError):
                scp_codes = {}

            label = 1 if any(code in self.mi_codes for code in scp_codes.keys()) else 0
            signal = self._load_ecg_signal(record_rel_path)

            visit_id = str(row["ptbxl/ecg_id"])

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": visit_id,
                    "record_id": idx + 1,
                    "signal": signal.tolist(),
                    "label": label,
                }
            )

        return samples