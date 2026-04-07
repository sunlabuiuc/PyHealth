import ast
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd

from pyhealth.tasks import BaseTask


class PTBXLMIClassificationTask(BaseTask):
    task_name = "ptbxl_mi_classification"
    input_schema = {
        "signal": "tensor",
    }
    output_schema = {
        "label": "binary",
    }

    def __init__(self, root: str):
        self.root = root

        scp_path = os.path.join(self.root, "scp_statements.csv")
        scp_df = pd.read_csv(scp_path, index_col=0)
        self.mi_codes = set(
            scp_df[scp_df["diagnostic_class"] == "MI"].index.astype(str).tolist()
        )

    def __call__(self, patient) -> List[Dict]:
        samples = []

        rows = patient.data_source.to_dicts()

        for idx, row in enumerate(rows):
            raw_label = row["ptbxl/scp_codes"]

            try:
                scp_codes = (
                    ast.literal_eval(raw_label)
                    if isinstance(raw_label, str)
                    else raw_label
                )
            except (ValueError, SyntaxError):
                scp_codes = {}

            label = 1 if any(code in self.mi_codes for code in scp_codes.keys()) else 0

            signal = np.zeros((12, 1000), dtype=np.float32)

            visit_id = str(row["ptbxl/ecg_id"])
            cache_dir = os.path.join("/tmp", "ptbxl_task_cache")
            os.makedirs(cache_dir, exist_ok=True)
            save_file_path = os.path.join(
                cache_dir, f"{patient.patient_id}-MI-{visit_id}.pkl"
            )

            with open(save_file_path, "wb") as f:
                pickle.dump({"signal": signal, "label": label}, f)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": visit_id,
                    "record_id": idx + 1,
                    "signal": signal.tolist(),
                    "label": label,
                    "epoch_path": save_file_path,
                }
            )

        return samples