import ast
import os
import pickle
from typing import Dict, List

import numpy as np

from pyhealth.tasks import BaseTask


class PTBXLMIClassificationTask(BaseTask):
    task_name = "ptbxl_mi_classification"
    input_schema = {
        "signal": "timeseries",
    }
    output_schema = {
        "label": "binary",
    }

    def __call__(self, patient) -> List[Dict]:
        samples = []

        patient_df = patient.data_source
        rows = patient_df.to_dicts()

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

            label = 1 if "MI" in scp_codes else 0

            signal = np.zeros((12, 1000), dtype=np.float32)

            visit_id = str(row["ptbxl/ecg_id"])
            cache_dir = os.path.join("/tmp", "ptbxl_task_cache")
            os.makedirs(cache_dir, exist_ok=True)
            save_file_path = os.path.join(
                cache_dir, f"{patient.patient_id}-MI-{visit_id}.pkl"
            )

            with open(save_file_path, "wb") as f:
                pickle.dump(
                    {
                        "signal": signal,
                        "label": label,
                    },
                    f,
                )

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