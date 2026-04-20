import numpy as np
from typing import Any, Dict, List

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


class PhysioNetMortalityTask(BaseTask):
    """In-hospital mortality prediction task for PhysioNet 2012.

    Dynamically bins time-series data into n timesteps based on the patient's
    last observation time. Defers normalization to the DuETTTimeSeriesProcessor.

    Args:
        n_timesteps (int): Number of time bins to generate. Defaults to 32.
        **kwargs: Additional keyword arguments for BaseTask.
    """

    task_name: str = "PhysioNetMortalityTask"
    input_schema: Dict[str, str] = {
        "x_ts": "duett_ts",
        "x_static": "tensor",
        "times": "tensor",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, n_timesteps: int = 32, **kwargs):
        super().__init__(**kwargs)
        self.n_timesteps = n_timesteps

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes a patient to create mortality prediction samples.

        Args:
            patient (Patient): The patient to process.

        Returns:
            List[Dict[str, Any]]: Single element list containing the feature sample.
        """
        samples =[]
        outcomes = patient.get_events("outcomes")
        if not outcomes:
            return[]

        label = int(float(getattr(outcomes[0], "in-hospital_death", 0)))

        ts_vars =[
            "Albumin", "ALP", "ALT", "AST", "Bilirubin", "BUN", "Cholesterol",
            "Creatinine", "DiasABP", "FiO2", "GCS", "Glucose", "HCO3", "HCT",
            "HR", "K", "Lactate", "Mg", "MAP", "MechVent", "Na", "NIDiasABP",
            "NIMAP", "NISysABP", "PaCO2", "PaO2", "pH", "Platelets", "RespRate",
            "SaO2", "SysABP", "Temp", "TroponinI", "TroponinT", "Urine", "WBC",
        ]

        x_ts_raw = np.zeros((self.n_timesteps, len(ts_vars)), dtype=np.float32)
        x_ts_counts = np.zeros((self.n_timesteps, len(ts_vars)), dtype=np.float32)
        x_static = np.zeros(8, dtype=np.float32)
        times = np.zeros(self.n_timesteps, dtype=np.float32)

        events = patient.get_events("events")
        if not events:
            return[]

        base_time = min(e.timestamp for e in events)
        last_time = max(e.timestamp for e in events)

        # Calculate total duration dynamically per patient
        total_duration_days = (last_time - base_time).total_seconds() / 86400.0
        if total_duration_days <= 0:
            total_duration_days = 1e-5

        for event in events:
            param = getattr(event, "parameter", None)
            val = float(getattr(event, "value", 0))

            if param == "ICUType":
                icu_type = int(val)
                if 1 <= icu_type <= 4:
                    x_static[2 + icu_type] = 1.0
            elif param == "Age":
                x_static[0] = val
            elif param == "Gender":
                x_static[1] = val
            elif param == "Height":
                x_static[2] = val
            elif param == "Weight":
                x_static[7] = val
            elif param in ts_vars:
                idx = ts_vars.index(param)
                t_offset_days = (event.timestamp - base_time).total_seconds() / 86400.0

                if t_offset_days >= total_duration_days:
                    bin_idx = self.n_timesteps - 1
                else:
                    bin_idx = int((t_offset_days / total_duration_days) * self.n_timesteps)

                bin_idx = max(0, min(bin_idx, self.n_timesteps - 1))

                # Assign raw values and increment counts
                x_ts_raw[bin_idx, idx] = val
                x_ts_counts[bin_idx, idx] += 1

        for b in range(self.n_timesteps):
            times[b] = ((b + 1) / self.n_timesteps) * total_duration_days

        samples.append({
            "patient_id": patient.patient_id,
            "x_ts": (x_ts_raw, x_ts_counts),
            "x_static": x_static,
            "times": times,
            "label": label,
        })
        return samples