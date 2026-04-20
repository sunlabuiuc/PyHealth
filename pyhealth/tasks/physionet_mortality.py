"""PhysioNet in-hospital mortality prediction task for PyHealth.

This module provides the in-hospital mortality prediction task for the
PhysioNet 2012 Challenge dataset (set A). The task dynamically bins
irregularly sampled time-series data into a fixed number of timesteps
based on the patient's last observation time. Normalization is deferred
to the ``DuETTTimeSeriesProcessor``.

Dataset paper:
    https://physionet.org/content/challenge-2012/1.0.0/

Author:
    Original implementation adapted for PyHealth.
"""

from typing import Any, Dict, List

import numpy as np

from pyhealth.data import Patient
from pyhealth.tasks.base_task import BaseTask


class PhysioNetMortalityTask(BaseTask):
    """In-hospital mortality prediction task for PhysioNet 2012.

    Dynamically bins time-series data into ``n_timesteps`` based on the
    patient's last observation time. Defers normalization to the
    ``DuETTTimeSeriesProcessor``.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
            - ``x_ts``: ``duett_ts`` (raw values + counts per bin)
            - ``x_static``: ``tensor`` (static features)
            - ``times``: ``tensor`` (normalized time bins)
        output_schema (Dict[str, str]): The output schema specifying outputs.
            - ``label``: ``binary`` (in-hospital death)

    Examples:
        >>> from pyhealth.datasets import PhysioNet2012Dataset
        >>> from pyhealth.tasks import PhysioNetMortalityTask
        >>> dataset = PhysioNet2012Dataset(
        ...     root="/path/to/physionet2012",
        ...     tables=["events", "outcomes"],
        ... )
        >>> task = PhysioNetMortalityTask(n_timesteps=32)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "PhysioNetMortalityTask"

    input_schema: Dict[str, str] = {
        "x_ts": "duett_ts",
        "x_static": "tensor",
        "times": "tensor",
    }
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self, n_timesteps: int = 32, **kwargs) -> None:
        """Initialize the PhysioNetMortalityTask.

        Args:
            n_timesteps: Number of time bins to generate. Defaults to 32.
            **kwargs: Additional keyword arguments for ``BaseTask``.
        """
        super().__init__(**kwargs)
        self.n_timesteps = n_timesteps

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes a patient to create in-hospital mortality samples.

        Args:
            patient: A Patient object containing PhysioNet 2012 data.

        Returns:
            List[Dict[str, Any]]: A list containing a single dictionary with
                - ``patient_id``: Patient identifier
                - ``x_ts``: Tuple of (raw time-series values, counts per bin)
                - ``x_static``: Static feature vector (8-dimensional)
                - ``times``: Normalized time bin centers
                - ``label``: Binary mortality label (0/1)

            Returns an empty list if the patient has no outcome or event data.
        """
        samples: List[Dict[str, Any]] = []

        # --- Outcome (label) ---
        outcomes = patient.get_events(event_type="outcomes")
        if not outcomes:
            return []

        label = int(float(getattr(outcomes[0], "in-hospital_death", 0)))

        # --- Time-series variables (36 variables) ---
        ts_vars = [
            "Albumin",
            "ALP",
            "ALT",
            "AST",
            "Bilirubin",
            "BUN",
            "Cholesterol",
            "Creatinine",
            "DiasABP",
            "FiO2",
            "GCS",
            "Glucose",
            "HCO3",
            "HCT",
            "HR",
            "K",
            "Lactate",
            "Mg",
            "MAP",
            "MechVent",
            "Na",
            "NIDiasABP",
            "NIMAP",
            "NISysABP",
            "PaCO2",
            "PaO2",
            "pH",
            "Platelets",
            "RespRate",
            "SaO2",
            "SysABP",
            "Temp",
            "TroponinI",
            "TroponinT",
            "Urine",
            "WBC",
        ]

        # Initialize containers
        x_ts_raw = np.zeros((self.n_timesteps, len(ts_vars)), dtype=np.float32)
        x_ts_counts = np.zeros((self.n_timesteps, len(ts_vars)), dtype=np.float32)
        x_static = np.zeros(8, dtype=np.float32)
        times = np.zeros(self.n_timesteps, dtype=np.float32)

        # --- Events ---
        events = patient.get_events(event_type="events")
        if not events:
            return []

        # Dynamic duration based on first/last timestamp
        base_time = min(e.timestamp for e in events)
        last_time = max(e.timestamp for e in events)
        total_duration_days = (last_time - base_time).total_seconds() / 86400.0
        if total_duration_days <= 0:
            total_duration_days = 1e-5

        for event in events:
            param = getattr(event, "parameter", None)
            val = float(getattr(event, "value", 0))

            # Static features
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

            # Time-series features
            elif param in ts_vars:
                idx = ts_vars.index(param)
                t_offset_days = (event.timestamp - base_time).total_seconds() / 86400.0

                # Clip to last bin if needed
                if t_offset_days >= total_duration_days:
                    bin_idx = self.n_timesteps - 1
                else:
                    bin_idx = int((t_offset_days / total_duration_days) * self.n_timesteps)

                bin_idx = max(0, min(bin_idx, self.n_timesteps - 1))

                x_ts_raw[bin_idx, idx] = val
                x_ts_counts[bin_idx, idx] += 1

        # Build normalized time bins
        for b in range(self.n_timesteps):
            times[b] = ((b + 1) / self.n_timesteps) * total_duration_days

        # Single sample per patient
        samples.append(
            {
                "patient_id": patient.patient_id,
                "x_ts": (x_ts_raw, x_ts_counts),
                "x_static": x_static,
                "times": times,
                "label": label,
            }
        )

        return samples