"""In-Hospital Mortality task for TLS-preprocessed MIMIC-III data.

This module defines the IHM (In-Hospital Mortality) prediction task for
MIMIC-III data preprocessed by the TLS pipeline. The task constructs
dense time-series samples from the preprocessed gridded data.

Reference:
    Kuznetsova et al., "On the Importance of Step-wise Embeddings for
    Heterogeneous Clinical Time-Series", JMLR 2023.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from .base_task import BaseTask


class InHospitalMortalityTLS(BaseTask):
    """In-Hospital Mortality prediction from TLS-preprocessed MIMIC-III.

    This task collects all timestep events for a patient's ICU stay,
    constructs a dense time-series tensor of shape ``(T, F)``, and
    extracts a binary in-hospital mortality label.

    The task supports ablations via configurable observation windows
    and feature subsets.

    Args:
        observation_hours: Number of hours to include in the observation
            window. The TLS pipeline uses a 1-hour grid, so this equals
            the number of timesteps. Default is 48 (matching the
            paper's IHM setup).
        feature_subset: Optional list of feature indices to include.
            If ``None``, all 42 features are used. Useful for ablation
            studies testing performance with different feature sets.

    Examples:
        >>> from pyhealth.datasets import MIMIC3TLSDataset
        >>> from pyhealth.tasks import InHospitalMortalityTLS
        >>> dataset = MIMIC3TLSDataset(root="/path/to/tls_output/")
        >>> task = InHospitalMortalityTLS(observation_hours=48)
        >>> samples = dataset.set_task(task)

        >>> # 24-hour ablation
        >>> task_24h = InHospitalMortalityTLS(observation_hours=24)
        >>> samples_24h = dataset.set_task(task_24h)

        >>> # Feature subset ablation (monitored features only)
        >>> task_monitored = InHospitalMortalityTLS(
        ...     feature_subset=[2, 3, 5, 6, 7, 8, 9, 10]
        ... )
    """

    task_name: str = "InHospitalMortalityTLS"
    input_schema: Dict[str, str] = {"time_series": "tensor"}
    output_schema: Dict[str, str] = {"ihm": "binary"}

    def __init__(
        self,
        observation_hours: int = 48,
        feature_subset: Optional[List[int]] = None,
    ):
        self.observation_hours = observation_hours
        self.feature_subset = feature_subset

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a single patient into IHM prediction samples.

        Extracts dense time-series features and the IHM label from the
        patient's timestep events.

        Args:
            patient: A :class:`~pyhealth.data.Patient` object containing
                TLS-preprocessed timestep events.

        Returns:
            A list containing zero or one sample dictionaries. Each
            sample has keys ``"patient_id"``, ``"time_series"`` (a
            list-of-lists of shape ``(T, F)``), and ``"ihm"`` (0 or 1).
        """
        events = patient.get_events(event_type="timeseries")
        if not events:
            return []

        # Sort events by timestamp
        events = sorted(events, key=lambda e: e.timestamp)

        # Truncate to observation window
        events = events[: self.observation_hours]

        if len(events) < 2:
            return []

        # Extract IHM label (same for all timesteps of a stay)
        try:
            ihm_label = int(float(events[0].ihm_label))
        except (ValueError, TypeError, AttributeError):
            return []

        if ihm_label not in (0, 1):
            return []

        # Build the feature names list (matching YAML attribute order)
        from pyhealth.datasets.mimic3_tls import MIMIC3TLSDataset

        feature_names = MIMIC3TLSDataset.FEATURE_NAMES

        # Extract feature values for each timestep
        values = []
        for event in events:
            row = []
            for name in feature_names:
                try:
                    val = float(getattr(event, name))
                except (ValueError, TypeError, AttributeError):
                    val = 0.0
                if np.isnan(val):
                    val = 0.0
                row.append(val)
            values.append(row)

        values_arr = np.array(values, dtype=np.float64)

        # Apply feature subset if specified
        if self.feature_subset is not None:
            values_arr = values_arr[:, self.feature_subset]

        return [
            {
                "patient_id": patient.patient_id,
                "time_series": values_arr.tolist(),
                "ihm": ihm_label,
            }
        ]
