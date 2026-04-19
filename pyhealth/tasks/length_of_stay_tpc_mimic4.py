from __future__ import annotations

from pyhealth.tasks import BaseTask
from pyhealth.data import Event, Patient
from typing import List, Dict, Any, Type, Union, cast

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

from pyhealth.data.data import Patient
from pyhealth.tasks.base_task import BaseTask

from pyhealth.processors import TemporalTimeseriesProcessor, TensorProcessor, SequenceProcessor
from pyhealth.processors.base_processor import Processor
import polars as pl

@dataclass
class RemainingLOSConfig:
    prediction_step_size: int = 1
    min_history_hours: int = 5
    min_remaining_hours: int = 1
    max_history_hours: int = 366

# Formally, our task is to predict the remaining LoS at regular
# timepoints 𝑦1, . . . , 𝑦𝑇 ∈ R>0 in the patient’s ICU stay, up to the
# discharge time 𝑇 , using the diagnoses (d ∈ R𝐷×1), static features
# (s ∈ R𝑆×1), and time series (x1, . . . , x𝑇 ∈ R𝐹 ×2). Initially, for every
# timepoint 𝑡, there are two ‘channels’ per time series feature: 𝐹 fea-
# ture values (x′𝑡 ∈ R𝐹 ×1), and their corresponding decay indicators
# (x′′𝑡 ∈ R𝐹 ×1). The decay indicators tell the model how recently
# the observation x′𝑡 was recorded.

class RemainingLOSMIMIC4(BaseTask):
    """
    Custom remaining length-of-stay regression task for MIMIC-IV.

    Each sample corresponds to one prediction cutoff time within a stay.
    Input:
        - ts: (timestamps, values) where values is shape (T, F)
        - optionally static / code features
    Target:
        - remaining_los_hours: float
    """

    task_name: str = "RemainingLOSMIMIC4"

    # Keep this conceptual unless you already know the exact schema names
    # your installed PyHealth version expects.
    input_schema: Dict[str, Any]  = {
        "timeseries": TensorProcessor(),
        "static": TensorProcessor(),
        "conditions": SequenceProcessor(),
    }

    output_schema: Dict[str, Any] = {"los": "tensor"}

    def __init__(self, config: Optional[RemainingLOSConfig] = None):
        self.config = config or RemainingLOSConfig()

        # TODO: These need to be fixed
        # 17 vitals from chartevents
        self.chart_itemids = [
            "220045", "220210", "220277", "220179", "220180", "220181", 
            "220050", "220051", "220052", "223761", "220739", "223900", 
            "223901", "226253", "220235", "224690", "220339"
        ]
        # 17 lab items from labevents
        self.lab_itemids = [
            "51006", "50912", "50931", "50902", "50882", "50868", 
            "50960", "50970", "51265", "51301", "50811", "51222", 
            "50813", "50820", "50818", "50821", "50825"
        ]
        self.all_itemids = self.chart_itemids + self.lab_itemids
        self.F = len(self.all_itemids)


    def __call__(self, patient: Patient) -> List[Dict]:
        samples: List[Dict] = []

        admissions_result = patient.get_events(event_type="icustays")
        # Handle both DataFrame and List[Event] return types
        if isinstance(admissions_result, list):
            admissions = admissions_result
        else:
            return []
        
        if len(admissions) == 0:
            return []
        
        patient_static_attributes = patient.get_events("patients")[0]
        
        static = np.array([patient_static_attributes['anchor_age'], 1. if patient_static_attributes['gender'] == 'F' else 0.], dtype=np.float32)

        for admission in admissions:

            admit_time = admission.timestamp
            # outtime is usually a string in attributes
            outtime_raw = admission.outtime
            discharge_time = datetime.strptime(outtime_raw, "%Y-%m-%d %H:%M:%S")

            los_hours = (discharge_time - admit_time).total_seconds() / 3600.0
            T = min(int(math.ceil(los_hours)), self.config.max_history_hours)

            if admit_time is None or discharge_time is None:
                continue
            if discharge_time <= admit_time:
                continue
            if los_hours < self.config.min_history_hours + self.config.min_remaining_hours:
                continue

            labevents = patient.get_events(
                event_type="labevents",
                start=admission.timestamp,
                end=discharge_time,
            )
            chartevents = patient.get_events(
                event_type="chartevents",
                start=admission.timestamp,
                end=discharge_time,
            )
            diagnoses_events = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", admission.hadm_id)],
            )
            conditions = [
                f"{getattr(event, 'icd_version', '10')}_{event.icd_code}" 
                for event in diagnoses_events if hasattr(event, "icd_code")
            ]

            all_events = labevents + chartevents

            if len(all_events) == 0:
                continue

            values_mat = np.zeros((self.F, T), dtype=np.float32)
            masks_mat = np.zeros((self.F, T), dtype=np.float32)
            
            id_to_idx = {id: i for i, id in enumerate(self.all_itemids)}
            # Pivot events into matrix
            for row in all_events:
                if row["itemid"] not in id_to_idx:
                    continue
                f_idx = id_to_idx.get(row["itemid"])
                h_idx = int((row["timestamp"].timestamp() - admit_time.timestamp() ) // 3600)
                if f_idx is not None and 0 <= h_idx < T:
                    values_mat[f_idx, h_idx] = row["valuenum"]
                    masks_mat[f_idx, h_idx] = 1.0

            # Forward-fill and calculate decay indicators
            # Decay = 0.75 ** (hours since last measurement)
            decay_mat = np.zeros((self.F, T), dtype=np.float32)
            for f in range(self.F):
                last_val = 0.0
                hours_since = math.inf # Initial large value for decay
                for t in range(T):
                    if masks_mat[f, t] > 0:
                        last_val = values_mat[f, t]
                        hours_since = 0.0
                    else:
                        values_mat[f, t] = last_val
                        hours_since += 1.0
                    decay_mat[f, t] = 0.75 ** hours_since

            # Elapsed time channel
            elapsed = np.arange(T, dtype=np.float32).reshape(1, T)

            # hour_of_day channel
            hour_of_day = np.array([
                (admit_time + timedelta(hours=t)).hour
                for t in range(T)
            ], dtype=np.float32).reshape(1, T)

            # Concatenate all channels: [elapsed (1), values (F), decays (F), hour_of_day (1)] -> (2F+2, T)
            timeseries = np.concatenate([elapsed, values_mat, decay_mat, hour_of_day], axis=0)
            
            # Label sequence: remaining LoS in hours at each hour
            labels = np.array([
                max(0.0, (discharge_time - (admit_time + timedelta(hours=t))).total_seconds() / (3600.0))
                for t in range(T)
            ], dtype=np.float32)

            samples.append({
                "patient_id": patient.patient_id,
                "visit_id": admission.stay_id ,
                "timeseries": timeseries,
                "static": static,
                "conditions": conditions,
                "los": labels,
            })

        return samples
