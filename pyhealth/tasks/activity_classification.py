"""Activity classification for PyHealth.

This module provides a task for classifying daily and sports activity using
motion sensor data from Daily and Sports Activities (DSA) dataset.
"""
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from .base_task import BaseTask

logger = logging.getLogger(__name__)

class ActivityClassification(BaseTask):
    """Task for classifying activity using motion sensor data.

    This task classifies which activity a patient is performing over a time 
    period from multi-sensor time-series data.

    Attributes:
        task_name (str): The name of the task.
        input_schema (Dict[str, str]): The input schema specifying required inputs.
        output_schema (Dict[str, str]): The output schema specifying outputs.

    Note:
        Each time-series is sampled over 5 seconds with 25 Hz frequency.

    Examples:
        >>> from pyhealth.datasets import DSADataset
        >>> from pyhealth.tasks import ActivityClassification
        >>> dataset = DSADataset(root="/path/to/dsa")
        >>> task = ActivityClassification()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ActivityClassification"
    input_schema: Dict[str, str] = {
        "T": "sequence",
        "RA": "sequence",
        "LA": "sequence",
        "RL": "sequence",
        "LL": "sequence",
    }
    output_schema: Dict[str, str] = {
        "label": "text",
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process daily and sports activities data for activity classification.

        Args:
            patient: A patient object containing activities data.

        Returns:
            List[Dict[str, Any]]: A list containing dictionaries with
            patient features and activity label. 

        Note:
            Returns empty list for patients with:
            - No activities
        """
        events = patient.get_events(event_type="activities")

        if len(events) == 0:
            return []
        
        df = pd.DataFrame([e.attr_dict for e in events])

        def extract_time_series(df, sensor):
            columns = [f"{d}" for d in range(125)]
            return df[df["sensor"].str.startswith(sensor + "_")][columns].reset_index(drop=True).sort_values("sensor").T.to_numpy()
        
        records = []
        for a in df["activity"].unique(): 
            for s in df["segment"].unique(): 

                df_one = df.query(f'activity == "{a}"').query(f'segment == "{s}"')
                
                if len(df_one) == 0: 
                    continue

                record = {
                    sensor: extract_time_series(df_one, sensor) for sensor in ["T", "LA", "RA", "LL", "RL"]
                }
                record["patient_id"] = patient.patient_id
                record["label"] = a
                records.append(record)
        
        return records
