# Name (s): Chris Yu, Jimmy Lee
# NetId (s) (If applicable for UIUC students): hmyu2, jl279
# The paper title : Revisit Deep Cox Mixtures For Survival Regression
# The paper link: https://github.com/chrisyu-uiuc/revisit-deepcoxmixtures-cs598-uiuc/blob/main/Revisit_DeepCoxMixuresForSurvivalRegression.pdf
# Implementation of the SurvivalAnalysisGBSG and TimeToEventGBSG task classes for defining survival analysis tasks on the GBSG dataset within the pyhealth framework.

from datetime import datetime
from typing import Any, Dict, List, Optional
from .base_task import BaseTask
from typing import Dict, Any, List

class SurvivalAnalysisGBSG(BaseTask):
    """
    Task for survival analysis using GBSG dataset with supported processor types.

    This task defines the input and output schema for performing standard
    survival analysis on the GBSG dataset, typically predicting time-to-event
    and event status.
    """    
    task_name: str = "SurvivalAnalysisGBSG"
    input_schema: Dict[str, str] = {
        "age": "sequence",       # Numerical features
        "meno": "binary",        # Binary features
        "size": "sequence",      # Numerical features
        "grade": "sequence",     # Treat grade as numerical sequence
        "nodes": "sequence",     # Numerical features
        "pgr": "sequence",       # Numerical features
        "er": "sequence",        # Numerical features
        "hormon": "binary"       # Binary features
    }
    output_schema: Dict[str, str] = {
        "event": "binary",       # Binary output
        "time": "sequence"       # Numerical output
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Processes a single patient record from the GBSG dataset for survival analysis.

        This function extracts relevant features, the time-to-event, and the event
        indicator from a patient's GBSG record and formats them into a sample
        dictionary according to the task's output schema.

        Args:
            patient: An object representing a single patient, expected to have
                     a method `get_events` and attributes accessible via records.

        Returns:
            A list containing a single dictionary representing the processed
            sample for the patient, or an empty list if no GBSG records are found
            or an error occurs during processing.

        Example Usage:
            # This function is typically called internally by the pyhealth dataset
            # processing pipeline after setting the task on a dataset instance.
            # Example (conceptual):
            # dataset = GBSGDataset(...)
            # task_dataset = dataset.set_task(SurvivalAnalysisGBSG())
            # sample = task_dataset.samples[0] # Accessing a processed sample
        """
        samples = []
        
        # Get all GBSG records for this patient
        records = patient.get_events(event_type="gbsg")
        
        if not records:
            return []
        
        record = records[0]  
        
        try:
            samples.append({
                "age": [float(record.age)],
                "meno": int(record.meno),
                "size": [float(record.size)],
                "grade": [int(record.grade)],  # Convert to int and wrap in list
                "nodes": [float(record.nodes)],
                "pgr": [float(getattr(record, 'pgr', 0))],
                "er": [float(getattr(record, 'er', 0))],
                "hormon": int(record.hormon),
                "patient_id": patient.patient_id,
                "event": int(record.status),
                "time": [float(record.rfstime)]
            })
            
        except (AttributeError, ValueError) as e:
            print(f"Error processing patient {patient.patient_id}: {e}")
        
        return samples
        
class TimeToEventGBSG(BaseTask):
    """
    Alternative formulation with separate time-to-event prediction.

    This task provides an alternative way to structure the GBSG data,
    formulating it for predicting both a binary event within a specific
    timeframe (e.g., 5 years) and the actual survival time (regression).
    """
    
    task_name: str = "TimeToEventGBSG"
    input_schema: Dict[str, str] = {
        "age": "numerical",
        "size": "numerical",
        "nodes": "numerical",
        "hormon": "binary"
    }
    output_schema: Dict[str, str] = {
        "event_within_5yrs": "binary",  # 1 if event occurs within 5 years
        "survival_time": "numerical"    # Actual survival time (days)
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """
        Processes a single patient record from the GBSG dataset for time-to-event prediction.

        This function extracts a subset of features and calculates two output targets:
        a binary indicator for whether an event occurred within 5 years and the
        actual survival time. It formats these into a sample dictionary.

        Args:
            patient: An object representing a single patient, expected to have
                     a method `get_events` and attributes accessible via records.

        Returns:
            A list containing a single dictionary representing the processed
            sample for the patient, or an empty list if no GBSG records are found
            or an error occurs during processing.

        Example Usage:
            # This function is typically called internally by the pyhealth dataset
            # processing pipeline when setting this task on a dataset instance.
            # Example (conceptual):
            # dataset = GBSGDataset(...)
            # task_dataset = dataset.set_task(TimeToEventGBSG())
            # sample = task_dataset.samples[0] # Accessing a processed sample
        """
        samples = []
        records = patient.get_events(event_type="gbsg")
        
        if not records:
            return []
        
        record = records[0]
        
        try:
            # Convert days to years for clinical relevance
            years_to_event = float(record.rfstime) / 365.25
            event_occurred = int(record.status)
            
            # Create both classification and regression targets
            samples.append({
                "age": float(record.age),
                "size": float(record.size),
                "nodes": float(record.nodes),
                "hormon": int(record.hormon),
                "patient_id": patient.patient_id,
                "event_within_5yrs": 1 if (event_occurred and years_to_event <= 5) else 0,
                "survival_time": float(record.rfstime)
            })
            
        except Exception as e:
            print(f"Error processing patient {patient.patient_id}: {e}")
            
        return samples
