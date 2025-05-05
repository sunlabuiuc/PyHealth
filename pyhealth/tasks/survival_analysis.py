from datetime import datetime
from typing import Any, Dict, List, Optional
from .base_task import BaseTask
from typing import Dict, Any, List

class SurvivalAnalysisGBSG(BaseTask):
    """Task for survival analysis using GBSG dataset with supported processor types."""
    
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
        """Processes a single patient for survival analysis."""
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
    """Alternative formulation with separate time-to-event prediction.
    
    Predicts:
    - Will recurrence/death occur within X days? (binary)
    - Time until event (regression)
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