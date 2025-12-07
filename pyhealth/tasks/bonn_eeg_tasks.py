from typing import Dict, List, Any
from pyhealth.tasks import BaseTask
import numpy as np

class BonnEEGSeizureDetection(BaseTask):
    """
	Binary classification task for Seizure Detection on Bonn EEG.
    
    This task streams signal data from disk on-the-fly to maintain low memory usage.
    It classifies 'Set E' (S) as Seizure (1) and all other sets as Normal/Interictal (0).
    """
    task_name: str = "SeizureDetection"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(self):
        super().__init__()

    def __call__(self, sample: Any) -> List[Dict[str, Any]]:
        samples = []
        events = sample.get_events('index')
        if len(events) != 1:
            return samples
        event =  events[0]
        fpath = getattr(event, "filepath")
        
        # We read the file only now, when the task requests it streaming it.
        try:
            signal = np.loadtxt(fpath).reshape(1, -1)
        except Exception:
            return []

        # 3. Process Label
        label_class = getattr(event, "label_class")
        label = 1 if label_class == "S" else 0

        samples.append({
            "patient_id": sample.patient_id,
            "visit_id": "0",
            "signal": signal,
            "label": int(label),
        })

        return samples