"""Patient-Level Drug Recommendation on MIMIC-III.

This example demonstrates a PATIENT-LEVEL task where we recommend drugs
for the CURRENT visit based on patient history from the PREVIOUS visit.

Key patterns:
- MIMIC3Dataset with tables (uppercase table names)
- Requires multiple visits per patient (len(visits) > 1)
- Iterates starting from visit 1 (i=1 to n)
- Features: current conditions/procedures + previous drug history
- Label: drugs prescribed at current visit (multilabel output)
"""

import os
from typing import Any, Dict, List

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.medcode import CrossMap
from pyhealth.tasks import BaseTask

# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class DrugRecommendationTask(BaseTask):
    """Patient-level drug recommendation task for MIMIC-III.
    
    Recommends drugs for current visit based on patient's history.
    Uses previous visit's medications as historical context.
    """

    task_name: str = "DrugRecommendationTask"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs_history": "sequence",  # From previous visit
    }

    output_schema: Dict[str, str] = {"drugs": "multilabel"}

    def __init__(self, **kwargs):
        # Load inside __init__, not at module scope: at module scope this
        # downloads on import and re-runs in every set_task worker process.
        super().__init__(**kwargs)
        self.ndc_to_atc = CrossMap.load("NDC", "ATC")

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return drug recommendation samples.
        
        PATIENT-LEVEL PATTERN:
        - Requires at least 2 visits
        - For visit i, use history from visit i-1
        - Uses temporal relationships between visits
        - Multilabel output (multiple drugs per visit)
        """
        samples = []

        visits = patient.get_events(event_type="admissions")

        # Patient-level task requires multiple visits
        if len(visits) <= 1:
            return []

        # Start from visit 1 (need previous visit for history)
        for i in range(1, len(visits)):
            prev_visit = visits[i - 1]
            curr_visit = visits[i]

            # Features: Current visit diagnoses and procedures
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", curr_visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", curr_visit.hadm_id)],
            )

            # Feature: Drug history from PREVIOUS visit
            prev_prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", prev_visit.hadm_id)],
            )

            # Label: Drugs prescribed at CURRENT visit
            curr_prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", curr_visit.hadm_id)],
            )

            # Extract features
            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            def _to_atc3(ndc):
                mapped = self.ndc_to_atc.map(ndc, target_kwargs={"level": 3})
                return mapped[0] if mapped else None

            drugs_history = [
                a for a in (
                    _to_atc3(e.ndc) for e in prev_prescriptions if e.ndc
                ) if a
            ]
            drugs_target = [
                a for a in (
                    _to_atc3(e.ndc) for e in curr_prescriptions if e.ndc
                ) if a
            ]

            # Skip if insufficient data
            if len(conditions) == 0 or len(drugs_target) == 0:
                continue

            samples.append(
                {
                    "visit_id": curr_visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs_history": drugs_history,
                    "drugs": drugs_target,  # Multilabel target
                }
            )

        return samples


# =============================================================================
# EXAMPLE USAGE
# =============================================================================
if __name__ == '__main__':
    # Dataset initialization
    DATA_ROOT = "/srv/local/data/physionet.org/files/mimiciii/1.4"
    CACHE_DIR = "./cache"
    TASK_CACHE_DIR = "./task_cache"

    base_dataset = MIMIC3Dataset(
        root=DATA_ROOT,
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir=CACHE_DIR,
        dev=False,
    )

    # Apply task
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    task = DrugRecommendationTask()
    sample_dataset = base_dataset.set_task(
        task,
        num_workers=num_workers
    )

    # Verify samples
    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        print(f"First sample keys: {list(sample_dataset[0].keys())}")
        print(f"First sample: {sample_dataset[0]}")
