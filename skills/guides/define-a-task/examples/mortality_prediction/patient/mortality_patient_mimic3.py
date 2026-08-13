"""Patient-Level Mortality Prediction on MIMIC-III.

This example demonstrates a PATIENT-LEVEL task where we predict mortality
at the NEXT visit based on data from the CURRENT visit.

Key patterns:
- MIMIC3Dataset with tables (uppercase table names)
- Requires multiple visits per patient (len(visits) > 1)
- Iterates over visit pairs (i, i+1)
- Label comes from next visit's hospital_expire_flag
- Features come from current visit
"""

import os
from typing import Any, Dict, List

from pyhealth.datasets import MIMIC3Dataset
from pyhealth.tasks import BaseTask


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class MortalityPredictionTask(BaseTask):
    """Patient-level mortality prediction task for MIMIC-III.
    
    Predicts mortality at next visit based on current visit data.
    """

    task_name: str = "MortalityPredictionTask"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }

    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return mortality prediction samples.
        
        PATIENT-LEVEL PATTERN:
        - Requires at least 2 visits
        - For each visit i, predict outcome at visit i+1
        - Uses temporal relationships between visits
        """
        samples = []

        # Get visits (MIMIC-III uses uppercase ADMISSIONS)
        visits = patient.get_events(event_type="admissions")

        # Patient-level task requires multiple visits
        if len(visits) <= 1:
            return []

        # Iterate over visit PAIRS (current visit -> next visit)
        for i in range(len(visits) - 1):
            visit = visits[i]
            next_visit = visits[i + 1]

            # Label: Mortality from NEXT visit
            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                continue
            mortality_label = (
                1 if next_visit.hospital_expire_flag in [1, "1"] else 0
            )

            # Features: Clinical codes from CURRENT visit
            # MIMIC-III uses uppercase event types
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )

            # Extract code lists (MIMIC-III uses icd9_code)
            conditions = [event.icd9_code for event in diagnoses]
            procedures_list = [event.icd9_code for event in procedures]
            drugs = [event.drug for event in prescriptions]

            # Skip visits without sufficient data
            if len(conditions) == 0 and len(procedures_list) == 0:
                continue

            samples.append(
                {
                    "visit_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
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
    task = MortalityPredictionTask()
    sample_dataset = base_dataset.set_task(
        task,
        num_workers=num_workers
    )

    # Verify samples
    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        print(f"First sample keys: {list(sample_dataset[0].keys())}")
        print(f"First sample: {sample_dataset[0]}")
