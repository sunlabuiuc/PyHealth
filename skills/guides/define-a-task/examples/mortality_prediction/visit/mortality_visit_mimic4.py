"""Next-Visit Mortality Prediction on MIMIC-IV.

This example demonstrates a TEMPORAL PREDICTION task where we predict mortality
in the NEXT hospital visit based on features from the CURRENT visit.

Key patterns:
- MIMIC4Dataset with ehr_root and ehr_tables (lowercase table names)
- TEMPORAL PATTERN: Features from visit i → Label from visit i+1
- Prevents temporal leakage (no same-visit prediction)
- Matches PyHealth canonical implementation (MortalityPredictionMIMIC4)
- Simple code-based features (conditions, procedures, drugs)

⚠️ AVOIDING TEMPORAL LEAKAGE:
DO NOT use same-visit features + same-visit label. This causes severe leakage
because in-hospital events (ICU transfers, medications) correlate with mortality
THAT IS HAPPENING, not mortality RISK at admission.

Correct: Features from visit N → Mortality in visit N+1 ✅
Wrong:   Features from visit N → Mortality in visit N ❌ (leakage!)
"""

import os
from typing import Any, Dict, List

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import BaseTask


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class NextVisitMortalityPredictionTask(BaseTask):
    """Next-visit mortality prediction task for MIMIC-IV.

    Predicts mortality in the NEXT hospital visit based on current visit data.
    This temporal pattern prevents data leakage.
    """

    task_name: str = "NextVisitMortalityPredictionTask"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }

    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return next-visit mortality samples.

        TEMPORAL PATTERN (No Leakage):
        - Features extracted from CURRENT visit (visit i)
        - Label extracted from NEXT visit (visit i+1)
        - Predicts: Will patient die in their next hospitalization?

        This prevents temporal leakage by ensuring prediction target is in the future.
        """
        samples = []

        # Get visits (MIMIC-IV uses lowercase admissions)
        visits = patient.get_events(event_type="admissions")

        # Need at least 2 visits for next-visit prediction
        if len(visits) <= 1:
            return samples

        # Process visit pairs: predict mortality in visit i+1 using features from visit i
        for i in range(len(visits) - 1):
            current_visit = visits[i]
            next_visit = visits[i + 1]
            # Label: Mortality from NEXT visit (temporal prediction)
            if next_visit.hospital_expire_flag not in [0, 1, "0", "1"]:
                continue
            mortality_label = 1 if next_visit.hospital_expire_flag in [1, "1"] else 0

            # Features: Clinical codes from CURRENT visit
            # MIMIC-IV uses lowercase event types
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )

            # Extract code lists (MIMIC-IV uses icd_code attribute)
            conditions = [
                str(event.icd_code)
                for event in diagnoses
                if getattr(event, "icd_code", None)
            ]
            procedures_list = [
                str(event.icd_code)
                for event in procedures
                if getattr(event, "icd_code", None)
            ]
            drugs = [
                str(event.drug)
                for event in prescriptions
                if getattr(event, "drug", None)
            ]

            # Skip visits without sufficient data
            if len(conditions) == 0 and len(procedures_list) == 0:
                continue

            samples.append(
                {
                    "visit_id": current_visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "prediction_for_visit": next_visit.hadm_id,  # Explicit: predicting for NEXT visit
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
if __name__ == "__main__":
    # Dataset initialization
    EHR_ROOT = "/srv/local/data/physionet.org/files/mimiciv/2.2"
    CACHE_DIR = "./cache"
    TASK_CACHE_DIR = "./task_cache"

    base_dataset = MIMIC4Dataset(
        ehr_root=EHR_ROOT,
        ehr_tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        cache_dir=CACHE_DIR,
        dev=False,
    )

    # Apply task
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    task = NextVisitMortalityPredictionTask()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    # Verify samples
    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        print(f"First sample keys: {list(sample_dataset[0].keys())}")
        print(f"First sample: {sample_dataset[0]}")
