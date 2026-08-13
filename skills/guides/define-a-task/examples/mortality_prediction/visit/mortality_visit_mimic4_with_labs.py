"""Next-Visit Mortality Prediction on MIMIC-IV with Lab Values.

This example demonstrates a TEMPORAL PREDICTION task with semantic features (labs).
Predicts mortality in the NEXT visit based on current visit features.

Key patterns:
- MIMIC4Dataset with ehr_root and ehr_tables (lowercase table names)
- TEMPORAL PATTERN: Features from visit i → Label from visit i+1
- Includes semantic features: filtered lab values aggregated as tensor
- Uses SELECTED_LAB_ITEMS to filter clinically relevant lab itemids
- Aggregates lab values as fixed-length numerical vector

⚠️ TEMPORAL SAFETY FOR LABS:
Labs from current visit are used to predict NEXT visit mortality (safe).
AVOID same-visit labs + same-visit mortality (leakage: labs during deterioration).
"""

import os
from typing import Any, Dict, List

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import BaseTask


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class NextVisitMortalityWithLabsTask(BaseTask):
    """Next-visit mortality prediction using codes + filtered lab values.

    Predicts mortality in the NEXT hospital visit based on current visit data.
    Includes semantic lab features filtered by clinically relevant itemids.
    """

    task_name = "NextVisitMortalityWithLabsTask"

    # Clinically relevant lab itemids for mortality prediction
    # (These would come from Phase 4 exploration in the agent workflow)
    SELECTED_LAB_ITEMS = [
        "50912",  # Creatinine
        "51006",  # BUN
        "50971",  # Potassium
        "50983",  # Sodium
        "51265",  # Platelets
        "51301",  # WBC
    ]

    input_schema = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
        "labs": "tensor",  # Numerical lab values (aggregated per visit)
    }

    output_schema = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return next-visit mortality samples.

        TEMPORAL PATTERN (No Leakage):
        - Features from CURRENT visit (visit i)
        - Label from NEXT visit (visit i+1)
        - Predicts: Will patient die in next hospitalization?

        LAB TEMPORAL SAFETY:
        - Using current visit labs to predict NEXT visit mortality is safe
        - Model learns: "These lab values indicate future risk"
        - NOT learning: "These lab values indicate dying right now"
        """
        samples = []
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

            # Features: Code features from CURRENT visit
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

            # Semantic features: Get and filter lab events from CURRENT visit
            labs = patient.get_events(
                event_type="labevents",
                filters=[("hadm_id", "==", current_visit.hadm_id)],
            )

            # FILTER labs by selected itemids
            filtered_labs = [
                lab for lab in labs if str(lab.itemid) in self.SELECTED_LAB_ITEMS
            ]

            # Build feature sequences
            conditions = [
                str(d.icd_code) for d in diagnoses if getattr(d, "icd_code", None)
            ]
            procedures_list = [
                str(p.icd_code) for p in procedures if getattr(p, "icd_code", None)
            ]
            drugs = [
                str(rx.ndc) if rx.ndc not in [None, "", "0"] else str(rx.drug)
                for rx in prescriptions
                if getattr(rx, "ndc", None) or getattr(rx, "drug", None)
            ]

            # Aggregate lab values: create a vector with one value per itemid
            # Initialize with zeros for all selected lab items
            lab_values = [0.0] * len(self.SELECTED_LAB_ITEMS)

            # For each filtered lab, take the last (most recent) value
            lab_dict = {}
            for lab in filtered_labs:
                if getattr(lab, "valuenum", None) is not None:
                    try:
                        lab_dict[str(lab.itemid)] = float(lab.valuenum)
                    except (ValueError, TypeError):
                        pass

            # Fill in the values for itemids we have
            for idx, itemid in enumerate(self.SELECTED_LAB_ITEMS):
                if itemid in lab_dict:
                    lab_values[idx] = lab_dict[itemid]

            # Compute label
            is_mortality = next_visit.hospital_expire_flag in [1, "1"]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": current_visit.hadm_id,
                    "prediction_for_visit": next_visit.hadm_id,  # Explicit: predicting for NEXT visit
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "labs": lab_values,
                    "mortality": 1 if is_mortality else 0,
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
        ehr_tables=[
            "diagnoses_icd",
            "procedures_icd",
            "prescriptions",
            "labevents",  # Include labs for semantic features
        ],
        cache_dir=CACHE_DIR,
        dev=False,
    )

    # Apply task
    num_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    task = NextVisitMortalityWithLabsTask()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    # Verify samples
    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        print(f"First sample keys: {list(sample_dataset[0].keys())}")
        print(f"First sample: {sample_dataset[0]}")
