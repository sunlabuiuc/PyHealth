"""Visit-Level Length of Stay Prediction on MIMIC-IV.

This example demonstrates a VISIT-LEVEL task where we predict the length
of stay category for each visit independently.

Key patterns:
- MIMIC4Dataset with ehr_root and ehr_tables (lowercase table names)
- Each visit is INDEPENDENT (no temporal relationships)
- Label: Computed from THIS visit's admission and discharge times
- Multi-class classification with 10 canonical PyHealth categories
- Features come from the SAME visit

Label categories (MUST follow this scheme exactly):
  0: < 1 day
  1: 1 day
  2: 2 days
  3: 3 days
  4: 4 days
  5: 5 days
  6: 6 days
  7: 7 days
  8: > 7 days and <= 14 days
  9: > 14 days
"""

import os
from datetime import datetime
from typing import Any, Dict, List

from pyhealth.datasets import MIMIC4Dataset
from pyhealth.tasks import BaseTask


def categorize_los(days: int) -> int:
    """Categorize length of stay into 10 canonical PyHealth categories.

    This is the STANDARD bucketing used across all PyHealth LOS tasks.
    Do NOT invent custom buckets — always use this function.

    Categories:
      0: < 1 day
      1-7: one bucket per day (days 1 through 7)
      8: > 7 days and <= 14 days
      9: > 14 days
    """
    if days < 1:
        return 0
    elif 1 <= days <= 7:
        return days
    elif 7 < days <= 14:
        return 8
    else:
        return 9


# =============================================================================
# TASK CLASS DEFINITION
# =============================================================================
class LengthOfStayTask(BaseTask):
    """Visit-level length of stay prediction task for MIMIC-IV.

    Predicts length of stay category for each visit independently.
    Uses 10 canonical PyHealth categories via categorize_los().
    """

    task_name: str = "LengthOfStayTask"

    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
    }

    output_schema: Dict[str, str] = {
        "los": "multiclass"  # 10 categories: 0 (<1d), 1-7 (per day), 8 (1-2wk), 9 (>2wk)
    }

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient and return length of stay samples.

        VISIT-LEVEL PATTERN:
        - Each visit is independent
        - Label computed from THIS visit's admit/discharge timestamps
        - No temporal relationships between visits
        - Multi-class output using canonical categorize_los() — 10 categories
        """
        samples = []

        visits = patient.get_events(event_type="admissions")

        # Process each visit independently
        for visit in visits:
            # Features: Clinical codes from THIS visit
            diagnoses = patient.get_events(
                event_type="diagnoses_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                filters=[("hadm_id", "==", visit.hadm_id)],
            )

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

            if len(conditions) == 0:
                continue

            # Label: Calculate LOS from THIS visit's timestamps
            try:
                dischtime = datetime.strptime(str(visit.dischtime), "%Y-%m-%d %H:%M:%S")
                admittime = visit.timestamp
                los_days = (dischtime - admittime).days
            except (ValueError, AttributeError):
                continue

            # Bucket into 10 canonical PyHealth categories
            los_label = categorize_los(los_days)

            samples.append(
                {
                    "visit_id": visit.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "los": los_label,
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
    task = LengthOfStayTask()
    sample_dataset = base_dataset.set_task(task, num_workers=num_workers)

    # Verify samples
    print(f"Total samples: {len(sample_dataset)}")
    if len(sample_dataset) > 0:
        print(f"First sample keys: {list(sample_dataset[0].keys())}")
        print(f"First sample: {sample_dataset[0]}")
