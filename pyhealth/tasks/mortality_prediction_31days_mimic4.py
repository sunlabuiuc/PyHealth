from datetime import datetime
from typing import Any, Dict, List, Optional

from .base_task import BaseTask


class MortalityPredictionMIMIC4(BaseTask):
    """Task for predicting mortality within a month (i.e. less than 32 days)
    using MIMIC-IV EHR data only.
    
    This task matches the clinical predictive task in EHRMamba in terms
    of label definition.
    """

    task_name: str = "MortalityPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_sequence(self, sequence: Optional[List[Any]]) -> List[str]:
        """
        Clean a sequence by:
        1. Removing None values
        2. Converting to strings
        3. Removing empty strings
        """
        if sequence is None:
            return []

        # Remove None, convert to strings, remove empty strings
        cleaned = [
            str(item).strip()
            for item in sequence
            if item is not None and str(item).strip()
        ]
        return cleaned

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Processes a single patient for the mortality prediction task."""
        samples = []

        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        anchor_age = getattr(demographics, "anchor_age", None)

        # Safely check age - fix potential bug with non-numeric ages
        try:
            if anchor_age is not None and int(float(anchor_age)) < 18:
                return []  # Skip patients under 18
        except (ValueError, TypeError):
            # If age can't be determined, we'll include the patient
            pass

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]
            next_admission = admissions[i + 1]

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                print("Error parsing admission discharge time:", admission.dischtime)
                continue

            # Check discharge status for mortality label, and if yes, whether
            # the time difference between the current admission and the next admission is less than 32 days
            if next_admission.hospital_expire_flag not in [0, 1, "0", "1"]:
                mortality_label = 0
            else:
                mortality_label = int(next_admission.hospital_expire_flag)

            if mortality_label == 1:
                next_admission_dischtime = datetime.strptime(
                    next_admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
                time_diff_hour = (
                    (next_admission_dischtime - admission_dischtime).total_seconds() / 3600
                )
                if time_diff_hour <= 32 * 24:
                    mortality_label = 1
                else:
                    mortality_label = 0

            # Get clinical codes
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            prescriptions = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime,
            )

            # Extract relevant data
            conditions = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in diagnoses_icd]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in procedures_icd]
            )
            drugs = self._clean_sequence(
                [getattr(event, "drug", None) for event in prescriptions]
            )

            # Exclude visits without condition, procedure, or drug code
            if len(conditions) * len(procedures_list) * len(drugs) == 0:
                continue

            samples.append(
                {
                    "visit_id": admission.hadm_id,
                    "patient_id": patient.patient_id,
                    "conditions": conditions,
                    "procedures": procedures_list,
                    "drugs": drugs,
                    "mortality": mortality_label,
                }
            )
        return samples