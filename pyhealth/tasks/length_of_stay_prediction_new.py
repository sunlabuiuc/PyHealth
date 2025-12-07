from pyhealth.data import Patient
from datetime import datetime
from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask

class MultiClassLengthOfStayPredictionMIMIC4(BaseTask):
    """Updated version of the length_of_stay_prediction_mimic4_fn()
    for PyHealth 2.0.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (conditions, procedures, and drugs).

    As in the old version, this task is defined as a multi-class classification task.

    """
    task_name: str = "MultiClassLengthOfStayPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"length_of_stay": "multiclass"}

    def _categorize_los(self, days: int):
        """Categorizes length of stay into 10 categories.

        One for ICU stays shorter than a day, seven day-long categories for each day of
        the first week, one for stays of over one week but less than two,
        and one for stays of over two weeks.

        Args:
            days: int, length of stay in days

        Returns:
            category: int, category of length of stay
        """
        # ICU stays shorter than a day
        if days < 1:
            return 0
        # each day of the first week
        elif 1 <= days <= 7:
            return days
        # stays of over one week but less than two
        elif 7 < days <= 14:
            return 8
        # stays of over two weeks
        else:
            return 9

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

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes a single patient for the length of stay prediction task."""
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

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                print("Error parsing admission discharge time:", admission.dischtime)
                continue

            los_days = (admission_dischtime - admission.timestamp).days
            los_category = self._categorize_los(los_days)

            conditions = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            drugs = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime,
            )

            # Extract relevant data
            conditions = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in conditions]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in procedures]
            )
            drugs = self._clean_sequence(
                [getattr(event, "drug", None) for event in drugs]
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
                    "length_of_stay": los_category,
                }
            )
        return samples


class BinaryLengthOfStayPredictionMIMIC4(BaseTask):
    """Updated binary version of the length_of_stay_prediction_mimic4_fn()
    for PyHealth 2.0.

    Length of stay prediction aims at predicting the length of stay (in days) of the
    current hospital visit based on the clinical information from the visit
    (conditions, procedures, and drugs).

    To reproduce the clinical tasks as in EHRMamba, this version predicts
    whether the length of stay will exceed one week.

    If length of stay exceeds 7 days, the label is 1, otherwise 0.
    """
    task_name: str = "BinaryLengthOfStayPredictionMIMIC4"
    input_schema: Dict[str, str] = {
        "conditions": "sequence",
        "procedures": "sequence",
        "drugs": "sequence",
    }
    output_schema: Dict[str, str] = {"length_of_stay": "binary"}

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

    def __call__(self, patient: Patient) -> List[Dict[str, Any]]:
        """Processes a single patient for the length of stay prediction task."""
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

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) <= 1:
            return []

        for i in range(len(admissions) - 1):
            admission = admissions[i]

            # Parse admission timestamps
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If date parsing fails, skip this admission
                print("Error parsing admission discharge time:", admission.dischtime)
                continue

            time_diff_hours = (admission_dischtime - admission.timestamp).total_seconds() / 3600 
            if time_diff_hours > 7 * 24:
                length_of_stay_label = 1
            else:
                length_of_stay_label = 0

            conditions = patient.get_events(
                event_type="diagnoses_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            procedures = patient.get_events(
                event_type="procedures_icd",
                start=admission.timestamp,
                end=admission_dischtime,
            )
            drugs = patient.get_events(
                event_type="prescriptions",
                start=admission.timestamp,
                end=admission_dischtime,
            )

            # Extract relevant data
            conditions = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in conditions]
            )
            procedures_list = self._clean_sequence(
                [getattr(event, "icd_code", None) for event in procedures]
            )
            drugs = self._clean_sequence(
                [getattr(event, "drug", None) for event in drugs]
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
                    "length_of_stay": length_of_stay_label,
                }
            )
        return samples

