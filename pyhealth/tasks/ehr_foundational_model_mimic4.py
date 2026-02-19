from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from pyhealth.tasks.base_task import BaseTask

class EHRFoundationalModelMIMIC4(BaseTask):
    
    task_name: str = "EHRFoundationalModelMIMIC4"
    TOKEN_REPRESENTING_MISSING_TEXT = "<missing>"
    TOKEN_REPRESENTING_MISSING_FLOAT = float("nan")
    
    def __init__(self):
        """Initialize the EHR Foundational Model task."""
        self.input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_name": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_name": "bert-base-uncased",
                    "type_tag": "note",
                },
            )
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]

        # Get visits
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        # Determine which admissions to process iteratively
        # Check each admission's NEXT admission for mortality flag
        admissions_to_process = []
        mortality_label = 0

        for i, admission in enumerate(admissions):
            # Check if THIS admission has the death flag
            if admission.hospital_expire_flag in [1, "1"]:
                # Patient died in this admission - set mortality label
                # but don't include this admission's data
                mortality_label = 1
                break

            # Check if there's a next admission with death flag
            if i + 1 < len(admissions):
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag in [1, "1"]:
                    # Next admission has death - include current, set mortality
                    admissions_to_process.append(admission)
                    mortality_label = 1
                    break

            # No death in current or next - include this admission
            admissions_to_process.append(admission)

        if len(admissions_to_process) == 0:
            return []

        # Aggregated notes and time offsets across all admissions (per hadm_id)
        all_discharge_texts: List[str] = []
        all_discharge_times_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_times_from_admission: List[float] = []

        # Process each admission independently (per hadm_id)
        for admission in admissions_to_process:
            admission_time = admission.timestamp

            # Get notes for this hadm_id only
            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

            for note in discharge_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_discharge_texts.append(note_text)
                        all_discharge_times_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not discharge_notes: # If we get an empty list
                all_discharge_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_discharge_times_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

            for note in radiology_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_radiology_texts.append(note_text)
                        all_radiology_times_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not radiology_notes: # If we receive empty list
                all_radiology_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_radiology_times_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

        discharge_note_times_from_admission = (all_discharge_texts, all_discharge_times_from_admission)
        radiology_note_times_from_admission = (all_radiology_texts, all_radiology_times_from_admission)

        return [
            {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times_from_admission,
                "radiology_note_times": radiology_note_times_from_admission,
                "mortality": mortality_label,
            }
        ]