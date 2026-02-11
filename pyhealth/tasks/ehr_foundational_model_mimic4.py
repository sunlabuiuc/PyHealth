from datetime import datetime
from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask

class EHRFoundationalModelMIMIC4(BaseTask):
    
    task_name: str = "EHRFoundationalModelMIMIC4"
    
    def __init__(self):
        """Initialize the EHR Foundational Model task."""
        self.input_schema: Dict[str, str] = {
            "discharge": "raw",
            "radiology": "raw",
            "discharge_note_time_diffs": "tensor",
            "radiology_note_time_diffs": "tensor",
        }
        self.output_schema: Dict[str, str] = {"mortality": "regression"}

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

        # Get first admission time as reference for lab time calculations
        first_admission_time = admissions_to_process[0].timestamp

        # Aggregated data across all admissions
        all_discharge_notes = []  # List of individual discharge notes
        all_radiology_notes = []  # List of individual radiology notes
        all_discharge_notes_timestamps = [] # List of individual discharge notes timestamps
        all_radiology_notes_timestamps = [] # List of individual discharge notes timestamps
        discharge_note_time_diffs = []
        radiology_notes_time_diffs = []
        

        # Process each admission and aggregate data
        for admission in admissions_to_process:
            # Parse admission discharge time for lab events filtering
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # If we can't parse discharge time, skip this admission
                continue

            # Skip if discharge is before admission (data quality issue)
            if admission_dischtime < admission.timestamp:
                continue

            # Get notes using hadm_id filtering
            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

        # Extract and aggregate notes as individual items in lists
            # Note: attribute is "text" (from mimic4_note.yaml), not "discharge"/"radiology"
            for note in discharge_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_discharge_notes.append(note_text)
                        all_discharge_notes_timestamps.append(note.timestamp)
                except AttributeError:
                    pass

            for note in radiology_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_radiology_notes.append(note_text)
                        all_radiology_notes_timestamps.append(note.timestamp)
                except AttributeError:
                    pass

        # Sort discharge_notes by timestamp 
        all_discharge_notes_timestamps.sort()
        all_radiology_notes_timestamps.sort()

        # Compute time difference for discharge notes (hours)
        discharge_note_time_diffs = [0.0] + [
            (curr - prev).total_seconds() / 3600
            for prev, curr in zip(all_discharge_notes_timestamps, all_discharge_notes_timestamps[1:])
        ]

        # Compute time difference for radiology notes (hours)
        radiology_note_time_diffs = [0.0] + [
            (curr - prev).total_seconds() / 3600
            for prev, curr in zip(all_radiology_notes_timestamps, all_radiology_notes_timestamps[1:])
        ]

        # ===== MODALITY REQUIREMENTS =====
        # Check notes - need at least one discharge OR radiology note
        has_notes = len(all_discharge_notes) > 0 or len(all_radiology_notes) > 0

        #Return empty list if any required modality is missing
        if not (
            has_notes
        ):
            return []


        return [
            {
                "patient_id": patient.patient_id,
                "discharge": all_discharge_notes,
                "discharge_note_time_diffs": discharge_note_time_diffs,
                "radiology": all_radiology_notes,
                "radiology_note_time_diffs": radiology_note_time_diffs,
                "mortality": mortality_label,
            }
        ]