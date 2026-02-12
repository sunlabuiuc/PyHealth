from datetime import datetime
from typing import Any, Dict, List, Optional

from pyhealth.tasks.base_task import BaseTask

class EHRFoundationalModelMIMIC4(BaseTask):
    
    task_name: str = "EHRFoundationalModelMIMIC4"
    
    def __init__(self):
        """Initialize the EHR Foundational Model task."""
        self.input_schema: Dict[str, str] = {
            "discharge_note_times": "tuple_time_text",
            "radiology_note_times": "tuple_time_text",
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def _compute_time_diffs(self, notes_with_timestamps, anchor_time=None): 
        # TODO: Add docstrings. 
        # anchor_time is in case we want it to normalize/center the time on admission time or something like that.
        
        if not notes_with_timestamps: # TODO: Maybe I should move this somewhere else as it's not relevant to time diffs
            return (["<missing>"], [0.0]) # TODO: How should we handle notes with missing timestamps? 
        result = []
        for i, (text, timestamp) in enumerate(notes_with_timestamps):
            if anchor_time is not None:
                diff = (timestamp - anchor_time).total_seconds() / 3600
            elif i == 0:
                diff = 0.0
            else:
                diff = (timestamp - notes_with_timestamps[i - 1][1]).total_seconds() / 3600
            result.append((text, diff))
        texts, time_diffs = zip(*result)
        return (list(texts), list(time_diffs))

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
        all_discharge_notes_timestamped = []  # List of (note_text, timestamp) tuples
        all_radiology_notes_timestamped = []  # List of (note_text, timestamp) tuples

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
                        all_discharge_notes_timestamped.append((note_text, note.timestamp))
                except AttributeError:
                    pass

            for note in radiology_notes:
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        all_radiology_notes_timestamped.append((note_text, note.timestamp))
                except AttributeError:
                    pass

        # Convert (note_text, timestamp) tuples to (note_text, time_diff_hours) tuples
        discharge_note_times = self._compute_time_diffs(all_discharge_notes_timestamped)
        radiology_note_times = self._compute_time_diffs(all_radiology_notes_timestamped)

        return [
            {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times,
                "radiology_note_times": radiology_note_times,
                "mortality": mortality_label,
            }
        ]