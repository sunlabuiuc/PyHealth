from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple

from pyhealth.tasks.base_task import BaseTask

class EHRFoundationalModelMIMIC4(BaseTask):
    
    task_name: str = "EHRFoundationalModelMIMIC4"
    
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
            ),
        }
        self.output_schema: Dict[str, str] = {"mortality": "binary"}

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def _compute_time_diffs(self, notes_with_timestamps, first_admission_time):
        """Compute hourly time offsets for notes relative to first admission.

        Sorts notes chronologically by timestamp, then computes each note's
        offset (in hours) from the first admission time.

        Args:
            notes_with_timestamps: List of (text, timestamp) tuples where
                text is the clinical note string and timestamp is a datetime.
            first_admission_time: datetime of the patient's first admission,
                used as the anchor (t=0) for all time offsets.

        Returns:
            Tuple of (texts, time_diffs) where:
                - texts: List[str] of note contents, sorted chronologically
                - time_diffs: List[float] of hours since first admission
            Returns (["<missing>"], [0.0]) if no notes are available.
        """
        result = []

        if not notes_with_timestamps:
            return (["<missing>"], [0.0]) # TODO: Need to also figure out how to tokenize missing timestamps
        notes_with_timestamps.sort(key=lambda x: x[1])
        result = [(text, (ts - first_admission_time).total_seconds() / 3600) for text, ts in notes_with_timestamps]
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

        # Get first admission time as reference for notes time offset
        first_admission_time = admissions_to_process[0].timestamp

        # Only for testing (delete later)
        if first_admission_time is None:
            print("oops, there are cases without admisison time")
            sys.exit()

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
        discharge_note_times = self._compute_time_diffs(all_discharge_notes_timestamped, first_admission_time)
        radiology_note_times = self._compute_time_diffs(all_radiology_notes_timestamped, first_admission_time)

        return [
            {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times,
                "radiology_note_times": radiology_note_times,
                "mortality": mortality_label,
            }
        ]