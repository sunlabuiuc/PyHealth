from datetime import datetime
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import polars as pl

from pyhealth.tasks.base_task import BaseTask

class EHRFoundationalModelMIMIC4(BaseTask):

    task_name: str = "EHRFoundationalModelMIMIC4"
    TOKEN_REPRESENTING_MISSING_TEXT = "<missing>"
    TOKEN_REPRESENTING_MISSING_FLOAT = float("nan")
    PADDING: int = 0

    LAB_CATEGORIES: ClassVar[Dict[str, List[str]]] = {
        "Sodium": ["50824", "52455", "50983", "52623"],
        "Potassium": ["50822", "52452", "50971", "52610"],
        "Chloride": ["50806", "52434", "50902", "52535"],
        "Bicarbonate": ["50803", "50804"],
        "Glucose": ["50809", "52027", "50931", "52569"],
        "Calcium": ["50808", "51624"],
        "Magnesium": ["50960"],
        "Anion Gap": ["50868", "52500"],
        "Osmolality": ["52031", "50964", "51701"],
        "Phosphate": ["50970"],
    }

    LAB_CATEGORY_NAMES: ClassVar[List[str]] = [
        "Sodium", "Potassium", "Chloride", "Bicarbonate", "Glucose",
        "Calcium", "Magnesium", "Anion Gap", "Osmolality", "Phosphate",
    ]

    LABITEMS: ClassVar[List[str]] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

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
            "icd_codes": ("stagenet", {"padding": self.PADDING}),
            "labs": ("stagenet_tensor", {}),
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
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[Any]] = []
        all_lab_times: List[float] = []
        previous_admission_time = None

        # Process each admission independently (per hadm_id)
        for admission in admissions_to_process:
            admission_time = admission.timestamp

            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                continue

            if admission_dischtime < admission_time:
                continue

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

            # ICD codes (diagnoses + procedures) with time relative to previous admission
            diagnoses_icd = patient.get_events(
                event_type="diagnoses_icd", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            procedures_icd = patient.get_events(
                event_type="procedures_icd", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            visit_icd_codes = (
                [e.icd_code for e in diagnoses_icd if hasattr(e, "icd_code") and e.icd_code] +
                [e.icd_code for e in procedures_icd if hasattr(e, "icd_code") and e.icd_code]
            )
            if visit_icd_codes:
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (admission_time - previous_admission_time).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)

            previous_admission_time = admission_time

            # Lab events with time relative to this admission's start
            labevents_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=admission_dischtime,
                return_df=True,
            )
            labevents_df = labevents_df.filter(
                pl.col("labevents/itemid").is_in(self.LABITEMS)
            )
            if labevents_df.height > 0:
                labevents_df = labevents_df.with_columns(
                    pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
                )
                labevents_df = labevents_df.filter(
                    pl.col("labevents/storetime") <= admission_dischtime
                )
                if labevents_df.height > 0:
                    labevents_df = labevents_df.select(
                        pl.col("timestamp"),
                        pl.col("labevents/itemid"),
                        pl.col("labevents/valuenum").cast(pl.Float64),
                    )
                    for lab_ts in sorted(labevents_df["timestamp"].unique().to_list()):
                        ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)
                        lab_vector: List[Any] = []
                        for category_name in self.LAB_CATEGORY_NAMES:
                            category_value = None
                            for itemid in self.LAB_CATEGORIES[category_name]:
                                matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                                if matching.height > 0:
                                    category_value = matching["labevents/valuenum"][0]
                                    break
                            lab_vector.append(category_value)
                        all_lab_values.append(lab_vector)
                        all_lab_times.append((lab_ts - admission_time).total_seconds() / 3600.0)

        discharge_note_times_from_admission = (all_discharge_texts, all_discharge_times_from_admission)
        radiology_note_times_from_admission = (all_radiology_texts, all_radiology_times_from_admission)

        return [
            {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times_from_admission,
                "radiology_note_times": radiology_note_times_from_admission,
                "icd_codes": (all_icd_times, all_icd_codes),
                "labs": (all_lab_times, all_lab_values),
                "mortality": mortality_label,
            }
        ]
