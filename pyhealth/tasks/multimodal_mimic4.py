from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, ClassVar

from pyhealth.tasks.base_task import BaseTask


class BaseMultimodalMIMIC4Task(BaseTask):
    """Base class for multimodal MIMIC-IV tasks.

    Provides shared constants and utility methods used across all multimodal
    task variants (notes, ICD codes, lab values).
    """

    MISSING_TEXT_TOKEN: ClassVar[str] = ""
    MISSING_FLOAT_TOKEN: ClassVar[float] = 0.0

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

    @staticmethod
    def _clean_text(text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(value, fmt)
                except ValueError:
                    continue
        return None

    @staticmethod
    def _to_hours(delta_seconds: float) -> float:
        return delta_seconds / 3600.0

    def _collect_icd_codes(self, patient: Any, hadm_id: Any) -> List[str]:
        """Collect ICD diagnosis and procedure codes for one admission.

        Returns:
            List of ICD code strings, or an empty list if none found.
        """
        diagnoses_icd = patient.get_events(
            event_type="diagnoses_icd", filters=[("hadm_id", "==", hadm_id)]
        )
        procedures_icd = patient.get_events(
            event_type="procedures_icd", filters=[("hadm_id", "==", hadm_id)]
        )
        return (
            [e.icd_code for e in diagnoses_icd if hasattr(e, "icd_code") and e.icd_code] +
            [e.icd_code for e in procedures_icd if hasattr(e, "icd_code") and e.icd_code]
        )

    def _collect_labs(
        self,
        patient: Any,
        admission_time: datetime,
        end_time: datetime,
    ) -> Tuple[List[float], List[List[float]], List[List[bool]]]:
        """Collect lab values and observation masks for one admission.

        Args:
            patient: Patient object.
            admission_time: Start of the window; times are relative to this.
            end_time: End of the window (inclusive).

        Returns:
            Tuple of (lab_times, lab_values, lab_masks). ``lab_masks`` is a
            parallel boolean tensor where ``True`` means observed and ``False``
            means imputed with 0.0. Falls back to a single missing placeholder
            row when no valid lab events are found.
        """
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars is required for lab collection."
            ) from exc

        labevents_df = patient.get_events(
            event_type="labevents",
            start=admission_time,
            end=end_time,
            return_df=True,
        )

        lab_times: List[float] = []
        lab_values: List[List[float]] = []
        lab_masks: List[List[bool]] = []

        labevents_df = labevents_df.filter(
            pl.col("labevents/itemid").is_in(self.LABITEMS)
        )
        if labevents_df.height > 0:
            labevents_df = labevents_df.with_columns(
                pl.col("labevents/storetime").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            )
            labevents_df = labevents_df.filter(
                pl.col("labevents/storetime") <= end_time
            )
            if labevents_df.height > 0:
                labevents_df = labevents_df.select(
                    pl.col("timestamp"),
                    pl.col("labevents/itemid"),
                    pl.col("labevents/valuenum").cast(pl.Float64),
                )
                for lab_ts in sorted(labevents_df["timestamp"].unique().to_list()):
                    ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)
                    lab_vector: List[float] = []
                    lab_mask: List[bool] = []
                    for category_name in self.LAB_CATEGORY_NAMES:
                        category_value = self.MISSING_FLOAT_TOKEN
                        observed = False
                        for itemid in self.LAB_CATEGORIES[category_name]:
                            matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                            if matching.height > 0:
                                category_value = matching["labevents/valuenum"][0]
                                observed = True
                                break
                        lab_vector.append(category_value)
                        lab_mask.append(observed)
                    lab_times.append(self._to_hours((lab_ts - admission_time).total_seconds()))
                    lab_values.append(lab_vector)
                    lab_masks.append(lab_mask)
            else:  # If missing lab for a given admission
                lab_values.append([self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES))
                lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
                lab_times.append(self.MISSING_FLOAT_TOKEN)

        if len(lab_values) == 0:  # If missing lab for ALL admissions
            lab_values.append([self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES))
            lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            lab_times.append(self.MISSING_FLOAT_TOKEN)
        return lab_times, lab_values, lab_masks

    def _collect_notes(
        self,
        patient: Any,
        note_event_type: str,
        hadm_id: Any,
        admission_time: datetime,
    ) -> Tuple[List[str], List[float]]:
        """Collect notes of a given type for one admission.

        Args:
            patient: Patient object.
            note_event_type: Event type string (e.g. "discharge", "radiology").
            hadm_id: Admission ID to filter by.
            admission_time: Admission start time; used to compute time offsets.

        Returns:
            Tuple of (texts, hours_from_admission). Falls back to
            ``([MISSING_TEXT_TOKEN], [MISSING_FLOAT_TOKEN])`` when the events
            list is empty.
        """
        notes = patient.get_events(
            event_type=note_event_type,
            filters=[("hadm_id", "==", hadm_id)],
        )

        texts: List[str] = []
        note_times: List[float] = []
        for note in notes:
            try:
                note_text = self._clean_text(note.text)
                if note_text:
                    time_from_admission = self._to_hours(
                        (note.timestamp - admission_time).total_seconds()
                    )
                    texts.append(note_text)
                    note_times.append(time_from_admission)
            except AttributeError:  # note object is missing .text or .timestamp attribute (e.g. malformed note)
                pass

        if not notes or not texts:  # If we get an empty list or all notes were malformed
            return [self.MISSING_TEXT_TOKEN], [self.MISSING_FLOAT_TOKEN]  # Token representing missing text/time
        return texts, note_times


class ClinicalNotesMIMIC4(BaseMultimodalMIMIC4Task):
    """Task for clinical notes-based mortality prediction using MIMIC-IV.

    Predicts patient-level mortality using discharge summaries and radiology
    notes paired with their timestamps. Notes are processed by
    ``TupleTimeTextProcessor`` to produce token IDs and
    time offsets relative to each admission.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     note_root="/path/to/mimic-iv-note/2.2",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd",
        ...                 "prescriptions", "labevents"],
        ...     note_tables=["discharge", "radiology"],
        ... )
        >>> task = ClinicalNotesMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ClinicalNotesMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            )
        }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self):
        """Initialize the EHR Foundational Model task."""

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

            discharge_texts, discharge_times = self._collect_notes(
                patient, "discharge", admission.hadm_id, admission_time
            )
            all_discharge_texts.extend(discharge_texts)
            all_discharge_times_from_admission.extend(discharge_times)

            radiology_texts, radiology_times = self._collect_notes(
                patient, "radiology", admission.hadm_id, admission_time
            )
            all_radiology_texts.extend(radiology_texts)
            all_radiology_times_from_admission.extend(radiology_times)

        discharge_note_times_from_admission = (all_discharge_texts, all_discharge_times_from_admission)
        radiology_note_times_from_admission = (all_radiology_texts, all_radiology_times_from_admission)

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times_from_admission,
                "radiology_note_times": radiology_note_times_from_admission,
                "mortality": mortality_label,
            }

        return [
            single_patient_longitudinal_record
        ]

class ClinicalNotesICDLabsMIMIC4(BaseMultimodalMIMIC4Task):
    """Task for multimodal mortality prediction combining clinical notes, ICD codes, and lab values using MIMIC-IV.

    Extends ``ClinicalNotesMIMIC4`` with two additional modalities:

    - **ICD codes**: diagnosis and procedure codes per admission, processed by
      ``StageNetProcessor`` with inter-admission time offsets.
    - **Lab values**: 10-dimensional lab vectors (one per lab category) at each
      measurement timestamp, processed by ``StageNetTensorProcessor``.

    Lab categories (10 dimensions):
        Sodium, Potassium, Chloride, Bicarbonate, Glucose, Calcium, Magnesium,
        Anion Gap, Osmolality, Phosphate.

    The ``labs_mask`` field is a parallel boolean tensor (same shape as ``labs``)
    where ``True`` means the value was observed and ``False`` means it was
    imputed w/ 0.0. Downstream models should use this mask to seperate
    real zeros from missing data fill values.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks.multimodal_mimic4 import ClinicalNotesICDLabsMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     note_root="/path/to/mimic-iv-note/2.2",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd",
        ...                 "prescriptions", "labevents"],
        ...     note_tables=["discharge", "radiology"],
        ... )
        >>> task = ClinicalNotesICDLabsMIMIC4()
        >>> samples = dataset.set_task(task)
    """
    PADDING: int = 0

    task_name: str = "ClinicalNotesICDLabsMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_note_times": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "icd_codes": ("stagenet", {"padding": PADDING}),
            "labs": ("stagenet_tensor", {}),
            "labs_mask": ("stagenet_tensor", {}),
        }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __init__(self):
        """Initialize the EHR Foundational Model task."""

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
        all_lab_values: List[List[float]] = []
        all_lab_masks: List[List[bool]] = []  # True = observed, False = imputed 0.0
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

            discharge_texts, discharge_times = self._collect_notes(
                patient, "discharge", admission.hadm_id, admission_time
            )
            all_discharge_texts.extend(discharge_texts)
            all_discharge_times_from_admission.extend(discharge_times)

            radiology_texts, radiology_times = self._collect_notes(
                patient, "radiology", admission.hadm_id, admission_time
            )
            all_radiology_texts.extend(radiology_texts)
            all_radiology_times_from_admission.extend(radiology_times)

            # ICD codes (diagnoses + procedures) with time relative to previous admission
            visit_icd_codes = self._collect_icd_codes(patient, admission.hadm_id)
            if visit_icd_codes:  # If there are ICD diagnosis/inpatient procedure codes
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (admission_time - previous_admission_time).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)
            else:  # Add missingness token if there are no ICD diagnosis/inpatient procedure codes
                all_icd_codes.append([self.MISSING_TEXT_TOKEN])
                all_icd_times.append(self.MISSING_FLOAT_TOKEN)

            previous_admission_time = admission_time

            # Lab events with time relative to this admission's start
            lab_times, lab_values, lab_masks = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                end_time=admission_dischtime,
            )
            all_lab_times.extend(lab_times)
            all_lab_values.extend(lab_values)
            all_lab_masks.extend(lab_masks)

        if len(all_lab_values) == 0: # If missing lab for ALL admissions
            all_lab_values.append([self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES))
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_times.append(self.MISSING_FLOAT_TOKEN)

        discharge_note_times_from_admission = (all_discharge_texts, all_discharge_times_from_admission)
        radiology_note_times_from_admission = (all_radiology_texts, all_radiology_times_from_admission)

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times_from_admission,
                "radiology_note_times": radiology_note_times_from_admission,
                "icd_codes": (all_icd_times, all_icd_codes),
                "labs": (all_lab_times, all_lab_values),
                "labs_mask": (all_lab_times, all_lab_masks),
                "mortality": mortality_label,
            }

        return [
            single_patient_longitudinal_record
        ]
