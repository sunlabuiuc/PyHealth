from datetime import datetime
from typing import Any, Dict, List, Optional, Union, Tuple, ClassVar

from pyhealth.tasks.base_task import BaseTask

class ClinicalNotesMIMIC4(BaseTask):
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

    Note on time encoding:
        - [Per Admission Granularity] Notes use time (hours) relative to each admission time.
    """
    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0

    task_name: str = "ClinicalNotesMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_notes": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_notes": (
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
        all_discharge_hours_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_hours_from_admission: List[float] = []

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

            # For all discharge notes in a single admission:
            for note in discharge_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_discharge_texts.append(note_text)
                        all_discharge_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not discharge_notes: # If we get an empty list
                all_discharge_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_discharge_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

            # For all radiology notes in a single admission:
            for note in radiology_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_radiology_texts.append(note_text)
                        all_radiology_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not radiology_notes: # If we receive empty list
                all_radiology_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_radiology_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_notes": (all_discharge_texts, all_discharge_hours_from_admission),
                "radiology_notes": (all_radiology_texts, all_radiology_hours_from_admission),
                "mortality": mortality_label,
            }

        return [
            single_patient_longitudinal_record
        ]

class ClinicalNotesICDLabsMIMIC4(BaseTask):
    """Task for multimodal mortality prediction combining clinical notes, ICD codes, and lab values using MIMIC-IV.

    Extends ``ClinicalNotesMIMIC4`` with two additional modalities:

    - **ICD codes**: diagnosis and procedure codes per admission, processed by
      ``StageNetProcessor`` with inter-admission time offsets.
    - **Lab values**: N_Lab_Category lab vectors (one per lab category) at each
      measurement timestamp, processed by ``StageNetTensorProcessor``.

    Lab categories (N_Lab_Category):
        Sodium, Potassium, Chloride, Bicarbonate, Glucose, Calcium, Magnesium,
        Anion Gap, Osmolality, Phosphate.

    The ``labs_mask`` field is a parallel boolean tensor (same shape as ``labs``)
    where ``True`` means the value was observed and ``False`` means it was
    imputed w/ 0.0. Downstream models should use this mask to seperate
    real zeros from missing data fill values.

    Note on time encoding:
        - [Per Admission Granularity] Notes and labs use time (hours) relative to each admission time.
        - [Per Admission Granularity] ICD codes use inter-admission gap (time (hours) between previous and current
          admission time), since ICD codes represent the whole visit and have no
          within-admission timestamp. This is intentionally inconsistent and may
          be revisited (e.g. time since first admission, or always 0.0).

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
    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0
    PADDING: int = 0

    task_name: str = "ClinicalNotesICDLabsMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_notes": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_notes": (
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

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for ClinicalNotesICDLabsMIMIC4."
            ) from None

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
        all_discharge_hours_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_hours_from_admission: List[float] = []
        all_icd_codes: List[List[str]] = []
        all_icd_inter_admission_hours: List[float] = []
        all_lab_values: List[List[Any]] = []
        all_lab_masks: List[List[bool]] = []  # True = observed, False = imputed 0.0
        all_lab_hours_from_admission: List[float] = []
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

            # For all discharge notes in a single admission:
            for note in discharge_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_discharge_texts.append(note_text)
                        all_discharge_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not discharge_notes: # If we get an empty list
                all_discharge_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_discharge_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

            # For all radiology notes in a single admission:
            for note in radiology_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_radiology_texts.append(note_text)
                        all_radiology_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not radiology_notes: # If we receive empty list
                all_radiology_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_radiology_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

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
            if visit_icd_codes: # If there are ICD diagnosis/inpatient procedure codes
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (admission_time - previous_admission_time).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_inter_admission_hours.append(time_from_previous)
            else: # Add missingness token if there are no ICD diagnosis/inpatient procedure codes
                all_icd_codes.append([self.TOKEN_REPRESENTING_MISSING_TEXT])
                all_icd_inter_admission_hours.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

            previous_admission_time = admission_time  # Advance rolling reference for next admission's time delta

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
                        lab_mask: List[bool] = []
                        for category_name in self.LAB_CATEGORY_NAMES:
                            category_value = self.TOKEN_REPRESENTING_MISSING_FLOAT
                            observed = False
                            for itemid in self.LAB_CATEGORIES[category_name]:
                                matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                                if matching.height > 0:
                                    category_value = matching["labevents/valuenum"][0]
                                    observed = True
                                    break
                            lab_vector.append(category_value)
                            lab_mask.append(observed)
                        all_lab_values.append(lab_vector)
                        all_lab_masks.append(lab_mask)
                        all_lab_hours_from_admission.append((lab_ts - admission_time).total_seconds() / 3600.0)
                else: # If missing lab for a given admission
                    all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        if len(all_lab_values) == 0: # If missing lab for ALL admissions
            all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_notes": (all_discharge_texts, all_discharge_hours_from_admission),
                "radiology_notes": (all_radiology_texts, all_radiology_hours_from_admission),
                "icd_codes": (all_icd_inter_admission_hours, all_icd_codes),
                "labs": (all_lab_hours_from_admission, all_lab_values),
                "labs_mask": (all_lab_hours_from_admission, all_lab_masks),
                "mortality": mortality_label,
            }

        return [
            single_patient_longitudinal_record
        ]

class ClinicalNotesICDLabsCXRMIMIC4(BaseTask):
    """Task for multimodal mortality prediction combining clinical notes, ICD codes, lab values, and chest X-rays using MIMIC-IV.

    Extends ``ClinicalNotesICDLabsMIMIC4`` with two additional CXR modalities:

    - **image_path**: path to the first available chest X-ray image for the patient.
    - **negbio_findings**: deduplicated list of positive NegBio findings across all X-rays.

    CXR data is processed at the patient level (not per-admission), since MIMIC-CXR
    studies are not always linked to a specific ``hadm_id``.

    Note on time encoding:
        - [Per Admission Granularity] Notes and labs use time (hours) relative to each admission time.
        - [Per Admission Granularity] ICD codes use inter-admission gap (time (hours) between previous and current
          admission time), since ICD codes represent the whole visit and have no
          within-admission timestamp. This is intentionally inconsistent and may
          be revisited (e.g. time since first admission, or always 0.0).
        - [Per Patient Granularity] CXR time is encoded as hours relative to the nearest admission start time
          in ``admissions_to_process``.
    """
    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0
    TOKEN_REPRESENTING_MISSING_PATH = ""
    PADDING: int = 0

    task_name: str = "ClinicalNotesICDLabsCXRMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
            "discharge_notes": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "radiology_notes": (
                "tuple_time_text",
                {
                    "tokenizer_model": "bert-base-uncased",
                    "type_tag": "note",
                },
            ),
            "icd_codes": ("stagenet", {"padding": PADDING}),
            "labs": ("stagenet_tensor", {}),
            "labs_mask": ("stagenet_tensor", {}),
            "cxrs": ("time_image", {"padding": TOKEN_REPRESENTING_MISSING_PATH}),
            "negbio_findings": ("stagenet", {"padding": PADDING}),
        }
    output_schema: Dict[str, str] = {"mortality": "binary"}

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

    NEGBIO_FINDING_NAMES: ClassVar[List[str]] = [
            "no finding",
            "enlarged cardiomediastinum",
            "cardiomegaly",
            "lung opacity",
            "lung lesion",
            "edema",
            "consolidation",
            "pneumonia",
            "atelectasis",
            "pneumothorax",
            "pleural effusion",
            "pleural other",
            "fracture",
            "support devices"
    ]

    def __init__(self):
        """Initialize the EHR Foundational Model task."""

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        try:
            import polars as pl
        except ImportError:
            raise ImportError(
                "Polars is required for ClinicalNotesICDLabsMIMIC4."
            ) from None

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
        all_discharge_hours_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_hours_from_admission: List[float] = []
        all_icd_codes: List[List[str]] = []
        all_icd_inter_admission_hours: List[float] = []
        all_lab_values: List[List[Any]] = []
        all_lab_masks: List[List[bool]] = []  # True = observed, False = imputed 0.0
        all_lab_hours_from_admission: List[float] = []
        all_negbio_findings: List[List[Any]] = []
        all_cxr_image_paths: List[str] = []
        all_cxr_hours_relative_to_nearest_admission: List[float] = []

        previous_admission_time = None

        # [Chest X-Rays (CXRs)]: Process at patient level, not admission-level
        negbio_events = patient.get_events(event_type="negbio")
        metadata_events = patient.get_events(event_type="metadata")
        
        for cxr in negbio_events: # Loop through each CXR
            negbio_vector = [] # Per CXR Vector
            try:
                for finding_name in self.NEGBIO_FINDING_NAMES: # Check each CXR's NEGBIO_FINDING_NAMES
                    try:
                        negbio_value = getattr(cxr, finding_name, self.TOKEN_REPRESENTING_MISSING_TEXT) 
                        if negbio_value!= self.TOKEN_REPRESENTING_MISSING_TEXT and float(negbio_value) > 0:
                            negbio_vector.append(finding_name)
                    except (ValueError, TypeError, AttributeError):
                        negbio_vector.append(self.TOKEN_REPRESENTING_MISSING_TEXT)
            except Exception: # Missing negbio for a given cxr returns a N-length vector of MISSING_TOKEN
                negbio_vector = [self.TOKEN_REPRESENTING_MISSING_TEXT] * len(self.NEGBIO_FINDING_NAMES)

            all_negbio_findings.append(negbio_vector)

        for cxr in metadata_events: # Loop through each CXR
            try:
                if cxr.image_path:
                    cxr_image_path = cxr.image_path
                    cxr_image_timestamp = cxr.timestamp

                    # TODO: Consider making this into a utility function
                    if cxr_image_timestamp is not None and admissions_to_process:
                        nearest_admission = min(
                            admissions_to_process,
                            key=lambda a: abs((cxr_image_timestamp - a.timestamp).total_seconds()),
                        )
                        image_hours_from_nearest_admission = (
                            (cxr_image_timestamp - nearest_admission.timestamp).total_seconds() / 3600.0
                        )

                    all_cxr_image_paths.append(cxr_image_path)
                    all_cxr_hours_relative_to_nearest_admission.append(image_hours_from_nearest_admission)
            except AttributeError:
                all_cxr_image_paths.append(self.TOKEN_REPRESENTING_MISSING_PATH)
                all_cxr_hours_relative_to_nearest_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        if len(all_cxr_image_paths) == 0: # If patient has no metadata events at all, insert a padding entry
            all_cxr_image_paths.append(self.TOKEN_REPRESENTING_MISSING_PATH)
            all_cxr_hours_relative_to_nearest_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        # [Clinical Notes, EHR, Labs]: Process each admission independently (per hadm_id)
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

            discharge_notes = patient.get_events(
                event_type="discharge", filters=[("hadm_id", "==", admission.hadm_id)]
            )
            radiology_notes = patient.get_events(
                event_type="radiology", filters=[("hadm_id", "==", admission.hadm_id)]
            )

            # For all discharge notes in a single admission:
            for note in discharge_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_discharge_texts.append(note_text)
                        all_discharge_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not discharge_notes: # If we get an empty list
                all_discharge_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_discharge_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

            # For all radiology notes in a single admission:
            for note in radiology_notes: #TODO: Maybe make this into a helper function?
                try:
                    note_text = self._clean_text(note.text)
                    if note_text:
                        time_from_admission = (
                            note.timestamp - admission_time
                        ).total_seconds() / 3600.0
                        all_radiology_texts.append(note_text)
                        all_radiology_hours_from_admission.append(time_from_admission)
                except AttributeError: # note object is missing .text or .timestamp attribute (e.g. malformed note)
                    pass
            if not radiology_notes: # If we receive empty list
                all_radiology_texts.append(self.TOKEN_REPRESENTING_MISSING_TEXT) # Token representing missing text
                all_radiology_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT) # Token representing missing time(?)

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
            if visit_icd_codes: # If there are ICD diagnosis/inpatient procedure codes
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (admission_time - previous_admission_time).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_inter_admission_hours.append(time_from_previous)
            else: # Add missingness token if there are no ICD diagnosis/inpatient procedure codes
                all_icd_codes.append([self.TOKEN_REPRESENTING_MISSING_TEXT])
                all_icd_inter_admission_hours.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

            previous_admission_time = admission_time  # Advance rolling reference for next admission's time delta

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
                        lab_mask: List[bool] = []
                        for category_name in self.LAB_CATEGORY_NAMES:
                            category_value = self.TOKEN_REPRESENTING_MISSING_FLOAT
                            observed = False
                            for itemid in self.LAB_CATEGORIES[category_name]:
                                matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                                if matching.height > 0:
                                    category_value = matching["labevents/valuenum"][0]
                                    observed = True
                                    break
                            lab_vector.append(category_value)
                            lab_mask.append(observed)
                        all_lab_values.append(lab_vector)
                        all_lab_masks.append(lab_mask)
                        all_lab_hours_from_admission.append((lab_ts - admission_time).total_seconds() / 3600.0)
                else: # If missing lab for a given admission
                    all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        if len(all_lab_values) == 0: # If missing lab for ALL admissions
            all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_hours_from_admission.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_notes": (all_discharge_texts, all_discharge_hours_from_admission),
                "radiology_notes": (all_radiology_texts, all_radiology_hours_from_admission),
                "icd_codes": (all_icd_inter_admission_hours, all_icd_codes),
                "labs": (all_lab_hours_from_admission, all_lab_values),
                "labs_mask": (all_lab_hours_from_admission, all_lab_masks),
                "cxrs": (all_cxr_image_paths, all_cxr_hours_relative_to_nearest_admission),
                "negbio_findings": (all_cxr_hours_relative_to_nearest_admission, all_negbio_findings),
                "mortality": mortality_label,
            }

        return [
            single_patient_longitudinal_record
        ]