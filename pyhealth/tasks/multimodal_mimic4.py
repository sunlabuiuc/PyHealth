from datetime import datetime, timedelta
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
    """   
    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0

    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0

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

        single_patient_longitudinal_record = {
                "patient_id": patient.patient_id,
                "discharge_note_times": discharge_note_times_from_admission,
                "radiology_note_times": radiology_note_times_from_admission,
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
    TOKEN_REPRESENTING_MISSING_TEXT = ""
    TOKEN_REPRESENTING_MISSING_FLOAT = 0.0
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
        all_discharge_times_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_times_from_admission: List[float] = []
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[Any]] = []
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
            if visit_icd_codes: # If there are ICD diagnosis/inpatient procedure codes
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (admission_time - previous_admission_time).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)
            else: # Add missingness token if there are no ICD diagnosis/inpatient procedure codes
                all_icd_codes.append([self.TOKEN_REPRESENTING_MISSING_TEXT])
                all_icd_times.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

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
                        all_lab_times.append((lab_ts - admission_time).total_seconds() / 3600.0)
                else: # If missing lab for a given admission
                    all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
                    all_lab_times.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

        if len(all_lab_values) == 0: # If missing lab for ALL admissions
            all_lab_values.append([self.TOKEN_REPRESENTING_MISSING_FLOAT] * len(self.LAB_CATEGORY_NAMES))
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_times.append(self.TOKEN_REPRESENTING_MISSING_FLOAT)

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


class MultimodalMortalityHorizonMIMIC4(BaseTask):
    """Admission-level in-hospital mortality prediction with a configurable time horizon.

    Observes the first ``observation_window_hours`` of an admission and labels
    whether the patient dies within the following ``prediction_horizon_hours``.
    Mortality time falls back to ``dischtime`` when ``hospital_expire_flag == 1``
    and ``deathtime`` is unavailable. Missing modality data uses placeholder
    tokens so every sample remains valid for unified multimodal embedding.

    Inputs: ICD codes (stagenet), lab vectors (stagenet_tensor), and optionally
    discharge/radiology notes (tuple_time_text).

    Args:
        observation_window_hours: Hours of EHR data collected from admission start.
        prediction_horizon_hours: Hours after the observation window to predict over.
        include_notes: Include discharge/radiology notes as a third modality.
        tokenizer_model: HuggingFace tokenizer for note tokenization.
        min_age: Minimum patient age (uses MIMIC-IV ``anchor_age``).
        padding: Padding token index for StageNet ICD sequences.

    Examples:
        >>> task = MultimodalMortalityHorizonMIMIC4(
        ...     observation_window_hours=24,
        ...     prediction_horizon_hours=12,
        ... )
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "MultimodalMortalityHorizonMIMIC4"
    output_schema: Dict[str, str] = {"mortality": "binary"}

    MISSING_TEXT_TOKEN: ClassVar[str] = ""
    MISSING_FLOAT_TOKEN: ClassVar[float] = 0.0
    MISSING_CODE_TOKEN: ClassVar[str] = "<missing_code>"

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
        "Sodium",
        "Potassium",
        "Chloride",
        "Bicarbonate",
        "Glucose",
        "Calcium",
        "Magnesium",
        "Anion Gap",
        "Osmolality",
        "Phosphate",
    ]

    LABITEMS: ClassVar[List[str]] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

    def __init__(
        self,
        observation_window_hours: int = 24,
        prediction_horizon_hours: int = 12,
        include_notes: bool = False,
        tokenizer_model: str = "bert-base-uncased",
        min_age: int = 18,
        padding: int = 0,
    ):
        self.observation_window_hours = observation_window_hours
        self.prediction_horizon_hours = prediction_horizon_hours
        self.include_notes = include_notes
        self.tokenizer_model = tokenizer_model
        self.min_age = min_age
        self.padding = padding

        self.input_schema: Dict[str, Union[str, Tuple[str, Dict[str, Any]]]] = {
            "icd_codes": ("stagenet", {"padding": padding}),
            "labs": ("stagenet_tensor", {}),
        }
        if include_notes:
            self.input_schema["discharge_note_times"] = (
                "tuple_time_text",
                {"tokenizer_model": tokenizer_model, "type_tag": "note"},
            )
            self.input_schema["radiology_note_times"] = (
                "tuple_time_text",
                {"tokenizer_model": tokenizer_model, "type_tag": "note"},
            )

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

    def _is_adult(self, patient: Any) -> bool:
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return False
        anchor_age = getattr(demographics[0], "anchor_age", None)
        try:
            return int(float(anchor_age)) >= self.min_age
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _to_hours(delta_seconds: float) -> float:
        return delta_seconds / 3600.0

    def _collect_icd_codes(self, patient: Any, hadm_id: Any) -> List[str]:
        diagnoses = patient.get_events(
            event_type="diagnoses_icd", filters=[("hadm_id", "==", hadm_id)]
        )
        procedures = patient.get_events(
            event_type="procedures_icd", filters=[("hadm_id", "==", hadm_id)]
        )

        codes = [
            str(code).strip()
            for code in [getattr(event, "icd_code", None) for event in diagnoses + procedures]
            if code is not None and str(code).strip()
        ]
        if len(codes) == 0:
            return [self.MISSING_CODE_TOKEN]
        return codes

    def _collect_labs(
        self,
        patient: Any,
        admission_time: datetime,
        prediction_time: datetime,
    ) -> Tuple[List[float], List[List[float]]]:
        try:
            import polars as pl
        except ImportError as exc:
            raise ImportError(
                "Polars is required for MultimodalMortalityHorizonMIMIC4."
            ) from exc

        labevents_df = patient.get_events(
            event_type="labevents",
            start=admission_time,
            end=prediction_time,
            return_df=True,
        )

        if labevents_df is None or labevents_df.height == 0:
            return (
                [self.MISSING_FLOAT_TOKEN],
                [[self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)],
            )

        labevents_df = labevents_df.filter(
            pl.col("labevents/itemid").is_in(self.LABITEMS)
        )
        if labevents_df.height == 0:
            return (
                [self.MISSING_FLOAT_TOKEN],
                [[self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)],
            )

        labevents_df = labevents_df.with_columns(
            pl.col("labevents/storetime")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S", strict=False)
            .alias("parsed_storetime")
        )
        labevents_df = labevents_df.filter(
            pl.col("parsed_storetime").is_not_null() & (pl.col("parsed_storetime") <= prediction_time)
        )
        if labevents_df.height == 0:
            return (
                [self.MISSING_FLOAT_TOKEN],
                [[self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)],
            )

        labevents_df = labevents_df.select(
            pl.col("timestamp"),
            pl.col("labevents/itemid"),
            pl.col("labevents/valuenum").cast(pl.Float64),
        )

        lab_times: List[float] = []
        lab_values: List[List[float]] = []
        unique_timestamps = sorted(labevents_df["timestamp"].unique().to_list())

        for lab_ts in unique_timestamps:
            ts_labs = labevents_df.filter(pl.col("timestamp") == lab_ts)
            lab_vector: List[float] = []

            for category_name in self.LAB_CATEGORY_NAMES:
                category_value = self.MISSING_FLOAT_TOKEN
                for itemid in self.LAB_CATEGORIES[category_name]:
                    matching = ts_labs.filter(pl.col("labevents/itemid") == itemid)
                    if matching.height > 0:
                        value = matching["labevents/valuenum"][0]
                        category_value = (
                            float(value)
                            if value is not None
                            else self.MISSING_FLOAT_TOKEN
                        )
                        break
                lab_vector.append(category_value)

            time_from_admission = self._to_hours(
                (lab_ts - admission_time).total_seconds()
            )
            if time_from_admission < 0:
                continue
            lab_times.append(time_from_admission)
            lab_values.append(lab_vector)

        if len(lab_values) == 0:
            return (
                [self.MISSING_FLOAT_TOKEN],
                [[self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)],
            )
        return lab_times, lab_values

    def _collect_notes(
        self,
        patient: Any,
        note_event_type: str,
        hadm_id: Any,
        admission_time: datetime,
        prediction_time: datetime,
    ) -> Tuple[List[str], List[float]]:
        notes = patient.get_events(
            event_type=note_event_type,
            filters=[("hadm_id", "==", hadm_id)],
        )

        texts: List[str] = []
        note_times: List[float] = []
        for note in notes:
            text = getattr(note, "text", None)
            timestamp = getattr(note, "timestamp", None)
            if not text or timestamp is None:
                continue
            if timestamp > prediction_time:
                continue

            time_from_admission = self._to_hours(
                (timestamp - admission_time).total_seconds()
            )
            if time_from_admission < 0:
                continue

            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue
            texts.append(cleaned_text)
            note_times.append(time_from_admission)

        if len(texts) == 0:
            return [self.MISSING_TEXT_TOKEN], [self.MISSING_FLOAT_TOKEN]
        return texts, note_times

    def _mortality_in_horizon(
        self,
        admission: Any,
        prediction_time: datetime,
        horizon_end_time: datetime,
        admission_dischtime: datetime,
    ) -> int:
        death_time = self._parse_datetime(getattr(admission, "deathtime", None))
        if death_time is None:
            try:
                if int(getattr(admission, "hospital_expire_flag", 0)) == 1:
                    death_time = admission_dischtime
            except (TypeError, ValueError):
                death_time = None
        if death_time is None:
            return 0
        return int(prediction_time < death_time <= horizon_end_time)

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        if not self._is_adult(patient):
            return []

        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return []

        samples: List[Dict[str, Any]] = []

        for admission in admissions:
            admission_time = getattr(admission, "timestamp", None)
            if admission_time is None:
                continue

            admission_dischtime = self._parse_datetime(getattr(admission, "dischtime", None))
            if admission_dischtime is None or admission_dischtime <= admission_time:
                continue

            prediction_time = admission_time + timedelta(hours=self.observation_window_hours)
            horizon_end_time = prediction_time + timedelta(hours=self.prediction_horizon_hours)
            # Skip only if the patient was discharged before the observation window ends —
            # we cannot collect meaningful observation data for them.
            # Do NOT skip if death occurs within the horizon window (those are our positives).
            if admission_dischtime <= prediction_time:
                continue

            hadm_id = getattr(admission, "hadm_id", None)
            if hadm_id is None:
                continue

            icd_codes = self._collect_icd_codes(patient, hadm_id)
            icd_times = [self.MISSING_FLOAT_TOKEN] * len(icd_codes)
            lab_times, lab_values = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                prediction_time=prediction_time,
            )
            mortality = self._mortality_in_horizon(
                admission=admission,
                prediction_time=prediction_time,
                horizon_end_time=horizon_end_time,
                admission_dischtime=admission_dischtime,
            )

            sample: Dict[str, Any] = {
                "patient_id": patient.patient_id,
                "visit_id": str(hadm_id),
                "prediction_time_hours": float(self.observation_window_hours),
                "icd_codes": (icd_times, icd_codes),
                "labs": (lab_times, lab_values),
                "mortality": mortality,
            }

            if self.include_notes:
                discharge_texts, discharge_times = self._collect_notes(
                    patient=patient,
                    note_event_type="discharge",
                    hadm_id=hadm_id,
                    admission_time=admission_time,
                    prediction_time=prediction_time,
                )
                radiology_texts, radiology_times = self._collect_notes(
                    patient=patient,
                    note_event_type="radiology",
                    hadm_id=hadm_id,
                    admission_time=admission_time,
                    prediction_time=prediction_time,
                )
                sample["discharge_note_times"] = (discharge_texts, discharge_times)
                sample["radiology_note_times"] = (radiology_texts, radiology_times)

            samples.append(sample)

        return samples
