import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple, ClassVar

from pyhealth.tasks.base_task import BaseTask


class BaseMultimodalMIMIC4Task(BaseTask):
    """Base class for multimodal MIMIC-IV tasks.

    Provides shared constants and utility methods used across all multimodal
    task variants (notes, ICD codes, lab values).
    """

    MISSING_TEXT_TOKEN: ClassVar[str] = ""
    MISSING_CODE_TOKEN: ClassVar[str] = "<missing>"
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
        window_hours: Optional[float] = None,
    ):
        self.window_hours = window_hours

    @staticmethod
    def _extract_section(
        text: str, section: Union[str, List[str]]
    ) -> Optional[str]:
        """Extract one or more named sections from a MIMIC-IV discharge note.
        """
        sections = [section] if isinstance(section, str) else section
        parts = []
        for name in sections:
            pattern = re.compile(
                rf"(?m)^[ \t]*{re.escape(name)}:[ \t]*\n(.*?)(?=^[ \t]*\S[^\n]*:[ \t]*\n|\Z)",
                re.DOTALL | re.IGNORECASE,
            )
            m = pattern.search(text)
            if m:
                extracted = m.group(1).strip()
                if extracted:
                    parts.append(extracted)
        return "\n\n".join(parts) if parts else None

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

    def _compute_effective_window(
        self,
        admissions_to_process: List[Any],
    ) -> Tuple[datetime, Optional[datetime]]:
        """Compute effective start/end from the global span of processed admissions.

        Returns:
            Tuple of (effective_start, effective_end).
        """
        global_start = admissions_to_process[0].timestamp
        global_end: Optional[datetime] = None

        for a in admissions_to_process:
            dt = self._parse_datetime(getattr(a, "dischtime", None))
            if dt is not None and (global_end is None or dt > global_end):
                global_end = dt

        if self.window_hours is not None:
            effective_start = global_start
            effective_end = effective_start + timedelta(hours=self.window_hours)
            return effective_start, effective_end

        effective_start = global_start
        effective_end = global_end

        return effective_start, effective_end

    def _build_admissions_to_process(self, patient: Any) -> Tuple[List[Any], int]:
        """Build admissions to process and derive mortality label.

        The task includes admissions until the first mortality event and labels
        the sample as positive if death occurs in the current or next admission.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return [], 0

        admissions_to_process: List[Any] = []
        mortality_label = 0

        for i, admission in enumerate(admissions):
            if admission.hospital_expire_flag in [1, "1"]:
                mortality_label = 1
                break

            if i + 1 < len(admissions):
                next_admission = admissions[i + 1]
                if next_admission.hospital_expire_flag in [1, "1"]:
                    admissions_to_process.append(admission)
                    mortality_label = 1
                    break

            admissions_to_process.append(admission)

        return admissions_to_process, mortality_label

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
        return [
            e.icd_code for e in diagnoses_icd if hasattr(e, "icd_code") and e.icd_code
        ] + [
            e.icd_code for e in procedures_icd if hasattr(e, "icd_code") and e.icd_code
        ]

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
            raise ImportError("Polars is required for lab collection.") from exc

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
                pl.col("labevents/storetime").str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S"
                )
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
                            matching = ts_labs.filter(
                                pl.col("labevents/itemid") == itemid
                            )
                            if matching.height > 0:
                                category_value = matching["labevents/valuenum"][0]
                                observed = True
                                break
                        lab_vector.append(category_value)
                        lab_mask.append(observed)
                    lab_times.append(
                        self._to_hours((lab_ts - admission_time).total_seconds())
                    )
                    lab_values.append(lab_vector)
                    lab_masks.append(lab_mask)
            else:  # If missing lab for a given admission
                lab_values.append(
                    [self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)
                )
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
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        section: Optional[Union[str, List[str]]] = None,
    ) -> Tuple[List[str], List[float]]:
        """Collect notes of a given type for one admission.

        Args:
            patient: Patient object.
            note_event_type: Event type string (e.g. "discharge", "radiology").
            hadm_id: Admission ID to filter by.
            admission_time: Admission start time; used to compute time offsets.
            start_time: Optional start of the time window
            end_time: Optional end of the time window
            section: Optional section name or list of section names to extract
                from each note

        Returns:
            Tuple of (texts, hours_from_admission). Falls back to
            ``([MISSING_TEXT_TOKEN], [MISSING_FLOAT_TOKEN])`` when the events
            list is empty.
        """
        notes = patient.get_events(
            event_type=note_event_type,
            start=start_time,
            end=end_time,
            filters=[("hadm_id", "==", hadm_id)],
        )

        texts: List[str] = []
        note_times: List[float] = []
        for note in notes:
            try:
                raw = note.text
                if section is not None and raw:
                    raw = self._extract_section(raw, section)
                note_text = self._clean_text(raw)
                if note_text:
                    time_from_admission = self._to_hours(
                        (note.timestamp - admission_time).total_seconds()
                    )
                    texts.append(note_text)
                    note_times.append(time_from_admission)
            except (
                AttributeError
            ):  # note object is missing .text or .timestamp attribute (e.g. malformed note)
                pass

        if (
            not notes or not texts
        ):  # If we get an empty list or all notes were malformed
            return [self.MISSING_TEXT_TOKEN], [
                self.MISSING_FLOAT_TOKEN
            ]  # Token representing missing text/time
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
        ),
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]

        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )

        if len(admissions_to_process) == 0:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )

        # Aggregated notes and time offsets across all admissions (per hadm_id)
        all_discharge_texts: List[str] = []
        all_discharge_times_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_times_from_admission: List[float] = []

        # Process each admission independently (per hadm_id)
        for admission in admissions_to_process:
            admission_time = admission.timestamp

            discharge_texts, discharge_times = self._collect_notes(
                patient,
                "discharge",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=effective_end,
                section=["Chief Complaint", "History of Present Illness"],
            )
            all_discharge_texts.extend(discharge_texts)
            all_discharge_times_from_admission.extend(discharge_times)

            radiology_texts, radiology_times = self._collect_notes(
                patient,
                "radiology",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=effective_end,
                section="Findings",
            )
            all_radiology_texts.extend(radiology_texts)
            all_radiology_times_from_admission.extend(radiology_times)

        discharge_note_times_from_admission = (
            all_discharge_texts,
            all_discharge_times_from_admission,
        )
        radiology_note_times_from_admission = (
            all_radiology_texts,
            all_radiology_times_from_admission,
        )

        if len(all_discharge_texts) == 0:
            discharge_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )
        if len(all_radiology_texts) == 0:
            radiology_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )

        single_patient_longitudinal_record = {
            "patient_id": patient.patient_id,
            "discharge_note_times": discharge_note_times_from_admission,
            "radiology_note_times": radiology_note_times_from_admission,
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        return [single_patient_longitudinal_record]


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

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        # Get demographic info to filter by age
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]

        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )

        if len(admissions_to_process) == 0:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )

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

            # Some rows can have missing/malformed discharge timestamps.
            # Do not skip the entire admission, or downstream temporal
            # processors may receive empty time sequences for this patient.
            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # Skip malformed admissions; note collectors are not called for these.
                continue

            if admission_dischtime < admission_time:
                # Guard against invalid chronology in source records.
                continue

            discharge_texts, discharge_times = self._collect_notes(
                patient,
                "discharge",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=effective_end,
                section=["Chief Complaint", "History of Present Illness"],
            )
            all_discharge_texts.extend(discharge_texts)
            all_discharge_times_from_admission.extend(discharge_times)

            radiology_texts, radiology_times = self._collect_notes(
                patient,
                "radiology",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=effective_end,
                section="Findings",
            )
            all_radiology_texts.extend(radiology_texts)
            all_radiology_times_from_admission.extend(radiology_times)

            # ICD codes (diagnoses + procedures) with time relative to previous admission
            visit_icd_codes = self._collect_icd_codes(patient, admission.hadm_id)
            if visit_icd_codes:  # If there are ICD diagnosis/inpatient procedure codes
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = (
                        admission_time - previous_admission_time
                    ).total_seconds() / 3600.0
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)
            else:  # Add missingness token if there are no ICD diagnosis/inpatient procedure codes
                all_icd_codes.append([self.MISSING_CODE_TOKEN])
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

        if len(all_lab_values) == 0:  # If missing lab for ALL admissions
            all_lab_values.append(
                [self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)
            )
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_times.append(self.MISSING_FLOAT_TOKEN)

        # If all admissions were skipped before ICD collection, ensure a
        # single placeholder step so StageNetProcessor does not emit None time.
        if len(all_icd_codes) == 0:
            all_icd_codes.append([self.MISSING_CODE_TOKEN])
            all_icd_times.append(self.MISSING_FLOAT_TOKEN)

        discharge_note_times_from_admission = (
            all_discharge_texts,
            all_discharge_times_from_admission,
        )
        radiology_note_times_from_admission = (
            all_radiology_texts,
            all_radiology_times_from_admission,
        )

        # Per-admission note fallback happens inside _collect_notes().
        # This final guard handles the edge case where every admission was
        # skipped before _collect_notes() was reached.
        if len(all_discharge_texts) == 0:
            discharge_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )
        if len(all_radiology_texts) == 0:
            radiology_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )

        single_patient_longitudinal_record = {
            "patient_id": patient.patient_id,
            "discharge_note_times": discharge_note_times_from_admission,
            "radiology_note_times": radiology_note_times_from_admission,
            "icd_codes": (all_icd_times, all_icd_codes),
            "labs": (all_lab_times, all_lab_values),
            "labs_mask": (all_lab_times, all_lab_masks),
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        return [single_patient_longitudinal_record]


class ICDLabsMIMIC4(BaseMultimodalMIMIC4Task):
    """Task for ICD codes + lab values mortality prediction using MIMIC-IV.

    A notes-free variant of ``ClinicalNotesICDLabsMIMIC4`` that uses only:

    - **ICD codes**: diagnosis and procedure codes per admission, processed by
      ``StageNetProcessor`` with inter-admission time offsets.
    - **Lab values**: 10-dimensional lab vectors (one per lab category) at each
      measurement timestamp, processed by ``StageNetTensorProcessor``.

    Examples:
        >>> from pyhealth.datasets import MIMIC4Dataset
        >>> from pyhealth.tasks.multimodal_mimic4 import ICDLabsMIMIC4
        >>> dataset = MIMIC4Dataset(
        ...     ehr_root="/path/to/mimic-iv/2.2",
        ...     ehr_tables=["diagnoses_icd", "procedures_icd", "labevents"],
        ... )
        >>> task = ICDLabsMIMIC4()
        >>> samples = dataset.set_task(task)
    """

    PADDING: int = 0

    task_name: str = "ICDLabsMIMIC4"
    input_schema: Dict[str, Union[str, Tuple[str, Dict]]] = {
        "icd_codes": ("stagenet", {"padding": PADDING}),
        "labs": ("stagenet_tensor", {}),
        "labs_mask": ("stagenet_tensor", {}),
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )

        if len(admissions_to_process) == 0:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )

        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_masks: List[List[bool]] = []
        all_lab_times: List[float] = []
        previous_admission_time = None

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

            visit_icd_codes = self._collect_icd_codes(patient, admission.hadm_id)
            if visit_icd_codes:
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = self._to_hours(
                        (admission_time - previous_admission_time).total_seconds()
                    )
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)
            else:
                all_icd_codes.append([self.MISSING_CODE_TOKEN])
                all_icd_times.append(self.MISSING_FLOAT_TOKEN)

            previous_admission_time = admission_time

            lab_times, lab_values, lab_masks = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                end_time=admission_dischtime,
            )
            all_lab_times.extend(lab_times)
            all_lab_values.extend(lab_values)
            all_lab_masks.extend(lab_masks)

        if len(all_lab_values) == 0:
            all_lab_values.append(
                [self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)
            )
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_times.append(self.MISSING_FLOAT_TOKEN)

        if len(all_icd_codes) == 0:
            all_icd_codes.append([self.MISSING_CODE_TOKEN])
            all_icd_times.append(self.MISSING_FLOAT_TOKEN)

        single_patient_longitudinal_record = {
            "patient_id": patient.patient_id,
            "icd_codes": (all_icd_times, all_icd_codes),
            "labs": (all_lab_times, all_lab_values),
            "labs_mask": (all_lab_times, all_lab_masks),
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        return [single_patient_longitudinal_record]


class ClinicalNotesICDLabsCXRMIMIC4(BaseMultimodalMIMIC4Task):
    """Task combining notes, ICD, labs, and CXR for MIMIC-IV mortality.

    Adds temporally filtered CXR signals on top of ``ClinicalNotesICDLabsMIMIC4``:
    - ``cxr_image_times``: ``(image_paths, hours_from_admission)`` processed by
      ``TimeImageProcessor``.

        CXR filtering uses event timestamps from the metadata table. Since
    timestamps are built from StudyDate+StudyTime in dataset configs, studytime
    is naturally respected in temporal windows.
    """

    PADDING: int = 0

    task_name: str = "ClinicalNotesICDLabsCXRMIMIC4"
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
        "cxr_image_times": (
            "time_image",
            {
                "image_size": 224,
                "mode": "RGB",
                "padding": "",
            },
        ),
    }
    output_schema: Dict[str, str] = {"mortality": "binary"}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )
        if len(admissions_to_process) == 0:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )

        all_discharge_texts: List[str] = []
        all_discharge_times_from_admission: List[float] = []
        all_radiology_texts: List[str] = []
        all_radiology_times_from_admission: List[float] = []
        all_icd_codes: List[List[str]] = []
        all_icd_times: List[float] = []
        all_lab_values: List[List[float]] = []
        all_lab_masks: List[List[bool]] = []
        all_lab_times: List[float] = []
        all_cxr_paths: List[str] = []
        all_cxr_times: List[float] = []
        previous_admission_time = None

        for admission in admissions_to_process:
            admission_time = admission.timestamp

            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                # Skip malformed admissions; note collectors are not called for these.
                continue

            if admission_dischtime < admission_time:
                # Guard against invalid chronology in source records.
                continue

            admission_end = admission_dischtime
            if effective_end is not None and effective_end < admission_end:
                admission_end = effective_end

            discharge_texts, discharge_times = self._collect_notes(
                patient,
                "discharge",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=admission_end,
                section=["Chief Complaint", "History of Present Illness"],
            )
            all_discharge_texts.extend(discharge_texts)
            all_discharge_times_from_admission.extend(discharge_times)

            radiology_texts, radiology_times = self._collect_notes(
                patient,
                "radiology",
                admission.hadm_id,
                admission_time,
                start_time=effective_start,
                end_time=admission_end,
            )
            all_radiology_texts.extend(radiology_texts)
            all_radiology_times_from_admission.extend(radiology_times)

            visit_icd_codes = self._collect_icd_codes(patient, admission.hadm_id)
            if visit_icd_codes:
                if previous_admission_time is None:
                    time_from_previous = 0.0
                else:
                    time_from_previous = self._to_hours(
                        (admission_time - previous_admission_time).total_seconds()
                    )
                all_icd_codes.append(visit_icd_codes)
                all_icd_times.append(time_from_previous)
            else:
                all_icd_codes.append([self.MISSING_CODE_TOKEN])
                all_icd_times.append(self.MISSING_FLOAT_TOKEN)

            previous_admission_time = admission_time

            lab_times, lab_values, lab_masks = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                end_time=admission_dischtime,
            )
            all_lab_times.extend(lab_times)
            all_lab_values.extend(lab_values)
            all_lab_masks.extend(lab_masks)

            # CXR metadata is filtered by timestamp; this includes StudyTime.
            metadata_events = patient.get_events(
                event_type="metadata",
                start=admission_time,
                end=admission_end,
            )
            for event in metadata_events:
                try:
                    if event.image_path:
                        all_cxr_paths.append(event.image_path)
                        all_cxr_times.append(
                            self._to_hours(
                                (event.timestamp - admission_time).total_seconds()
                            )
                        )
                except AttributeError:
                    continue

        if len(all_lab_values) == 0:
            all_lab_values.append(
                [self.MISSING_FLOAT_TOKEN] * len(self.LAB_CATEGORY_NAMES)
            )
            all_lab_masks.append([False] * len(self.LAB_CATEGORY_NAMES))
            all_lab_times.append(self.MISSING_FLOAT_TOKEN)

        # If all admissions were skipped before ICD collection, ensure a
        # single placeholder step so StageNetProcessor does not emit None time.
        if len(all_icd_codes) == 0:
            all_icd_codes.append([self.MISSING_CODE_TOKEN])
            all_icd_times.append(self.MISSING_FLOAT_TOKEN)

        # time_image processor expects at least one path/time pair.
        if len(all_cxr_paths) == 0:
            all_cxr_paths = [self.MISSING_TEXT_TOKEN]
            all_cxr_times = [self.MISSING_FLOAT_TOKEN]

        discharge_note_times_from_admission = (
            all_discharge_texts,
            all_discharge_times_from_admission,
        )
        radiology_note_times_from_admission = (
            all_radiology_texts,
            all_radiology_times_from_admission,
        )

        # Per-admission note fallback happens inside _collect_notes().
        # This final guard handles the edge case where every admission was
        # skipped before _collect_notes() was reached.
        if len(all_discharge_texts) == 0:
            discharge_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )
        if len(all_radiology_texts) == 0:
            radiology_note_times_from_admission = (
                [self.MISSING_TEXT_TOKEN],
                [self.MISSING_FLOAT_TOKEN],
            )

        single_patient_longitudinal_record = {
            "patient_id": patient.patient_id,
            "discharge_note_times": discharge_note_times_from_admission,
            "radiology_note_times": radiology_note_times_from_admission,
            "icd_codes": (all_icd_times, all_icd_codes),
            "labs": (all_lab_times, all_lab_values),
            "labs_mask": (all_lab_times, all_lab_masks),
            "cxr_image_times": (all_cxr_paths, all_cxr_times),
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        return [single_patient_longitudinal_record]
