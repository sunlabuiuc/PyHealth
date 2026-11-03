import logging
import re
from datetime import datetime, timedelta
from typing import Any, ClassVar

from pyhealth.tasks.base_task import BaseTask

logger = logging.getLogger(__name__)


class BaseMultimodalMIMIC4Task(BaseTask):
    """Base class for multimodal MIMIC-IV tasks.

    Provides shared constants and utility methods used across all multimodal
    task variants (notes, ICD codes, lab values). CXR task classes build on
    this base in a follow-up commit: they add a ``cxr_image_times`` field
    and bound per-admission ``metadata`` CXR events with
    :meth:`_admission_window_end`.

    The shared observation window is ``window_hours``: ``0`` (default)
    means no cap (full stay through discharge); a positive value caps each
    admission at ``min(admit + N h, discharge)``. Discharge notes stay
    unbounded by design: the collected sections are admission context
    stamped at admit, so windowing them would drop context without
    removing any post-admit information.

    Example:
        >>> from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4
        >>> task = LabsMIMIC4()
        >>> task.window_hours
        0
    """

    MISSING_TEXT_TOKEN: ClassVar[str] = ""
    MISSING_CODE_TOKEN: ClassVar[str] = "<missing>"
    MISSING_FLOAT_TOKEN: ClassVar[float] = 0.0

    LAB_CATEGORIES: ClassVar[dict[str, list[str]]] = {
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

    LAB_CATEGORY_NAMES: ClassVar[list[str]] = [
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

    LABITEMS: ClassVar[list[str]] = [
        item for itemids in LAB_CATEGORIES.values() for item in itemids
    ]

    RADIOLOGY_CLINICAL_HEADERS: ClassVar[list[str]] = [
        "indication",
        "impression",
        # "findings",
        # "clinical history",
        # "history",
        # "comparison",
        # "technique",
        # "conclusion",
        # "summary"
    ]

    DISCHARGE_CLINICAL_HEADERS: ClassVar[list[str]] = [
        "chief complaint",
        # "history of present illness",
        # "hpi",
        # "past medical history",
        # "past medical and surgical history",
        # "past medical/surgical history",
        # "past surgical history",
        # "medications on admission",
        # "admission medications",
        # "home medications",
        # "social history",
        # "family history",
        # "allergies",
        # "review of systems",
    ]

    def __init__(
        self,
        window_hours: float = 0,
    ):
        if window_hours is None or window_hours < 0:
            raise ValueError(
                "window_hours must be a non-negative number of hours "
                f"(0 = full stay through discharge); got {window_hours!r}."
            )
        self.window_hours = window_hours
        # The task cache key is uuid5 over {**vars(task), schemas}, so this
        # field participates in it. Bump it whenever the emitted samples
        # change so stale caches cannot be reused.
        self.emitted_data_version = 1

    @staticmethod
    def _clean_text(text: str | None) -> str | None:
        """Return text if non-empty, otherwise None."""
        return text if text else None

    @staticmethod
    def _parse_note_sections(text: str, note_type: str) -> dict[str, str]:
        """Split a note into {lowercased_header: content_text} pairs."""
        ext_text = text + '\n\n'
        if note_type == "radiology":
            section_re = re.compile(r'([a-zA-Z ]+):[ \t\n]+(.+?)\n{2,}', re.DOTALL)
        elif note_type == "discharge":
            section_re = re.compile(r'([a-zA-Z ]+):\n+(.+?)\n{2,}', re.DOTALL)
        else:
            raise ValueError(f"Note Type '{note_type}' not supported.")
        return {
            m.group(1).strip().lower(): m.group(2).strip()
            for m in section_re.finditer(ext_text)
            if m.end() - m.start() > 0
        }

    @staticmethod
    def _parse_datetime(value: Any) -> datetime | None:
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

    @classmethod
    def _hours_since(cls, timestamp: datetime, origin: datetime) -> float:
        """Hours from ``origin`` to ``timestamp``.

        Collection windows stay per admission (full stay, or admit+window
        when ``window_hours`` is set). The value written onto the unified
        timeline is hours from the first stay in this sample, so a later
        stay at +6h does not sort with the first stay at +6h.
        """
        return cls._to_hours((timestamp - origin).total_seconds())

    def _compute_effective_window(
        self,
        admissions_to_process: list[Any],
    ) -> tuple[datetime, datetime | None]:
        """Compute effective start/end from the global span of processed admissions.

        Returns:
            Tuple of (effective_start, effective_end).
        """
        global_start = admissions_to_process[0].timestamp
        global_end: datetime | None = None

        for a in admissions_to_process:
            dt = self._parse_datetime(getattr(a, "dischtime", None))
            if dt is not None and (global_end is None or dt > global_end):
                global_end = dt

        if self.window_hours > 0:
            effective_start = global_start
            effective_end = effective_start + timedelta(hours=self.window_hours)
            return effective_start, effective_end

        effective_start = global_start
        effective_end = global_end

        return effective_start, effective_end

    def _admission_window_end(
        self,
        admission_time: datetime,
        admission_dischtime: datetime,
    ) -> datetime:
        """End of the observation window for one admission.

        Callers previously passed ``admission_dischtime`` directly, so
        ``window_hours`` was inert and labs were collected through discharge.
        For a mortality label that reads the outcome. Re-anchor per admission
        and clamp to discharge so a later stay cannot inherit the first
        admission's window.
        """
        if self.window_hours == 0:
            return admission_dischtime
        end = admission_time + timedelta(hours=self.window_hours)
        return min(end, admission_dischtime) if admission_dischtime else end

    def _build_admissions_to_process(self, patient: Any) -> tuple[list[Any], int]:
        """Build admissions to process and derive mortality label.

        Includes all admissions up to and including the first death admission.
        Patients who die in their first (and only) admission are included as
        positives — previously they were dropped, which silently discarded most
        ICU mortality positives and collapsed positive rate from ~10% to ~2.7%.
        This now matches stagenet's semantics: use all available admission data
        and label as positive if any admission has hospital_expire_flag=1.
        """
        admissions = patient.get_events(event_type="admissions")
        if len(admissions) == 0:
            return [], 0

        admissions_to_process: list[Any] = []
        mortality_label = 0

        for admission in admissions:
            admissions_to_process.append(admission)
            if admission.hospital_expire_flag in [1, "1"]:
                mortality_label = 1
                break

        return admissions_to_process, mortality_label

    def _collect_icd_codes(self, patient: Any, hadm_id: Any) -> list[str]:
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
        time_origin: datetime,
    ) -> tuple[list[float], list[list[float]], list[list[bool]]]:
        """Collect lab values and observation masks for one admission.

        Args:
            patient: Patient object.
            admission_time: Start of this stay's collection window.
            end_time: End of the window (inclusive).
            time_origin: First stay in this sample. Event times are hours
                from here, not from ``admission_time``.

        Returns:
            Tuple of (lab_times, lab_values, lab_masks). ``lab_masks`` is a
            parallel boolean tensor where ``True`` means observed and ``False``
            means imputed with 0.0. Returns empty lists when no valid lab
            events are found; do not invent a placeholder row.
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

        lab_times: list[float] = []
        lab_values: list[list[float]] = []
        lab_masks: list[list[bool]] = []

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
                    lab_vector: list[float] = []
                    lab_mask: list[bool] = []
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
                    lab_times.append(self._hours_since(lab_ts, time_origin))
                    lab_values.append(lab_vector)
                    lab_masks.append(lab_mask)
        return lab_times, lab_values, lab_masks

    def _collect_notes(
        self,
        patient: Any,
        note_event_type: str,
        hadm_id: Any,
        admission_time: datetime,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
        section_headers: list[str] | None = None,
        fallback_to_full_note: bool = False,
        time_origin: datetime | None = None,
        event_time: datetime | None = None,
    ) -> tuple[list[str], list[float]]:
        """Collect notes of a given type for one admission.

        Args:
            patient: Patient object.
            note_event_type: Event type string (e.g. "discharge", "radiology").
            hadm_id: Admission ID to filter by.
            admission_time: This stay's admit time. Used as the timeline
                origin when ``time_origin`` is omitted.
            start_time: Optional start of the time window.
            end_time: Optional end of the time window.
            section_headers: When provided, extract only these named sections
                from each note (lowercased match against parsed headers).
            fallback_to_full_note: When True, falls back to the full note text
                if no matching sections are found. When False, notes with no
                matching sections are dropped entirely.
            time_origin: Timeline origin. Hours are measured from here.
            event_time: If set, stamp every collected note at this instant.
                Discharge-section text is admission-context and should pass
                this stay's admit. Radiology omits it and uses exam
                ``charttime``.

        Returns:
            Tuple of (texts, hours from the sample's first stay). Empty lists
            when the events list is empty; do not invent a placeholder note.
        """
        notes = patient.get_events(
            event_type=note_event_type,
            start=start_time,
            end=end_time,
            filters=[("hadm_id", "==", hadm_id)],
        )

        texts: list[str] = []
        note_times: list[float] = []
        for note in notes:
            try:
                note_text = self._clean_text(note.text)
                if note_text:
                    if section_headers is not None:
                        parsed = self._parse_note_sections(note_text, note_type=note_event_type)
                        extracted = [f"{k}: {v}" for k, v in parsed.items() if k in section_headers and v]
                        if extracted:
                            note_text = " [SEP] ".join(extracted)
                        elif not fallback_to_full_note:
                            continue

                    origin = time_origin if time_origin is not None else admission_time
                    stamp = event_time if event_time is not None else note.timestamp
                    texts.append(note_text)
                    note_times.append(self._hours_since(stamp, origin))
            except (
                AttributeError
            ):  # note object is missing .text or .timestamp attribute (e.g. malformed note)
                pass

        return texts, note_times


class NotesLabsMIMIC4(BaseMultimodalMIMIC4Task):
    """Mortality prediction from admission-context notes and lab values.

    Follows the approach of Lee et al. (2023): use text that is clinically
    available *at admission* rather than discharge notes. ICD codes are
    excluded by default but can be re-enabled for ablation experiments.

    Text is extracted from the MIMIC-IV discharge note by parsing the Chief
    Complaint, History of Present Illness, Past Medical History, and Medications
    on Admission sections — all of which describe the patient's state at the
    start of the stay. The extracted text is stamped at that stay's admit.

    Radiology reports are also included, parsed for their Indication and
    Impression sections and bounded to the same observation window as labs
    (rather than that stay's admit), since — unlike the discharge summary — they
    are written at exam time and describe findings from later in the stay.

    Fields:
        admission_note_times: Admission-context discharge-note text stamped
            at that stay's admit (hours from the first stay), plus in-window
            radiology note text at exam time.
        labs: 10-dim lab vectors at each measurement timestamp.
        labs_mask: Boolean observation mask parallel to ``labs``.
        icd_codes: (only when ``include_icd=True``) Diagnosis + procedure codes
            per admission with inter-admission time offsets.

    Args:
        window_hours: Hours from admission for lab collection. ``0``
            (default) collects for the full admission span.
        include_icd: When ``True``, collect discharge-coded ICD codes and add
            ``icd_codes`` to the sample dict / input schema. Default: ``False``.
            MIMIC-IV timestamps those codes at ``dischtime``, so this leaks
            the in-hospital mortality label. Keep it off except as an
            explicit leaky ablation.

    Example:
        >>> from pyhealth.tasks.multimodal_mimic4 import NotesLabsMIMIC4
        >>> task = NotesLabsMIMIC4()
        >>> (task.task_name, task.window_hours, task.include_icd)
        ('NotesLabsMIMIC4', 0, False)
    """

    PADDING: int = 0

    task_name: str = "NotesLabsMIMIC4"

    _BASE_INPUT_SCHEMA: ClassVar[dict] = {
        "admission_note_times": (
            "tuple_time_text",
            {
                "tokenizer_model": "emilyalsentzer/Bio_ClinicalBERT",
                "type_tag": "note",
                "max_length": 512,
            },
        ),
        "labs": ("stagenet_tensor", {}),
        "labs_mask": ("stagenet_tensor", {"forward_fill": False}),
    }

    input_schema: ClassVar[dict[str, str | tuple[str, dict]]] = _BASE_INPUT_SCHEMA
    output_schema: ClassVar[dict[str, str]] = {"mortality": "binary"}

    def __init__(
        self,
        window_hours: float = 0,
        include_icd: bool = False,
    ) -> None:
        super().__init__(window_hours=window_hours)
        self.include_icd = include_icd
        schema = dict(self._BASE_INPUT_SCHEMA)
        if include_icd:
            schema["icd_codes"] = ("stagenet", {"padding": self.PADDING})
        self.input_schema = schema
        logger.info(
            "NotesLabsMIMIC4: filtering discharge notes to sections: %s; "
            "radiology notes to sections: %s",
            self.DISCHARGE_CLINICAL_HEADERS,
            self.RADIOLOGY_CLINICAL_HEADERS,
        )

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        if not patient.get_events(event_type="patients"):
            return []

        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )
        if not admissions_to_process:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )
        time_origin = admissions_to_process[0].timestamp

        all_note_texts: list[str] = []
        all_note_times: list[float] = []
        all_lab_values: list[list[float]] = []
        all_lab_masks: list[list[bool]] = []
        all_lab_times: list[float] = []
        all_icd_codes: list[list[str]] = []
        all_icd_times: list[float] = []

        for admission in admissions_to_process:
            admission_time = admission.timestamp

            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                admission_dischtime = admission_time
            admission_dischtime = max(admission_dischtime, admission_time)

            note_texts, note_times = self._collect_notes(
                patient,
                "discharge",
                admission.hadm_id,
                admission_time,
                section_headers=self.DISCHARGE_CLINICAL_HEADERS,
                time_origin=time_origin,
                event_time=admission_time,
            )
            all_note_texts.extend(note_texts)
            all_note_times.extend(note_times)

            # Labs within the observation window of THIS admission.
            lab_end = self._admission_window_end(admission_time, admission_dischtime)
            lab_times, lab_values, lab_masks = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                end_time=lab_end,
                time_origin=time_origin,
            )
            all_lab_times.extend(lab_times)
            all_lab_values.extend(lab_values)
            all_lab_masks.extend(lab_masks)

            # Radiology notes within the observation window. Unlike the
            # discharge note (a retrospective summary parsed for its
            # admission-context sections), radiology reports are written at
            # exam time, so they're bounded to the same window as labs
            # to avoid pulling in findings from later in the stay.
            radiology_texts, radiology_times = self._collect_notes(
                patient,
                "radiology",
                admission.hadm_id,
                admission_time,
                start_time=admission_time,
                end_time=lab_end,
                section_headers=self.RADIOLOGY_CLINICAL_HEADERS,
                time_origin=time_origin,
            )
            all_note_texts.extend(radiology_texts)
            all_note_times.extend(radiology_times)

            if self.include_icd:
                visit_icd_codes = self._collect_icd_codes(patient, admission.hadm_id)
                if visit_icd_codes:
                    all_icd_codes.append(visit_icd_codes)
                    all_icd_times.append(
                        self._hours_since(admission_time, time_origin)
                    )

        record: dict[str, Any] = {
            "patient_id": patient.patient_id,
            "admission_note_times": (all_note_texts, all_note_times),
            "labs": (all_lab_times, all_lab_values),
            "labs_mask": (all_lab_times, all_lab_masks),
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        if self.include_icd:
            record["icd_codes"] = (all_icd_times, all_icd_codes)

        return [record]


class LabsMIMIC4(BaseMultimodalMIMIC4Task):
    """EHR-only mortality prediction using lab values — no notes, no ICD codes.

    Serves as the structured-EHR reference baseline for multimodal ablations.
    Collecting only ``labevents`` keeps the dataset loader fast and avoids any
    leakage from discharge-coded ICD tables.

    Schema mirrors the ``labs`` / ``labs_mask`` fields from ``NotesLabsMIMIC4``
    so the same backbone models (MLP, RNN, Transformer, etc.) work unchanged.

    Args:
        window_hours: Hours from admission to collect lab measurements.
            ``0`` (default) collects for the full admission span.

    Example:
        >>> from pyhealth.tasks.multimodal_mimic4 import LabsMIMIC4
        >>> task = LabsMIMIC4()
        >>> (task.task_name, task.window_hours)
        ('LabsMIMIC4', 0)
    """

    PADDING: int = 0

    task_name: str = "LabsMIMIC4"

    input_schema: ClassVar[dict] = {
        "labs": ("stagenet_tensor", {}),
        "labs_mask": ("stagenet_tensor", {"forward_fill": False}),
    }
    output_schema: ClassVar[dict] = {"mortality": "binary"}

    def __init__(self, window_hours: float = 0) -> None:
        super().__init__(window_hours=window_hours)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:  # type: ignore[override]
        admissions_to_process, mortality_label = self._build_admissions_to_process(
            patient
        )
        if not admissions_to_process:
            return []

        effective_start, effective_end = self._compute_effective_window(
            admissions_to_process
        )
        time_origin = admissions_to_process[0].timestamp

        all_lab_times: list[float] = []
        all_lab_values: list[list[float]] = []
        all_lab_masks: list[list[bool]] = []

        for admission in admissions_to_process:
            admission_time = admission.timestamp

            try:
                admission_dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                admission_dischtime = admission_time
            admission_dischtime = max(admission_dischtime, admission_time)

            lab_times, lab_values, lab_masks = self._collect_labs(
                patient=patient,
                admission_time=admission_time,
                end_time=self._admission_window_end(
                    admission_time, admission_dischtime
                ),
                time_origin=time_origin,
            )
            all_lab_times.extend(lab_times)
            all_lab_values.extend(lab_values)
            all_lab_masks.extend(lab_masks)

        single_patient_longitudinal_record = {
            "patient_id": patient.patient_id,
            "labs": (all_lab_times, all_lab_values),
            "labs_mask": (all_lab_times, all_lab_masks),
            "mortality": mortality_label,
            "window_start": effective_start,
            "window_end": effective_end,
        }

        return [single_patient_longitudinal_record]

