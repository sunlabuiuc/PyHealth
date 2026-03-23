from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

from .base_task import BaseTask


class MultimodalMortalityHorizonMIMIC4(BaseTask):
    """Multimodal MIMIC-IV task for mortality prediction within a time horizon.

    This task creates admission-level samples using a fixed observation window
    and predicts whether in-hospital mortality occurs in the next horizon
    window. The default setting mirrors the paper-style setup conceptually:
    observe first 24h, predict mortality within next 12h.

    Modalities:
        - ICD codes (diagnoses + procedures) as StageNet sequences
        - Lab vectors as StageNet tensor sequences
        - Optional discharge/radiology notes as tuple_time_text sequences

    Notes:
        - If ``admission.deathtime`` is unavailable, the task approximates
          mortality time with ``dischtime`` when ``hospital_expire_flag == 1``.
        - Missing modality data is represented by placeholder tokens so each
          sample remains usable for unified multimodal embedding.
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
