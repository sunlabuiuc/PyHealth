"""Standalone MIMIC-III task for left-against-medical-advice prediction.

This task is designed for reproducibility-oriented experiments inspired by:
"Racial Disparities and Mistrust in End-of-Life Care" (Boag et al.).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from .base_task import BaseTask


class AgainstMedicalAdvicePredictionMIMIC3(BaseTask):
    """Predict whether an admission ends with left against medical advice.

    The task emits configurable feature groups so users can run focused
    ablations:
    1. Baseline demographics and numeric features.
    2. Race-only additions.
    3. Lightweight note-derived mistrust proxies.
    4. Optional diagnosis/procedure/drug code features.

    Notes:
        - The MIMIC-III discharge value is stored as
          ``LEFT AGAINST MEDICAL ADVI`` (truncated in source table).
        - Mistrust features are heuristic and intentionally lightweight.

    Examples:
        >>> from pyhealth.datasets import MIMIC3Dataset
        >>> from pyhealth.tasks import AgainstMedicalAdvicePredictionMIMIC3
        >>> dataset = MIMIC3Dataset(
        ...     root="/path/to/mimic-iii/1.4",
        ...     tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
        ... )
        >>> task = AgainstMedicalAdvicePredictionMIMIC3(
        ...     include_race=True,
        ...     include_mistrust=True,
        ... )
        >>> sample_dataset = dataset.set_task(task)
    """

    task_name: str = "AgainstMedicalAdvicePredictionMIMIC3"
    input_schema: Dict[str, str] = {}
    output_schema: Dict[str, str] = {"left_ama": "binary"}

    NONCOMPLIANCE_PATTERNS = [
        r"\bnon[\s\-]?compliant\b",
        r"\brefus(?:e|ed|ing|al)\b",
        r"\bdeclin(?:e|ed|ing)\b",
        r"\bagainst\s+medical\s+advice\b",
        r"\bleft\s+ama\b",
        r"\buncooperative\b",
        r"\bhostile\b",
        r"\bcombative\b",
    ]
    NEGATIVE_LEXICON = {
        "angry",
        "argumentative",
        "belligerent",
        "combative",
        "confrontational",
        "declined",
        "delirious",
        "distrustful",
        "hostile",
        "noncompliant",
        "refused",
        "refusing",
        "uncooperative",
        "upset",
        "verbally",
        "violent",
    }
    EMPTY_CODE_TOKENS = {
        "conditions": "NO_CONDITION",
        "procedures": "NO_PROCEDURE",
        "drugs": "NO_DRUG",
    }

    def __init__(
        self,
        exclude_minors: bool = True,
        min_icu_hours: Optional[float] = None,
        include_baseline: bool = True,
        include_race: bool = False,
        include_mistrust: bool = False,
        mistrust_feature_set: str = "all",
        note_categories: Optional[List[str]] = None,
        include_codes: bool = False,
        code_mapping: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the task.

        Args:
            exclude_minors: Whether to exclude admissions where age is under 18.
            min_icu_hours: Optional minimum ICU overlap duration in hours.
            include_baseline: Include baseline numeric and demographic features.
            include_race: Include normalized race token feature.
            include_mistrust: Include note-derived mistrust proxy tensor.
            mistrust_feature_set: One of ``"noncompliance"``,
                ``"negative_sentiment"``, or ``"all"``.
            note_categories: Optional note category whitelist (case-insensitive).
            include_codes: Include diagnosis/procedure/drug sequence features.
            code_mapping: Optional PyHealth code mapping config.
        """
        if mistrust_feature_set not in {"noncompliance", "negative_sentiment", "all"}:
            raise ValueError(
                "mistrust_feature_set must be one of "
                "{'noncompliance', 'negative_sentiment', 'all'}"
            )

        self.exclude_minors = exclude_minors
        self.min_icu_hours = min_icu_hours
        self.include_baseline = include_baseline
        self.include_race = include_race
        self.include_mistrust = include_mistrust
        self.mistrust_feature_set = mistrust_feature_set
        self.note_categories = (
            {self._normalize_text(c) for c in note_categories}
            if note_categories
            else None
        )
        self.include_codes = include_codes
        self._noncompliance_regex = [
            re.compile(pattern, flags=re.IGNORECASE)
            for pattern in self.NONCOMPLIANCE_PATTERNS
        ]

        schema: Dict[str, str] = {}
        if self.include_baseline:
            schema["baseline_numeric"] = "tensor"
            schema["baseline_demographics"] = "multi_hot"
        if self.include_race:
            schema["race_tokens"] = "multi_hot"
        if self.include_mistrust:
            schema["mistrust_features"] = "tensor"
        if self.include_codes:
            schema["conditions"] = "sequence"
            schema["procedures"] = "sequence"
            schema["drugs"] = "sequence"
        self.input_schema = schema

        super().__init__(code_mapping=code_mapping)

    @staticmethod
    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip().lower()

    @staticmethod
    def _parse_datetime(value: Any) -> Optional[datetime]:
        """Parse timestamp-like values into ``datetime``."""
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        as_str = str(value).strip()
        if not as_str or as_str.lower() == "none":
            return None
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(as_str, fmt)
            except ValueError:
                continue
        try:
            return datetime.fromisoformat(as_str)
        except ValueError:
            return None

    @staticmethod
    def _normalize_insurance(value: Any) -> str:
        """Map raw insurance to coarse buckets."""
        v = str(value or "").strip().upper()
        if not v:
            return "unknown"
        if v == "PRIVATE":
            return "private"
        if v in {"MEDICARE", "MEDICAID", "GOVERNMENT"}:
            return "public"
        if v == "SELF PAY":
            return "self_pay"
        return "other"

    @staticmethod
    def _normalize_race(value: Any) -> str:
        """Collapse ethnicity text into broad race groups."""
        v = str(value or "").strip().upper()
        if "BLACK" in v:
            return "black"
        if "WHITE" in v:
            return "white"
        if "HISPANIC" in v or "LATINO" in v:
            return "hispanic"
        if "ASIAN" in v:
            return "asian"
        return "other"

    def _count_noncompliance_hits(self, text: str) -> int:
        return sum(len(regex.findall(text)) for regex in self._noncompliance_regex)

    def _negative_sentiment_proxy(self, text: str) -> float:
        tokens = re.findall(r"\b[a-z]+\b", text.lower())
        if not tokens:
            return 0.0
        negative_hits = sum(token in self.NEGATIVE_LEXICON for token in tokens)
        return float(negative_hits / len(tokens))

    def _compute_age_years(
        self,
        patient: Any,
        admission_time: datetime,
    ) -> Optional[float]:
        patient_events = patient.get_events(event_type="patients")
        if not patient_events:
            return None
        dob = self._parse_datetime(getattr(patient_events[0], "dob", None))
        if dob is None:
            return None
        age_years = admission_time.year - dob.year
        if (admission_time.month, admission_time.day) < (dob.month, dob.day):
            age_years -= 1
        if age_years < 0:
            return None
        if age_years > 89:
            age_years = 89
        return float(age_years)

    def _compute_icu_overlap_hours(
        self,
        patient: Any,
        admission_time: datetime,
        discharge_time: datetime,
    ) -> float:
        icu_events = patient.get_events(event_type="icustays")
        overlap_hours = 0.0
        for icu_event in icu_events:
            intime = self._parse_datetime(getattr(icu_event, "intime", None))
            if intime is None:
                intime = self._parse_datetime(getattr(icu_event, "timestamp", None))
            outtime = self._parse_datetime(getattr(icu_event, "outtime", None))
            if intime is None or outtime is None:
                continue
            if outtime <= intime:
                continue
            overlap_start = max(intime, admission_time)
            overlap_end = min(outtime, discharge_time)
            if overlap_end <= overlap_start:
                continue
            overlap_hours += (overlap_end - overlap_start).total_seconds() / 3600.0
        return overlap_hours

    def _collect_note_text(self, patient: Any, hadm_id: Any) -> tuple[str, int]:
        note_events = patient.get_events(
            event_type="noteevents",
            filters=[("hadm_id", "==", hadm_id)],
        )
        filtered_text_parts: List[str] = []
        retained_notes = 0
        for event in note_events:
            category = self._normalize_text(getattr(event, "category", ""))
            if (
                self.note_categories is not None
                and category not in self.note_categories
            ):
                continue
            text = str(getattr(event, "text", "") or "").strip()
            if not text:
                continue
            retained_notes += 1
            filtered_text_parts.append(text)
        return " ".join(filtered_text_parts), retained_notes

    @classmethod
    def _extract_code_sequence(
        cls,
        events: Iterable[Any],
        attribute: str,
        placeholder_key: str,
    ) -> List[str]:
        """Extract a non-empty code sequence for categorical features.

        Some admissions legitimately have no diagnosis, procedure, or
        prescription code rows in the subset of tables loaded for the task.
        Returning an explicit placeholder token keeps these sequence features
        well-defined for simple baseline models that do not handle empty
        categorical sequences robustly.
        """
        values = [
            str(getattr(event, attribute, "")).strip()
            for event in events
            if str(getattr(event, attribute, "")).strip()
        ]
        if values:
            return values
        return [cls.EMPTY_CODE_TOKENS[placeholder_key]]

    @staticmethod
    def _is_left_ama(discharge_location: Any) -> int:
        value = str(discharge_location or "").strip().upper()
        if value == "LEFT AGAINST MEDICAL ADVI":
            return 1
        if value.startswith("LEFT AGAINST MEDICAL ADVI"):
            return 1
        return 0

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Generate admission-level samples for one patient."""
        admissions = patient.get_events(event_type="admissions")
        patient_events = patient.get_events(event_type="patients")
        patient_gender = self._normalize_text(
            getattr(patient_events[0], "gender", "unknown")
            if patient_events
            else "unknown"
        )
        if not patient_gender:
            patient_gender = "unknown"
        samples: List[Dict[str, Any]] = []

        for admission in admissions:
            hadm_id = getattr(admission, "hadm_id", None)
            admission_time = self._parse_datetime(getattr(admission, "timestamp", None))
            discharge_time = self._parse_datetime(getattr(admission, "dischtime", None))
            if hadm_id is None or admission_time is None or discharge_time is None:
                continue
            if discharge_time <= admission_time:
                continue

            los_days = (discharge_time - admission_time).total_seconds() / 86400.0
            if los_days < 0:
                continue

            age_years = self._compute_age_years(patient, admission_time)
            if self.exclude_minors:
                if age_years is None or age_years < 18:
                    continue

            if self.min_icu_hours is not None:
                icu_hours = self._compute_icu_overlap_hours(
                    patient=patient,
                    admission_time=admission_time,
                    discharge_time=discharge_time,
                )
                if icu_hours < self.min_icu_hours:
                    continue

            left_ama = self._is_left_ama(getattr(admission, "discharge_location", None))
            sample: Dict[str, Any] = {
                "visit_id": str(hadm_id),
                "hadm_id": str(hadm_id),
                "patient_id": str(patient.patient_id),
                "left_ama": left_ama,
            }

            if self.include_baseline:
                sample["baseline_numeric"] = [float(age_years or 0.0), float(los_days)]
                insurance = self._normalize_insurance(
                    getattr(admission, "insurance", None)
                )
                sample["baseline_demographics"] = [
                    f"gender:{patient_gender}",
                    f"insurance:{insurance}",
                ]

            if self.include_race:
                race = self._normalize_race(getattr(admission, "ethnicity", None))
                sample["race_tokens"] = [f"race:{race}"]

            if self.include_mistrust:
                note_text, note_count = self._collect_note_text(patient, hadm_id)
                noncompliance_count = float(self._count_noncompliance_hits(note_text))
                noncompliance_any = float(noncompliance_count > 0.0)
                negative_proxy = float(self._negative_sentiment_proxy(note_text))
                note_present = float(note_count > 0)

                if self.mistrust_feature_set == "noncompliance":
                    mistrust_features = [
                        noncompliance_any,
                        noncompliance_count,
                        0.0,
                        note_present,
                    ]
                elif self.mistrust_feature_set == "negative_sentiment":
                    mistrust_features = [0.0, 0.0, negative_proxy, note_present]
                else:
                    mistrust_features = [
                        noncompliance_any,
                        noncompliance_count,
                        negative_proxy,
                        note_present,
                    ]
                sample["mistrust_features"] = mistrust_features

            if self.include_codes:
                diagnoses = patient.get_events(
                    event_type="diagnoses_icd",
                    filters=[("hadm_id", "==", hadm_id)],
                )
                procedures = patient.get_events(
                    event_type="procedures_icd",
                    filters=[("hadm_id", "==", hadm_id)],
                )
                prescriptions = patient.get_events(
                    event_type="prescriptions",
                    filters=[("hadm_id", "==", hadm_id)],
                )
                sample["conditions"] = self._extract_code_sequence(
                    diagnoses,
                    attribute="icd9_code",
                    placeholder_key="conditions",
                )
                sample["procedures"] = self._extract_code_sequence(
                    procedures,
                    attribute="icd9_code",
                    placeholder_key="procedures",
                )
                sample["drugs"] = self._extract_code_sequence(
                    prescriptions,
                    attribute="ndc",
                    placeholder_key="drugs",
                )

            samples.append(sample)

        return samples
