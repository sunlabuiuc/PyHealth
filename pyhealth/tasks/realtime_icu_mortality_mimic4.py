# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: Real-time ICU mortality task on MIMIC-IV. Every measurement time is a
#   "state marker"; each sample holds the irregular measurement tuples observed before
#   it, the tuples of the next state (for temporal-difference learning), a terminal
#   flag and mortality labels at several horizons.
"""Real-time ICU mortality prediction from irregular measurements (MIMIC-IV)."""

from __future__ import annotations

import bisect
import math
import random
from collections.abc import Callable, Sequence
from datetime import datetime, timedelta
from typing import Any, ClassVar

import numpy as np
import polars as pl

from .base_task import BaseTask

_LAB, _RATE, _BOLUS = 0, 1, 2
_MINUTE = 60.0  # seconds
_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d")


def _parse_time(value: Any) -> datetime | None:
    """Parses a MIMIC-IV timestamp string (or passes a datetime through).

    Args:
        value: A ``datetime``, a timestamp string, or ``None``.

    Returns:
        The parsed ``datetime``, or ``None`` if the value is missing or
        cannot be parsed.
    """
    if value is None or isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in ("none", "nan", "nat", "null"):
        return None
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _to_float(value: Any) -> float:
    """Converts a string attribute to float, returning NaN when impossible."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return math.nan
    return out


class RealTimeICUMortalityMIMIC4(BaseTask):
    """Real-time ICU mortality prediction with next-state observations (MIMIC-IV).

    This task reproduces the data framing of Frost et al. (ML4H 2024). Every
    unique time at which a patient has an ICU lab value or an IV drug event is
    a *state marker*. For each (optionally subsampled) marker the task emits
    one sample containing:

    * ``measurements``: the observation :math:`O_t` as an ``(L, 5)`` list of
      tuples ``[feature_id, minutes_before, value, delta_minutes,
      delta_value]`` taken from the ``lookback_hours`` before the marker,
      ordered oldest to newest. Feature ids start at 1 (0 is reserved for
      padding); ``delta_*`` are the change since the previous measurement of
      the same feature inside the window, or NaN. At most
      ``max_measurements`` tuples are kept, prioritising age/gender/weight,
      then currently running infusions, then features by novelty and
      recency (paper Appendix C).
    * ``next_measurements``: the observation at the *next state*, the first
      marker inside ``[t + next_state_delay_hours, t + next_state_delay_hours
      + next_state_window_hours]``. If there is none the state is terminal
      and this field repeats ``measurements`` (a placeholder that models
      must ignore).
    * ``terminal``: 1.0 if the state is terminal, else 0.0.
    * ``terminal_reward``: 1.0 if the patient died within
      ``terminal_reward_days`` of the marker (the TD terminal reward).
    * ``mortality``: binary label, death within ``label_horizon_days`` of the
      marker (the label used for training supervised models and for model
      selection).
    * ``mortality_horizons``: death within each of ``horizons_days`` (default
      1, 3, 7, 14 and 28 days), for the multi-horizon AUROC evaluation.

    Features follow the authors' public pipeline: 32 laboratory values from
    ICU ``chartevents`` and 72 IV drugs from ``inputevents``. Drug rows given
    over at most one minute become ``"<drug> bolus"`` events (antibiotics,
    IVIG and mannitol as binary events); longer rows become per-minute
    ``"<drug> rate"`` events, with overlapping infusions of the same drug
    summed and an explicit zero rate when an infusion stops. Values are kept
    in (harmonised) clinical units; per-feature standardisation is left to
    the model so that its statistics come from the training split only.

    Works with :class:`~pyhealth.datasets.MIMIC4EHRDataset` (or
    :class:`~pyhealth.datasets.MIMIC4Dataset`) loaded with the
    ``patients``, ``admissions``, ``chartevents`` and ``inputevents`` tables.

    Args:
        lookback_hours: Length of the observation window before each marker.
            Defaults to 168 (7 days), as in the paper.
        max_measurements: Maximum number of tuples per observation, including
            age, gender and weight. Defaults to 400, as in the authors' code.
        next_state_delay_hours: Start of the next-state eligibility window,
            in hours after the marker. Defaults to 24 (the paper's TD-24hr).
        next_state_window_hours: Length of the eligibility window. Defaults
            to 24.
        label_horizon_days: Horizon of the ``mortality`` label. Defaults to 28.
        terminal_reward_days: Horizon of the TD terminal reward. Defaults to
            28, as in the paper.
        horizons_days: Horizons of ``mortality_horizons``. Defaults to
            ``(1, 3, 7, 14, 28)``.
        feature_set: ``"all"`` (labs and drugs), ``"labs"`` or ``"drugs"``.
            Age, gender and weight are always included. Defaults to ``"all"``.
        state_sample_prob: Probability of keeping each state marker as a
            sample (next states are still searched among *all* markers).
            Use values below 1 to reduce dataset size. Defaults to 1.0.
        max_states_per_patient: Optional cap on samples per patient, drawn at
            random after ``state_sample_prob``. Defaults to ``None``.
        min_measurements: Markers whose window has fewer clinical
            measurements than this are skipped. Defaults to 1.
        seed: Seed of the per-patient subsampling RNG. Defaults to 0.
        include_features: Optional names (from :attr:`feature_names`) of the
            clinical features to keep; all others are dropped. Useful to
            restrict an external test set to the features a model was trained
            on. Age, gender and weight are always kept. Defaults to ``None``
            (all features of ``feature_set``).

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit
        >>> from pyhealth.tasks import RealTimeICUMortalityMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimiciv/3.0",
        ...     tables=["patients", "admissions", "chartevents", "inputevents"],
        ... )
        >>> task = RealTimeICUMortalityMIMIC4(state_sample_prob=0.1)
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.8, 0.1, 0.1), seed=42)
        ... )
        >>> train[0]["measurements"].shape  # doctest: +SKIP
        torch.Size([400, 5])
    """

    task_name: str = "RealTimeICUMortalityMIMIC4"
    input_schema: ClassVar[dict[str, str]] = {
        "measurements": "tensor",
        "next_measurements": "tensor",
    }
    output_schema: ClassVar[dict[str, str]] = {
        "mortality": "binary",
        "terminal": "tensor",
        "terminal_reward": "tensor",
        "mortality_horizons": "tensor",
    }

    #: ICU ``chartevents`` lab itemids -> feature name (paper Appendix B).
    LAB_ITEMIDS: ClassVar[dict[str, str]] = {
        "220644": "Alanine transaminase",
        "227456": "Albumin",
        "225612": "Alkaline phosphatase",
        "220581": "Amylase",
        "227073": "Anion gap",
        "220587": "Aspartate transferase",
        "224828": "Base excess",
        "227443": "Bicarbonate",
        "220235": "Blood CO2 pressure",
        "226062": "Blood CO2 pressure",
        "220224": "Blood O2 pressure",
        "226063": "Blood O2 pressure",
        "220227": "Blood O2 saturation",
        "220274": "Blood pH",
        "223830": "Blood pH",
        "227444": "C-reactive protein",
        "225667": "Calcium (ionised)",
        "225625": "Calcium (total)",
        "220602": "Chloride",
        "226536": "Chloride",
        "220615": "Creatinine",
        "225664": "Glucose (bedside)",
        "226537": "Glucose (bedside)",
        "220621": "Glucose (serum)",
        "220545": "Haematocrit",
        "220228": "Haemoglobin",
        "225668": "Lactate",
        "220632": "Lactate dehydrogenase",
        "225672": "Lipase",
        "227457": "Platelet count",
        "227442": "Potassium",
        "227464": "Potassium",
        "227465": "Prothrombin time",
        "220645": "Sodium",
        "226534": "Sodium",
        "225690": "Total bilirubin",
        "227429": "Troponin-T",
        "225624": "Urea",
        "220546": "White blood cell count",
    }

    #: Multiplicative factors converting MIMIC lab units to SI-style units.
    LAB_UNIT_FACTORS: ClassVar[dict[str, float]] = {
        "Albumin": 10.0,  # g/dL -> g/L
        "Glucose (serum)": 1.0 / 18.0,  # mg/dL -> mmol/L
        "Glucose (bedside)": 1.0 / 18.0,
        "Total bilirubin": 17.1,  # mg/dL -> umol/L
        "Blood O2 pressure": 0.133322,  # mmHg -> kPa
        "Blood CO2 pressure": 0.133322,
        "Calcium (total)": 0.2495,  # mg/dL -> mmol/L
        "Creatinine": 88.42,  # mg/dL -> umol/L
        "Haemoglobin": 10.0,  # g/dL -> g/L
        "Urea": 0.357,  # mg/dL -> mmol/L
    }

    #: ICU ``inputevents`` drug itemids -> drug name (paper Appendix B).
    DRUG_ITEMIDS: ClassVar[dict[str, str]] = {
        "221347": "Amiodarone",
        "228339": "Amiodarone",
        "229654": "Amiodarone",
        "230034": "Amiodarone",
        "225837": "Aciclovir",
        "225838": "Ambisome",
        "225840": "Amikacin",
        "225842": "Ampicillin",
        "225843": "Ampicillin-Sulbactam",
        "225845": "Azithromycin",
        "225847": "Aztreonam",
        "225848": "Caspofungin",
        "225850": "Cefazolin",
        "225851": "Cefepime",
        "225853": "Ceftazidime",
        "225855": "Ceftriaxone",
        "229059": "Chloramphenicol",
        "225859": "Ciprofloxacin",
        "225860": "Clindamycin",
        "225862": "Colistin",
        "225899": "Co-trimoxazole",
        "225863": "Daptomycin",
        "225865": "Doxycycline",
        "229061": "Ertapenem",
        "225866": "Erythromycin",
        "225875": "Gentamicin",
        "225879": "Levofloxacin",
        "225881": "Linezolid",
        "225883": "Meropenem",
        "225885": "Micafungin",
        "225884": "Metronidazole",
        "225888": "Nafcillin",
        "225889": "Oxacillin",
        "225892": "Piperacillin",
        "225893": "Piperacillin-Tazobactam",
        "225898": "Rifampin",
        "229064": "Tigecycline",
        "225902": "Tobramycin",
        "225798": "Vancomycin",
        "225905": "Voriconazole",
        "225152": "Unfractionated Heparin",
        "228316": "Levetiracetam",
        "227690": "Phenytoin",
        "222056": "Nitroglycerin",
        "222051": "Nitroprusside",
        "225153": "Labetalol",
        "220970": "Fresh Frozen Plasma",
        "225168": "Red Blood Cells",
        "220862": "Human Albumin Solution",
        "225170": "Platelets",
        "227530": "IV Immune Globulin",
        "221794": "Furosemide",
        "228340": "Furosemide",
        "229639": "Bumetanide",
        "223258": "Regular Insulin",
        "225161": "Hypertonic Saline 3%",
        "227531": "Mannitol",
        "221342": "Aminophylline",
        "220995": "Sodium bicarbonate 8.4%",
        "227533": "Sodium bicarbonate 8.4%",
        "221744": "Fentanyl",
        "225942": "Fentanyl",
        "225972": "Fentanyl",
        "225154": "Morphine sulphate",
        "221555": "Cisatracurium",
        "229233": "Rocuronium",
        "222062": "Vecuronium",
        "225150": "Dexmedetomidine",
        "229420": "Dexmedetomidine",
        "221712": "Ketamine",
        "221385": "Lorazepam",
        "221668": "Midazolam",
        "222168": "Propofol",
        "221319": "Alteplase",
        "221289": "Adrenaline",
        "221653": "Dobutamine",
        "221662": "Dopamine",
        "221986": "Milrinone",
        "221906": "Noradrenaline",
        "222315": "Vasopressin",
    }

    #: Drugs recorded as binary events (dose ignored), as in the authors' code.
    BINARY_DRUGS: ClassVar[tuple[str, ...]] = (
        "Aciclovir",
        "Ambisome",
        "Amikacin",
        "Ampicillin",
        "Ampicillin-Sulbactam",
        "Azithromycin",
        "Aztreonam",
        "Caspofungin",
        "Cefazolin",
        "Cefepime",
        "Ceftazidime",
        "Ceftriaxone",
        "Chloramphenicol",
        "Ciprofloxacin",
        "Clindamycin",
        "Colistin",
        "Co-trimoxazole",
        "Daptomycin",
        "Doxycycline",
        "Ertapenem",
        "Erythromycin",
        "Gentamicin",
        "Levofloxacin",
        "Linezolid",
        "Meropenem",
        "Micafungin",
        "Metronidazole",
        "Nafcillin",
        "Oxacillin",
        "Piperacillin",
        "Piperacillin-Tazobactam",
        "Rifampin",
        "Tigecycline",
        "Tobramycin",
        "Vancomycin",
        "Voriconazole",
        "IV Immune Globulin",
        "Mannitol",
    )

    #: Amount-unit conversion to one canonical unit per unit family.
    AMOUNT_UNIT_FACTORS: ClassVar[dict[str, tuple[str, float]]] = {
        "mg": ("mg", 1.0),
        "mcg": ("mg", 1e-3),
        "ug": ("mg", 1e-3),
        "grams": ("mg", 1e3),
        "gm": ("mg", 1e3),
        "g": ("mg", 1e3),
        "ng": ("mg", 1e-6),
        "units": ("units", 1.0),
        "ml": ("ml", 1.0),
        "meq": ("ml", 1.0),  # sodium bicarbonate 8.4% is 1 mEq/mL
        "dose": ("dose", 1.0),
    }

    #: Albumin 25% is charted in mL; convert to grams of albumin, then mg.
    _ALBUMIN_ML_TO_MG: ClassVar[float] = 0.25 * 1e3

    DEMOGRAPHICS: ClassVar[tuple[str, ...]] = ("age", "gender", "weight")
    DEFAULT_WEIGHT_KG: ClassVar[dict[int, float]] = {0: 86.0, 1: 74.0}
    VALID_WEIGHT_KG: ClassVar[tuple[float, float]] = (20.0, 350.0)

    def __init__(
        self,
        lookback_hours: float = 168.0,
        max_measurements: int = 400,
        next_state_delay_hours: float = 24.0,
        next_state_window_hours: float = 24.0,
        label_horizon_days: int = 28,
        terminal_reward_days: int = 28,
        horizons_days: Sequence[int] = (1, 3, 7, 14, 28),
        feature_set: str = "all",
        state_sample_prob: float = 1.0,
        max_states_per_patient: int | None = None,
        min_measurements: int = 1,
        seed: int = 0,
        include_features: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        if feature_set not in ("all", "labs", "drugs"):
            raise ValueError("feature_set must be 'all', 'labs' or 'drugs'")
        if max_measurements < len(self.DEMOGRAPHICS) + 1:
            raise ValueError("max_measurements must leave room for measurements")
        if not 0.0 < state_sample_prob <= 1.0:
            raise ValueError("state_sample_prob must be in (0, 1]")
        if lookback_hours <= 0 or next_state_window_hours <= 0:
            raise ValueError("window lengths must be positive")
        if next_state_delay_hours < 0:
            raise ValueError("next_state_delay_hours must be non-negative")
        self.lookback_hours = float(lookback_hours)
        self.max_measurements = int(max_measurements)
        self.next_state_delay_hours = float(next_state_delay_hours)
        self.next_state_window_hours = float(next_state_window_hours)
        self.label_horizon_days = int(label_horizon_days)
        self.terminal_reward_days = int(terminal_reward_days)
        self.horizons_days = tuple(int(d) for d in horizons_days)
        self.feature_set = feature_set
        self.state_sample_prob = float(state_sample_prob)
        self.max_states_per_patient = max_states_per_patient
        self.min_measurements = int(min_measurements)
        self.seed = int(seed)
        self.feature_names: list[str] = self.build_feature_names()
        self._feature_index: dict[str, int] = {
            name: i + 1 for i, name in enumerate(self.feature_names)
        }
        self.include_features: list[str] | None = None
        if include_features is not None:
            unknown = set(include_features) - set(self.feature_names)
            if unknown:
                raise ValueError(f"unknown features in include_features: {unknown}")
            self.include_features = sorted(include_features)
        self._lab_ids = sorted(self.LAB_ITEMIDS)
        self._drug_ids = sorted(self.DRUG_ITEMIDS)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------
    @classmethod
    def build_feature_names(cls) -> list[str]:
        """Returns the fixed feature vocabulary (index ``i`` has id ``i + 1``).

        Returns:
            Lab names, then ``"<drug> rate"``/``"<drug> bolus"`` names, then
            ``"age"``, ``"gender"`` and ``"weight"``.
        """
        labs = sorted(set(cls.LAB_ITEMIDS.values()))
        names: list[str] = list(labs)
        for drug in sorted(set(cls.DRUG_ITEMIDS.values())):
            if drug not in cls.BINARY_DRUGS:
                names.append(f"{drug} rate")
            names.append(f"{drug} bolus")
        names.extend(cls.DEMOGRAPHICS)
        return names

    @property
    def num_features(self) -> int:
        """Number of feature ids (excluding the padding id 0)."""
        return len(self.feature_names)

    # ------------------------------------------------------------------
    # Dataset-level filtering
    # ------------------------------------------------------------------
    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps only the events this task reads, before patients are built.

        Args:
            df: The dataset's global event frame.

        Returns:
            The frame restricted to demographics, admissions, the lab
            ``chartevents`` and the drug ``inputevents`` of this task.
        """
        columns = set(df.collect_schema().names())
        keep = pl.col("event_type").is_in(["patients", "admissions"])
        if "chartevents/itemid" in columns:
            keep = keep | (
                (pl.col("event_type") == "chartevents")
                & pl.col("chartevents/itemid").cast(pl.Utf8).is_in(self._lab_ids)
            )
        if "inputevents/itemid" in columns:
            keep = keep | (
                (pl.col("event_type") == "inputevents")
                & pl.col("inputevents/itemid").cast(pl.Utf8).is_in(self._drug_ids)
            )
        return df.filter(keep)

    # ------------------------------------------------------------------
    # Event extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _column(df: pl.DataFrame, table: str, attr: str) -> list[Any]:
        """Returns ``table/attr`` as a list, or Nones if the column is absent."""
        name = f"{table}/{attr}"
        if name not in df.columns:
            return [None] * df.height
        return df[name].to_list()

    def _lab_events(self, patient: Any) -> list[tuple[datetime, str, float, int]]:
        """Extracts lab measurements from ICU ``chartevents``.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            A list of ``(time, feature, value, kind)`` tuples.
        """
        if self.feature_set == "drugs":
            return []
        df = patient.get_events(event_type="chartevents", return_df=True)
        if df.height == 0:
            return []
        events = []
        times = df["timestamp"].to_list()
        itemids = self._column(df, "chartevents", "itemid")
        values = self._column(df, "chartevents", "valuenum")
        for time, itemid, value in zip(times, itemids, values):
            feature = self.LAB_ITEMIDS.get(str(itemid))
            value = _to_float(value)
            if feature is None or time is None or not math.isfinite(value):
                continue
            value *= self.LAB_UNIT_FACTORS.get(feature, 1.0)
            events.append((time, feature, value, _LAB))
        return events

    def _drug_amount(self, drug: str, amount: float, unit: Any) -> float:
        """Converts a drug amount to the drug's canonical unit (NaN if unknown).

        Args:
            drug: Drug name.
            amount: Raw amount.
            unit: Raw ``amountuom`` string.

        Returns:
            The amount in the canonical unit, or NaN when the unit is unknown.
        """
        unit_key = str(unit).strip().lower() if unit is not None else ""
        if drug == "Human Albumin Solution" and unit_key == "ml":
            return amount * self._ALBUMIN_ML_TO_MG
        if unit_key not in self.AMOUNT_UNIT_FACTORS:
            return math.nan
        return amount * self.AMOUNT_UNIT_FACTORS[unit_key][1]

    def _drug_events(
        self, patient: Any
    ) -> tuple[list[tuple[datetime, str, float, int]], list[tuple[datetime, float]]]:
        """Extracts bolus and rate events from ICU ``inputevents``.

        Rows delivered over at most one minute are boluses; longer rows are
        infusions whose per-minute rate is ``amount / duration``. Overlapping
        infusions of the same drug are summed into a net rate, emitted at
        every infusion start and every change, including an explicit zero
        when the drug stops.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.

        Returns:
            ``(events, weights)``: drug events as ``(time, feature, value,
            kind)`` tuples, and ``(time, weight_kg)`` pairs from the rows'
            ``patientweight``.
        """
        df = patient.get_events(event_type="inputevents", return_df=True)
        if df.height == 0:
            return [], []
        events: list[tuple[datetime, str, float, int]] = []
        weights: list[tuple[datetime, float]] = []
        infusions: dict[str, list[tuple[datetime, datetime, float]]] = {}
        starts = df["timestamp"].to_list()
        itemids = self._column(df, "inputevents", "itemid")
        ends = self._column(df, "inputevents", "endtime")
        amounts = self._column(df, "inputevents", "amount")
        units = self._column(df, "inputevents", "amountuom")
        pweights = self._column(df, "inputevents", "patientweight")
        low_w, high_w = self.VALID_WEIGHT_KG
        for start, itemid, end, amount, unit, pweight in zip(
            starts, itemids, ends, amounts, units, pweights
        ):
            if start is None:
                continue
            weight = _to_float(pweight)
            if low_w <= weight <= high_w:
                weights.append((start, weight))
            drug = self.DRUG_ITEMIDS.get(str(itemid))
            if drug is None or self.feature_set == "labs":
                continue
            end_time = _parse_time(end)
            minutes = (
                (end_time - start).total_seconds() / _MINUTE
                if end_time is not None
                else 0.0
            )
            if drug in self.BINARY_DRUGS:
                events.append((start, f"{drug} bolus", 1.0, _BOLUS))
                continue
            value = self._drug_amount(drug, _to_float(amount), unit)
            if not math.isfinite(value) or value <= 0:
                continue
            if minutes <= 1.0:
                events.append((start, f"{drug} bolus", value, _BOLUS))
            else:
                infusions.setdefault(drug, []).append(
                    (start, end_time, value / minutes)
                )
        for drug, intervals in infusions.items():
            events.extend(self._net_rate_events(f"{drug} rate", intervals))
        return events, weights

    @staticmethod
    def _net_rate_events(
        feature: str, intervals: list[tuple[datetime, datetime, float]]
    ) -> list[tuple[datetime, str, float, int]]:
        """Turns possibly overlapping infusions into net-rate change events.

        Args:
            feature: The ``"<drug> rate"`` feature name.
            intervals: ``(start, end, rate)`` infusions of one drug.

        Returns:
            ``(time, feature, net_rate, kind)`` events at every infusion start
            and at every time the summed rate changes (zero when stopped).
        """
        deltas: dict[datetime, float] = {}
        starts = set()
        for start, end, rate in intervals:
            deltas[start] = deltas.get(start, 0.0) + rate
            deltas[end] = deltas.get(end, 0.0) - rate
            starts.add(start)
        events = []
        total, previous = 0.0, 0.0
        for time in sorted(deltas):
            total += deltas[time]
            if abs(total) < 1e-9 * max(1.0, abs(previous)):
                total = 0.0
            if time in starts or total != previous:
                events.append((time, feature, max(total, 0.0), _RATE))
            previous = total
        return events

    @staticmethod
    def _death_time(patient: Any, last_event: datetime | None) -> datetime | None:
        """Determines the time of death (in-hospital death time, else ``dod``).

        Mirrors the authors' rule: if only the (date-resolution) ``dod`` is
        known and some measurement falls on or after it, the last measurement
        time is used as the time of death.

        Args:
            patient: A :class:`~pyhealth.data.Patient`.
            last_event: Time of the patient's last measurement, if any.

        Returns:
            The time of death, or ``None`` if the patient is not known to
            have died.
        """
        death_times = [
            _parse_time(getattr(adm, "deathtime", None))
            for adm in patient.get_events(event_type="admissions")
        ]
        death_times = [t for t in death_times if t is not None]
        if death_times:
            return max(death_times)
        demographics = patient.get_events(event_type="patients")
        dod = (
            _parse_time(getattr(demographics[0], "dod", None)) if demographics else None
        )
        if dod is None:
            return None
        if last_event is not None and last_event >= dod:
            return last_event
        return dod

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------
    def _observation(
        self,
        state_index: int,
        times: np.ndarray,
        feature_ids: np.ndarray,
        values: np.ndarray,
        kinds: np.ndarray,
        previous: np.ndarray,
        demographics: list[float],
    ) -> tuple[list[list[float]], int]:
        """Builds the observation tuples for one state marker.

        Args:
            state_index: Index (into the sorted event arrays) of the last
                event at the marker time.
            times: Event times in minutes, sorted ascending.
            feature_ids: Feature id of each event.
            values: Value of each event.
            kinds: ``_LAB``, ``_RATE`` or ``_BOLUS`` for each event.
            previous: Index of the previous event of the same feature, or -1.
            demographics: ``[age, gender, weight]`` at the marker.

        Returns:
            ``(tuples, n_clinical)``: the ``[feature_id, minutes_before,
            value, delta_minutes, delta_value]`` rows ordered oldest to
            newest, and the number of clinical (non-demographic) rows.
        """
        now = times[state_index]
        start = int(np.searchsorted(times, now - self.lookback_hours * 60.0, "left"))
        idx = np.arange(start, state_index + 1)
        minutes_before = now - times[idx]
        prev = previous[idx]
        has_delta = (prev >= start) & (kinds[idx] != _BOLUS)
        safe_prev = np.where(has_delta, prev, idx)
        delta_t = np.where(has_delta, times[idx] - times[safe_prev], np.nan)
        delta_v = np.where(has_delta, values[idx] - values[safe_prev], np.nan)

        # Rank each feature's measurements by recency (1 = newest), then order
        # by (rank, recency): novel features first, newer before older.
        feats = feature_ids[idx]
        by_feature = np.lexsort((minutes_before, feats))
        sorted_feats = feats[by_feature]
        group_start = np.flatnonzero(np.r_[True, sorted_feats[1:] != sorted_feats[:-1]])
        group_sizes = np.diff(np.r_[group_start, len(idx)])
        rank = np.empty(len(idx), dtype=np.int64)
        rank[by_feature] = np.arange(len(idx)) - np.repeat(group_start, group_sizes)
        priority = np.lexsort((minutes_before, rank))

        # Currently running infusions: the latest rate of each drug, if > 0.
        rate_pos = np.flatnonzero(kinds[idx] == _RATE)
        rate_feats = feats[rate_pos]
        # Positions are in ascending time, so the last occurrence is the latest.
        _, last_rev = np.unique(rate_feats[::-1], return_index=True)
        latest = rate_pos[len(rate_pos) - 1 - last_rev] if rate_pos.size else rate_pos
        latest = latest[values[idx[latest]] > 0]
        current = np.column_stack(
            [
                feats[latest].astype(np.float64),
                np.zeros(len(latest)),
                values[idx[latest]],
                np.zeros(len(latest)),
                np.zeros(len(latest)),
            ]
        )

        budget = self.max_measurements - len(self.DEMOGRAPHICS)
        current = current[:budget]
        chosen = priority[: max(budget - len(current), 0)]
        history = np.column_stack(
            [
                feats[chosen].astype(np.float64),
                minutes_before[chosen],
                values[idx[chosen]],
                delta_t[chosen],
                delta_v[chosen],
            ]
        )
        clinical = np.concatenate([current, history], axis=0)
        # Oldest -> newest; the stable sort keeps priority order within ties.
        clinical = clinical[np.argsort(-clinical[:, 1], kind="stable")]
        demo = np.array(
            [
                [float(self._feature_index[name]), 0.0, value, math.nan, math.nan]
                for name, value in zip(self.DEMOGRAPHICS, demographics)
            ]
        )
        rows = np.concatenate([clinical, demo], axis=0)
        return rows.tolist(), len(clinical)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates one sample per (subsampled) state marker of a patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` with ``patients``,
                ``admissions``, ``chartevents`` and ``inputevents`` events.

        Returns:
            A list of sample dictionaries (see the class docstring).
        """
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []
        demo = demographics[0]
        gender = 1.0 if str(getattr(demo, "gender", "")).upper() == "F" else 0.0
        anchor_age = _to_float(getattr(demo, "anchor_age", None))
        anchor_year = _to_float(getattr(demo, "anchor_year", None))
        if not (math.isfinite(anchor_age) and math.isfinite(anchor_year)):
            return []

        drug_events, weights = self._drug_events(patient)
        events = self._lab_events(patient) + drug_events
        events = [e for e in events if e[1] in self._feature_index]
        if not events:
            return []
        events.sort(key=lambda e: (e[0], e[1]))
        death = self._death_time(patient, events[-1][0])

        def age_at(when: datetime) -> float:
            return when.year - anchor_year + anchor_age

        return self._timeline_samples(
            patient.patient_id, events, weights, death, gender, age_at
        )

    def _timeline_samples(
        self,
        patient_id: str,
        events: list[tuple[datetime, str, float, int]],
        weights: list[tuple[datetime, float]],
        death: datetime | None,
        gender: float,
        age_at: Callable[[datetime], float],
        rng_key: str | None = None,
    ) -> list[dict[str, Any]]:
        """Builds the samples of one patient timeline (dataset-independent).

        Args:
            patient_id: Patient identifier stored in every sample.
            events: ``(time, feature, value, kind)`` events sorted by time.
            weights: ``(time, weight_kg)`` observations.
            death: Time of death, or ``None``.
            gender: 1.0 for female, 0.0 otherwise.
            age_at: Function returning the age at a given time.
            rng_key: Key of the subsampling RNG. Defaults to ``patient_id``.

        Returns:
            One sample dictionary per (subsampled) state marker.
        """
        if self.include_features is not None:
            keep = set(self.include_features)
            events = [e for e in events if e[1] in keep]
        if not events:
            return []
        epoch = events[0][0]
        times = np.array(
            [(e[0] - epoch).total_seconds() / _MINUTE for e in events], dtype=np.float64
        )
        feature_ids = np.array(
            [self._feature_index[e[1]] for e in events], dtype=np.int64
        )
        values = np.array([e[2] for e in events], dtype=np.float64)
        kinds = np.array([e[3] for e in events], dtype=np.int64)
        previous = np.full(len(events), -1, dtype=np.int64)
        last_seen: dict[int, int] = {}
        for i, f in enumerate(feature_ids.tolist()):
            previous[i] = last_seen.get(f, -1)
            last_seen[f] = i

        # State markers: unique event times; each maps to its last event index.
        marker_times = np.unique(times)
        marker_last = np.searchsorted(times, marker_times, "right") - 1
        weights = sorted(weights, key=lambda w: w[0])
        weight_times = [w[0] for w in weights]

        def demographics_at(when: datetime) -> list[float]:
            age = age_at(when)
            pos = bisect.bisect_right(weight_times, when) - 1
            weight = (
                weights[pos][1] if pos >= 0 else self.DEFAULT_WEIGHT_KG[int(gender)]
            )
            return [float(age), gender, float(weight)]

        def died_within(when: datetime, days: int) -> int:
            return int(death is not None and death <= when + timedelta(days=days))

        delay = self.next_state_delay_hours * 60.0
        window = self.next_state_window_hours * 60.0
        rng = random.Random(f"{self.seed}-{rng_key or patient_id}")
        chosen = [
            k
            for k in range(len(marker_times))
            if self.state_sample_prob >= 1.0 or rng.random() < self.state_sample_prob
        ]
        if (
            self.max_states_per_patient is not None
            and len(chosen) > self.max_states_per_patient
        ):
            chosen = sorted(rng.sample(chosen, self.max_states_per_patient))

        cache: dict[int, tuple[list[list[float]], int]] = {}

        def observation_at(marker: int) -> tuple[list[list[float]], int]:
            if marker not in cache:
                when = epoch + timedelta(minutes=float(marker_times[marker]))
                cache[marker] = self._observation(
                    int(marker_last[marker]),
                    times,
                    feature_ids,
                    values,
                    kinds,
                    previous,
                    demographics_at(when),
                )
            return cache[marker]

        samples = []
        for k in chosen:
            when = epoch + timedelta(minutes=float(marker_times[k]))
            obs, n_clinical = observation_at(k)
            if n_clinical < self.min_measurements:
                continue
            lo = int(np.searchsorted(marker_times, marker_times[k] + delay, "left"))
            lo = max(lo, k + 1)  # a state is never its own next state
            hi = int(
                np.searchsorted(marker_times, marker_times[k] + delay + window, "right")
            )
            terminal = lo >= hi
            next_obs = obs if terminal else observation_at(lo)[0]
            samples.append(
                {
                    "patient_id": patient_id,
                    "measurements": obs,
                    "next_measurements": next_obs,
                    "terminal": float(terminal),
                    "terminal_reward": float(
                        died_within(when, self.terminal_reward_days)
                    ),
                    "mortality": died_within(when, self.label_horizon_days),
                    "mortality_horizons": [
                        float(died_within(when, d)) for d in self.horizons_days
                    ],
                }
            )
        return samples
