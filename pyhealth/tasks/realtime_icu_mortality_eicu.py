# Authors: Hanyang Chen (hc81), Siqi Song (siqis3)
# Paper: Frost, Li & Harris. Robust Real-Time Mortality Prediction in the Intensive
#   Care Unit using Temporal Difference Learning. ML4H 2024, PMLR 259:350-363.
# Paper link: https://proceedings.mlr.press/v259/frost25a.html
# Description: eICU-CRD version of the real-time ICU mortality task, sharing the
#   feature vocabulary of RealTimeICUMortalityMIMIC4 so that models trained on MIMIC-IV
#   can be validated externally on eICU (labs only).
"""Real-time ICU mortality prediction on eICU-CRD (external validation)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta
from typing import Any, ClassVar

import polars as pl

from .realtime_icu_mortality_mimic4 import (
    _LAB,
    RealTimeICUMortalityMIMIC4,
    _to_float,
)


def _parse_age(value: Any) -> float:
    """Parses eICU ages such as ``"67"`` or ``"> 89"`` (NaN if missing)."""
    text = str(value).strip() if value is not None else ""
    if text.startswith(">"):
        return 90.0
    return _to_float(text)


class RealTimeICUMortalityEICU(RealTimeICUMortalityMIMIC4):
    """Real-time ICU mortality with next-state observations (eICU-CRD).

    Produces exactly the same sample format and feature ids as
    :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4`, so a model trained
    on MIMIC-IV can be evaluated on eICU without changes. This supports the
    external validation of Frost et al. (ML4H 2024), who used SICdb; eICU is
    a multi-centre alternative that needs only standard PhysioNet
    credentialing.

    Differences from the MIMIC-IV task, all forced by eICU's structure:

    * **Labs only.** The 32 laboratory features are mapped from eICU
      ``lab.labname``; eICU drug charting (infusion rates in mL/hr without
      concentrations, free-text medication names) does not map reliably to
      the MIMIC-IV drug features. Train the MIMIC-IV model with
      ``feature_set="labs"`` for external validation.
    * **Timeline per hospitalisation.** eICU only stores minute offsets from
      each ICU-unit admission. Stays of one hospitalisation
      (``patienthealthsystemstayid``) are aligned on the hospital clock
      (``offset - hospitaladmitoffset``); separate hospitalisations are
      separate timelines.
    * **In-hospital death only.** The time of death is the hospital
      discharge offset when ``hospitaldischargestatus == "Expired"``;
      deaths after discharge are not recorded, so longer horizons are
      censored at discharge.
    * **Demographics.** Age is the admission age (``"> 89"`` becomes 90);
      weight is ``admissionweight``.

    Units are converted to those of the MIMIC-IV task: the same factors for
    shared US units, C-reactive protein mg/dL -> mg/L, and ionised calcium
    mg/dL -> mmol/L (values below 2.5 are taken to be mmol/L already).

    Works with :class:`~pyhealth.datasets.eICUDataset` loaded with the
    ``lab`` table (``patient`` is always loaded).

    The lab event kind (``_LAB``) and the value parser (``_to_float``) are
    imported from the MIMIC-IV task module on purpose, so that the three
    real-time tasks build identical tuples and share one feature vocabulary.

    Args:
        **kwargs: Any argument of
            :class:`~pyhealth.tasks.RealTimeICUMortalityMIMIC4` except
            ``feature_set``, which must be ``"labs"`` (the default here).

    Examples:
        >>> from pyhealth.datasets import eICUDataset
        >>> from pyhealth.tasks import RealTimeICUMortalityEICU
        >>> dataset = eICUDataset(  # doctest: +SKIP
        ...     root="/path/to/eicu-crd/2.0", tables=["lab"]
        ... )
        >>> samples = dataset.set_task(RealTimeICUMortalityEICU())  # doctest: +SKIP
    """

    task_name: str = "RealTimeICUMortalityEICU"

    #: eICU ``lab.labname`` -> feature name of the MIMIC-IV task.
    LAB_NAMES: ClassVar[dict[str, str]] = {
        "ALT (SGPT)": "Alanine transaminase",
        "albumin": "Albumin",
        "alkaline phos.": "Alkaline phosphatase",
        "amylase": "Amylase",
        "AST (SGOT)": "Aspartate transferase",
        "anion gap": "Anion gap",
        "Base Excess": "Base excess",
        "bicarbonate": "Bicarbonate",
        "paCO2": "Blood CO2 pressure",
        "paO2": "Blood O2 pressure",
        "O2 Sat (%)": "Blood O2 saturation",
        "pH": "Blood pH",
        "CRP": "C-reactive protein",
        "ionized calcium": "Calcium (ionised)",
        "calcium": "Calcium (total)",
        "chloride": "Chloride",
        "creatinine": "Creatinine",
        "glucose": "Glucose (serum)",
        "bedside glucose": "Glucose (bedside)",
        "Hct": "Haematocrit",
        "Hgb": "Haemoglobin",
        "lactate": "Lactate",
        "LDH": "Lactate dehydrogenase",
        "lipase": "Lipase",
        "platelets x 1000": "Platelet count",
        "potassium": "Potassium",
        "PT": "Prothrombin time",
        "sodium": "Sodium",
        "total bilirubin": "Total bilirubin",
        "troponin - T": "Troponin-T",
        "BUN": "Urea",
        "WBC x 1000": "White blood cell count",
    }

    #: Extra eICU-specific factors applied before the shared lab factors.
    EICU_UNIT_FACTORS: ClassVar[dict[str, float]] = {
        "C-reactive protein": 10.0,  # mg/dL -> mg/L (MIMIC-IV chartevents)
    }

    #: Arbitrary origin of the hospital clock (eICU has no absolute dates).
    EPOCH: ClassVar[datetime] = datetime(2000, 1, 1)

    def __init__(self, **kwargs: Any) -> None:
        feature_set = kwargs.pop("feature_set", "labs")
        if feature_set != "labs":
            raise ValueError("RealTimeICUMortalityEICU supports feature_set='labs'")
        super().__init__(feature_set="labs", **kwargs)

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Keeps the ``patient`` rows and the task's ``lab`` rows.

        Args:
            df: The dataset's global event frame.

        Returns:
            The filtered frame.
        """
        columns = set(df.collect_schema().names())
        keep = pl.col("event_type") == "patient"
        if "lab/labname" in columns:
            keep = keep | (
                (pl.col("event_type") == "lab")
                & pl.col("lab/labname").is_in(list(self.LAB_NAMES))
            )
        return df.filter(keep)

    def _lab_value(self, feature: str, value: float) -> float:
        """Converts an eICU lab value to the MIMIC-IV task's unit.

        Args:
            feature: Feature name.
            value: Raw eICU value.

        Returns:
            The converted value.
        """
        if feature == "Calcium (ionised)":
            return value * 0.2495 if value >= 2.5 else value
        value *= self.EICU_UNIT_FACTORS.get(feature, 1.0)
        return value * self.LAB_UNIT_FACTORS.get(feature, 1.0)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Generates samples for every hospitalisation of an eICU patient.

        Args:
            patient: A :class:`~pyhealth.data.Patient` from
                :class:`~pyhealth.datasets.eICUDataset` with ``patient`` and
                ``lab`` events.

        Returns:
            A list of sample dictionaries (same format as the MIMIC-IV task).
        """
        stays = patient.get_events(event_type="patient", return_df=True)
        if stays.height == 0:
            return []
        labs = patient.get_events(event_type="lab", return_df=True)

        def column(df: pl.DataFrame, name: str) -> list[Any]:
            return df[name].to_list() if name in df.columns else [None] * df.height

        # Per-stay information and the hospitalisation each stay belongs to.
        stay_info: dict[str, dict[str, Any]] = {}
        for row in zip(
            column(stays, "patient/patientunitstayid"),
            column(stays, "patient/patienthealthsystemstayid"),
            column(stays, "patient/hospitaladmitoffset"),
            column(stays, "patient/hospitaldischargeoffset"),
            column(stays, "patient/hospitaldischargestatus"),
            column(stays, "patient/age"),
            column(stays, "patient/gender"),
            column(stays, "patient/admissionweight"),
        ):
            stay, hosp, admit, disch, status, age, gender, weight = row
            admit = _to_float(admit)
            if stay is None or not math.isfinite(admit):
                continue
            stay_info[str(stay)] = {
                "hosp": str(hosp),
                "admit": admit,  # unit admit -> hospital admit, in minutes (<= 0)
                "discharge": _to_float(disch),
                "expired": str(status).strip().lower() == "expired",
                "age": _parse_age(age),
                "female": str(gender).strip().lower() == "female",
                "weight": _to_float(weight),
            }

        events: dict[str, list[tuple[datetime, str, float, int]]] = {}
        for stay, offset, name, value in zip(
            column(labs, "lab/patientunitstayid"),
            column(labs, "lab/labresultoffset"),
            column(labs, "lab/labname"),
            column(labs, "lab/labresult"),
        ):
            info = stay_info.get(str(stay))
            feature = self.LAB_NAMES.get(str(name))
            offset, value = _to_float(offset), _to_float(value)
            if info is None or feature is None:
                continue
            if not (math.isfinite(offset) and math.isfinite(value)):
                continue
            when = self.EPOCH + timedelta(minutes=offset - info["admit"])
            events.setdefault(info["hosp"], []).append(
                (when, feature, self._lab_value(feature, value), _LAB)
            )

        samples: list[dict[str, Any]] = []
        low_w, high_w = self.VALID_WEIGHT_KG
        for hosp, hosp_events in sorted(events.items()):
            infos = [i for i in stay_info.values() if i["hosp"] == hosp]
            hosp_events.sort(key=lambda e: (e[0], e[1]))
            death: datetime | None = None
            for info in infos:
                if info["expired"] and math.isfinite(info["discharge"]):
                    death = self.EPOCH + timedelta(
                        minutes=info["discharge"] - info["admit"]
                    )
            # Admission weight of each unit stay from its admission onwards; the
            # first one also covers labs drawn before the first ICU admission.
            weights = sorted(
                (self.EPOCH + timedelta(minutes=-info["admit"]), info["weight"])
                for info in infos
                if low_w <= info["weight"] <= high_w
            )
            if weights:
                weights[0] = (self.EPOCH - timedelta(days=1), weights[0][1])
            ages = [i["age"] for i in infos if math.isfinite(i["age"])]
            if not ages:
                continue
            age = ages[0]
            gender = 1.0 if infos[0]["female"] else 0.0
            samples.extend(
                self._timeline_samples(
                    patient.patient_id,
                    hosp_events,
                    weights,
                    death,
                    gender,
                    lambda _when, age=age: age,
                    rng_key=f"{patient.patient_id}-{hosp}",
                )
            )
        return samples
