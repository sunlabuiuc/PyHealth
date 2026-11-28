"""Vasopressor-onset prediction task on MIMIC-IV (after Wu et al. 2022).

The task of "Learning Optimal Summaries of Clinical Time-series with
Concept Bottleneck Models" (MLHC 2022,
https://proceedings.mlr.press/v182/wu22a.html) moved from MIMIC-III,
where the paper defines it, to MIMIC-IV. The cohort rules, the 28
variables, the indicators and the 10 statics are those of
:class:`~pyhealth.tasks.VasopressorOnsetMIMIC3`; both tasks share one
implementation and differ only in the tables, item ids and the coding of
age, admission type and care unit.

The paper does not use MIMIC-IV, so every MIMIC-IV-specific choice here
is ours, not the authors'. They are listed on
:class:`VasopressorOnsetMIMIC4`.
"""

from collections.abc import Sequence
from typing import Any, ClassVar

from .vasopressor_onset_mimic3 import (
    TIME_LEN,
    _to_float,
    _VasopressorOnsetTask,
)

# MIMIC-IV item ids of the 28 variables, the GCS sub-scores and the
# weight (our choice): the MetaVision ``chartevents`` items and the
# ``labevents`` items of ``DEFAULT_ITEMIDS`` in the MIMIC-III task, with
# the CareVue-only ids dropped and ``po2`` given its MIMIC-IV items
# (the MIMIC-III candidate map only had a CareVue id for it). Every id
# was checked against the labels in ``d_items`` / ``d_labitems`` of the
# open MIMIC-IV demo v2.2. Pass ``itemid_map=`` to the task to replace it.
MIMIC4_ITEMIDS: dict[str, tuple[int, ...]] = {
    "dbp": (220051, 220180),
    "fio2": (223835,),
    "GCS_eye": (220739,),
    "GCS_motor": (223901,),
    "GCS_verbal": (223900,),
    "hr": (220045,),
    "map": (220052, 220181),
    "sbp": (220050, 220179),
    "spontaneousrr": (224689,),
    "spo2": (220277,),
    "temp": (223762, 223761),
    "urine": (226559, 226560),
    "bun": (51006, 225624),
    "magnesium": (50960, 220635),
    "platelets": (51265, 227457),
    "sodium": (50983, 220645),
    "alt": (50861, 220644),
    "hct": (51221, 220545),
    "po2": (50821, 220224),
    "ast": (50878, 220587),
    "potassium": (50971, 227464),
    "wbc": (51301, 220546),
    "bicarbonate": (50882, 227443),
    "creatinine": (50912, 220615),
    "lactate": (50813, 225668),
    "pco2": (50818, 220235),
    "glucose": (50931, 220621),
    "inr": (51237, 227467),
    "hgb": (51222, 220228),
    "bilirubin_total": (50885, 225690),
    "weight": (224639, 226512, 226531),
}

# MIMIC-IV temperature item charted in Fahrenheit.
MIMIC4_FAHRENHEIT_ITEMIDS: tuple[int, ...] = (223761,)

# MIMIC-IV weight item charted in pounds ("Admission Weight (lbs.)"),
# converted to kg. Added (our choice) because on the open demo a third of
# the first ICU stays have no weight in kg charted during the stay.
MIMIC4_POUND_ITEMIDS: tuple[int, ...] = (226531,)

# MIMIC-IV ``inputevents`` items counted as a vasopressor (our choice):
# the seven drugs of ``DEFAULT_VASOPRESSOR_ITEMIDS`` in the MIMIC-III
# task (norepinephrine, epinephrine, dopamine, phenylephrine,
# vasopressin, dobutamine, milrinone) plus the items MIMIC-IV added for
# two of the same drugs: 229617 "Epinephrine." and 229630 / 229631 /
# 229632 "Phenylephrine (50/250)" / "(200/250)_OLD_1" / "(200/250)".
# Angiotensin II (229709, 229764) is left out, as it is not one of the
# seven. Pass ``vasopressor_itemids=`` to the task to replace it.
MIMIC4_VASOPRESSOR_ITEMIDS: tuple[int, ...] = (
    221906,
    221289,
    229617,
    221662,
    221749,
    229630,
    229631,
    229632,
    222315,
    221653,
    221986,
)

# MIMIC-IV ``admission_type`` values mapped to the two MIMIC-III flags
# (our choice). MIMIC-III has EMERGENCY / URGENT / ELECTIVE / NEWBORN;
# MIMIC-IV has nine values. The observation and elective types set
# neither flag.
MIMIC4_URGENT_TYPES: tuple[str, ...] = ("URGENT",)
MIMIC4_EMERGENCY_TYPES: tuple[str, ...] = ("EW EMER.", "DIRECT EMER.")

# MIMIC-IV ``first_careunit`` values mapped to the four MIMIC-III unit
# flags (our choice). The mixed MICU/SICU sets both ``is_micu`` and
# ``is_sicu``; the neuro SICU counts as a SICU; the CVICU stands in for
# MIMIC-III's cardiac surgery recovery unit (CSRU). Units not listed
# (CCU, neuro intermediate / stepdown, and any other) set no flag, as
# the CCU does in MIMIC-III.
MIMIC4_CAREUNIT_FLAGS: dict[str, tuple[str, ...]] = {
    "Surgical Intensive Care Unit (SICU)": ("is_sicu",),
    "Neuro Surgical Intensive Care Unit (Neuro SICU)": ("is_sicu",),
    "Trauma SICU (TSICU)": ("is_tsicu",),
    "Medical Intensive Care Unit (MICU)": ("is_micu",),
    "Medical/Surgical Intensive Care Unit (MICU/SICU)": ("is_micu", "is_sicu"),
    "Cardiac Vascular Intensive Care Unit (CVICU)": ("is_csru",),
}

_UNIT_FLAG_ORDER = ("is_sicu", "is_tsicu", "is_micu", "is_csru")


class VasopressorOnsetMIMIC4(_VasopressorOnsetTask):
    """Vasopressor-onset prediction from the first 6 ICU hours (MIMIC-IV).

    The task of Wu et al., "Learning Optimal Summaries of Clinical
    Time-series with Concept Bottleneck Models" (MLHC 2022,
    https://proceedings.mlr.press/v182/wu22a.html), on MIMIC-IV instead
    of the paper's MIMIC-III. Works with
    :class:`~pyhealth.datasets.MIMIC4EHRDataset` loaded with the tables
    ``["chartevents", "labevents", "outputevents", "inputevents"]``.

    One sample per patient, from the first ICU stay, with the same
    fields as :class:`~pyhealth.tasks.VasopressorOnsetMIMIC3`:

    - ``values``: ``(6, 28)`` hourly vitals/labs in the order of
      ``CHANGING_VARS``.
    - ``indicators``: ``(6, 28)``, 1 where the variable was measured in
      that hour.
    - ``statics``: the 10 variables of ``STATIC_VARS``.
    - ``vasopressor_onset``: 1 if a vasopressor is started between
      ``time_len`` and ``max_los_hours`` hours after ICU admission.

    Cohort rules, as in the MIMIC-III task: first ICU stay only; ICU
    length of stay between 6 and 600 hours; patients given a vasopressor
    in the first 6 hours are excluded; patients with no weight
    measurement are excluded.

    Deviations that are ours (the paper has no MIMIC-IV version; each
    is an argument or a module constant, so it can be changed):

    - No MetaVision filter: all of MIMIC-IV is MetaVision.
    - Item ids: ``MIMIC4_ITEMIDS`` (MetaVision chart items plus the lab
      items of the MIMIC-III candidate map) and
      ``MIMIC4_VASOPRESSOR_ITEMIDS`` (the same seven drugs, plus the
      items MIMIC-IV added for epinephrine and phenylephrine).
    - Weight: daily weight and admission weight in kg, plus the
      admission weight in pounds (``MIMIC4_POUND_ITEMIDS``), converted.
    - Age: MIMIC-IV has no date of birth. Age is ``anchor_age`` plus
      the years from ``anchor_year`` to the ICU admission, in whole
      years, capped at 90 (``anchor_age`` itself is capped at 91).
    - ``is_urgent`` is 1 for admission type ``URGENT``; ``is_emergency``
      is 1 for ``EW EMER.`` and ``DIRECT EMER.``
      (``MIMIC4_URGENT_TYPES``, ``MIMIC4_EMERGENCY_TYPES``).
    - Care units follow ``MIMIC4_CAREUNIT_FLAGS``: MICU/SICU sets both
      ``is_micu`` and ``is_sicu``, the neuro SICU counts as a SICU and
      the CVICU stands in for the CSRU.

    Scaling, carry-forward and unit conversions are those of the
    MIMIC-III task.

    Args:
        normalization: ``"zscore"`` or ``None`` (raw values, NaN where
            unmeasured). Default is ``"zscore"``.
        time_len: number of hourly input steps. Default is 6.
        min_los_hours: minimum ICU length of stay. Default is 6.
        max_los_hours: maximum ICU length of stay, and the latest onset
            counted as positive. Default is 600.
        forward_fill: carry values forward within the window.
            Default is True.
        require_weight: drop patients with no weight measurement.
            Default is True.
        itemid_map: variable name to itemids, replacing
            ``MIMIC4_ITEMIDS``. Default is None.
        vasopressor_itemids: ``inputevents`` itemids that count as a
            vasopressor, replacing ``MIMIC4_VASOPRESSOR_ITEMIDS``.
            Default is None.
        careunit_flags: ``first_careunit`` value to the unit flags it
            sets, replacing ``MIMIC4_CAREUNIT_FLAGS``. Default is None.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset, PatientSplit
        >>> from pyhealth.tasks import VasopressorOnsetMIMIC4
        >>> dataset = MIMIC4EHRDataset(  # doctest: +SKIP
        ...     root="/path/to/mimic-iv/3.1",
        ...     tables=["chartevents", "labevents", "outputevents",
        ...             "inputevents"],
        ... )
        >>> task = VasopressorOnsetMIMIC4()
        >>> task.output_schema
        {'vasopressor_onset': 'binary'}
        >>> task._unit_flags("Trauma SICU (TSICU)")
        (0.0, 1.0, 0.0, 0.0)
        >>> train, val, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=PatientSplit(ratios=(0.68, 0.17, 0.15), seed=1)
        ... )
    """

    task_name: str = "VasopressorOnsetMIMIC4"

    _INPUT_TABLE: ClassVar[str] = "inputevents"
    _STAY_ID_KEY: ClassVar[str] = "stay_id"
    _DEFAULT_ITEMIDS: ClassVar[dict[str, tuple[int, ...]]] = MIMIC4_ITEMIDS
    _DEFAULT_VASOPRESSOR_ITEMIDS: ClassVar[tuple[int, ...]] = (
        MIMIC4_VASOPRESSOR_ITEMIDS
    )
    _FAHRENHEIT_ITEMIDS: ClassVar[tuple[int, ...]] = MIMIC4_FAHRENHEIT_ITEMIDS
    _POUND_ITEMIDS: ClassVar[tuple[int, ...]] = MIMIC4_POUND_ITEMIDS

    def __init__(
        self,
        normalization: str | None = "zscore",
        time_len: int = TIME_LEN,
        min_los_hours: float = 6.0,
        max_los_hours: float = 600.0,
        forward_fill: bool = True,
        require_weight: bool = True,
        itemid_map: dict[str, Sequence[int]] | None = None,
        vasopressor_itemids: Sequence[int] | None = None,
        careunit_flags: dict[str, Sequence[str]] | None = None,
    ) -> None:
        super().__init__(
            normalization=normalization,
            time_len=time_len,
            min_los_hours=min_los_hours,
            max_los_hours=max_los_hours,
            forward_fill=forward_fill,
            require_weight=require_weight,
            itemid_map=itemid_map,
            vasopressor_itemids=vasopressor_itemids,
        )
        flags = MIMIC4_CAREUNIT_FLAGS if careunit_flags is None else careunit_flags
        unknown = {
            flag for names in flags.values() for flag in names
        } - set(_UNIT_FLAG_ORDER)
        if unknown:
            raise ValueError(
                f"careunit_flags may only set {_UNIT_FLAG_ORDER}, got {unknown}"
            )
        self.careunit_flags = {
            unit.strip().lower(): tuple(names) for unit, names in flags.items()
        }

    def _age(self, demographics: Any, stay: Any) -> float:
        """Age in whole years at ICU admission, capped at 90.

        Args:
            demographics: the patient's ``patients`` event (with
                ``anchor_age`` and ``anchor_year``), or None.
            stay: the ICU stay event.

        Returns:
            ``anchor_age + (ICU admission year - anchor_year)`` capped at
            90, or NaN when either anchor is missing.
        """
        anchor_age = _to_float(getattr(demographics, "anchor_age", None))
        anchor_year = _to_float(getattr(demographics, "anchor_year", None))
        if anchor_age is None or anchor_year is None:
            return float("nan")
        return min(anchor_age + stay.timestamp.year - anchor_year, 90.0)

    def _admission_flags(self, admission_type: str) -> tuple[float, float]:
        """``(is_urgent, is_emergency)`` of a MIMIC-IV admission type.

        Args:
            admission_type: the admission's ``admission_type``.

        Returns:
            The two 0/1 flags, per ``MIMIC4_URGENT_TYPES`` and
            ``MIMIC4_EMERGENCY_TYPES``.
        """
        kind = admission_type.strip().upper()
        return (
            float(kind in MIMIC4_URGENT_TYPES),
            float(kind in MIMIC4_EMERGENCY_TYPES),
        )

    def _unit_flags(self, unit: str) -> tuple[float, float, float, float]:
        """``(is_sicu, is_tsicu, is_micu, is_csru)`` of a MIMIC-IV unit.

        Args:
            unit: the stay's ``first_careunit``.

        Returns:
            The four 0/1 flags, per ``careunit_flags``; all 0 for a unit
            that is not listed.
        """
        flags = self.careunit_flags.get(unit.strip().lower(), ())
        return tuple(float(name in flags) for name in _UNIT_FLAG_ORDER)
