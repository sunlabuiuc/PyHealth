"""Day-one continuous LoS adapter for TPC using existing YAIB ICU extraction."""

from typing import Any, ClassVar, Literal

import numpy as np
import polars as pl
import torch

from pyhealth.data import Patient

from .base_task import BaseTask
from .yaib_eicu import YAIBLengthOfStayEICU
from .yaib_mimic4 import _DYNAMIC_FEATURES as MIMIC4_FEATURES
from .yaib_mimic4 import YAIBLengthOfStayMIMIC4


class TPCDayOne(BaseTask):
    """Predicts remaining ICU days after observing the first 24 hours.

    Reuses YAIB's feature definitions, unit conversions, hourly aggregation,
    and retrospective cohort exclusions, but not its seven-day label cap or
    normalized hourly labels. Keeps stays longer than 24 hours and returns
    one window of bins 0..23 per stay. The prediction time is hour 24; labels
    are ``(actual ICU duration in hours - 24) / 24``, clipped to 30 minutes
    through 100 days. This is a day-one adaptation, not the paper's all-hour
    training protocol or exact cohort.

    Raw missing values and their original measurement mask are retained.
    Tensor processors learn z-score statistics on training patients when
    used with ``set_task(..., split=PatientSplit(...))``. Configure TPC with
    ``observed_key="observed"``, ``decay_key=None``, ``static_key="static"``
    and this task's ``lab_indices``. The model then fills standardized values
    causally and derives decay from the pre-fill mask. Static missing values
    become zero (the training mean). Lab-only features are selected for the
    no-forward-fill ablation; mixed lab/vital features are not lab-only.

    Args:
        dataset: ``"eicu"`` or ``"mimic4"``; must match the source dataset.

    Examples:
        >>> task = TPCDayOne("eicu")
        >>> task.task_name
        'TPCDayOne_eicu_v1'
        >>> task.output_schema
        {'los': 'regression'}
    """

    input_schema: ClassVar[dict[str, Any]] = {
        "time_series": ("tensor", {"normalization": "zscore"}),
        "observed": ("tensor", {"dtype": torch.bool}),
        "static": (
            "tensor",
            {"normalization": "zscore", "impute_strategy": "zero"},
        ),
    }
    output_schema: ClassVar[dict[str, Any]] = {"los": "regression"}
    observation_hours: ClassVar[int] = 24

    def __init__(self, dataset: Literal["eicu", "mimic4"] = "eicu") -> None:
        """Selects existing ICU extraction without loading any patient data."""
        super().__init__()
        if dataset not in ("eicu", "mimic4"):
            raise ValueError("dataset must be 'eicu' or 'mimic4'.")
        self.dataset = dataset
        source = self.source
        self.task_name = f"TPCDayOne_{dataset}_v1"
        self.feature_names = list(source.DYNAMIC_FEATURES)
        definitions = (
            source._features_read()[0] if dataset == "eicu" else MIMIC4_FEATURES
        )
        self.lab_indices = [
            self.feature_names.index(name)
            for name, (_, _, items) in definitions.items()
            if items and all(item.table in ("lab", "labevents") for item in items)
        ]

    @property
    def source(self) -> YAIBLengthOfStayEICU | YAIBLengthOfStayMIMIC4:
        """Returns a stateless extractor without adding object IDs to cache keys."""
        return (
            YAIBLengthOfStayEICU()
            if self.dataset == "eicu"
            else YAIBLengthOfStayMIMIC4()
        )

    def pre_filter(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Delegates required-table validation and row selection to YAIB.

        Args:
            df: Source event frame, before patient sample extraction.

        Returns:
            The existing YAIB task's filtered event frame.
        """
        return self.source.pre_filter(df)

    def __call__(self, patient: Patient) -> list[dict[str, Any]]:
        """Extracts day-one windows without using post-hour-24 measurements.

        Args:
            patient: One patient's events from the selected PyHealth dataset.

        Returns:
            One sample per eligible stay, or an empty list for excluded stays.
        """
        source = self.source
        stays = source._stays(patient)
        if stays.height == 0:
            return []
        stays, hourly, static = source._base_cohort(patient, stays)
        samples = []
        for stay_id, los_hours in stays.select("stay_id", "los_hours").iter_rows():
            if los_hours <= self.observation_hours:
                continue
            dynamic = source._dynamic(hourly, stay_id, self.observation_hours - 1)
            values = np.asarray(dynamic["values"], dtype=np.float32)
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(stay_id),
                    "time_series": values,
                    "observed": np.isfinite(values),
                    "static": static[stay_id],
                    "los": float(
                        np.clip(
                            (los_hours - self.observation_hours) / 24,
                            1 / 48,
                            100,
                        )
                    ),
                }
            )
        return samples
