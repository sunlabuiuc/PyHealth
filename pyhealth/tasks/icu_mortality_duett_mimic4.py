"""ICU Mortality Prediction Task for DuETT using MIMIC-IV.

Author: Shubham Srivastava (ss253@illinois.edu)

Description:
    This task converts irregular MIMIC-IV lab events into the fixed
    event-by-time tensor format required by DuETT. Observations are
    binned into uniform time windows within a configurable observation
    period after admission. Each cell stores the mean lab value and
    the observation count, allowing the model to distinguish true zeros
    from missing entries.

    The task produces patient-level samples with ICU mortality labels
    derived from the hospital_expire_flag field in MIMIC-IV admissions.
"""

from datetime import datetime, timedelta
from typing import Any, ClassVar, Dict, List, Optional, Tuple

import polars as pl

from .base_task import BaseTask


class ICUMortalityDuETTMIMIC4(BaseTask):
    """Task for ICU mortality prediction using DuETT format on MIMIC-IV.

    Converts irregular lab events into a fixed event-by-time tensor by
    binning into uniform time windows. Produces two tensors per sample:
    binned mean values and observation counts, plus static features
    (age, sex) and bin endpoint times.

    Args:
        n_time_bins: Number of uniform time bins within the observation
            window. Default is 24.
        input_window_hours: Hours after admission to observe.
            Default is 48.

    Attributes:
        task_name (str): Name of the task.
        input_schema (Dict[str, str]): Schema for input data.
        output_schema (Dict[str, str]): Schema for output data.

    Examples:
        >>> from pyhealth.datasets import MIMIC4EHRDataset
        >>> from pyhealth.tasks import ICUMortalityDuETTMIMIC4
        >>> dataset = MIMIC4EHRDataset(
        ...     root="/path/to/mimic-iv/2.2",
        ...     tables=["patients", "admissions", "labevents"],
        ... )
        >>> task = ICUMortalityDuETTMIMIC4(n_time_bins=24)
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "ICUMortalityDuETTMIMIC4"

    input_schema: ClassVar[Dict[str, str]] = {
        "ts_values": "tensor",
        "ts_counts": "tensor",
        "static": "tensor",
        "times": "tensor",
    }
    output_schema: ClassVar[Dict[str, str]] = {"mortality": "binary"}

    # 10 lab categories matching existing MIMIC-IV mortality tasks
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
        item
        for itemids in LAB_CATEGORIES.values()
        for item in itemids
    ]

    # Map each itemid to its category index for fast lookup
    _ITEMID_TO_CAT_IDX: ClassVar[Dict[str, int]] = {}
    for _idx, _cat in enumerate(LAB_CATEGORY_NAMES):
        for _itemid in LAB_CATEGORIES[_cat]:
            _ITEMID_TO_CAT_IDX[_itemid] = _idx
    del _idx, _cat, _itemid  # clean up loop variables

    D_VARS: ClassVar[int] = len(LAB_CATEGORY_NAMES)  # 10

    def __init__(
        self,
        n_time_bins: int = 24,
        input_window_hours: int = 48,
    ):
        """Initialize the task.

        Args:
            n_time_bins: Number of uniform time bins. Default is 24.
            input_window_hours: Observation window in hours after
                admission. Default is 48.
        """
        self.n_time_bins = n_time_bins
        self.input_window_hours = input_window_hours

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        """Process a patient to create DuETT-format mortality samples.

        Creates one sample per qualifying admission. Bins lab events
        into a fixed event-by-time tensor with observation counts.

        Args:
            patient: Patient object with get_events method.

        Returns:
            List of sample dicts with ts_values, ts_counts, static,
            times, and mortality label.
        """
        # Filter by age >= 18
        demographics = patient.get_events(event_type="patients")
        if not demographics:
            return []

        demographics = demographics[0]
        try:
            anchor_age = int(demographics.anchor_age)
            if anchor_age < 18:
                return []
        except (ValueError, TypeError, AttributeError):
            return []

        # Determine gender for static features
        try:
            gender = demographics.gender
            gender_val = 1.0 if gender == "M" else 0.0
        except AttributeError:
            gender_val = 0.0

        admissions = patient.get_events(event_type="admissions")
        if not admissions:
            return []

        samples = []
        window_td = timedelta(hours=self.input_window_hours)

        for admission in admissions:
            try:
                admission_time = admission.timestamp
                dischtime = datetime.strptime(
                    admission.dischtime, "%Y-%m-%d %H:%M:%S"
                )
            except (ValueError, AttributeError):
                continue

            if dischtime < admission_time:
                continue

            # Require admission longer than observation window
            duration_hours = (
                dischtime - admission_time
            ).total_seconds() / 3600.0
            if duration_hours < self.input_window_hours:
                continue

            # Get mortality label
            try:
                mortality = int(admission.hospital_expire_flag)
            except (ValueError, TypeError, AttributeError):
                mortality = 0

            # Get lab events within observation window
            window_end = admission_time + window_td
            labevents_df = patient.get_events(
                event_type="labevents",
                start=admission_time,
                end=window_end,
                return_df=True,
            )

            # Filter to relevant lab items
            labevents_df = labevents_df.filter(
                pl.col("labevents/itemid").is_in(self.LABITEMS)
            )

            if labevents_df.height == 0:
                continue

            # Parse storetime
            labevents_df = labevents_df.with_columns(
                pl.col("labevents/storetime").str.strptime(
                    pl.Datetime, "%Y-%m-%d %H:%M:%S"
                )
            )
            labevents_df = labevents_df.filter(
                pl.col("labevents/storetime") <= window_end
            )

            if labevents_df.height == 0:
                continue

            # Build event-by-time tensor via binning
            ts_values, ts_counts = self._bin_observations(
                labevents_df, admission_time
            )

            # Compute bin endpoint times in fractional days
            bin_duration_days = (
                self.input_window_hours / self.n_time_bins / 24.0
            )
            times = [
                (b + 1) * bin_duration_days
                for b in range(self.n_time_bins)
            ]

            # Static features: normalized age + binary sex
            static = [anchor_age / 100.0, gender_val]

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "ts_values": ts_values,
                    "ts_counts": ts_counts,
                    "static": static,
                    "times": times,
                    "mortality": mortality,
                }
            )

        return samples

    def _bin_observations(
        self,
        labevents_df: pl.DataFrame,
        admission_time: datetime,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Bin lab events into a fixed event-by-time tensor.

        Args:
            labevents_df: Filtered lab events DataFrame with itemid and
                valuenum columns.
            admission_time: Admission timestamp used as the reference
                point for time-bin offset calculation.

        Returns:
            A tuple ``(ts_values, ts_counts)``:

            - ``ts_values``: Nested list of shape ``(n_time_bins, D_VARS)``
              containing the mean lab value within each (bin, variable)
              cell. Cells with zero observations contain ``0.0``.
            - ``ts_counts``: Nested list of shape ``(n_time_bins, D_VARS)``
              containing the number of observations that fell into each
              (bin, variable) cell, preserving the distinction between
              truly zero values and missing measurements.
        """
        n_bins = self.n_time_bins
        d_vars = self.D_VARS
        window_seconds = self.input_window_hours * 3600.0

        # Accumulators: sum and count per (bin, variable)
        sums = [[0.0] * d_vars for _ in range(n_bins)]
        counts = [[0.0] * d_vars for _ in range(n_bins)]

        # Iterate through lab events
        for row in labevents_df.iter_rows(named=True):
            itemid = row["labevents/itemid"]
            valuenum = row.get("labevents/valuenum")

            if itemid not in self._ITEMID_TO_CAT_IDX:
                continue
            if valuenum is None:
                continue

            try:
                value = float(valuenum)
            except (ValueError, TypeError):
                continue

            # Compute time offset from admission
            event_time = row["timestamp"]
            offset_seconds = (
                event_time - admission_time
            ).total_seconds()
            if offset_seconds < 0 or offset_seconds >= window_seconds:
                continue

            # Determine bin index
            bin_idx = int(
                offset_seconds / window_seconds * n_bins
            )
            bin_idx = min(bin_idx, n_bins - 1)

            # Get category index
            cat_idx = self._ITEMID_TO_CAT_IDX[itemid]

            sums[bin_idx][cat_idx] += value
            counts[bin_idx][cat_idx] += 1.0

        # Compute means (zero-impute where count is 0)
        ts_values = []
        ts_counts = []
        for b in range(n_bins):
            row_vals = []
            row_cnts = []
            for v in range(d_vars):
                cnt = counts[b][v]
                if cnt > 0:
                    row_vals.append(sums[b][v] / cnt)
                else:
                    row_vals.append(0.0)
                row_cnts.append(cnt)
            ts_values.append(row_vals)
            ts_counts.append(row_cnts)

        return ts_values, ts_counts
