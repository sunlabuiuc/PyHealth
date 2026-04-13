"""
PyHealth task for FAMEWS-style fairness analysis on HiRID.

Dataset link:
    https://physionet.org/content/hirid/1.1.1/

Dataset paper: (please cite if you use this dataset)
    Hoche, M.; Mineeva, O.; Burger, M.; Blasimme, A.; and
    Ratsch, G. 2024. FAMEWS: a Fairness Auditing tool for
    Medical Early-Warning Systems. In Proceedings of Ma-
    chine Learning Research, volume 248, 297–311. Confer-
    ence on Health, Inference, and Learning (CHIL) 2024.

Dataset paper link:
    https://proceedings.mlr.press/v248/hoche24a.html

Author:
    John Doll (doll3@illinois.edu)
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import polars as pl

from pyhealth.tasks import BaseTask


class FAMEWS_fairness_audit(BaseTask):
    """Build per-patient samples for FAMEWS-style fairness analysis on HiRID.

    This task is designed for :class:`~pyhealth.datasets.HiRIDDataset` and
    returns one sample per patient containing:

    - A multivariate timeseries (for model features)
    - Demographic metadata used for fairness subgrouping

    By default it reads from ``imputed_stage``. For merged data, set
    ``stage_table="merged_stage"``.

    The HiRID subgroup configuration used by the original FAMEWS fairness
    pipeline is bundled at ``FAMEWS_fairness_audit.group_config_path``.

    Examples:
        >>> from pyhealth.datasets import HiRIDDataset
        >>> from pyhealth.tasks import FAMEWS_fairness_audit
        >>> dataset = HiRIDDataset(
        ...     root="/path/to/hirid/1.1.1",
        ...     stage="imputed",
        ... )
        >>> task = FAMEWS_fairness_audit(stage_table="imputed_stage")
        >>> samples = dataset.set_task(task)
    """

    task_name: str = "FAMEWS_fairness_audit"
    input_schema: Dict[str, str] = {"signals": "timeseries"}
    output_schema: Dict[str, str] = {}
    group_config_path: str = str(
        Path(__file__).parent / "configs" / "group_hirid_complete.yaml"
    )

    DEFAULT_FEATURE_COLUMNS: List[str] = [
        "heart_rate",
        "systolic_bp_invasive",
        "diastolic_bp_invasive",
        "mean_arterial_pressure",
        "cardiac_output",
        "spo2",
        "rass",
        "peak_inspiratory_pressure",
        "lactate_arterial",
        "lactate_venous",
        "inr",
        "serum_glucose",
        "c_reactive_protein",
        "dobutamine",
        "milrinone",
        "levosimendan",
        "theophyllin",
        "non_opioid_analgesics",
    ]

    def __init__(
        self,
        stage_table: str = "imputed_stage",
        feature_columns: Optional[Iterable[str]] = None,
    ) -> None:
        if stage_table not in {"merged_stage", "imputed_stage"}:
            raise ValueError(
                "stage_table must be one of {'merged_stage', 'imputed_stage'}, "
                f"got '{stage_table}'"
            )
        self.stage_table = stage_table
        self.feature_columns = list(feature_columns) if feature_columns else list(
            self.DEFAULT_FEATURE_COLUMNS
        )

    @staticmethod
    def _build_age_group(age_value: Any) -> Optional[str]:
        try:
            age = float(age_value)
        except (TypeError, ValueError):
            return None

        if age < 50:
            return "<50"
        if age < 65:
            return "50-65"
        if age < 75:
            return "65-75"
        if age < 85:
            return "75-85"
        return ">85"

    def _extract_time_axis(self, events_df: pl.DataFrame) -> List[Any]:
        if "timestamp" in events_df.columns:
            timestamp_series = events_df.get_column("timestamp")
            if timestamp_series.null_count() < events_df.height:
                return timestamp_series.to_list()

        relative_time_col = f"{self.stage_table}/reldatetime"
        if relative_time_col in events_df.columns:
            return events_df.get_column(relative_time_col).to_list()

        return list(range(events_df.height))

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        patient_general = patient.get_events(event_type="general_table")
        if len(patient_general) == 0:
            return []

        stage_df = patient.get_events(event_type=self.stage_table, return_df=True)
        if stage_df.height == 0:
            return []

        available_features = [
            col
            for col in self.feature_columns
            if f"{self.stage_table}/{col}" in stage_df.columns
        ]
        if len(available_features) == 0:
            return []

        value_df = stage_df.select(
            [
                pl.col(f"{self.stage_table}/{col}").cast(pl.Float64)
                for col in available_features
            ]
        )

        general_event = patient_general[0]
        age_group = self._build_age_group(getattr(general_event, "age", None))

        sample = {
            "patient_id": patient.patient_id,
            "signals": (self._extract_time_axis(stage_df), value_df.to_numpy()),
            "feature_columns": available_features,
            "sex": getattr(general_event, "sex", None),
            "age": getattr(general_event, "age", None),
            "age_group": age_group,
            "discharge_status": getattr(general_event, "discharge_status", None),
        }

        return [sample]
