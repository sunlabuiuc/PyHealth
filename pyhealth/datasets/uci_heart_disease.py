"""
Names: Desi Nainar, Sayed Sarwary, Kelong Wu
Net IDs: ds34, sarwary2, kwu18

Dataset: UCI Heart Disease (Cleveland)
Link: https://archive.ics.uci.edu/dataset/45/heart+disease

This class implements the HeartDiseaseDataset for PyHealth.

It loads the UCI Heart Disease dataset directly from Python via
`ucimlrepo`, converts it into PyHealth's patient-event format,
and provides utilities for downstream tasks such as heart disease
risk prediction.
"""

import logging
from typing import List, Optional

import polars as pl
from ucimlrepo import fetch_ucirepo
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class HeartDiseaseDataset(BaseDataset):
    """
    UCI Heart Disease Dataset using ucimlrepo.

    Attributes:
        root (str): Root directory (not used for CSV in this version).
        tables (List[str]): Only 'heart_disease'.
        dataset_name (str): Name of the dataset.
        dev (bool): Whether to run in dev mode (limit to 1000 patients).
    """

    def __init__(
        self,
        root: Optional[str] = None,
        tables: Optional[List[str]] = None,
        dataset_name: str = "uci_heart_disease",
        dev: bool = False,
    ):
        """Initialize the HeartDiseaseDataset using ucimlrepo.

        Args:
            root: Root directory (not used but kept for BaseDataset compatibility).
            tables: List of tables to load (default ['heart_disease']).
            dataset_name: Name of the dataset.
            dev: Whether to limit to 1000 patients for dev mode.
        """
        if tables is None:
            tables = ["heart_disease"]

        logger.info("Fetching UCI Heart Disease dataset using ucimlrepo")
        super().__init__(
            root=root or "",
            tables=tables,
            dataset_name=dataset_name,
            config_path=None,
            dev=dev,
        )

    def load_table(self, table_name: str) -> pl.LazyFrame:
        """Load heart disease data via ucimlrepo and convert to PyHealth events.

        Args:
            table_name: Must be 'heart_disease'.

        Returns:
            pl.LazyFrame: LazyFrame with patient_id, event_type, timestamp,
                          and feature columns prefixed with table_name.
        """
        if table_name != "heart_disease":
            raise ValueError(f"Unknown table {table_name} for HeartDiseaseDataset")

        dataset = fetch_ucirepo(id=45)
        X = dataset.data.features
        y = dataset.data.targets

        import pandas as pd

        df = X.copy()
        df["target"] = y
        df["patient_id"] = df.index.astype(str)

        base_columns = [
            "patient_id",
        ]
        base_df = pl.from_pandas(df[base_columns])
        base_df = base_df.with_columns(
            pl.lit("heart_disease").alias("event_type"),
            pl.lit(None).cast(pl.Datetime).alias("timestamp"),
        )

        attribute_columns = [
            pl.col(col).alias(f"{table_name}/{col}") for col in df.columns
            if col not in ["patient_id"]
        ]
        df_pl = pl.from_pandas(df)
        df_pl = df_pl.select(base_columns + ["event_type", "timestamp"] + [
            col for col in df_pl.columns if col not in base_columns
        ])

        for col in df.columns:
            if col not in ["patient_id"]:
                df_pl = df_pl.rename({col: f"{table_name}/{col}"})

        return df_pl.lazy()

    @property
    def default_task(self):
        """Returns the default task for the dataset (none by default)."""
        return None

