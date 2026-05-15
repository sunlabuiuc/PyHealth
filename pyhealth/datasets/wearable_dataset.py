from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import dask.dataframe as dd
import pandas as pd

from pyhealth.datasets import BaseDataset


class WearableDataset(BaseDataset):
    """Dataset for daily wearable data used to detect illness trends over time."""

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: str = "WearableDataset",
        dev: bool = False,
    ) -> None:
        # Default to a single table of daily wearable records
        if tables is None:
            tables = ["wearable_daily"]

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name,
            dev=dev,
        )

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Load wearable.csv and convert it into PyHealth event format."""
        file_path = f"{self.root}/wearable.csv"
        df = pd.read_csv(file_path)

        # Basic check to make sure required fields are present
        required_columns = {
            "patient_id",
            "day_index",
            "resting_heart_rate",
            "sleep_duration",
            "is_ill",
        }
        missing = required_columns - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df.copy()
        df["patient_id"] = df["patient_id"].astype("string")
        df["event_type"] = table_name

        # Convert day index into a simple timestamp so events can be ordered
        df["timestamp"] = df["day_index"].apply(
            lambda x: datetime(2020, 1, 1) + timedelta(days=int(x))
        )

        # Add prefixes so PyHealth treats these as event attributes
        df[f"{table_name}/day_index"] = df["day_index"]
        df[f"{table_name}/resting_heart_rate"] = df["resting_heart_rate"]
        df[f"{table_name}/sleep_duration"] = df["sleep_duration"]
        df[f"{table_name}/is_ill"] = df["is_ill"]

        # Only keep the columns PyHealth expects
        final_cols = [
            "patient_id",
            "event_type",
            "timestamp",
            f"{table_name}/day_index",
            f"{table_name}/resting_heart_rate",
            f"{table_name}/sleep_duration",
            f"{table_name}/is_ill",
        ]

        return dd.from_pandas(df[final_cols], npartitions=1)