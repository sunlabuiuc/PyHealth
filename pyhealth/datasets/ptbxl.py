import os
from typing import Optional

import dask.dataframe as dd
import pandas as pd

from pyhealth.datasets import BaseDataset


class PTBXLDataset(BaseDataset):
    """PTB-XL ECG dataset represented as an event table."""

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = "PTBXL",
        dev: bool = False,
        cache_dir: Optional[str] = None,
        num_workers: int = 1,
        use_high_resolution: bool = False,
    ):
        self.use_high_resolution = use_high_resolution
        super().__init__(
            root=root,
            tables=["ptbxl"],
            dataset_name=dataset_name,
            cache_dir=cache_dir,
            num_workers=num_workers,
            dev=dev,
        )

    def load_data(self) -> dd.DataFrame:
        metadata_path = os.path.join(self.root, "ptbxl_database.csv")
        df = pd.read_csv(metadata_path)

        record_path_col = "filename_hr" if self.use_high_resolution else "filename_lr"

        event_df = pd.DataFrame(
            {
                "patient_id": df["patient_id"].astype(str),
                "event_type": "ptbxl",
                "timestamp": pd.NaT,
                "ptbxl/ecg_id": df["ecg_id"],
                "ptbxl/record_path": df[record_path_col],
                "ptbxl/scp_codes": df["scp_codes"],
            }
        )

        return dd.from_pandas(event_df, npartitions=1)