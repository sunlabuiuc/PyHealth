import logging
import warnings
from pathlib import Path
from typing import List, Optional

import ast
import numpy as np
import wfdb
import polars as pl
from collections import Counter

from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PTBXL(BaseDataset):
    """
    A dataset class for handling PTB-XL ECG data https://physionet.org/content/ptb-xl/1.0.1/ 
    using PyHealth's configuration system.

    Attributes:
        root (str): Root directory containing the dataset and signal files.
        tables (List[str]): Tables to include (default: ['ecg']).
        dataset_name (Optional[str]): Name of the dataset (default: 'ptbxl').
        config_path (Optional[str]): Path to the dataset YAML configuration file.
    """

    def __init__(
        self,
        root: str,
        tables: List[str] = [],
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        use_high_res: bool = False,
        **kwargs
    ) -> None:
        if config_path is None:
            logger.info("No config path provided, using default ptbxl.yaml")
            config_path = Path(__file__).parent / "configs" / "ptbxl.yaml"

        default_tables = ["ecg"]
        tables = default_tables + tables

        self.use_high_res = use_high_res
        self.samples = []

        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            **kwargs
        )

    def post_init(self) -> None:
        """Called after event table and tables are fully initialized."""
        self.build_from_events()

    def build_from_events(self, limit: int = None) -> None:
        """Processes the event table and loads WFDB signals into self.samples."""
        """Loads ECG metadata and waveforms from the 'ecg' table."""
        print("Calling load_table('ecg')...")
        df = self.load_table("ecg")

        # Make sure it's a regular DataFrame (not LazyFrame)
        if hasattr(df, "collect"):
            df = df.collect()
        print(f"df columns: {df.columns}")
        required_cols = {"ecg/filename_lr", "ecg/filename_hr", "ecg/scp_codes", "ecg/age", "ecg/sex"}
        if not required_cols.issubset(df.columns):
            raise ValueError(f"Missing required columns: {required_cols - set(df.columns)}")

        skipped = 0
        self.samples = []

        for i, row in enumerate(df.iter_rows(named=True)):
            if limit is not None and i >= limit:
                break
            record_path = Path(self.root) / row["ecg/filename_lr"]
            try:
                signal, _ = wfdb.rdsamp(str(record_path))
            except Exception as e:
                logger.warning(f"Error reading {record_path}: {e}")
                skipped += 1
                continue

            if signal.shape[0] < 2496:
                # Pad with zeros at the end
                pad_length = 2496 - signal.shape[0]
                signal = np.pad(signal, ((0, pad_length), (0, 0)), mode="constant")
            elif signal.shape[0] > 2496:
                # Truncate to 2496
                signal = signal[:2496]
            if signal.shape[0] != 2496:
                logger.warning(f"Skipping signal of shape {signal.shape}")
                skipped += 1
                continue

            scp_dict = ast.literal_eval(row["ecg/scp_codes"])
            label = self.extract_label(scp_dict)

            self.samples.append({
                "record_id": row["patient_id"],  # originally ecg_id
                "signal": np.transpose(signal).astype(np.float32),  # shape (12, 2496)
                "label": label,
                "age": row["ecg/age"],
                "sex": row["ecg/sex"],
            })
        print(f"Finished loading {len(self.samples)} samples. Skipped {skipped} rows.")

    def extract_label(self, scp_dict: dict) -> int:
        """Extracts a binary label: 1 if only NORM, otherwise 0."""
        filtered_codes = {code for code, score in scp_dict.items() if score > 0.0 and code != "SR"}
        return 1 if filtered_codes == {"NORM"} else 0

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]

    def __len__(self) -> int:
        return len(self.samples)

    def summary(self) -> None:
        """Prints a summary of the dataset's structure and contents."""
        print("PTBXL Dataset Summary")
        print(f"Total loaded samples: {len(self.samples)}")

        if not self.samples:
            print("No samples loaded.")
            return

        print(f"Available sample fields: {list(self.samples[0].keys())}")

        # Signal shape consistency
        signal_shapes = {sample["signal"].shape for sample in self.samples}
        print(f"Signal shapes found: {signal_shapes}")

        # Label distribution
        labels = [sample["label"] for sample in self.samples]
        label_counts = Counter(labels)
        print(f"Label distribution: {label_counts}")

        # Age stats
        ages = [float(sample["age"]) for sample in self.samples if sample["age"] not in (None, '', 'NA')]
        if ages:
            print(f"Age: mean={np.mean(ages):.1f}, range=({min(ages)}–{max(ages)})")

        # Sex distribution
        sexes = [sample["sex"] for sample in self.samples]
        sex_counts = Counter(sexes)
        print(f"Sex distribution: {sex_counts}")
def main():
    import logging
    logging.basicConfig(level=logging.DEBUG)

    print("Running PTBXL main()")
    dataset = PTBXL(
        root="/home/sbgray2/ptbxl_data/" \
        "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1",
        config_path="pyhealth/datasets/configs/ptbxl.yaml",
        use_high_res=False,
    )
    print("Manually calling build_from_events()...")
    dataset.build_from_events(limit=25)

    print("Calling summary:")
    dataset.summary()
if __name__ == "__main__" or __name__.endswith("ptbxl"):
    main()