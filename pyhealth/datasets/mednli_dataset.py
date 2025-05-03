"""
MedNLI Dataset Implementation for PyHealth
Author: Abraham Arellano, Umesh Kumar
NetID: aa107, umesh2
Paper: Lessons from natural language inference in the clinical domain
Paper Link: https://arxiv.org/abs/1808.06752
Description: Implementation of the Medical Natural Language Inference dataset for sentence pair classification in the clinical domain.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
import polars as pl
from tqdm import tqdm

from ..tasks import MedNLITask
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class MedNLIDataset(BaseDataset):
    """Dataset for Medical Natural Language Inference (MedNLI).

    MedNLI is a dataset for natural language inference in the clinical domain,
    consisting of sentence pairs (premise and hypothesis) manually annotated 
    for textual entailment.

    The dataset is available at:
    https://jgc128.github.io/mednli/

    Paper:
    Romanov, A., & Shivade, C. (2018). Lessons from natural language inference 
    in the clinical domain. arXiv preprint arXiv:1808.06752.

    Args:
        root: Root directory of the raw data (should contain train.jsonl, dev.jsonl, test.jsonl).
        dataset_name: Optional name of the dataset. Defaults to "mednli".
        config_path: Optional path to the configuration file. If not provided,
            uses the default config in the configs directory.
        dev: Whether to enable dev mode (only use a small subset of the data).
            Default is False.
        data_fraction: Optional fraction of the data to use (0.01, 0.05, 0.1, 0.25, 1.0).
            Only applied to the training set. Default is 1.0.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.
        dev: Whether dev mode is enabled.
        data_fraction: Fraction of the training data to use.

    Examples:
        >>> from pyhealth.datasets import MedNLIDataset
        >>> dataset = MedNLIDataset(
        ...     root="/path/to/mednli"
        ... )
        >>> dataset.stat()
        >>> samples = dataset.set_task()
        >>> print(samples[0])
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        data_fraction: float = 1.0,
    ) -> None:
        """Initialize the MedNLI dataset.

        Creates a dataset instance for Medical Natural Language Inference (MedNLI),
        which consists of sentence pairs annotated for textual entailment in the
        clinical domain.

        Args:
            root: Root directory containing the MedNLI JSONL files (train.jsonl, 
                dev.jsonl, test.jsonl).
            dataset_name: Optional name for the dataset. Defaults to "mednli".
            config_path: Optional path to configuration file. Defaults to the built-in 
                config.
            dev: Whether to use development mode with limited samples. Defaults to False.
            data_fraction: Fraction of training data to use (0.01, 0.05, 0.1, 0.25, 1.0).
                Defaults to 1.0.

        Returns:
            None
        """
        self.data_fraction = data_fraction

        if config_path is None:
            logger.info("No config path provided, using default config")
            config_path = Path(__file__).parent / "configs" / "mednli.yaml"

        default_tables = ["mednli"]
        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "mednli",
            config_path=config_path,
            dev=dev,
        )
        return

    def load_table(self, table_name: str) -> pl.LazyFrame:
        """Loads a table and processes JSONL files for MedNLI data.

        Args:
            table_name: The name of the table to load.

        Returns:
            pl.LazyFrame: The processed lazy frame for the table.

        Raises:
            ValueError: If the table is not found in the config.
            FileNotFoundError: If required JSONL files are not found.
        """
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]

        # Check for required files
        train_path = os.path.join(self.root, "train.jsonl")
        dev_path = os.path.join(self.root, "dev.jsonl")
        test_path = os.path.join(self.root, "test.jsonl")

        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training file not found: {train_path}")
        if not os.path.exists(dev_path):
            raise FileNotFoundError(f"Development file not found: {dev_path}")
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test file not found: {test_path}")

        # Process files and create dataframes
        train_data = self._process_jsonl_file(train_path, "train")
        dev_data = self._process_jsonl_file(dev_path, "dev")
        test_data = self._process_jsonl_file(test_path, "test")

        # Apply data fraction if needed
        if self.data_fraction < 1.0:
            logger.info(f"Using {self.data_fraction * 100}% of training data")
            train_data = train_data.sample(frac=self.data_fraction, seed=42)

        # Limit data size in dev mode
        if self.dev:
            logger.info("Dev mode enabled: limiting to 100 samples per split")
            train_data = train_data.head(100)
            dev_data = dev_data.head(100)
            test_data = test_data.head(100)

        # Combine all data
        all_data = pd.concat([train_data, dev_data, test_data],
                             ignore_index=True)

        # Convert to LazyFrame
        df = pl.from_pandas(all_data).lazy()

        # Add required columns for BaseDataset
        df = df.with_columns([
            pl.lit(None).alias("patient_id"),
            pl.lit(table_name).cast(pl.Utf8).alias("event_type"),
            pl.col("pairID").cast(pl.Utf8).alias("timestamp"),
        ])

        # Add column prefix for attributes
        attribute_cols = table_cfg.attributes
        attribute_columns = [
            pl.col(attr).alias(f"{table_name}/{attr}")
            for attr in attribute_cols
        ]

        event_frame = df.select(["patient_id", "event_type", "timestamp"] +
                                attribute_columns)

        return event_frame

    def _process_jsonl_file(self, file_path: str, split: str) -> pd.DataFrame:
        """Process a JSONL file into a pandas DataFrame.

        Reads a JSONL file containing MedNLI data and converts it to a pandas DataFrame
        with the appropriate structure.

        Args:
            file_path: Path to the JSONL file to process.
            split: Data split identifier (train, dev, test).

        Returns:
            pd.DataFrame: DataFrame containing processed MedNLI data with split information.
        """
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line)
                item['split'] = split
                data.append(item)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        return df

    def stat(self) -> str:
        """Returns statistics about the dataset.

        Calculates and returns statistics about the MedNLI dataset, including
        sample counts by split and label distribution.

        Returns:
            str: Formatted string containing dataset statistics.
        """
        df = self.collected_global_event_df

        # Get basic counts
        train_count = df.filter(pl.col("mednli/split") == "train").height
        dev_count = df.filter(pl.col("mednli/split") == "dev").height
        test_count = df.filter(pl.col("mednli/split") == "test").height

        # Count label distributions
        label_distribution = df.group_by("mednli/gold_label").count()

        # Format the stats message
        stats = [
            f"Dataset: {self.dataset_name}",
            f"Total samples: {df.height}",
            f"Training samples: {train_count}",
            f"Development samples: {dev_count}",
            f"Test samples: {test_count}",
            f"Label distribution:",
        ]

        for row in label_distribution.rows():
            stats.append(f"  - {row[0]}: {row[1]}")

        stats_str = "\n".join(stats)
        print(stats_str)
        return stats_str

    @property
    def default_task(self) -> MedNLITask:
        """Returns the default task for the dataset.

        Provides the default MedNLITask instance configured for this dataset.

        Returns:
            MedNLITask: The default task instance for MedNLI classification.
        """
        return MedNLITask()
