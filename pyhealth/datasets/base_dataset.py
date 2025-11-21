import logging
import os
import pickle
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse, urlunparse
import uuid
import json
import functools
import operator

import polars as pl
import pandas as pd
import dask.dataframe as dd
import pyarrow.csv as pv
import pyarrow.parquet as pq
import requests
from tqdm import tqdm
import platformdirs

from ..data import Patient
from ..tasks import BaseTask
from ..processors.base_processor import FeatureProcessor
from .configs import load_yaml_config
from .sample_dataset import SampleDataset
from .utils import _convert_for_cache, _restore_from_cache

logger = logging.getLogger(__name__)


def is_url(path: str) -> bool:
    """URL detection."""
    result = urlparse(path)
    # Both scheme and netloc must be present for a valid URL
    return all([result.scheme, result.netloc])


def clean_path(path: str) -> str:
    """Clean a path string."""
    if is_url(path):
        parsed = urlparse(path)
        cleaned_path = os.path.normpath(parsed.path)
        # Rebuild the full URL
        return urlunparse(parsed._replace(path=cleaned_path))
    else:
        # It's a local path — resolve and normalize
        return str(Path(path).expanduser().resolve())


def path_exists(path: str) -> bool:
    """
    Check if a path exists.
    If the path is a URL, it will send a HEAD request.
    If the path is a local file, it will use the Path.exists().
    """
    if is_url(path):
        try:
            response = requests.head(path, timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False
    else:
        return Path(path).exists()

def alt_path(path: str) -> str:
    """
    Get the alternative path by switching between .csv.gz, .csv, .tsv.gz, and .tsv extensions.

    Args:
        path (str): Original file path.

    Returns:
        str: Alternative file path.
    """
    if path.endswith(".csv.gz"):
        return path[:-3]  # Remove .gz -> try .csv
    elif path.endswith(".csv"):
        return f"{path}.gz"  # Add .gz -> try .csv.gz
    elif path.endswith(".tsv.gz"):
        return path[:-3]  # Remove .gz -> try .tsv
    elif path.endswith(".tsv"):
        return f"{path}.gz"  # Add .gz -> try .tsv.gz
    else:
        raise ValueError(f"Path does not have expected extension: {path}")

def scan_csv_gz_or_csv_tsv(path: str) -> pl.LazyFrame:
    """
    Scan a CSV.gz, CSV, TSV.gz, or TSV file and returns a LazyFrame.
    It will fall back to the other extension if not found.

    Args:
        path (str): URL or local path to a .csv, .csv.gz, .tsv, or .tsv.gz file

    Returns:
        pl.LazyFrame: The LazyFrame for the CSV.gz, CSV, TSV.gz, or TSV file.
    """

    def scan_file(file_path: str) -> pl.LazyFrame:
        separator = "\t" if ".tsv" in file_path else ","
        return pl.scan_csv(file_path, separator=separator, infer_schema=False)

    if path_exists(path):
        return scan_file(path)

    # Try the alternative extension
    if path.endswith(".csv.gz"):
        alt_path = path[:-3]  # Remove .gz -> try .csv
    elif path.endswith(".csv"):
        alt_path = f"{path}.gz"  # Add .gz -> try .csv.gz
    elif path.endswith(".tsv.gz"):
        alt_path = path[:-3]  # Remove .gz -> try .tsv
    elif path.endswith(".tsv"):
        alt_path = f"{path}.gz"  # Add .gz -> try .tsv.gz
    else:
        raise FileNotFoundError(f"Path does not have expected extension: {path}")

    if path_exists(alt_path):
        logger.info(f"Original path does not exist. Using alternative: {alt_path}")
        return scan_file(alt_path)

    raise FileNotFoundError(f"Neither path exists: {path} or {alt_path}")


class BaseDataset(ABC):
    """Abstract base class for all PyHealth datasets.

    Attributes:
        root (Path): The root directory where dataset files are stored.
        tables (List[str]): List of table names to load.
        dataset_name (str): Name of the dataset.
        config (dict): Configuration loaded from a YAML file.
        global_event_df (pl.LazyFrame): The global event data frame.
        dev (bool): Whether to enable dev mode (limit to 1000 patients).
    """

    def __init__(
        self,
        root: str,
        tables: List[str],
        dataset_name: str | None = None,
        config_path: str | None = None,
        cache_dir: str | Path | None = None,
        dev: bool = False,
    ):
        """Initializes the BaseDataset.

        Args:
            root (str): The root directory where dataset files are stored.
            tables (List[str]): List of table names to load.
            dataset_name (str | None): Name of the dataset. Defaults to class name.
            config_path (str | None): Path to the configuration YAML file.
            cache_dir (str | Path | None): Directory to cache processed data. If None, a default
                cache directory will be created under the platform's cache directory.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
        """
        if config_path is None:
            raise ValueError("config_path must be provided")

        if len(set(tables)) != len(tables):
            logger.warning("Duplicate table names in tables list. Removing duplicates.")
            tables = list(set(tables))

        self.root = root
        self.tables = tables
        self.dataset_name = dataset_name or self.__class__.__name__
        self.config = load_yaml_config(config_path)
        self.dev = dev

        subfolder = self.cache_subfolder(self.root, self.tables, self.dataset_name, self.dev)
        self.setup_cache_dir(cache_dir=cache_dir, subfolder=subfolder)

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        self.global_event_df = self.load_data()

        # Cached attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

    @staticmethod
    def cache_subfolder(root: str, tables: List[str], dataset_name: str, dev: bool) -> str:
        """Generates a unique identifier for the dataset instance. This is used for creating 
        cache directories. The UUID is based on the root path, tables, dataset name, and dev mode.

        Returns:
            str: A unique identifier string.
        """
        id_str = json.dumps({
            "root": root,
            "tables": sorted(tables),
            "dataset_name": dataset_name,
            "dev": dev,
        }, sort_keys=True)
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

    def setup_cache_dir(self, cache_dir: str | Path | None = None, subfolder: str = str(uuid.uuid4())) -> None:
        """Creates the cache directory structure.

        Args:
            cache_dir (str | Path | None): The base cache directory. If None, a default cache
                directory will be created under the platform's cache directory.
            subfolder (str): Subfolder name for this dataset instance's cache.
        """
        if cache_dir is None:
            cache_dir = platformdirs.user_cache_dir(appname='pyhealth')
            logger.info(f"No cache_dir provided. Using default cache for PyHealth: {cache_dir}")
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir / subfolder

        self.cache_dir.mkdir(parents=True, exist_ok=True)        
        # Create tables subdirectory to store cached table files
        (self.cache_dir / "tables").mkdir(parents=True, exist_ok=True)
        # Create global_event_df subdirectory to store cached global event dataframe
        (self.cache_dir / "global_event_df").mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initializing {self.dataset_name} dataset cache directory to {self.cache_dir}")

    @property
    def collected_global_event_df(self) -> dd.DataFrame:
        """Collects and returns the global event data frame.

        Returns:
            dd.DataFrame: The collected global event data frame.
        """
        path = self.cache_dir / "global_event_df" / "cached.parquet"

        if not path_exists(str(path)):
            if self.dev:
                patients = self.global_event_df["patient_id"].unique().head(1000).tolist()
                filter = self.global_event_df["patient_id"].isin(patients)
                self.global_event_df[filter].to_parquet(path)
            else:
                self.global_event_df.to_parquet(path)

        return dd.read_parquet(str(path))

    def load_data(self) -> dd.DataFrame:
        """Loads data from the specified tables.

        Returns:
            dd.DataFrame: A concatenated lazy frame of all tables.
        """
        frames = [self.load_table(table.lower()) for table in self.tables]
        return dd.concat(frames, axis=0, join="outer")

    def load_table(self, table_name: str) -> dd.DataFrame:
        """Loads a table and processes joins if specified.

        Args:
            table_name (str): The name of the table to load.

        Returns:
            dd.DataFrame: The processed lazy frame for the table.

        Raises:
            ValueError: If the table is not found in the config.
            FileNotFoundError: If the CSV file for the table or join is not found.
        """
        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]
        csv_path = f"{self.root}/{table_cfg.file_path}"
        csv_path = clean_path(csv_path)

        logger.info(f"Scanning table: {table_name} from {csv_path}")
        df: dd.DataFrame = self.load_csv_or_tsv(table_name, csv_path)

        # Check if there is a preprocessing function for this table
        # TODO: we need to update the preprocess function to work with Dask DataFrame
        #   for all datasets. Only care about MIMIC4 for now.
        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                f"Preprocessing table: {table_name} with {preprocess_func.__name__}"
            )
            df: dd.DataFrame = preprocess_func(df)

        # Handle joins
        for i, join_cfg in enumerate(table_cfg.join):
            other_csv_path = f"{self.root}/{join_cfg.file_path}"
            other_csv_path = clean_path(other_csv_path)
            logger.info(f"Joining with table: {other_csv_path}")
            join_df: dd.DataFrame = self.load_csv_or_tsv(f"{table_name}_join_{i}", other_csv_path)
            join_key = join_cfg.on
            columns = join_cfg.columns
            how = join_cfg.how

            df: dd.DataFrame = df.merge(join_df[[join_key] + columns], on=join_key, how=how)

        patient_id_col = table_cfg.patient_id
        timestamp_col = table_cfg.timestamp
        timestamp_format = table_cfg.timestamp_format
        attribute_cols = table_cfg.attributes

        # Timestamp expression
        if timestamp_col:
            if isinstance(timestamp_col, list):
                # Concatenate all timestamp parts in order with no separator
                timestamp_series: dd.Series = functools.reduce(operator.add, (df[col].astype(str) for col in timestamp_col))
            else:
                # Single timestamp column
                timestamp_series: dd.Series = df[timestamp_col].astype(str)
            timestamp_series: dd.Series = dd.to_datetime(
                timestamp_series,
                format=timestamp_format,
                errors="raise",
            )
            df: dd.DataFrame = df.assign(timestamp=timestamp_series.astype("datetime64[ms]"))
        else:
            df: dd.DataFrame = df.assign(timestamp=pd.NaT)

        # If patient_id_col is None, use row index as patient_id
        if patient_id_col:
            df: dd.DataFrame = df.assign(patient_id=df[patient_id_col].astype(str))
        else:
            df: dd.DataFrame = df.reset_index(drop=True)
            df: dd.DataFrame = df.assign(patient_id=df.index.astype(str))
        
        df: dd.DataFrame = df.assign(event_type=table_name)

        rename_attr = {attr: f"{table_name}/{attr}" for attr in attribute_cols}
        df: dd.DataFrame = df.rename(columns=rename_attr)

        attr_cols = [rename_attr[attr] for attr in attribute_cols]
        final_cols = ["patient_id", "event_type", "timestamp"] + attr_cols
        event_frame = df[final_cols]

        return event_frame

    def load_csv_or_tsv(self, table_name: str, path: str) -> dd.DataFrame:
        """Loads a CSV.gz, CSV, TSV.gz, or TSV file into a Dask DataFrame.

        Args:
            table_name (str): The name of the table.
            path (str): The URL or local path to the .csv, .csv.gz, .tsv, or .tsv.gz file.
        Returns:
            dd.DataFrame: The loaded Dask DataFrame.
        """
        parquet_path = self.cache_dir / "tables" / f"{table_name}.parquet"

        if not path_exists(str(parquet_path)):
            # convert .gz file to .parquet file since Dask cannot split on gz files directly
            if not path_exists(path):
                if not path_exists(alt_path(path)):
                    raise FileNotFoundError(f"Neither path exists: {path} or {alt_path(path)}")
                path = alt_path(path)
            
            delimiter = '\t' if path.endswith(".tsv") or path.endswith(".tsv.gz") else ','
            # TODO: this may give incorrect type inference for some columns 
            # if the first block is not representative
            csv_reader = pv.open_csv(
                path, 
                read_options=pv.ReadOptions(block_size=1 << 26), # 64 MB
                parse_options=pv.ParseOptions(delimiter=delimiter)
            )
            with pq.ParquetWriter(parquet_path, csv_reader.schema) as writer:
                for batch in csv_reader:
                    writer.write_batch(batch)

            pass
        return dd.read_parquet(
            self.cache_dir / "tables" / f"{table_name}.parquet",
            split_row_groups=True, # type: ignore
            blocksize="64MB",
        )

    @property
    def unique_patient_ids(self) -> List[str]:
        """Returns a list of unique patient IDs.

        Returns:
            List[str]: List of unique patient IDs.
        """
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.collected_global_event_df["patient_id"]
                .unique()
                .compute()
                .tolist()
            )
            logger.info(f"Found {len(self._unique_patient_ids)} unique patient IDs")
        return self._unique_patient_ids

    def get_patient(self, patient_id: str) -> Patient:
        """Retrieves a Patient object for the given patient ID.

        Args:
            patient_id (str): The ID of the patient to retrieve.

        Returns:
            Patient: The Patient object for the given ID.

        Raises:
            AssertionError: If the patient ID is not found in the dataset.
        """
        assert (
            patient_id in self.unique_patient_ids
        ), f"Patient {patient_id} not found in dataset"
        df = self.collected_global_event_df
        if not isinstance(df, dd.DataFrame):
            raise TypeError("collected_global_event_df must be a Dask DataFrame")

        patient_df = df[df["patient_id"] == patient_id]
        return Patient(patient_id=patient_id, data_source=patient_df)

    def iter_patients(self, df: Optional[dd.DataFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset. 
        This method is inefficient, you should prefer to use 
        `self.colllected_global_event_df.groupby(("patient_id", )).apply(...)` directly
        if possible.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            df = self.collected_global_event_df

        for patitent_id in self.unique_patient_ids:
            yield self.get_patient(patitent_id)

    def stats(self) -> None:
        """Prints statistics about the dataset."""
        df = self.collected_global_event_df
        n_patients = df["patient_id"].nunique().compute()
        n_events = df.shape[0].compute()
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"Number of patients: {n_patients}")
        print(f"Number of events: {n_events}")

    @property
    def default_task(self) -> Optional[BaseTask]:
        """Returns the default task for the dataset.

        Returns:
            Optional[BaseTask]: The default task, if any.
        """
        return None

    def set_task(
        self,
        task: Optional[BaseTask] = None,
        num_workers: int = 1,
        cache_dir: Optional[str] = None,
        cache_format: str = "parquet",
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> SampleDataset:
        """Processes the base dataset to generate the task-specific sample dataset.

        Args:
            task (Optional[BaseTask]): The task to set. Uses default task if None.
            num_workers (int): Number of workers for multi-threading. Default is 1.
                This is because the task function is usually CPU-bound. And using
                multi-threading may not speed up the task function.
            cache_dir (Optional[str]): Directory to cache processed samples.
                Default is None (no caching).
            cache_format (str): Format for caching ('parquet' or 'pickle').
                Default is 'parquet'.
            input_processors (Optional[Dict[str, FeatureProcessor]]):
                Pre-fitted input processors. If provided, these will be used
                instead of creating new ones from task's input_schema. Defaults to None.
            output_processors (Optional[Dict[str, FeatureProcessor]]):
                Pre-fitted output processors. If provided, these will be used
                instead of creating new ones from task's output_schema. Defaults to None.

        Returns:
            SampleDataset: The generated sample dataset.

        Raises:
            AssertionError: If no default task is found and task is None.
        """
        if task is None:
            assert self.default_task is not None, "No default tasks found"
            task = self.default_task

        logger.info(
            f"Setting task {task.task_name} for {self.dataset_name} base dataset..."
        )

        if cache_dir is not None:
            logger.warning(f"This argument cache_dir is deprecated. Use dataset cache_dir instead.")
        if cache_format != "parquet":
            logger.warning(f"Only 'parquet' cache_format is officially supported now.")

        # Check for cached data if cache_dir is provided
        samples = None
        if cache_dir is not None:
            cache_filename = f"{task.task_name}.{cache_format}"
            cache_path = Path(cache_dir) / cache_filename
            if cache_path.exists():
                logger.info(f"Loading cached samples from {cache_path}")
                try:
                    if cache_format == "parquet":
                        # Load samples from parquet file
                        cached_df = pl.read_parquet(cache_path)
                        samples = [
                            _restore_from_cache(row) for row in cached_df.to_dicts()
                        ]
                    elif cache_format == "pickle":
                        # Load samples from pickle file
                        with open(cache_path, "rb") as f:
                            samples = pickle.load(f)
                    else:
                        msg = f"Unsupported cache format: {cache_format}"
                        raise ValueError(msg)
                    logger.info(f"Loaded {len(samples)} cached samples")
                except Exception as e:
                    logger.warning(
                        "Failed to load cached data: %s. Regenerating...",
                        e,
                    )
                    samples = None

        # Generate samples if not loaded from cache
        if samples is None:
            logger.info(f"Generating samples with {num_workers} worker(s)...")
            filtered_global_event_df = task.pre_filter(self.collected_global_event_df)
            samples = []

            if num_workers == 1:
                # single-threading (by default)
                for patient in tqdm(
                    self.iter_patients(filtered_global_event_df),
                    total=filtered_global_event_df["patient_id"].n_unique(),
                    desc=(f"Generating samples for {task.task_name} " "with 1 worker"),
                    smoothing=0,
                ):
                    samples.extend(task(patient))
            else:
                # multi-threading (not recommended)
                logger.info(
                    f"Generating samples for {task.task_name} with "
                    f"{num_workers} workers"
                )
                patients = list(self.iter_patients(filtered_global_event_df))
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(task, patient) for patient in patients]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=(
                            f"Collecting samples for {task.task_name} "
                            f"from {num_workers} workers"
                        ),
                    ):
                        samples.extend(future.result())

            # Cache the samples if cache_dir is provided
            if cache_dir is not None:
                cache_path = Path(cache_dir) / cache_filename
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Caching samples to {cache_path}")
                try:
                    if cache_format == "parquet":
                        # Save samples as parquet file
                        samples_for_cache = [
                            _convert_for_cache(sample) for sample in samples
                        ]
                        samples_df = pl.DataFrame(samples_for_cache)
                        samples_df.write_parquet(cache_path)
                    elif cache_format == "pickle":
                        # Save samples as pickle file
                        with open(cache_path, "wb") as f:
                            pickle.dump(samples, f)
                    else:
                        msg = f"Unsupported cache format: {cache_format}"
                        raise ValueError(msg)
                    logger.info(f"Successfully cached {len(samples)} samples")
                except Exception as e:
                    logger.warning(f"Failed to cache samples: {e}")

        sample_dataset = SampleDataset(
            samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task,
            input_processors=input_processors,
            output_processors=output_processors,
        )

        logger.info(f"Generated {len(samples)} samples for task {task.task_name}")
        return sample_dataset
