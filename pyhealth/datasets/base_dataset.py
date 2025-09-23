import logging
import os
import pickle
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator, List, Optional
from urllib.parse import urlparse, urlunparse

import polars as pl
import requests
from tqdm import tqdm

from ..data import Patient
from ..tasks import BaseTask
from .configs import load_yaml_config
from .sample_dataset import SampleDataset

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
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
    ):
        """Initializes the BaseDataset.

        Args:
            root (str): The root directory where dataset files are stored.
            tables (List[str]): List of table names to load.
            dataset_name (Optional[str]): Name of the dataset. Defaults to class name.
            config_path (Optional[str]): Path to the configuration YAML file.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
        """
        if len(set(tables)) != len(tables):
            logger.warning("Duplicate table names in tables list. Removing duplicates.")
            tables = list(set(tables))
        self.root = root
        self.tables = tables
        self.dataset_name = dataset_name or self.__class__.__name__
        self.config = load_yaml_config(config_path)
        self.dev = dev

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        self.global_event_df = self.load_data()

        # Cached attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

    @property
    def collected_global_event_df(self) -> pl.DataFrame:
        """Collects and returns the global event data frame.

        Returns:
            pl.DataFrame: The collected global event data frame.
        """
        if self._collected_global_event_df is None:
            logger.info("Collecting global event dataframe...")

            # Collect the dataframe - with dev mode limiting if applicable
            df = self.global_event_df
            # TODO: dev doesn't seem to improve the speed / memory usage
            if self.dev:
                # Limit the number of patients in dev mode
                logger.info("Dev mode enabled: limiting to 1000 patients")
                limited_patients = df.select(pl.col("patient_id")).unique().limit(1000)
                df = df.join(limited_patients, on="patient_id", how="inner")

            self._collected_global_event_df = df.collect()

            # Profile the Polars collect() operation (commented out by default)
            # self._collected_global_event_df, profile = df.profile()
            # profile = profile.with_columns([
            #     (pl.col("end") - pl.col("start")).alias("duration"),
            # ])
            # profile = profile.with_columns([
            #     (pl.col("duration") / profile["duration"].sum() * 100).alias("percentage")
            # ])
            # profile = profile.sort("duration", descending=True)
            # with pl.Config() as cfg:
            #     cfg.set_tbl_rows(-1)
            #     cfg.set_fmt_str_lengths(200)
            #     print(profile)

            logger.info(
                f"Collected dataframe with shape: {self._collected_global_event_df.shape}"
            )

        return self._collected_global_event_df

    def load_data(self) -> pl.LazyFrame:
        """Loads data from the specified tables.

        Returns:
            pl.LazyFrame: A concatenated lazy frame of all tables.
        """
        frames = [self.load_table(table.lower()) for table in self.tables]
        return pl.concat(frames, how="diagonal")

    def load_table(self, table_name: str) -> pl.LazyFrame:
        """Loads a table and processes joins if specified.

        Args:
            table_name (str): The name of the table to load.

        Returns:
            pl.LazyFrame: The processed lazy frame for the table.

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
        df = scan_csv_gz_or_csv_tsv(csv_path)

        # Convert column names to lowercase before calling preprocess_func
        col_names = df.collect_schema().names()
        if any(col != col.lower() for col in col_names):
            logger.warning("Some column names were converted to lowercase")
        df = df.with_columns([pl.col(col).alias(col.lower()) for col in col_names])

        # Check if there is a preprocessing function for this table
        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                f"Preprocessing table: {table_name} with {preprocess_func.__name__}"
            )
            df = preprocess_func(df)

        # Handle joins
        for join_cfg in table_cfg.join:
            other_csv_path = f"{self.root}/{join_cfg.file_path}"
            other_csv_path = clean_path(other_csv_path)
            logger.info(f"Joining with table: {other_csv_path}")
            join_df = scan_csv_gz_or_csv_tsv(other_csv_path)
            join_df = join_df.with_columns(
                [
                    pl.col(col).alias(col.lower())
                    for col in join_df.collect_schema().names()
                ]
            )
            join_key = join_cfg.on
            columns = join_cfg.columns
            how = join_cfg.how

            df = df.join(join_df.select([join_key] + columns), on=join_key, how=how)

        patient_id_col = table_cfg.patient_id
        timestamp_col = table_cfg.timestamp
        timestamp_format = table_cfg.timestamp_format
        attribute_cols = table_cfg.attributes

        # Timestamp expression
        if timestamp_col:
            if isinstance(timestamp_col, list):
                # Concatenate all timestamp parts in order with no separator
                combined_timestamp = pl.concat_str(
                    [pl.col(col) for col in timestamp_col]
                ).str.strptime(pl.Datetime, format=timestamp_format, strict=True)
                timestamp_expr = combined_timestamp
            else:
                # Single timestamp column
                timestamp_expr = pl.col(timestamp_col).str.strptime(
                    pl.Datetime, format=timestamp_format, strict=True
                )
        else:
            timestamp_expr = pl.lit(None, dtype=pl.Datetime)

        # If patient_id_col is None, use row index as patient_id
        patient_id_expr = (
            pl.col(patient_id_col).cast(pl.Utf8)
            if patient_id_col
            else pl.int_range(0, pl.count()).cast(pl.Utf8)
        )
        base_columns = [
            patient_id_expr.alias("patient_id"),
            pl.lit(table_name).cast(pl.Utf8).alias("event_type"),
            # ms should be sufficient for most cases
            timestamp_expr.cast(pl.Datetime(time_unit="ms")).alias("timestamp"),
        ]

        # Flatten attribute columns with event_type prefix
        attribute_columns = [
            pl.col(attr).alias(f"{table_name}/{attr}") for attr in attribute_cols
        ]

        event_frame = df.select(base_columns + attribute_columns)

        return event_frame

    @property
    def unique_patient_ids(self) -> List[str]:
        """Returns a list of unique patient IDs.

        Returns:
            List[str]: List of unique patient IDs.
        """
        if self._unique_patient_ids is None:
            self._unique_patient_ids = (
                self.collected_global_event_df.select("patient_id")
                .unique()
                .to_series()
                .to_list()
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
        df = self.collected_global_event_df.filter(pl.col("patient_id") == patient_id)
        return Patient(patient_id=patient_id, data_source=df)

    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            df = self.collected_global_event_df
        grouped = df.group_by("patient_id")

        for patient_id, patient_df in grouped:
            patient_id = patient_id[0]
            yield Patient(patient_id=patient_id, data_source=patient_df)

    def stats(self) -> None:
        """Prints statistics about the dataset."""
        df = self.collected_global_event_df
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"Number of patients: {df['patient_id'].n_unique()}")
        print(f"Number of events: {df.height}")

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

        filtered_global_event_df = task.pre_filter(self.collected_global_event_df)

        # Check for cached data if cache_dir is provided
        samples = None
        if cache_dir is not None:
            cache_path = Path(cache_dir) / f"{task.task_name}.{cache_format}"
            if cache_path.exists():
                logger.info(f"Loading cached samples from {cache_path}")
                try:
                    if cache_format == "parquet":
                        # Load samples from parquet file
                        cached_df = pl.read_parquet(cache_path)
                        samples = cached_df.to_dicts()
                    elif cache_format == "pickle":
                        # Load samples from pickle file
                        with open(cache_path, "rb") as f:
                            samples = pickle.load(f)
                    else:
                        raise ValueError(f"Unsupported cache format: {cache_format}")
                    logger.info(f"Loaded {len(samples)} cached samples")
                except Exception as e:
                    logger.warning(f"Failed to load cached data: {e}. Regenerating...")
                    samples = None

        # Generate samples if not loaded from cache
        if samples is None:
            logger.info(f"Generating samples with {num_workers} worker(s)...")

            samples = []

            if num_workers == 1:
                # single-threading (by default)
                for patient in tqdm(
                    self.iter_patients(filtered_global_event_df),
                    total=filtered_global_event_df["patient_id"].n_unique(),
                    desc=f"Generating samples for {task.task_name} with 1 worker",
                    smoothing=0,
                ):
                    samples.extend(task(patient))
            else:
                # multi-threading (not recommended)
                logger.info(
                    f"Generating samples for {task.task_name} with {num_workers} workers"
                )
                patients = list(self.iter_patients(filtered_global_event_df))
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = [executor.submit(task, patient) for patient in patients]
                    for future in tqdm(
                        as_completed(futures),
                        total=len(futures),
                        desc=f"Collecting samples for {task.task_name} from {num_workers} workers",
                    ):
                        samples.extend(future.result())

            # Cache the samples if cache_dir is provided
            if cache_dir is not None:
                cache_path = Path(cache_dir) / f"{task.task_name}.{cache_format}"
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Caching samples to {cache_path}")
                try:
                    if cache_format == "parquet":
                        # Save samples as parquet file
                        samples_df = pl.DataFrame(samples)
                        samples_df.write_parquet(cache_path)
                    elif cache_format == "pickle":
                        # Save samples as pickle file
                        with open(cache_path, "wb") as f:
                            pickle.dump(samples, f)
                    else:
                        raise ValueError(f"Unsupported cache format: {cache_format}")
                    logger.info(f"Successfully cached {len(samples)} samples")
                except Exception as e:
                    logger.warning(f"Failed to cache samples: {e}")

        sample_dataset = SampleDataset(
            samples,
            input_schema=task.input_schema,
            output_schema=task.output_schema,
            dataset_name=self.dataset_name,
            task_name=task,
        )

        logger.info(f"Generated {len(samples)} samples for task {task.task_name}")
        return sample_dataset
