import logging
import os
import pickle
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any
from urllib.parse import urlparse, urlunparse
import json
import uuid
import platformdirs
import tempfile

import litdata
from litdata.streaming.item_loader import ParquetLoader
import pyarrow as pa
import pyarrow.parquet as pq
import polars as pl
import requests
from tqdm import tqdm

from ..data import Patient
from ..tasks import BaseTask
from ..processors.base_processor import FeatureProcessor
from .configs import load_yaml_config
from .sample_dataset import SampleDataset, SampleBuilder
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
        return pl.scan_csv(
            file_path,
            separator=separator,
            infer_schema=False,
            low_memory=True,
        )

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


def unpickle_sample(sample_bytes: dict[str, bytes]) -> dict[str, Any]:
    return pickle.loads(sample_bytes["sample"])


class StreamingParquetWriter:
    """
    Stream-write rows into a Parquet file in chunked (row-group) fashion.

    Usage:
        writer = StreamingParquetWriter(Path("out.parquet"), schema, chunk_size=10000)
        writer.append({"id": 1, "val": 3.14})
        writer.append({"id": 2, "val": 1.23})
        writer.close()
    """

    def __init__(self, path: Path | str, schema: pa.Schema, chunk_size: int = 8_192):
        """
        Args:
            path: output Parquet file path
            schema: pyarrow.Schema (required)
            chunk_size: flush buffer every N rows
        """
        self.path = Path(path)
        self.schema = schema
        self.chunk_size = chunk_size

        if self.schema is None:
            raise ValueError(
                "schema must be provided — no automatic inference allowed."
            )

        self._writer: pq.ParquetWriter | None = None
        self._buffer: list[dict] = []
        self._closed = False

    # --------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------
    def append(self, row: dict) -> None:
        """Append a single row (a Python dict)."""
        if self._closed:
            raise RuntimeError("Cannot append to a closed StreamingParquetWriter")

        self._buffer.append(row)
        if len(self._buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        """Flush buffered rows into a Parquet row-group."""
        if not self._buffer:
            return

        # Convert list[dict] → Arrow RecordBatch
        batch = pa.RecordBatch.from_pylist(self._buffer, schema=self.schema)

        # Lazy-initialize writer
        if self._writer is None:
            self._writer = pq.ParquetWriter(self.path, self.schema)

        self._writer.write_batch(batch)
        self._buffer.clear()

    def close(self) -> None:
        """Flush and close the Parquet writer."""
        if self._closed:
            return
        self.flush()
        if self._writer is not None:
            self._writer.close()
        self._closed = True

    # --------------------------------------------------------------
    # Context manager support
    # --------------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


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
        cache_dir: str | Path | None = None,
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
        self.dev = dev
        self.config = load_yaml_config(config_path) if config_path else None

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        # Cached attributes
        self._cache_dir = cache_dir
        self._event_df_path = None
        self._unique_patient_ids = None

    @property
    def cache_dir(self) -> Path:
        """Returns the cache directory path.
        Returns:
            Path: The cache directory path.
        """
        if self._cache_dir is None:
            id_str = json.dumps(
                {
                    "root": self.root,
                    "tables": sorted(self.tables),
                    "dataset_name": self.dataset_name,
                    "dev": self.dev,
                },
                sort_keys=True,
            )
            cache_dir = Path(platformdirs.user_cache_dir(appname="pyhealth")) / str(
                uuid.uuid5(uuid.NAMESPACE_DNS, id_str)
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            print(f"No cache_dir provided. Using default cache dir: {cache_dir}")
            self._cache_dir = cache_dir
        return Path(self._cache_dir)

    @property
    def global_event_df(self) -> pl.LazyFrame:
        """Returns the path to the cached event dataframe.

        Returns:
            Path: The path to the cached event dataframe.
        """
        if self._event_df_path is None:
            path = self.cache_dir / "event_df.parquet"
            if not path.exists():
                df = self.load_data()
                if self.dev:
                    logger.info("Dev mode enabled: limiting to 1000 patients")
                    limited_patients = (
                        df.select(pl.col("patient_id").shuffle(seed=0))
                        .unique()
                        .limit(1000)
                    )
                    df = df.join(limited_patients, on="patient_id", how="inner")

                logger.info(f"Caching event dataframe to {path}...")
                df.sort("patient_id").sink_parquet(
                    path,
                    compression="lz4",  # use lz4 compression for faster read/write
                    row_group_size=8_192,
                    maintain_order=True,  # Important for sorted writes
                )
            self._event_df_path = path

        return pl.scan_parquet(
            self._event_df_path,
            low_memory=True,
        ).set_sorted(
            "patient_id"
        )  # Guarantee sorted read, see sink_parquet above

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
        assert self.config is not None, "Config must be provided to load tables"

        if table_name not in self.config.tables:
            raise ValueError(f"Table {table_name} not found in config")

        table_cfg = self.config.tables[table_name]
        csv_path = f"{self.root}/{table_cfg.file_path}"
        csv_path = clean_path(csv_path)

        logger.info(f"Scanning table: {table_name} from {csv_path}")
        df = scan_csv_gz_or_csv_tsv(csv_path)

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
            join_key = join_cfg.on
            columns = join_cfg.columns
            how = join_cfg.how

            df = df.join(join_df.select([join_key] + columns), on=join_key, how=how)  # type: ignore

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
                self.global_event_df.select("patient_id")
                .unique()
                .collect(engine="streaming")
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

        data_source = self.global_event_df.filter(
            pl.col("patient_id") == patient_id
        ).collect(engine="streaming")
        return Patient(patient_id=patient_id, data_source=data_source)

    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            df = self.global_event_df
        patient_ids = (
            df.select("patient_id")
            .unique(maintain_order=True)
            .collect(engine="streaming")
            .to_series()
        )

        for patient_id in patient_ids:
            patient_df = df.filter(pl.col("patient_id") == patient_id).collect(
                engine="streaming"
            )
            yield Patient(patient_id=patient_id, data_source=patient_df)

    def stats(self) -> None:
        """Prints statistics about the dataset."""
        stats = self.global_event_df.select(
            pl.len().alias("n_events"),
            pl.col("patient_id").n_unique().alias("n_patients"),
        ).collect(engine="streaming")
        print(f"Dataset: {self.dataset_name}")
        print(f"Dev mode: {self.dev}")
        print(f"Number of patients: {stats['n_patients'][0]}")
        print(f"Number of events: {stats['n_events'][0]}")

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
        cache_dir: str | Path | None = None,
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
            cache_format (str): Deprecated. Only "parquet" is supported now.
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

        if cache_format != "parquet":
            logger.warning("Only 'parquet' cache_format is supported now. ")

        logger.info(
            f"Setting task {task.task_name} for {self.dataset_name} base dataset..."
        )

        if cache_dir is None:
            cache_dir = self.cache_dir / "tasks" / task.task_name
            cache_dir.mkdir(parents=True, exist_ok=True)

        path = Path(cache_dir)

        # Check if index.json exists to verify cache integrity, this
        # is the standard file for litdata.StreamingDataset
        if not (path / "index.json").exists():
            event_df = task.pre_filter(self.global_event_df)
            schema = pa.schema([("sample", pa.binary())])
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_dir = (
                    "./test_task_cache"  # For debugging purposes, keep the temp dir
                )
                with StreamingParquetWriter(
                    f"{tmp_dir}/samples.parquet", schema
                ) as writer:
                    logger.info(f"Applying task transformations on data...")

                    patient_ids = (
                        event_df.select("patient_id")
                        .unique()
                        .collect(engine="streaming")
                        .to_series()
                    )
                    for patient_id in tqdm(patient_ids):
                        patient_df = event_df.filter(
                            pl.col("patient_id") == patient_id
                        ).collect(engine="streaming")
                        patient = Patient(patient_id=patient_id, data_source=patient_df)
                        for sample in task(patient):
                            writer.append({"sample": pickle.dumps(sample)})
                litdata.index_parquet_dataset(tmp_dir)
                dataset = litdata.StreamingDataset(
                    tmp_dir,
                    item_loader=ParquetLoader(),
                )
                builder = SampleBuilder(
                    input_schema=task.input_schema,  # type: ignore
                    output_schema=task.output_schema,  # type: ignore
                    input_processors=input_processors,
                    output_processors=output_processors,
                )
                builder.fit(map(unpickle_sample, iter(dataset)))
                return dataset, builder
                # litdata.optimize(
                #     fn=lambda x: builder.transform(x),
                #     inputs=Streadataset,

                # )

        # sample_dataset = SampleDataset(
        #     samples,
        #     input_schema=task.input_schema,
        #     output_schema=task.output_schema,
        #     dataset_name=self.dataset_name,
        #     task_name=task,
        #     input_processors=input_processors,
        #     output_processors=output_processors,
        # )

        # logger.info(f"Generated {len(samples)} samples for task {task.task_name}")
        # return sample_dataset
