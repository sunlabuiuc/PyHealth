import logging
import os
from abc import ABC
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Callable
from urllib.parse import urlparse, urlunparse
import uuid
import json
import functools
import operator
from collections import namedtuple
import pickle

from distributed import get_client
import polars as pl
import pandas as pd
import dask.dataframe as dd
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import requests
import platformdirs
from litdata.streaming import StreamingDataset
from dask.distributed import progress
import xxhash

from ..data import Patient
from ..tasks import BaseTask
from ..processors.base_processor import FeatureProcessor
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

def _pickle(datum: dict[str, Any]) -> dict[str, bytes]:
    return {k: pickle.dumps(v) for k,v in datum.items()}

def _unpickle(datum: dict[str, bytes]) -> dict[str, Any]:
    return {k: pickle.loads(v) for k,v in datum.items()}

def _transform_fn(input: tuple[str, str, BaseTask]) -> Iterator[Dict[str, Any]]:
    (patient_id, path, task) = input
    with open(f"{path}/index.json", "rb") as f:
        n_partitions = json.load(f)["n_partitions"]
    bucket = xxhash.xxh64_intdigest(patient_id) % n_partitions
    path = f"{path}/bucket={bucket}"
    patient = Patient(
        patient_id=patient_id,
        data_source=pl.read_parquet(path).filter(pl.col("patient_id") == patient_id),
    )
    for sample in task(patient):
        # Schema is too complex to be handled by LitData, so we pickle the sample here
        yield _pickle(sample)

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

        subfolder = self.cache_subfolder(
            self.root, self.tables, self.dataset_name, self.dev
        )
        self.setup_cache_dir(cache_dir=cache_dir, subfolder=subfolder)

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        # Cached attributes
        self._unique_patient_ids = None

    @staticmethod
    def cache_subfolder(
        root: str, tables: List[str], dataset_name: str, dev: bool
    ) -> str:
        """Generates a unique identifier for the dataset instance. This is used for creating
        cache directories. The UUID is based on the root path, tables, dataset name, and dev mode.

        Returns:
            str: A unique identifier string.
        """
        id_str = json.dumps(
            {
                "root": root,
                "tables": sorted(tables),
                "dataset_name": dataset_name,
                "dev": dev,
            },
            sort_keys=True,
        )
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))

    def setup_cache_dir(
        self, cache_dir: str | Path | None = None, subfolder: str = str(uuid.uuid4())
    ) -> None:
        """Creates the cache directory structure.

        Args:
            cache_dir (str | Path | None): The base cache directory. If None, a default cache
                directory will be created under the platform's cache directory.
            subfolder (str): Subfolder name for this dataset instance's cache.
        """
        if cache_dir is None:
            cache_dir = platformdirs.user_cache_dir(appname="pyhealth")
            logger.info(
                f"No cache_dir provided. Using default cache for PyHealth: {cache_dir}"
            )
        cache_dir = Path(cache_dir)
        self.cache_dir = cache_dir / subfolder
        logger.info(
            f"Initializing {self.dataset_name} dataset cache directory to {self.cache_dir}"
        )

    def _table_cache(self, table_name: str) -> str:
        """Generates the cache path for a specific table.

        Args:
            table_name (str): The name of the table.

        Returns:
            str: The cache path for the table.
        """
        (self.cache_dir / "tables").mkdir(parents=True, exist_ok=True)
        return str(self.cache_dir / "tables" / f"{table_name}.parquet")

    def _dataset_cache(self) -> str:
        """Generates the cache path for the global event dataframe.

        Returns:
            str: The cache path for the global event dataframe.
        """
        return str(self.cache_dir / "global_event_df.parquet")

    def _task_cache(self, task_name: str) -> str:
        """Generates the cache path for a specific task.

        Args:
            task_name (str): The name of the task.
        Returns:
            str: The cache path for the task.
        """
        return str(self.cache_dir / "tasks" / task_name)

    @property
    def collected_global_event_df(self) -> pl.LazyFrame:
        """Collects and returns the global event data frame.

        Returns:
            dd.DataFrame: The collected global event data frame.
        """

        if not path_exists(self._dataset_cache()):
            global_event_df = self.load_data()
            if self.dev:
                patients = global_event_df["patient_id"].unique().head(1000).tolist()
                filter = global_event_df["patient_id"].isin(patients)
                global_event_df: dd.DataFrame = global_event_df[filter]

            mem_usage = global_event_df.memory_usage(deep=False).compute().sum()
            n_partitions = mem_usage // (256 * 1024 * 1024) + 1
            bucket = global_event_df["patient_id"].apply(
                xxhash.xxh64_intdigest, meta=("patient_id", "int")
            ) % n_partitions
            global_event_df = global_event_df.assign(bucket=bucket)

            logger.info(f"Estimated full global event dataframe size {mem_usage / (1024**3):.2f} GB")
            logger.info(f"Repartitioning global event dataframe into {n_partitions} partitions for caching.")

            client = get_client()
            handle = global_event_df.to_parquet(
                self._dataset_cache(),
                partition_on=["bucket"],
                write_index=False,
                compute=False,
            )
            future = client.compute(handle)
            progress(future)
            with open(self._dataset_cache() + "/index.json", "w") as future:
                json.dump({"n_partitions": int(n_partitions)}, future)
        return pl.scan_parquet(self._dataset_cache())

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
            join_df: dd.DataFrame = self.load_csv_or_tsv(
                f"{table_name}_join_{i}", other_csv_path
            )
            join_key = join_cfg.on
            columns = join_cfg.columns
            how = join_cfg.how

            df: dd.DataFrame = df.merge(
                join_df[[join_key] + columns], on=join_key, how=how
            )

        patient_id_col = table_cfg.patient_id
        timestamp_col = table_cfg.timestamp
        timestamp_format = table_cfg.timestamp_format
        attribute_cols = table_cfg.attributes

        # Timestamp expression
        if timestamp_col:
            if isinstance(timestamp_col, list):
                # Concatenate all timestamp parts in order with no separator
                timestamp_series: dd.Series = functools.reduce(
                    operator.add, (df[col].astype(str) for col in timestamp_col)
                )
            else:
                # Single timestamp column
                timestamp_series: dd.Series = df[timestamp_col].astype(str)
            timestamp_series: dd.Series = dd.to_datetime(
                timestamp_series,
                format=timestamp_format,
                errors="raise",
            )
            df: dd.DataFrame = df.assign(
                timestamp=timestamp_series.astype("datetime64[ms]")
            )
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
        if not path_exists(self._table_cache(table_name)):
            # convert .gz file to .parquet file since Dask cannot split on gz files directly
            if not path_exists(path):
                if not path_exists(alt_path(path)):
                    raise FileNotFoundError(
                        f"Neither path exists: {path} or {alt_path(path)}"
                    )
                path = alt_path(path)

            delimiter = (
                "\t" if path.endswith(".tsv") or path.endswith(".tsv.gz") else ","
            )

            # Always infer schema as string to avoid incorrect type inference
            schema_reader = pv.open_csv(
                path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(delimiter=delimiter),
            )
            schema = pa.schema(
                [pa.field(name, pa.string()) for name in schema_reader.schema.names]
            )
            csv_reader = pv.open_csv(
                path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(delimiter=delimiter),
                convert_options=pv.ConvertOptions(column_types=schema),
            )
            with pq.ParquetWriter(
                self._table_cache(table_name), csv_reader.schema
            ) as writer:
                for batch in csv_reader:
                    writer.write_batch(batch)

            pass
        return dd.read_parquet(
            self._table_cache(table_name),
            split_row_groups=True,  # type: ignore
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
                self.collected_global_event_df.select(pl.col("patient_id"))
                .unique()
                .collect()
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
        patient_df: pl.DataFrame = self.collected_global_event_df.filter(
            pl.col("patient_id") == patient_id
        ).collect()
        return Patient(patient_id=patient_id, data_source=patient_df)

    def iter_patients(self, df: Optional[pl.LazyFrame] = None) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

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
        n_patients = len(self.unique_patient_ids)
        n_events = df.select(pl.count()).collect().item()
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

        if cache_format != "parquet":
            logger.warning(
                f"This argument is no longer supported: cache_format={cache_format}"
            )
        if cache_dir is None:
            cache_dir = self._task_cache(task.task_name)
            logger.info(
                "No cache_dir provided. Using default task cache dir: %s", cache_dir
            )

        if not path_exists(str(cache_dir)):
            import litdata as ld

            ld.optimize(
                fn=_transform_fn,
                inputs=[
                    (patient_id, self._dataset_cache(), task)
                    for patient_id in self.unique_patient_ids
                ],
                output_dir=str(cache_dir),
                num_workers=num_workers,
                chunk_bytes="64MB",
            )

        streaming_dataset = StreamingDataset(str(cache_dir), transform=_unpickle)

        sample_dataset = SampleDataset(
            streaming_dataset,
            input_schema=task.input_schema, # type: ignore
            output_schema=task.output_schema, # type: ignore
            dataset_name=self.dataset_name,
            task_name=task.task_name,
            input_processors=input_processors,
            output_processors=output_processors,
        )

        logger.info(f"Generated {len(sample_dataset)} samples for task {task.task_name}")
        return sample_dataset
