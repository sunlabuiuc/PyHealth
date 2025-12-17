import logging
import os
import pickle
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Any, Callable
import functools
import operator
from urllib.parse import urlparse, urlunparse
from urllib.request import urlretrieve
import json
import uuid
import platformdirs
import tempfile
import multiprocessing

import litdata
from litdata.streaming.item_loader import ParquetLoader
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
import requests
from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster, progress
import narwhals as nw

from ..data import Patient
from ..tasks import BaseTask
from ..processors.base_processor import FeatureProcessor
from .configs import load_yaml_config
from .sample_dataset import SampleDataset, SampleBuilder

# Set logging level for distributed to ERROR to reduce verbosity
logging.getLogger("distributed").setLevel(logging.ERROR)
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

def _csv_tsv_gz_path(path: str) -> str:
    """
    Get the path to the file, trying the original path first, then the alternative path
    by switching between .csv.gz, .csv, .tsv.gz, and .tsv extensions.

    Args:
        path (str): Original file path.

    Returns:
        str: The file path that exists.

    Raises:
        FileNotFoundError: If neither the original nor the alternative path exists.
        ValueError: If the path does not have an expected extension.
    """
    if path_exists(path):
        return path

    if path.endswith(".csv.gz"):
        alt_path = path[:-3]  # Remove .gz -> try .csv
    elif path.endswith(".csv"):
        alt_path = f"{path}.gz"  # Add .gz -> try .csv.gz
    elif path.endswith(".tsv.gz"):
        alt_path = path[:-3]  # Remove .gz -> try .tsv
    elif path.endswith(".tsv"):
        alt_path = f"{path}.gz"  # Add .gz -> try .tsv.gz
    else:
        raise ValueError(f"Path does not have expected extension: {path}")
    
    if path_exists(alt_path):
        return alt_path
    
    raise FileNotFoundError(f"Neither path exists: {path} or {alt_path}")

def _uncollate(x: list[Any]) -> Any:
    return x[0] if isinstance(x, list) and len(x) == 1 else x


class _ParquetWriter:
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
        self._global_event_df = None
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
        else:
            # Ensure the explicitly provided cache_dir exists
            cache_dir = Path(self._cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._cache_dir = cache_dir
        return Path(self._cache_dir)
    
    @property
    def temp_dir(self) -> Path:
        return self.cache_dir / "temp"

    def _scan_csv_tsv_gz(self, table_name: str, source_path: str | None = None) -> dd.DataFrame:
        """Scans a CSV/TSV file (possibly gzipped) and returns a Dask DataFrame.

        If the cached Parquet file does not exist, it converts the source CSV/TSV file
        to Parquet and saves it to the cache.

        Args:
            table_name (str): The name of the table.
            source_path (str | None): The source CSV/TSV file path. If None, assumes the
                Parquet file already exists in the cache.

        Returns:
            dd.DataFrame: The Dask DataFrame loaded from the cached Parquet file.

        Raises:
            FileNotFoundError: If source_path is None and the cached Parquet file does not exist; 
                or if neither the original nor the alternative path of source_path exists.
            ValueError: If the path does not have an expected extension.
        """
        # Ensure the tables cache directory exists
        (self.temp_dir / "tables").mkdir(parents=True, exist_ok=True)
        ret_path = str(self.temp_dir / "tables" / f"{table_name}.parquet")

        if not path_exists(ret_path):
            if source_path is None:
                raise FileNotFoundError(
                    f"Table {table_name} not found in cache and no source_path provided."
                )

            source_path = _csv_tsv_gz_path(source_path)

            if is_url(source_path):
                local_filename = os.path.basename(source_path)
                download_dir = self.temp_dir / "downloads"
                download_dir.mkdir(parents=True, exist_ok=True)
                local_path = download_dir / local_filename
                if not local_path.exists():
                    logger.info(f"Downloading {source_path} to {local_path}")
                    urlretrieve(source_path, local_path)
                source_path = str(local_path)

            # Determine delimiter based on file extension
            delimiter = (
                "\t"
                if source_path.endswith(".tsv") or source_path.endswith(".tsv.gz")
                else ","
            )

            # Always infer schema as string to avoid incorrect type inference
            schema_reader = pv.open_csv(
                source_path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(delimiter=delimiter),
            )
            schema = pa.schema(
                [pa.field(name, pa.string()) for name in schema_reader.schema.names]
            )

            # Convert CSV/TSV to Parquet
            csv_reader = pv.open_csv(
                source_path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(delimiter=delimiter),
                convert_options=pv.ConvertOptions(column_types=schema),
            )
            with pq.ParquetWriter(ret_path, csv_reader.schema) as writer:
                for batch in csv_reader:
                    writer.write_batch(batch)

        df: dd.DataFrame = dd.read_parquet(
            ret_path,
            split_row_groups=True,  # type: ignore
            blocksize="64MB",
        )
        return df.replace("", pd.NA)  # Replace empty strings with NaN

    @property
    def global_event_df(self) -> pl.LazyFrame:
        """Returns the path to the cached event dataframe.

        Returns:
            Path: The path to the cached event dataframe.
        """
        if not multiprocessing.current_process().name == "MainProcess":
            logger.warning(
                "global_event_df property accessed from a non-main process. This may lead to unexpected behavior.\n" + 
                "Consider use __name__ == '__main__' guard when using multiprocessing."
            )
            return None  # type: ignore

        if self._global_event_df is None:
            ret_path = self.cache_dir / "global_event_df.parquet"
            if not ret_path.exists():
                # TODO: auto select processes=True/False based on if it's in jupyter notebook
                #   The processes=True will crash in jupyter notebook.
                # TODO: make the n_workers configurable
                with LocalCluster(
                    n_workers=1,
                    threads_per_worker=1,
                    processes=False,
                ) as cluster:
                    with Client(cluster) as client:
                        df: dd.DataFrame = self.load_data()
                        if self.dev:
                            logger.info("Dev mode enabled: limiting to 1000 patients")
                            patients = (
                                df["patient_id"].unique().head(1000).tolist()
                            )
                            filter = df["patient_id"].isin(patients)
                            df = df[filter]

                        logger.info(f"Caching event dataframe to {ret_path}...")
                        collection = df.sort_values("patient_id").to_parquet(
                            ret_path,
                            write_index=False,
                            compute=False,
                        )
                        handle = client.compute(collection)
                        progress(handle)
                        handle.result() # type: ignore
            self._global_event_df = ret_path

        return pl.scan_parquet(
            self._global_event_df,
            low_memory=True,
        )

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
            dd.DataFrame: The processed Dask dataframe for the table.

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
        df = self._scan_csv_tsv_gz(table_name, csv_path)

        # Convert column names to lowercase before calling preprocess_func
        df = df.rename(columns=str.lower)

        # Check if there is a preprocessing function for this table
        preprocess_func: Optional[Callable[[nw.LazyFrame], nw.LazyFrame]]
        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                f"Preprocessing table: {table_name} with {preprocess_func.__name__}"
            )
            df = preprocess_func(nw.from_native(df)).to_native() # type: ignore

        # Handle joins
        for i, join_cfg in enumerate(table_cfg.join):
            other_csv_path = f"{self.root}/{join_cfg.file_path}"
            other_csv_path = clean_path(other_csv_path)
            logger.info(f"Joining with table: {other_csv_path}")
            join_df = self._scan_csv_tsv_gz(f"{table_name}_join_{i}", other_csv_path)
            join_df = join_df.rename(columns=str.lower)
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

        rename_attr = {attr.lower(): f"{table_name}/{attr}" for attr in attribute_cols}
        df: dd.DataFrame = df.rename(columns=rename_attr)

        attr_cols = [rename_attr[attr.lower()] for attr in attribute_cols]
        final_cols = ["patient_id", "event_type", "timestamp"] + attr_cols
        event_frame = df[final_cols]

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
        if not multiprocessing.current_process().name == "MainProcess":
            logger.warning(
                "set_task method accessed from a non-main process. This may lead to unexpected behavior.\n" + 
                "Consider use __name__ == '__main__' guard when using multiprocessing."
            )
            return None # type: ignore

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
        else:
            # Ensure the explicitly provided cache_dir exists
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        path = Path(cache_dir)

        # Check if index.json exists to verify cache integrity, this
        # is the standard file for litdata.StreamingDataset
        if not (path / "index.json").exists():
            global_event_df = task.pre_filter(self.global_event_df)
            schema = pa.schema([("sample", pa.binary())])
            with tempfile.TemporaryDirectory() as tmp_dir:
                # Create Parquet file with samples
                logger.info(f"Applying task transformations on data...")
                with _ParquetWriter(f"{tmp_dir}/samples.parquet", schema) as writer:
                    # TODO: this can be further optimized.
                    patient_ids = (
                        global_event_df.select("patient_id")
                        .unique()
                        .collect(engine="streaming")
                        .to_series()
                    )
                    for patient_id in tqdm(patient_ids):
                        patient_df = global_event_df.filter(
                            pl.col("patient_id") == patient_id
                        ).collect(engine="streaming")
                        patient = Patient(patient_id=patient_id, data_source=patient_df)
                        for sample in task(patient):
                            writer.append({"sample": pickle.dumps(sample)})
                litdata.index_parquet_dataset(tmp_dir)

                # Build processors and fit on the dataset
                logger.info(f"Fitting processors on the dataset...")
                dataset = litdata.StreamingDataset(
                    tmp_dir,
                    item_loader=ParquetLoader(),
                    transform=lambda x: pickle.loads(x["sample"]),
                )
                builder = SampleBuilder(
                    input_schema=task.input_schema,  # type: ignore
                    output_schema=task.output_schema,  # type: ignore
                    input_processors=input_processors,
                    output_processors=output_processors,
                )
                builder.fit(dataset)
                builder.save(str(path / "schema.pkl"))

                # Apply processors and save final samples to cache_dir
                logger.info(f"Processing samples and saving to {path}...")
                dataset = litdata.StreamingDataset(
                    tmp_dir,
                    item_loader=ParquetLoader(),
                )
                litdata.optimize(
                    fn=builder.transform,
                    inputs=litdata.StreamingDataLoader(
                        dataset,
                        batch_size=1,
                        collate_fn=_uncollate,
                    ),
                    output_dir=str(path),
                    chunk_bytes="64MB",
                    num_workers=num_workers,
                )
                logger.info(f"Cached processed samples to {path}")

        return SampleDataset(
            path=str(path),
            dataset_name=self.dataset_name,
            task_name=task.task_name,
        )
