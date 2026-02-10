import logging
import os
import pickle
from abc import ABC
from pathlib import Path
from typing import Dict, Iterator, Iterable, List, Optional, Any, Callable
import functools
import operator
from urllib.parse import urlparse, urlunparse
from urllib.request import urlretrieve
import json
import uuid
import platformdirs
import multiprocessing
import multiprocessing.queues
import shutil

import litdata
from litdata.streaming.item_loader import ParquetLoader
from litdata.processing.data_processor import in_notebook
from litdata.streaming.writer import BinaryWriter
import pyarrow as pa
import pyarrow.csv as pv
import pyarrow.parquet as pq
import pandas as pd
import polars as pl
import requests
from tqdm import tqdm
import dask.dataframe as dd
from dask.distributed import Client as DaskClient, LocalCluster as DaskCluster, progress as dask_progress
import narwhals as nw
import itertools
import numpy as np
import more_itertools

from ..data import Patient
from ..tasks import BaseTask
from ..processors.base_processor import FeatureProcessor
from .configs import load_yaml_config
from .sample_dataset import SampleDataset, SampleBuilder
from ..utils import set_env

# Set logging level for distributed to ERROR to reduce verbosity
logging.getLogger("distributed").setLevel(logging.ERROR)
logger = logging.getLogger(__name__)
# Remove LitData version check to avoid unnecessary warnings
os.environ["LITDATA_DISABLE_VERSION_CHECK"] = "1"

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


def _litdata_merge(cache_dir: Path) -> None:
    """
    Merges LitData binary writer index files in the given cache directory.

    Args:
        cache_dir (Path): The cache directory containing LitData binary writer files.
    """
    from litdata.streaming.writer import _INDEX_FILENAME
    files = os.listdir(cache_dir)
    
    # Return if the index already exists
    if _INDEX_FILENAME in files:
        return

    index_files = [f for f in files if f.endswith(_INDEX_FILENAME)]
    
    # Return if there are no index files to merge
    if len(index_files) == 0:
        raise ValueError("There are zero samples in the dataset, please check the task and processors.")
    
    BinaryWriter(cache_dir=str(cache_dir), chunk_bytes="64MB").merge(num_workers=len(index_files))


class _ProgressContext:
    def __init__(self, queue: multiprocessing.queues.Queue | None, total: int, **kwargs):
        """
        :param queue: An existing queue (e.g., from multiprocessing). If provided,
                      this class acts as a passthrough.
        :param total: Total items for the progress bar (only used if queue is None).
        :param kwargs: Extra arguments for tqdm (e.g., desc="Processing").
        """
        self.queue = queue
        self.total = total
        self.kwargs = kwargs
        self.progress = None

    def put(self, n):
        if self.progress:
            self.progress.update(n)

    def __enter__(self):
        if self.queue:
            return self.queue

        self.progress = tqdm(total=self.total, **self.kwargs)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.close()

_task_transform_progress: multiprocessing.queues.Queue | None = None

def _task_transform_init(queue: multiprocessing.queues.Queue) -> None:
    """
    Initializer for worker processes to set up a global queue.

    Args:
        queue (multiprocessing.queues.Queue): The queue for progress tracking.
    """
    global _task_transform_progress
    _task_transform_progress = queue

def _task_transform_fn(args: tuple[int, BaseTask, Iterable[str], pl.LazyFrame, Path]) -> None:
    """
    Worker function to apply task transformation on a chunk of patients.

    Args:
        args (tuple): A tuple containing:
            worker_id (int): The ID of the worker.
            task (BaseTask): The task to apply.
            patient_ids (Iterable[str]): The patient IDs to process.
            global_event_df (pl.LazyFrame): The global event dataframe.
            output_dir (Path): The output directory to save results.
    """
    BATCH_SIZE = 128 # Use a batch size 128 can reduce runtime by 30%.
    worker_id, task, patient_ids, global_event_df, output_dir = args
    total_patients = len(list(patient_ids))
    logger.info(f"Worker {worker_id} started processing {total_patients} patients. (Polars threads: {pl.thread_pool_size()})")

    with (
        set_env(DATA_OPTIMIZER_GLOBAL_RANK=str(worker_id)),
        _ProgressContext(_task_transform_progress, total=total_patients) as progress
    ):
        writer = BinaryWriter(cache_dir=str(output_dir), chunk_bytes="64MB")

        write_index = 0
        batches = itertools.batched(patient_ids, BATCH_SIZE)
        for batch in batches:
            complete = 0
            patients = (
                global_event_df.filter(pl.col("patient_id").is_in(batch))
                    .collect(engine="streaming")
                    .partition_by("patient_id", as_dict=True)
            )
            for patient_id, patient_df in patients.items():
                patient_id = patient_id[0]  # Extract string from single-element list
                patient = Patient(patient_id=patient_id, data_source=patient_df)
                for sample in task(patient):
                    writer.add_item(write_index, {"sample": pickle.dumps(sample)})
                    write_index += 1
                complete += 1
            progress.put(complete)
        writer.done()

    logger.info(f"Worker {worker_id} finished processing patients.")

_proc_transform_progress: multiprocessing.queues.Queue | None = None

def _proc_transform_init(queue: multiprocessing.queues.Queue) -> None:
    """
    Initializer for worker processes to set up a global queue.

    Args:
        queue (multiprocessing.queues.Queue): The queue for progress tracking.
    """
    global _proc_transform_progress
    _proc_transform_progress = queue

def _proc_transform_fn(args: tuple[int, Path, int, int, Path]) -> None:
    """
    Worker function to apply processors on a chunk of samples.

    Args:
        args (tuple): A tuple containing:
            worker_id (int): The ID of the worker.
            task_df (Path): The path to the task dataframe.
            start_idx (int): The start index of samples to process.
            end_idx (int): The end index of samples to process.
            output_dir (Path): The output directory to save results.
    """
    BATCH_SIZE = 128
    worker_id, task_df, start_idx, end_idx, output_dir = args
    total_samples = end_idx - start_idx
    logger.info(f"Worker {worker_id} started processing {total_samples} samples. ({start_idx} to {end_idx})")

    with (
        set_env(DATA_OPTIMIZER_GLOBAL_RANK=str(worker_id)),
        _ProgressContext(_proc_transform_progress, total=total_samples) as progress
    ):
        writer = BinaryWriter(cache_dir=str(output_dir), chunk_bytes="64MB")

        dataset = litdata.StreamingDataset(str(task_df))
        builder = SampleBuilder.load(f"{output_dir}/schema.pkl")

        complete = 0
        write_index = 0
        for i in range(start_idx, end_idx):
            transformed: Dict[str, Any] = builder.transform(dataset[i])
            writer.add_item(write_index, transformed)
            write_index += 1
            complete += 1

            if complete >= BATCH_SIZE:
                progress.put(complete)
                complete = 0

        if complete > 0:
            progress.put(complete)
        writer.done()

    logger.info(f"Worker {worker_id} finished processing samples.")


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
        num_workers: int = 1,
        dev: bool = False,
    ):
        """Initializes the BaseDataset.

        Args:
            root (str): The root directory where dataset files are stored.
            tables (List[str]): List of table names to load.
            dataset_name (Optional[str]): Name of the dataset. Defaults to class name.
            config_path (Optional[str]): Path to the configuration YAML file.
            cache_dir (Optional[str | Path]): Directory for caching processed data.
                Behavior depends on the type passed:
                
                - **None** (default): Auto-generates a cache path under the default 
                  pyhealth cache directory. Cache files include a UUID in their 
                  filenames (e.g., ``global_event_df_{uuid}.parquet``) derived from 
                  the dataset configuration, so different table sets don't collide.
                - **str**: Used as the cache directory path. Cache files include a 
                  UUID in their filenames to prevent collisions between different 
                  table configurations sharing the same directory.
                - **Path**: Used as-is with NO modification. Cache files still include
                  UUID in their filenames for isolation.
            num_workers (int): Number of worker processes for parallel operations.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
        """
        if len(set(tables)) != len(tables):
            logger.warning("Duplicate table names in tables list. Removing duplicates.")
            tables = list(set(tables))
        self.root = root
        self.tables = tables
        self.dataset_name = dataset_name or self.__class__.__name__
        self.num_workers = num_workers
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

        The cache directory is determined by the type of ``cache_dir`` passed
        to ``__init__``:

        - **None**: Auto-generated under default pyhealth cache directory.
        - **str**: Used as-is as the cache directory path.
        - **Path**: Used exactly as-is (no modification).

        Cache files within the directory include UUID suffixes in their
        filenames (e.g., ``global_event_df_{uuid}.parquet``) to prevent
        collisions between different table configurations.

        The cache structure within the directory is::

            tmp/                                   # Temporary files during processing
            global_event_df_{uuid}.parquet/         # Cached global event dataframe
            tasks/                                  # Cached task-specific data

        Returns:
            Path: The resolved cache directory path.
        """
        # If already computed (Path object), return it directly.
        # This also handles the case where the user passed Path() explicitly
        # at init time -- it's used as-is with no modification.
        if isinstance(self._cache_dir, Path):
            return self._cache_dir

        if self._cache_dir is None:
            # No cache_dir provided: use default pyhealth cache directory
            cache_dir = Path(platformdirs.user_cache_dir(appname="pyhealth")) / "datasets"
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"No cache_dir provided. Using default cache dir: {cache_dir}")
            self._cache_dir = cache_dir
        else:
            # String provided: use as-is (file-based isolation via UUID in filenames)
            cache_dir = Path(self._cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                f"Using cache dir: {cache_dir} "
                f"(cache files will include UUID suffix for table isolation)"
            )
            self._cache_dir = cache_dir

        return self._cache_dir

    def _get_cache_uuid(self) -> str:
        """Get the cache UUID for this dataset configuration.

        Returns a deterministic UUID computed from tables, root, dataset_name,
        and dev mode. This is used to create unique filenames within the cache
        directory so that different table configurations don't collide.
        """
        if not hasattr(self, '_cache_uuid') or self._cache_uuid is None:
            id_str = json.dumps(
                {
                    "root": self.root,
                    "tables": sorted(self.tables),
                    "dataset_name": self.dataset_name,
                    "dev": self.dev,
                },
                sort_keys=True,
            )
            self._cache_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, id_str))
        return self._cache_uuid

    def create_tmpdir(self) -> Path:
        """Creates and returns a new temporary directory within the cache.

        Returns:
            Path: The path to the new temporary directory.
        """
        tmp_dir = self.cache_dir / "tmp" / str(uuid.uuid4())
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

    def clean_tmpdir(self) -> None:
        """Cleans up the temporary directory within the cache."""
        tmp_dir = self.cache_dir / "tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

    def _scan_csv_tsv_gz(
        self, source_path: str
    ) -> dd.DataFrame:
        """Scans a CSV/TSV file (possibly gzipped) and returns a Dask DataFrame.

        If the cached Parquet file does not exist, it converts the source CSV/TSV file
        to Parquet and saves it to the cache.

        Args:
            source_path (str): The source CSV/TSV file path.

        Returns:
            dd.DataFrame: The Dask DataFrame loaded from the cached Parquet file.

        Raises:
            FileNotFoundError: If source_path is None and the cached Parquet file does not exist;
                or if neither the original nor the alternative path of source_path exists.
            ValueError: If the path does not have an expected extension.
        """
        # Ensure the tables cache directory exists
        ret_path = self.create_tmpdir() / "table.parquet"

        if not ret_path.exists():
            source_path = _csv_tsv_gz_path(source_path)

            if is_url(source_path):
                local_filename = os.path.basename(source_path)
                local_path = self.create_tmpdir() / local_filename
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
            # Enable newlines_in_values for clinical notes with multi-line text
            schema_reader = pv.open_csv(
                source_path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(
                    delimiter=delimiter, newlines_in_values=True
                ),
            )
            schema = pa.schema(
                [pa.field(name, pa.string()) for name in schema_reader.schema.names]
            )

            # Convert CSV/TSV to Parquet
            csv_reader = pv.open_csv(
                source_path,
                read_options=pv.ReadOptions(block_size=1 << 26),  # 64 MB
                parse_options=pv.ParseOptions(
                    delimiter=delimiter, newlines_in_values=True
                ),
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

    def _event_transform(self, output_dir: Path) -> None:
        try:
            df = self.load_data()
            with DaskCluster(
                n_workers=self.num_workers,
                threads_per_worker=1,
                processes=not in_notebook(),
                # Use cache_dir for Dask's scratch space to avoid filling up /tmp or home directory
                local_directory=str(self.create_tmpdir()),
            ) as cluster:
                with DaskClient(cluster) as client:
                    if self.dev:
                        logger.info("Dev mode enabled: limiting to 1000 patients")
                        patients = df["patient_id"].unique().head(1000).tolist()
                        filter = df["patient_id"].isin(patients)
                        df = df[filter]

                    logger.info(f"Caching event dataframe to {output_dir}...")
                    collection = df.sort_values("patient_id").to_parquet(
                        output_dir,
                        write_index=False,
                        compute=False,
                    )
                    handle = client.compute(collection)
                    dask_progress(handle)
                    handle.result()  # type: ignore
        except Exception as e:
            if output_dir.exists():
                logger.error(f"Error during caching, removing incomplete file {output_dir}")
                shutil.rmtree(output_dir)
            raise e
        finally:
            self.clean_tmpdir()
        pass

    @property
    def global_event_df(self) -> pl.LazyFrame:
        """Returns the path to the cached event dataframe.

        Returns:
            Path: The path to the cached event dataframe.
        """
        self._main_guard(type(self).global_event_df.fget.__name__) # type: ignore

        if self._global_event_df is None:
            ret_path = self.cache_dir / f"global_event_df_{self._get_cache_uuid()}.parquet"
            if not ret_path.exists():
                logger.info(f"No cached event dataframe found. Creating: {ret_path}")
                self._event_transform(ret_path)
            else:
                logger.info(f"Found cached event dataframe: {ret_path}")
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
        df = self._scan_csv_tsv_gz(csv_path)

        # Convert column names to lowercase before calling preprocess_func
        df = df.rename(columns=str.lower)

        # Check if there is a preprocessing function for this table
        preprocess_func: Optional[Callable[[nw.LazyFrame], nw.LazyFrame]]
        preprocess_func = getattr(self, f"preprocess_{table_name}", None)
        if preprocess_func is not None:
            logger.info(
                f"Preprocessing table: {table_name} with {preprocess_func.__name__}"
            )
            df = preprocess_func(nw.from_native(df)).to_native()  # type: ignore

        # Handle joins
        for join_cfg in table_cfg.join:
            other_csv_path = f"{self.root}/{join_cfg.file_path}"
            other_csv_path = clean_path(other_csv_path)
            logger.info(f"Joining with table: {other_csv_path}")
            join_df = self._scan_csv_tsv_gz(other_csv_path)
            join_df = join_df.rename(columns=str.lower)
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
        # .astype(str) will convert `pd.NA` to "<NA>", which will raise error in to_datetime
        #   use .astype("string") instead, which keeps `pd.NA` as is.
        if timestamp_col:
            if isinstance(timestamp_col, list):
                # Concatenate all timestamp parts in order with no separator
                timestamp_series: dd.Series = functools.reduce(
                    operator.add, (df[col].astype("string") for col in timestamp_col)
                )
            else:
                timestamp_series: dd.Series = df[timestamp_col].astype("string")

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
            df: dd.DataFrame = df.assign(patient_id=df[patient_id_col].astype("string"))
        else:
            df: dd.DataFrame = df.reset_index(drop=True)
            df: dd.DataFrame = df.assign(patient_id=df.index.astype("string"))

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

    def _task_transform(self, task: BaseTask, output_dir: Path, num_workers: int) -> None:
        self._main_guard(self._task_transform.__name__)

        logger.info(f"Applying task transformations on data with {num_workers} workers...")
        global_event_df = task.pre_filter(self.global_event_df)
        patient_ids = (
            global_event_df.select("patient_id")
            .unique()
            .collect(engine="streaming")
            .to_series()
            # .sort can reduce runtime by 5%.
            .sort()
        )

        if in_notebook():
            logger.info("Detected Jupyter notebook environment, setting num_workers to 1")
            num_workers = 1
        num_workers = min(num_workers, len(patient_ids)) # Avoid spawning empty workers

        # This ensures worker's polars threads are limited to avoid oversubscription,
        # which can lead to additional 75% speedup when num_workers is large.
        threads_per_worker = max(1, (os.cpu_count() or 1) // num_workers)

        try:
            with set_env(POLARS_MAX_THREADS=str(threads_per_worker), DATA_OPTIMIZER_NUM_WORKERS=str(num_workers)):
                if num_workers == 1:
                    logger.info("Single worker mode, processing sequentially")
                    _task_transform_fn((0, task, patient_ids, global_event_df, output_dir))
                    _litdata_merge(output_dir)
                    return

                # spwan is required for polars in multiprocessing, see https://docs.pola.rs/user-guide/misc/multiprocessing/#summary
                ctx = multiprocessing.get_context("spawn")
                queue = ctx.Queue()
                args_list = [(
                    worker_id,
                    task,
                    pids,
                    global_event_df,
                    output_dir,
                ) for worker_id, pids in enumerate(itertools.batched(patient_ids, len(patient_ids) // num_workers + 1))]
                with ctx.Pool(processes=num_workers, initializer=_task_transform_init, initargs=(queue,)) as pool:
                    result = pool.map_async(_task_transform_fn, args_list) # type: ignore
                    with tqdm(total=len(patient_ids)) as progress:
                        while not result.ready():
                            try:
                                progress.update(queue.get(timeout=1))
                            except:
                                pass

                        # remaining items
                        while not queue.empty():
                            progress.update(queue.get())
                    result.get() # ensure exceptions are raised
                _litdata_merge(output_dir)

                logger.info(f"Task transformation completed and saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error during task transformation, cleaning up output directory: {output_dir}")
            shutil.rmtree(output_dir)
            raise e

    def _proc_transform(self, task_df: Path, output_dir: Path, num_workers: int) -> None:
        self._main_guard(self._proc_transform.__name__)

        logger.info(f"Applying processors on data with {num_workers} workers...")
        num_samples = len(litdata.StreamingDataset(str(task_df)))

        if in_notebook():
            logger.info("Detected Jupyter notebook environment, setting num_workers to 1")
            num_workers = 1

        num_workers = min(num_workers, num_samples) # Avoid spawning empty workers
        try:
            with set_env(DATA_OPTIMIZER_NUM_WORKERS=str(num_workers)):
                if num_workers == 1:
                    logger.info("Single worker mode, processing sequentially")
                    _proc_transform_fn((0, task_df, 0, num_samples, output_dir))
                    _litdata_merge(output_dir)
                    return

                ctx = multiprocessing.get_context("spawn")
                queue = ctx.Queue()
                linspace = more_itertools.sliding_window(np.linspace(0, num_samples, num_workers + 1, dtype=int), 2)
                args_list = [(
                    worker_id,
                    task_df,
                    start,
                    end,
                    output_dir,
                ) for worker_id, (start, end) in enumerate(linspace)]
                with ctx.Pool(processes=num_workers, initializer=_proc_transform_init, initargs=(queue,)) as pool:
                    result = pool.map_async(_proc_transform_fn, args_list) # type: ignore
                    with tqdm(total=num_samples) as progress:
                        while not result.ready():
                            try:
                                progress.update(queue.get(timeout=1))
                            except:
                                pass

                        # remaining items
                        while not queue.empty():
                            progress.update(queue.get())
                    result.get() # ensure exceptions are raised
                _litdata_merge(output_dir)

                logger.info(f"Processor transformation completed and saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error during processor transformation.")
            shutil.rmtree(output_dir)
            raise e
        finally:
            self.clean_tmpdir()

    def set_task(
        self,
        task: Optional[BaseTask] = None,
        num_workers: Optional[int] = None,
        cache_dir: str | Path | None = None,
        cache_format: str = "parquet",
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> SampleDataset:
        """Processes the base dataset to generate the task-specific sample dataset.
        The cache structure is as follows::

            task_df_{schema_uuid}.ld/  # Intermediate task dataframe (schema-aware)
            samples_{proc_uuid}.ld/    # Final processed samples after applying processors
                schema.pkl             # Saved SampleBuilder schema
                *.bin                  # Processed sample files
            samples_{proc_uuid}.ld/
                ...

        The task_df path includes a hash of the task's input/output schemas,
        so changing schemas automatically invalidates the cached task dataframe.

        Args:
            task (Optional[BaseTask]): The task to set. Uses default task if None.
            num_workers (int): Number of workers for multi-threading. Default is `self.num_workers`.
            cache_dir (Optional[str]): Directory to cache samples after task transformation,
                but without applying processors. Default is {self.cache_dir}/tasks/{task_name}_{uuid5(vars(task))}.
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
        self._main_guard(self.set_task.__name__)

        if task is None:
            assert self.default_task is not None, "No default tasks found"
            task = self.default_task

        if num_workers is None:
            num_workers = self.num_workers

        if cache_format != "parquet":
            logger.warning("Only 'parquet' cache_format is supported now. ")

        logger.info(
            f"Setting task {task.task_name} for {self.dataset_name} base dataset..."
        )

        task_params = json.dumps(
            vars(task),
            sort_keys=True,
            default=str
        )

        if cache_dir is None:
            cache_dir = self.cache_dir / "tasks" / f"{task.task_name}_{uuid.uuid5(uuid.NAMESPACE_DNS, task_params)}"
            cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Ensure the explicitly provided cache_dir exists
            cache_dir = Path(cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

        proc_params = json.dumps(
            {
                "input_schema": task.input_schema,
                "output_schema": task.output_schema,
                "input_processors": (
                    {
                        f"{k}_{v.__class__.__name__}": vars(v)
                        for k, v in input_processors.items()
                    }
                    if input_processors
                    else None
                ),
                "output_processors": (
                    {
                        f"{k}_{v.__class__.__name__}": vars(v)
                        for k, v in output_processors.items()
                    }
                    if output_processors
                    else None
                ),
            },
            sort_keys=True,
            default=str
        )

        # Hash based ONLY on task schemas (not the task instance) to avoid
        # recursion issues. This ensures task_df is invalidated when schemas change.
        task_schema_params = json.dumps(
            {
                "input_schema": task.input_schema,
                "output_schema": task.output_schema,
            },
            sort_keys=True,
            default=str
        )
        task_schema_hash = uuid.uuid5(uuid.NAMESPACE_DNS, task_schema_params)

        task_df_path = Path(cache_dir) / f"task_df_{task_schema_hash}.ld"
        samples_path = Path(cache_dir) / f"samples_{uuid.uuid5(uuid.NAMESPACE_DNS, proc_params)}.ld"

        logger.info(f"Task cache paths: task_df={task_df_path}, samples={samples_path}")

        task_df_path.mkdir(parents=True, exist_ok=True)
        samples_path.mkdir(parents=True, exist_ok=True)
        
        if not (samples_path / "index.json").exists():
            # Check if index.json exists to verify cache integrity, this
            # is the standard file for litdata.StreamingDataset
            if not (task_df_path / "index.json").exists():
                self._task_transform(
                    task,
                    task_df_path,
                    num_workers,
                )
            else:
                logger.info(f"Found cached task dataframe at {task_df_path}, skipping task transformation.")

            # Build processors and fit on the dataset
            logger.info(f"Fitting processors on the dataset...")
            dataset = litdata.StreamingDataset(
                str(task_df_path),
                transform=lambda x: pickle.loads(x["sample"]),
            )
            builder = SampleBuilder(
                input_schema=task.input_schema,  # type: ignore
                output_schema=task.output_schema,  # type: ignore
                input_processors=input_processors,
                output_processors=output_processors,
            )
            builder.fit(dataset)
            builder.save(str(samples_path / "schema.pkl"))

            # Apply processors and save final samples to cache_dir
            logger.info(f"Processing samples and saving to {samples_path}...")
            self._proc_transform(
                task_df_path,
                samples_path,
                num_workers,
            )
            logger.info(f"Cached processed samples to {samples_path}")
        else:
            logger.info(f"Found cached processed samples at {samples_path}, skipping processing.")

        return SampleDataset(
            path=str(samples_path),
            dataset_name=self.dataset_name,
            task_name=task.task_name,
        )

    def _main_guard(self, func_name: str):
        """Warn if method is accessed from a non-main process."""

        if not multiprocessing.current_process().name == "MainProcess":
            logger.warning(
                f"{func_name} method accessed from a non-main process. This may lead to unexpected behavior.\n"
                + "Consider use __name__ == '__main__' guard when using multiprocessing."
            )
            exit(1)
