import logging
import os
import pickle
from abc import ABC
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterator, List, Optional
from urllib.parse import urlparse, urlunparse

import polars as pl
import requests
from tqdm import tqdm

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
        stream: bool = False,
        cache_dir: Optional[str] = None,
    ):
        """Initializes the BaseDataset.

        Args:
            root (str): The root directory where dataset files are stored.
            tables (List[str]): List of table names to load.
            dataset_name (Optional[str]): Name of the dataset. Defaults to class name.
            config_path (Optional[str]): Path to the configuration YAML file.
            dev (bool): Whether to run in dev mode (limits to 1000 patients).
            stream (bool): Whether to enable streaming mode for memory efficiency.
                When True, data is loaded from disk on-demand rather than kept in memory.
                Default is False for backward compatibility.
            cache_dir (Optional[str]): Directory for streaming cache. If None, uses
                {root}/.pyhealth_cache. Only used when stream=True.
        """
        if len(set(tables)) != len(tables):
            logger.warning("Duplicate table names in tables list. Removing duplicates.")
            tables = list(set(tables))
        self.root = root
        self.tables = tables
        self.dataset_name = dataset_name or self.__class__.__name__
        self.config = load_yaml_config(config_path)
        self.dev = dev
        self.stream = stream

        # Setup cache directory
        if cache_dir is None:
            self.cache_dir = Path(root) / ".pyhealth_cache"
        else:
            self.cache_dir = Path(cache_dir)

        if self.stream:
            logger.info(f"Stream mode enabled - using disk cache at {self.cache_dir}")
            self._setup_streaming_cache()

        logger.info(
            f"Initializing {self.dataset_name} dataset from {self.root} (dev mode: {self.dev})"
        )

        self.global_event_df = self.load_data()

        # Cached attributes
        self._collected_global_event_df = None
        self._unique_patient_ids = None

        # Streaming-specific attributes
        if self.stream:
            self._patient_cache_path = None
            self._patient_index_path = None
            self._patient_index = None

    def _setup_streaming_cache(self) -> None:
        """Setup disk-backed cache directory structure for streaming mode.

        Creates cache directory and defines paths for patient cache and index.
        Called during __init__ when stream=True.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Define cache file paths
        self._patient_cache_path = (
            self.cache_dir / f"{self.dataset_name}_patients.parquet"
        )
        self._patient_index_path = (
            self.cache_dir / f"{self.dataset_name}_patient_index.parquet"
        )

        logger.info(f"Streaming cache directory: {self.cache_dir}")

    def _build_patient_cache(
        self,
        filtered_df: Optional[pl.LazyFrame] = None,
        force_rebuild: bool = False,
    ) -> None:
        """Build disk-backed patient cache with efficient indexing.

        This method uses Polars' native streaming execution via sink_parquet to process
        the data without loading everything into memory. According to the Polars docs
        (https://docs.pola.rs/user-guide/concepts/streaming/), sink_parquet automatically
        uses the streaming engine to process data in batches.

        Args:
            filtered_df (Optional[pl.LazyFrame]): Pre-filtered LazyFrame (e.g., from
                task.pre_filter()). If None, uses self.global_event_df.
            force_rebuild (bool): If True, rebuild cache even if it exists.
                Default is False.

        Implementation Notes:
            - Uses Polars' sink_parquet for automatic streaming execution
            - Sorts by patient_id for efficient patient-level reads
            - Creates index for O(1) patient lookup
            - Row group size tuned for typical patient size (~100 events)
        """
        if self._patient_cache_path.exists() and not force_rebuild:
            logger.info(f"Using existing patient cache: {self._patient_cache_path}")
            return

        logger.info("Building patient cache using Polars streaming engine...")

        # Use filtered_df if provided, otherwise use global_event_df
        df = filtered_df if filtered_df is not None else self.global_event_df

        # Apply dev mode filtering at the LazyFrame level
        if self.dev:
            logger.info("Dev mode enabled: limiting to 1000 patients")
            limited_patients = df.select(pl.col("patient_id")).unique().limit(1000)
            df = df.join(limited_patients, on="patient_id", how="inner")

        # CRITICAL: Sort by patient_id for efficient patient-level access
        # This enables:
        # 1. Efficient patient-level reads (via row group filtering)
        # 2. Polars can use merge joins on subsequent operations
        # 3. Better compression (similar data grouped together)
        df = df.sort("patient_id", "timestamp")

        # Use sink_parquet for memory-efficient writing
        # According to https://www.rhosignal.com/posts/streaming-in-polars/,
        # sink_parquet automatically uses Polars' streaming engine and never
        # loads the full dataset into memory
        df.sink_parquet(
            self._patient_cache_path,
            # Row group size tuned for patient-level access
            # Assuming ~100 events per patient, 10000 events ≈ 100 patients per row group
            row_group_size=10000,
            compression="zstd",  # Good balance of compression ratio and speed
            statistics=True,  # Enable statistics for better predicate pushdown
        )

        # Build patient index for fast lookups using streaming
        logger.info("Building patient index with streaming...")
        patient_index = (
            pl.scan_parquet(self._patient_cache_path)
            .group_by("patient_id")
            .agg([
                pl.count().alias("event_count"),
                pl.first("timestamp").alias("first_timestamp"),
                pl.last("timestamp").alias("last_timestamp"),
            ])
            .sort("patient_id")
        )
        # sink_parquet uses streaming automatically
        patient_index.sink_parquet(self._patient_index_path)

        cache_size_mb = self._patient_cache_path.stat().st_size / 1e6
        logger.info(f"Patient cache built: {cache_size_mb:.2f} MB")

    @property
    def collected_global_event_df(self) -> pl.DataFrame:
        """Collects and returns the global event data frame.

        WARNING: This property is NOT available in stream mode as it would
        load the entire dataset into memory, defeating the purpose of streaming.

        Returns:
            pl.DataFrame: The collected global event data frame.

        Raises:
            RuntimeError: If called in stream mode.
        """
        if self.stream:
            raise RuntimeError(
                "collected_global_event_df is not available in stream mode "
                "as it would load the entire dataset into memory. "
                "Use iter_patients() or get_patient() for memory-efficient access."
            )

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

        In streaming mode, loads the patient from disk cache.
        In normal mode, filters from the collected DataFrame.

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
        
        if self.stream:
            # Streaming mode: Load patient from disk cache
            if not self._patient_cache_path.exists():
                self._build_patient_cache()
            
            patient_df = (
                pl.scan_parquet(self._patient_cache_path)
                .filter(pl.col("patient_id") == patient_id)
                .collect()
            )
            return Patient(patient_id=patient_id, data_source=patient_df)
        else:
            # Normal mode: Filter from collected DataFrame
            df = self.collected_global_event_df.filter(pl.col("patient_id") == patient_id)
            return Patient(patient_id=patient_id, data_source=df)

    def iter_patients(
        self, 
        df: Optional[pl.DataFrame] = None,
        patient_ids: Optional[List[str]] = None,
        preload: int = 1,
    ) -> Iterator[Patient]:
        """Yields Patient objects for each unique patient in the dataset.

        Automatically uses streaming iteration when stream=True and df is None.
        In normal mode, loads data into memory and iterates.

        Args:
            df (Optional[pl.DataFrame]): Optional pre-filtered DataFrame.
                If None, behavior depends on stream mode:
                - stream=False: Uses collected_global_event_df (loads to memory)
                - stream=True: Uses disk-backed streaming iteration
            patient_ids (Optional[List[str]]): Optional list of specific patient IDs 
                to iterate over. Only used in streaming mode when df is None.
            preload (int): Number of patients to preload ahead in streaming mode.
                Only used when stream=True and df is None. Default is 1.

        Yields:
            Iterator[Patient]: An iterator over Patient objects.
        """
        if df is None:
            if self.stream:
                # Streaming mode: Use disk-backed iteration
                # Ensure cache exists
                if not self._patient_cache_path.exists():
                    self._build_patient_cache()

                # Load patient index for efficient filtering
                if self._patient_index is None:
                    self._patient_index = pl.read_parquet(self._patient_index_path)

                patient_index_df = self._patient_index

                # Filter to specific patients if requested
                if patient_ids is not None:
                    patient_index_df = patient_index_df.filter(
                        pl.col("patient_id").is_in(patient_ids)
                    )

                patient_list = patient_index_df["patient_id"].to_list()

                def load_patient(patient_id: str) -> Patient:
                    """Load a single patient from disk cache."""
                    patient_df = (
                        pl.scan_parquet(self._patient_cache_path)
                        .filter(pl.col("patient_id") == patient_id)
                        .collect()
                    )
                    return Patient(patient_id=patient_id, data_source=patient_df)

                if preload <= 1:
                    # No preloading - simple sequential iteration
                    for patient_id in patient_list:
                        yield load_patient(patient_id)
                else:
                    # Preload patients in background using ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=min(preload, 4)) as executor:
                        # Submit initial batch of futures
                        futures = []
                        for i in range(min(preload, len(patient_list))):
                            futures.append(executor.submit(load_patient, patient_list[i]))

                        # Main iteration loop
                        for i in range(len(patient_list)):
                            # Wait for and yield the oldest future
                            patient = futures.pop(0).result()
                            yield patient

                            # Submit next patient if available
                            next_idx = i + preload
                            if next_idx < len(patient_list):
                                futures.append(
                                    executor.submit(load_patient, patient_list[next_idx])
                                )

                        # Yield any remaining preloaded patients
                        for future in futures:
                            yield future.result()
            else:
                # Normal mode: Load all data to memory
                df = self.collected_global_event_df
                grouped = df.group_by("patient_id")
                for patient_id, patient_df in grouped:
                    patient_id = patient_id[0]
                    yield Patient(patient_id=patient_id, data_source=patient_df)
        else:
            # DataFrame provided: Use it regardless of mode
            grouped = df.group_by("patient_id")
            for patient_id, patient_df in grouped:
                patient_id = patient_id[0]
                yield Patient(patient_id=patient_id, data_source=patient_df)

    def iter_patients_streaming(
        self,
        patient_ids: Optional[List[str]] = None,
        preload: int = 1,
    ) -> Iterator[Patient]:
        """[DEPRECATED] Use iter_patients() instead - it automatically handles streaming.

        This method is kept for backward compatibility but now simply calls
        iter_patients() with the same parameters.

        Args:
            patient_ids (Optional[List[str]]): Optional list of specific patient IDs.
            preload (int): Number of patients to preload ahead. Default is 1.

        Yields:
            Patient: Patient objects

        Example:
            >>> # Old way (still works)
            >>> for patient in dataset.iter_patients_streaming():
            ...     process(patient)
            
            >>> # New way (recommended)
            >>> for patient in dataset.iter_patients():
            ...     process(patient)  # Automatically streams if stream=True
        """
        import warnings
        warnings.warn(
            "iter_patients_streaming() is deprecated. Use iter_patients() instead - "
            "it automatically handles streaming when stream=True.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.iter_patients(patient_ids=patient_ids, preload=preload)

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
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ):
        """Processes the base dataset to generate the task-specific sample dataset.

        Args:
            task (Optional[BaseTask]): The task to set. Uses default task if None.
            num_workers (int): Number of workers for multi-threading. Default is 1.
                Only used in non-streaming mode.
            cache_dir (Optional[str]): Directory to cache processed samples.
                Default is None (no caching).
            cache_format (str): Format for caching ('parquet' or 'pickle').
                Default is 'parquet'.
            input_processors (Optional[Dict[str, FeatureProcessor]]):
                Pre-fitted input processors.
            output_processors (Optional[Dict[str, FeatureProcessor]]):
                Pre-fitted output processors.

        Returns:
            Union[SampleDataset, IterableSampleDataset]: The generated sample dataset.
                Returns SampleDataset in non-streaming mode.
                Returns IterableSampleDataset in streaming mode.

        Raises:
            AssertionError: If no default task is found and task is None.
        """
        if task is None:
            assert self.default_task is not None, "No default tasks found"
            task = self.default_task

        logger.info(
            f"Setting task {task.task_name} for {self.dataset_name} base dataset..."
        )

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
            if self.stream:
                # ============================================================
                # STREAMING MODE: Process patients iteratively
                # ============================================================
                from .sample_dataset import IterableSampleDataset
                
                logger.info("Generating samples in streaming mode...")

                # Apply task's pre_filter on LazyFrame (no data loaded yet!)
                filtered_lazy_df = task.pre_filter(self.global_event_df)

                # Build patient cache if not exists (lazy execution with sink_parquet)
                if not self._patient_cache_path.exists():
                    self._build_patient_cache(filtered_lazy_df)

                # Create streaming sample dataset
                sample_dataset = IterableSampleDataset(
                    input_schema=task.input_schema,
                    output_schema=task.output_schema,
                    dataset_name=self.dataset_name,
                    task_name=task.task_name,
                    input_processors=input_processors,
                    output_processors=output_processors,
                    cache_dir=cache_dir or str(self.cache_dir),
                )

                # Process patients iteratively and write samples to disk
                batch_samples = []
                batch_size = 100  # Write every 100 patients worth of samples

                # Get total patient count for progress bar
                patient_index = pl.read_parquet(self._patient_index_path)
                total_patients = len(patient_index)

                for patient in tqdm(
                    self.iter_patients(preload=3),  # Now uses unified API!
                    total=total_patients,
                    desc=f"Generating samples for {task.task_name}",
                ):
                    # Call task function (SAME AS BEFORE!)
                    patient_samples = task(patient)
                    batch_samples.extend(patient_samples)

                    # Write batch to disk when it gets large enough
                    if len(batch_samples) >= batch_size:
                        sample_dataset.add_samples_streaming(batch_samples)
                        batch_samples = []

                # Write remaining samples
                if batch_samples:
                    sample_dataset.add_samples_streaming(batch_samples)

                # Finalize sample cache
                sample_dataset.finalize_samples()

                # Build processors (must happen after all samples written)
                sample_dataset.build_streaming()

                logger.info(f"Generated {len(sample_dataset)} samples in streaming mode")

                return sample_dataset

            else:
                # ============================================================
                # NORMAL MODE: Original implementation (UNCHANGED)
                # ============================================================
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

        # Create sample dataset from cached samples (normal mode only)
        if samples is not None:
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
