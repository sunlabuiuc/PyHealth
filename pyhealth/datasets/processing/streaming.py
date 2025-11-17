"""Streaming mode implementation for BaseDataset.

This module contains the implementation for processing datasets in streaming mode,
where data is processed in batches and stored on disk. This enables memory-efficient
processing of large datasets that don't fit in memory.

All streaming-specific logic is centralized here, including:
- Cache setup and management
- Patient cache building with disk-backed storage
- Streaming iteration over patients
- Task processing in streaming mode
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Union

import polars as pl
from tqdm import tqdm

from ...data import Patient
from ...processors.base_processor import FeatureProcessor
from ...tasks import BaseTask
from ..iterable_sample_dataset import IterableSampleDataset

logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions
# ============================================================================


def _create_patients_from_dataframe(
    df: pl.DataFrame, lazy_partition: bool = False
) -> List[Patient]:
    """Convert a DataFrame with multiple patients into Patient objects.

    Args:
        df: DataFrame containing events for one or more patients
        lazy_partition: Whether to enable lazy partitioning for streaming

    Returns:
        List of Patient objects
    """
    batch_patients = []
    grouped = df.group_by("patient_id")
    for patient_id, patient_df in grouped:
        patient_id = patient_id[0]
        batch_patients.append(
            Patient(
                patient_id=patient_id,
                data_source=patient_df,
                lazy_partition=lazy_partition,
            )
        )
    return batch_patients


# ============================================================================
# Cache Setup and Management
# ============================================================================


def setup_streaming_cache(dataset) -> None:
    """Setup disk-backed cache directory structure for streaming mode.

    Creates cache directory and defines paths for patient cache and index.
    Called during BaseDataset.__init__ when stream=True.

    Dev mode uses separate cache files to avoid conflicts with full dataset.
    Cache filenames include dev_max_patients to support different dev sizes.

    Args:
        dataset: The BaseDataset instance with cache_dir, dataset_name,
            dev, and dev_max_patients attributes
    """
    dataset.cache_dir.mkdir(parents=True, exist_ok=True)

    # Add dev suffix with patient count to cache paths
    # This ensures different dev configurations use different cache files
    if dataset.dev:
        suffix = f"_dev_{dataset.dev_max_patients}"
    else:
        suffix = ""

    # Define cache file paths
    dataset._patient_cache_path = (
        dataset.cache_dir / f"{dataset.dataset_name}_patients{suffix}.parquet"
    )
    dataset._patient_index_path = (
        dataset.cache_dir / f"{dataset.dataset_name}_patient_index{suffix}.parquet"
    )

    logger.info(f"Streaming cache directory: {dataset.cache_dir}")


def build_patient_cache(
    dataset,
    filtered_df: Optional[pl.LazyFrame] = None,
    force_rebuild: bool = False,
) -> None:
    """Build disk-backed patient cache with efficient indexing.

    This method uses Polars' native streaming execution via sink_parquet to process
    the data without loading everything into memory. According to the Polars docs
    (https://docs.pola.rs/user-guide/concepts/streaming/), sink_parquet automatically
    uses the streaming engine to process data in batches.

    Args:
        dataset: The BaseDataset instance
        filtered_df: Pre-filtered LazyFrame (e.g., from task.pre_filter()).
            If None, uses dataset.global_event_df.
        force_rebuild: If True, rebuild cache even if it exists.
            Default is False.

    Implementation Notes:
        - Uses Polars' sink_parquet for automatic streaming execution
        - Sorts by patient_id for efficient patient-level reads
        - Creates index for O(1) patient lookup
        - Row group size tuned for typical patient size (~100 events)
    """
    # Check if both cache and index exist
    cache_exists = dataset._patient_cache_path.exists()
    index_exists = dataset._patient_index_path.exists()

    if cache_exists and index_exists and not force_rebuild:
        logger.info(f"Using existing patient cache: {dataset._patient_cache_path}")
        return

    logger.info("Building patient cache using Polars streaming engine...")

    # Use filtered_df if provided, otherwise use global_event_df
    df = filtered_df if filtered_df is not None else dataset.global_event_df

    # Filter to synchronized patient IDs if set.
    # This applies to:
    # 1. Sub-datasets (Notes/CXR) - filtered to parent's patient set
    # 2. Composite datasets (MIMIC4) - filtered to synchronized patient set
    # 3. Any dataset with manually set _unique_patient_ids
    if dataset._unique_patient_ids is not None:
        logger.info(
            f"Filtering to {len(dataset._unique_patient_ids)} " f"synchronized patients"
        )
        reference_patients = pl.DataFrame(
            {"patient_id": dataset._unique_patient_ids}
        ).lazy()
        df = df.join(reference_patients, on="patient_id", how="inner")

    # Apply dev mode filtering at the LazyFrame level
    if dataset.dev:
        logger.info(
            f"Dev mode enabled: limiting to {dataset.dev_max_patients} patients"
        )
        limited_patients = (
            df.select(pl.col("patient_id")).unique().limit(dataset.dev_max_patients)
        )
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
        dataset._patient_cache_path,
        # Row group size tuned for patient-level access
        # Assuming ~100 events per patient, 10000 events ≈ 100 patients per row group
        row_group_size=10000,
        compression="zstd",  # Good balance of compression ratio and speed
        statistics=True,  # Enable statistics for better predicate pushdown
    )

    # Build patient index for fast lookups using streaming
    logger.info("Building patient index with streaming...")
    patient_index = (
        pl.scan_parquet(dataset._patient_cache_path)
        .group_by("patient_id")
        .agg(
            [
                pl.count().alias("event_count"),
                pl.first("timestamp").alias("first_timestamp"),
                pl.last("timestamp").alias("last_timestamp"),
            ]
        )
        .sort("patient_id")
    )
    # sink_parquet uses streaming automatically
    patient_index.sink_parquet(dataset._patient_index_path)

    # Load index with streaming for verification
    dataset._patient_index = pl.scan_parquet(dataset._patient_index_path).collect(
        streaming=True
    )

    cache_size_mb = dataset._patient_cache_path.stat().st_size / 1e6
    logger.info(f"Patient cache built: {cache_size_mb:.2f} MB")


# ============================================================================
# Streaming Patient Iteration
# ============================================================================


def iter_patients_streaming(
    dataset,
    patient_ids: Optional[List[str]],
    batch_size: int = 1,
) -> Iterator[Union[Patient, List[Patient]]]:
    """Iterate over patients in streaming mode.

    Loads patients from disk cache in batches for I/O efficiency.
    When batch_size=1, yields individual patients. When batch_size>1,
    yields lists of patients.

    Args:
        dataset: The BaseDataset instance
        patient_ids: Optional list of specific patient IDs to iterate over.
            If None, iterates over all patients in the cache.
        batch_size: Number of patients to load per disk query. Default is 1.
            Larger batches = better I/O efficiency but more memory.
            Use batch_size=1 for single-patient iteration.

    Yields:
        Patient objects (if batch_size=1) or lists of Patient objects (if batch_size>1)

    Example:
        >>> # Single patients (batch_size=1)
        >>> for patient in iter_patients_streaming(dataset, None, batch_size=1):
        ...     print(patient.patient_id)

        >>> # Batches of 100 (much more efficient!)
        >>> for batch in iter_patients_streaming(dataset, None, batch_size=100):
        ...     print(f"Processing {len(batch)} patients")
    """
    # Ensure cache exists
    if not dataset._patient_cache_path.exists():
        build_patient_cache(dataset)

    # Load patient index
    if dataset._patient_index is None:
        dataset._patient_index = pl.scan_parquet(dataset._patient_index_path).collect(
            streaming=True
        )

    patient_index_df = dataset._patient_index

    # Filter to specific patients if requested
    if patient_ids is not None:
        patient_index_df = patient_index_df.filter(
            pl.col("patient_id").is_in(patient_ids)
        )

    patient_list = patient_index_df["patient_id"].to_list()

    # Process in batches (even if batch_size=1, this is still efficient)
    for i in range(0, len(patient_list), batch_size):
        batch_patient_ids = patient_list[i : i + batch_size]

        # Load entire batch in one disk query (efficient even for single patient)
        batch_df = (
            pl.scan_parquet(dataset._patient_cache_path)
            .filter(pl.col("patient_id").is_in(batch_patient_ids))
            .collect(streaming=True)
        )

        # Convert DataFrame to Patient objects
        batch_patients = _create_patients_from_dataframe(batch_df, lazy_partition=True)

        # Yield batch or individual patient depending on batch_size
        if batch_size == 1:
            # Single patient mode - yield individual patient
            if batch_patients:
                yield batch_patients[0]
        else:
            # Batch mode - yield list of patients
            yield batch_patients

        # Explicitly clear batch to help garbage collection
        del batch_df


# ============================================================================
# Task Processing in Streaming Mode
# ============================================================================
# ============================================================================


def set_task_streaming(
    dataset,
    task: BaseTask,
    batch_size: int,
    cache_dir: Optional[str],
    input_processors: Optional[Dict[str, FeatureProcessor]],
    output_processors: Optional[Dict[str, FeatureProcessor]],
) -> IterableSampleDataset:
    """Execute set_task in streaming mode.

    This mode processes patients in batches and writes samples to disk,
    enabling memory-efficient processing of large datasets (>100k samples).

    Args:
        dataset: The BaseDataset instance
        task: The task to execute
        batch_size: Number of patients to process per batch
        cache_dir: Directory for caching processed samples
        input_processors: Pre-fitted input processors
        output_processors: Pre-fitted output processors

    Returns:
        IterableSampleDataset with samples stored on disk
    """
    logger.info("Generating samples in streaming mode...")

    # Apply task's pre_filter on LazyFrame (no data loaded yet!)
    filtered_lazy_df = task.pre_filter(dataset.global_event_df)

    # Build patient cache if not exists (lazy execution with sink_parquet)
    if not dataset._patient_cache_path.exists():
        build_patient_cache(dataset, filtered_lazy_df)

    # Create streaming sample dataset
    sample_dataset = IterableSampleDataset(
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=dataset.dataset_name,
        task_name=task.task_name,
        input_processors=input_processors,
        output_processors=output_processors,
        cache_dir=cache_dir or str(dataset.cache_dir),
        dev=dataset.dev,
        dev_max_patients=dataset.dev_max_patients,
    )

    # Process patients in batches and write samples to disk
    write_batch_samples = []
    write_batch_size = 500  # Write every 500 samples

    # Get total patient count for progress bar
    patient_index = pl.scan_parquet(dataset._patient_index_path).collect(streaming=True)
    total_patients = len(patient_index)

    logger.info(
        f"Processing patients in batches of {batch_size} " f"for better I/O efficiency"
    )

    # Use streaming batch iteration for better performance
    for patient_batch in tqdm(
        iter_patients_streaming(dataset, None, batch_size),
        total=(total_patients + batch_size - 1) // batch_size,
        desc=f"Generating samples for {task.task_name}",
        unit="batch",
    ):
        # Process all patients in the batch
        for patient in patient_batch:
            patient_samples = task(patient)
            write_batch_samples.extend(patient_samples)

        # Write to disk when batch gets large enough
        if len(write_batch_samples) >= write_batch_size:
            sample_dataset.add_samples_streaming(write_batch_samples)
            write_batch_samples = []

        # Explicitly clear patient_batch to free memory immediately
        # This helps garbage collector reclaim Patient DataFrames
        del patient_batch

    # Write remaining samples
    if write_batch_samples:
        sample_dataset.add_samples_streaming(write_batch_samples)

    # Finalize sample cache
    sample_dataset.finalize_samples()

    # Build processors (must happen after all samples written)
    sample_dataset.build_streaming()

    logger.info(f"Generated {len(sample_dataset)} samples in streaming mode")

    return sample_dataset
