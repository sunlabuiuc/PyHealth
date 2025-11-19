"""Normal (in-memory) mode implementation for BaseDataset.

This module contains the implementation for processing datasets in normal mode,
where all data is loaded into memory. This is the traditional PyHealth approach
suitable for smaller datasets that fit in memory.
"""

import json
import logging
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
from tqdm import tqdm

from ...processors.base_processor import FeatureProcessor
from ...tasks import BaseTask
from ..sample_dataset import SampleDataset
from ..utils import _convert_for_cache, save_processors, load_processors

logger = logging.getLogger(__name__)


def load_cached_samples_normal(
    cache_path: Path, cache_format: str
) -> Optional[List[Dict]]:
    """Load cached samples from disk (normal mode).

    Args:
        cache_path: Path to cached samples file
        cache_format: Format of cache ('parquet' or 'pickle')

    Returns:
        List of sample dictionaries if successful, None if loading failed
    """
    if not cache_path.exists():
        return None

    logger.info(f"Loading cached samples from {cache_path}")
    try:
        if cache_format == "parquet":
            from ..utils import deserialize_sample_from_parquet

            cached_df = pl.read_parquet(cache_path)
            samples = [
                deserialize_sample_from_parquet(row) for row in cached_df.to_dicts()
            ]
        elif cache_format == "pickle":
            with open(cache_path, "rb") as f:
                samples = pickle.load(f)
        else:
            raise ValueError(f"Unsupported cache format: {cache_format}")

        logger.info(f"Loaded {len(samples)} cached samples")
        return samples
    except Exception as e:
        logger.warning("Failed to load cached data: %s. Regenerating...", e)
        return None


def generate_samples_normal(
    dataset,
    task: BaseTask,
    num_workers: int = 1,
) -> List[Dict]:
    """Generate samples in normal (in-memory) mode.

    Args:
        dataset: The BaseDataset instance
        task: The task to generate samples for
        num_workers: Number of worker threads (1 = single-threaded)

    Returns:
        List of generated sample dictionaries
    """
    logger.info(f"Generating samples with {num_workers} worker(s)...")
    filtered_event_df = task.pre_filter(dataset.collected_global_event_df)
    samples = []

    if num_workers == 1:
        # Single-threading (default and recommended)
        for patient in tqdm(
            dataset.iter_patients(filtered_event_df),
            total=filtered_event_df["patient_id"].n_unique(),
            desc=f"Generating samples for {task.task_name} with 1 worker",
            smoothing=0,
        ):
            samples.extend(task(patient))
    else:
        # Multi-threading (not recommended but available)
        logger.info(
            f"Generating samples for {task.task_name} " f"with {num_workers} workers"
        )
        # Load all patients first (with progress bar)
        patients = list(
            tqdm(
                dataset.iter_patients(filtered_event_df),
                total=filtered_event_df["patient_id"].n_unique(),
                desc=f"Loading patients for {task.task_name}",
                smoothing=0,
            )
        )

        # Process patients in parallel (with progress bar)
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

    return samples


def cache_samples_normal(
    samples: List[Dict], cache_path: Path, cache_format: str
) -> None:
    """Cache samples to disk (normal mode).

    Args:
        samples: List of sample dictionaries to cache
        cache_path: Path where to save the cache
        cache_format: Format to use ('parquet' or 'pickle')
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Caching samples to {cache_path}")

    try:
        if cache_format == "parquet":
            # Convert samples to cache-friendly format
            samples_for_cache = [_convert_for_cache(sample) for sample in samples]

            # Serialize nested dicts to JSON for parquet compatibility
            # Avoids type inference issues like:
            # "failed to determine supertype of list[f64] and list[list[str]]"
            serialized_samples = []
            for sample in samples_for_cache:
                serialized_sample = {}
                for key, value in sample.items():
                    if isinstance(value, dict) and "__stagenet_cache__" in value:
                        # Serialize StageNet cache dicts to JSON strings
                        serialized_sample[key] = json.dumps(value)
                    else:
                        serialized_sample[key] = value
                serialized_samples.append(serialized_sample)

            samples_df = pl.DataFrame(serialized_samples)
            samples_df.write_parquet(cache_path)
        elif cache_format == "pickle":
            # Save samples as pickle file
            with open(cache_path, "wb") as f:
                pickle.dump(samples, f)
        else:
            # Don't raise – just warn and skip caching
            logger.warning(
                "Unsupported cache format '%s'. Skipping caching.",
                cache_format,
            )
            return

        logger.info(f"Successfully cached {len(samples)} samples")
    except Exception as e:
        logger.warning(f"Failed to cache samples: {e}")


def set_task_normal(
    dataset,
    task: BaseTask,
    num_workers: int,
    cache_dir: Optional[str],
    cache_format: str,
    input_processors: Optional[Dict[str, FeatureProcessor]],
    output_processors: Optional[Dict[str, FeatureProcessor]],
) -> SampleDataset:
    """Execute set_task in normal (in-memory) mode.

    This is the traditional PyHealth approach where all data is loaded into
    memory. Suitable for datasets that fit in memory (<100k samples typically).

    Implements smart caching:
    1. Checks if sample cache exists
    2. Checks if processors are cached
    3. Only regenerates what's missing

    Args:
        dataset: The BaseDataset instance
        task: The task to execute
        num_workers: Number of worker threads for parallel processing
        cache_dir: Directory for caching processed samples
        cache_format: Format for cache files ('parquet' or 'pickle')
        input_processors: Pre-fitted input processors
        output_processors: Pre-fitted output processors

    Returns:
        SampleDataset with processed samples loaded in memory
    """
    # Determine cache filename (include dev params to avoid conflicts)
    cache_filename = None
    cache_path = None
    if cache_dir is not None:
        if dataset.dev:
            cache_filename = (
                f"{task.task_name}_dev_{dataset.dev_max_patients}" f".{cache_format}"
            )
        else:
            cache_filename = f"{task.task_name}.{cache_format}"
        cache_path = Path(cache_dir) / cache_filename

    # Check if processors are cached
    processor_cache_dir = None
    processors_cached = False
    if cache_dir is not None:
        processor_cache_dir = Path(cache_dir) / "processors"
        processors_cached = (
            processor_cache_dir / "input_processors.pkl"
        ).exists() and (processor_cache_dir / "output_processors.pkl").exists()

    # Load cached processors if available and not provided
    if processors_cached and input_processors is None and output_processors is None:
        logger.info(f"Loading cached processors from {processor_cache_dir}")
        try:
            input_processors, output_processors = load_processors(
                str(processor_cache_dir)
            )
        except Exception as e:
            logger.warning(f"Failed to load cached processors: {e}. Will rebuild.")
            processors_cached = False
            input_processors = None
            output_processors = None

    # Try to load from cache
    samples = None
    if cache_path is not None:
        samples = load_cached_samples_normal(cache_path, cache_format)

    # Generate samples if not cached
    if samples is None:
        logger.info("No sample cache found. Generating samples...")
        samples = generate_samples_normal(dataset, task, num_workers)

        # Cache the generated samples
        if cache_path is not None:
            cache_samples_normal(samples, cache_path, cache_format)
    else:
        logger.info(f"Using {len(samples)} cached samples from {cache_path}")

    # Create and return SampleDataset
    sample_dataset = SampleDataset(
        samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name=dataset.dataset_name,
        task_name=task.task_name,
        input_processors=input_processors,
        output_processors=output_processors,
    )

    # Save processors if they were just built (and not loaded from cache)
    if (
        cache_dir is not None
        and not processors_cached
        and (input_processors is None or output_processors is None)
    ):
        logger.info(f"Saving processors to {processor_cache_dir}")
        save_processors(sample_dataset, str(processor_cache_dir))

    logger.info(f"Dataset ready with {len(samples)} samples")
    return sample_dataset
