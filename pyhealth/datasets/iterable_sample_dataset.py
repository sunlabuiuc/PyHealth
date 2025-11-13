from typing import Dict, List, Optional, Union, Type, Iterator
import inspect
import json
import logging
from pathlib import Path

from torch.utils.data import IterableDataset
import torch
import polars as pl

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor
from .utils import _convert_for_cache, deserialize_sample_from_parquet

logger = logging.getLogger(__name__)


class IterableSampleDataset(IterableDataset):
    """Iterable sample dataset class for streaming mode.

    This class provides memory-efficient iteration over samples stored on disk.
    It is designed for streaming mode and is the recommended approach for
    large datasets that don't fit in memory.

    Key differences from SampleDataset:
    - Inherits from IterableDataset (not Dataset)
    - No __getitem__ support (iteration only)
    - Samples stored on disk, loaded in batches
    - Memory usage independent of dataset size

    Attributes:
        input_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
            Schema for input data.
        output_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
            Schema for output data.
        dataset_name (Optional[str]): Name of the dataset.
        task_name (Optional[str]): Name of the task.
        cache_dir (Path): Directory for disk-backed cache.
    """

    def __init__(
        self,
        input_schema: Dict[str, Union[str, Type[FeatureProcessor]]],
        output_schema: Dict[str, Union[str, Type[FeatureProcessor]]],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
        cache_dir: Optional[str] = None,
        dev: bool = False,
        dev_max_patients: int = 1000,
    ) -> None:
        """Initializes the IterableSampleDataset.

        Args:
            input_schema: Schema for input data.
            output_schema: Schema for output data.
            dataset_name: Name of the dataset.
            task_name: Name of the task.
            input_processors: Pre-fitted input processors.
            output_processors: Pre-fitted output processors.
            cache_dir: Directory for disk-backed cache.
            dev: Whether dev mode is enabled (for separate caching).
            dev_max_patients: Max patients for dev mode (used in cache naming).
        """
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.dataset_name = dataset_name or ""
        self.task_name = task_name or ""
        self.dev = dev
        self.dev_max_patients = dev_max_patients

        # Processor dictionaries
        self.input_processors = input_processors or {}
        self.output_processors = output_processors or {}

        # Setup streaming storage
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".pyhealth_cache")
        self._setup_streaming_storage()

    def _setup_streaming_storage(self) -> None:
        """Setup disk-backed sample storage for streaming mode."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Add dev suffix to separate dev and full caches
        if self.dev:
            suffix = f"_dev_{self.dev_max_patients}"
        else:
            suffix = ""

        self._sample_cache_path = (
            self.cache_dir
            / f"{self.dataset_name}_{self.task_name}_samples{suffix}.parquet"
        )

        # Directory for storing individual batch files before combining
        self._sample_batch_dir = (
            self.cache_dir / f"{self.dataset_name}_{self.task_name}_batches{suffix}"
        )

        # Track number of samples for __len__
        self._num_samples = 0
        self._samples_finalized = False
        self._batch_counter = 0  # Track number of batch files

        logger.info(f"Streaming sample cache: {self._sample_cache_path}")

    def add_samples_streaming(self, samples: List[Dict]) -> None:
        """Add samples to disk-backed storage in streaming mode.

        This method writes samples to parquet file incrementally,
        allowing processing of datasets larger than memory.

        Args:
            samples: List of sample dictionaries to add

        Raises:
            RuntimeError: If called after finalize_samples()
        """
        if self._samples_finalized:
            raise RuntimeError(
                "Cannot add more samples after finalize_samples() has been called"
            )

        if not samples:
            return  # Nothing to add

        # Convert samples for cache-friendly storage
        converted_samples = [_convert_for_cache(s) for s in samples]

        # Serialize complex nested structures to JSON strings for Parquet storage
        # This avoids Polars type inference issues with mixed nested types
        serialized_samples = []
        for sample in converted_samples:
            serialized_sample = {}
            for key, value in sample.items():
                if isinstance(value, dict) and "__stagenet_cache__" in value:
                    # Serialize StageNet cache dicts to JSON strings
                    serialized_sample[key] = json.dumps(value)
                else:
                    serialized_sample[key] = value
            serialized_samples.append(serialized_sample)

        # DEBUG: Inspect first converted sample to understand structure
        if self._num_samples == 0 and len(serialized_samples) > 0:
            logger.info("DEBUG: Inspecting first serialized sample structure:")
            first_sample = serialized_samples[0]
            for key, value in first_sample.items():
                value_type = type(value).__name__
                if isinstance(value, str) and len(value) > 100:
                    logger.info(f"  {key}: {value_type} (JSON, len={len(value)})")
                else:
                    logger.info(f"  {key}: {value_type} = {value}")

        # Convert samples to DataFrame
        try:
            sample_df = pl.DataFrame(serialized_samples)
        except Exception as e:
            logger.error(f"Failed to create DataFrame from samples: {e}")
            logger.error("Sample structure causing issue:")
            for key in serialized_samples[0].keys():
                values = [s[key] for s in serialized_samples[:3]]
                logger.error(f"  {key}: {[type(v).__name__ for v in values]}")
            raise

        # Write batch to individual file (safer than appending)
        # Create batch directory if needed
        self._sample_batch_dir.mkdir(parents=True, exist_ok=True)

        # Write this batch to a separate file
        batch_file = self._sample_batch_dir / f"batch_{self._batch_counter:06d}.parquet"
        sample_df.write_parquet(batch_file, compression="zstd")

        self._batch_counter += 1
        self._num_samples += len(samples)
        logger.debug(f"Added {len(samples)} samples (total: {self._num_samples})")

    def finalize_samples(self) -> None:
        """Finalize sample writing and prepare for reading.

        Call this after all samples have been added via add_samples_streaming().
        This combines all batch files into a single parquet file.
        """
        if self._samples_finalized:
            logger.warning("finalize_samples() called multiple times")
            return

        # Combine all batch files into final cache file
        if self._batch_counter > 0:
            logger.info(f"Combining {self._batch_counter} batch files...")

            # Read all batch files and concatenate using streaming
            batch_files = sorted(self._sample_batch_dir.glob("batch_*.parquet"))

            if len(batch_files) == 1:
                # Only one batch - just rename it
                import shutil

                shutil.move(str(batch_files[0]), str(self._sample_cache_path))
            else:
                # Multiple batches - concatenate using scan for memory efficiency
                lazy_frames = [pl.scan_parquet(f) for f in batch_files]
                combined = pl.concat(lazy_frames, how="diagonal")

                # Write final file using streaming
                combined.sink_parquet(self._sample_cache_path, compression="zstd")

            # Clean up batch files with more robust error handling
            import shutil
            import time

            try:
                shutil.rmtree(self._sample_batch_dir)
            except OSError as e:
                # If deletion fails, try again after a brief pause
                # (files might still be held by OS)
                logger.warning(
                    f"Failed to delete batch dir on first try: {e}. Retrying..."
                )
                time.sleep(0.5)
                try:
                    shutil.rmtree(self._sample_batch_dir)
                except OSError as e:
                    # If it still fails, just log a warning and continue
                    # The directory will be overwritten on next run anyway
                    logger.warning(
                        f"Could not delete batch directory "
                        f"{self._sample_batch_dir}: {e}"
                    )

            logger.info(f"Combined into {self._sample_cache_path}")

        self._samples_finalized = True
        logger.info(f"Finalized {self._num_samples} samples in streaming mode")

    def build_streaming(self) -> None:
        """Build processors in streaming mode.

        Strategy: Read samples in batches to fit processors without
        loading everything to memory. Samples remain in cache as raw data
        and are processed on-the-fly during iteration.

        This method requires that samples have been finalized.
        """
        if not self._samples_finalized:
            raise RuntimeError("Must call finalize_samples() before build_streaming()")

        logger.info("Building processors in streaming mode...")

        # Step 1: Create processor instances (only for missing ones)
        # Track which processors need fitting
        processors_to_fit = {}

        if not self.input_processors:
            for k, v in self.input_schema.items():
                processor = self._get_processor_instance(v)
                self.input_processors[k] = processor
                processors_to_fit[k] = processor
        if not self.output_processors:
            for k, v in self.output_schema.items():
                processor = self._get_processor_instance(v)
                self.output_processors[k] = processor
                processors_to_fit[k] = processor

        # Step 2: Fit processors by reading samples in batches
        # Only fit processors that were just created (not pre-fitted ones)
        if processors_to_fit:
            logger.info(f"Fitting {len(processors_to_fit)} processors on samples...")
            batch_size = 1000

            # Warn about large batch size and Polars streaming issues
            if batch_size > 200:
                logger.warning(
                    f"Using batch_size={batch_size} for processor fitting. "
                    f"Note: Polars streaming is disabled for slice operations "
                    f"to avoid race conditions in async parquet reader. "
                    f"For better performance, consider batch_size <= 200."
                )

            lf = pl.scan_parquet(self._sample_cache_path)

            # Check if we have enough memory to load all samples
            # For now, use streaming fit if more than 100k samples
            use_streaming_fit = self._num_samples > 100_000

            if use_streaming_fit:
                # Streaming fit: Process in batches without accumulating
                logger.info(
                    f"Using streaming fit for {self._num_samples} samples "
                    f"(memory-efficient mode)"
                )
                num_batches = (self._num_samples + batch_size - 1) // batch_size

                for i in range(num_batches):
                    batch = lf.slice(i * batch_size, batch_size).collect(
                        streaming=False
                    )
                    batch_samples = batch.to_dicts()
                    restored_samples = [
                        deserialize_sample_from_parquet(s) for s in batch_samples
                    ]

                    # Fit incrementally on each batch
                    for key, processor in processors_to_fit.items():
                        processor.fit(restored_samples, key, stream=True)

                    # Free batch memory immediately
                    del restored_samples

                # Finalize all processors
                for key, processor in processors_to_fit.items():
                    if hasattr(processor, "finalize_fit"):
                        processor.finalize_fit()
            else:
                # Non-streaming fit: Load all samples then fit once
                logger.info("Loading all samples for processor fitting...")
                all_restored_samples = []
                num_batches = (self._num_samples + batch_size - 1) // batch_size
                for i in range(num_batches):
                    batch = lf.slice(i * batch_size, batch_size).collect(
                        streaming=False
                    )
                    batch_samples = batch.to_dicts()
                    restored_samples = [
                        deserialize_sample_from_parquet(s) for s in batch_samples
                    ]
                    all_restored_samples.extend(restored_samples)

                # Fit each processor once on all samples
                for key, processor in processors_to_fit.items():
                    logger.debug(f"Fitting processor for key: {key}")
                    processor.fit(all_restored_samples, key, stream=False)

                # Clean up to free memory
                del all_restored_samples
        else:
            logger.info("Using pre-fitted processors (skipping fit)")

        logger.info(
            "Processors ready! Processing will happen on-the-fly during iteration."
        )

    def _get_processor_instance(
        self, processor_spec: Union[str, Type[FeatureProcessor]]
    ) -> FeatureProcessor:
        """Get processor instance from schema value.

        Args:
            processor_spec: Either a string alias or a processor class

        Returns:
            Processor instance
        """
        if isinstance(processor_spec, str):
            return get_processor(processor_spec)()
        elif inspect.isclass(processor_spec) and issubclass(
            processor_spec, FeatureProcessor
        ):
            return processor_spec()
        else:
            raise ValueError(
                f"Processor spec must be either a string alias or a "
                f"FeatureProcessor class, got {type(processor_spec)}"
            )

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over samples efficiently.

        This is the main method for accessing samples in streaming mode.
        Samples are read from disk in batches, deserialized, and processed
        on-the-fly with fitted processors.

        Yields:
            Sample dictionaries with processed features (as tensors)

        Example:
            >>> for sample in dataset:
            ...     train_on_sample(sample)
        """
        if not self._samples_finalized:
            raise RuntimeError("Cannot iterate before finalize_samples()")

        # Get worker info for distributed training
        worker_info = torch.utils.data.get_worker_info()

        batch_size = 32  # Default batch size for reading

        # Warn about large batch sizes that may trigger Polars streaming bugs
        if batch_size > 200:
            logger.warning(
                f"batch_size={batch_size} may trigger Polars streaming bugs. "
                f"Consider using batch_size <= 200 for better stability. "
                f"Note: streaming is disabled for slice operations to avoid "
                f"known Polars parquet reader race conditions."
            )

        lf = pl.scan_parquet(self._sample_cache_path)

        if worker_info is None:
            # Single worker - iterate all samples
            num_batches = (self._num_samples + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                offset = batch_idx * batch_size
                length = min(batch_size, self._num_samples - offset)
                # Disable streaming for slice operations due to Polars bug:
                # Large batches (>200) trigger race conditions in async
                # parquet reader causing "range end index out of bounds" errors.
                # Using streaming=False forces synchronous reads, avoiding bug.
                batch = lf.slice(offset, length).collect(streaming=False)
                for sample in batch.to_dicts():
                    # Deserialize from parquet format
                    restored_sample = deserialize_sample_from_parquet(sample)

                    # Process with fitted processors
                    for k, v in restored_sample.items():
                        if k in self.input_processors:
                            restored_sample[k] = self.input_processors[k].process(v)
                        elif k in self.output_processors:
                            restored_sample[k] = self.output_processors[k].process(v)

                    yield restored_sample
        else:
            # Multiple workers - partition samples
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

            # Each worker processes every nth batch
            num_batches = (self._num_samples + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                if batch_idx % num_workers == worker_id:
                    offset = batch_idx * batch_size
                    length = min(batch_size, self._num_samples - offset)
                    # Disable streaming for slice operations (see comment above)
                    batch = lf.slice(offset, length).collect(streaming=False)
                    for sample in batch.to_dicts():
                        # Deserialize from parquet format
                        restored_sample = deserialize_sample_from_parquet(sample)

                        # Process with fitted processors
                        for k, v in restored_sample.items():
                            if k in self.input_processors:
                                restored_sample[k] = self.input_processors[k].process(v)
                            elif k in self.output_processors:
                                restored_sample[k] = self.output_processors[k].process(
                                    v
                                )

                        yield restored_sample

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return self._num_samples

    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            str: A string with the dataset and task names.
        """
        return f"Iterable sample dataset {self.dataset_name} {self.task_name}"
