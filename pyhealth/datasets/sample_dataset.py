from typing import Dict, List, Optional, Union, Type, Iterator
import inspect
import json
import logging
from pathlib import Path

from torch.utils.data import Dataset, IterableDataset
import torch
from tqdm import tqdm
import polars as pl

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor
from .utils import _convert_for_cache, deserialize_sample_from_parquet

logger = logging.getLogger(__name__)


class SampleDataset(Dataset):
    """Sample dataset class for handling and processing data samples.

    Attributes:
        samples (List[Dict]): List of data samples.
        input_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
            Schema for input data.
        output_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
            Schema for output data.
        dataset_name (Optional[str]): Name of the dataset.
        task_name (Optional[str]): Name of the task.
    """

    def __init__(
        self,
        samples: List[Dict],
        input_schema: Dict[str, Union[str, Type[FeatureProcessor]]],
        output_schema: Dict[str, Union[str, Type[FeatureProcessor]]],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> None:
        """Initializes the SampleDataset with samples and schemas.

        Args:
            samples (List[Dict]): List of data samples.
            input_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
                Schema for input data. Values can be string aliases or
                processor classes.
            output_schema (Dict[str, Union[str, Type[FeatureProcessor]]]):
                Schema for output data. Values can be string aliases or
                processor classes.
            dataset_name (Optional[str], optional): Name of the dataset.
                Defaults to None.
            task_name (Optional[str], optional): Name of the task.
                Defaults to None.
            input_processors (Optional[Dict[str, FeatureProcessor]],
                optional): Pre-fitted input processors. If provided, these
                will be used instead of creating new ones from input_schema.
                Defaults to None.
            output_processors (Optional[Dict[str, FeatureProcessor]],
                optional): Pre-fitted output processors. If provided, these
                will be used instead of creating new ones from output_schema.
                Defaults to None.
        """
        if dataset_name is None:
            dataset_name = ""
        if task_name is None:
            task_name = ""
        self.samples = samples
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_processors = input_processors if input_processors is not None else {}
        self.output_processors = (
            output_processors if output_processors is not None else {}
        )
        self.dataset_name = dataset_name
        self.task_name = task_name
        # Create patient_to_index and record_to_index mappings
        self.patient_to_index = {}
        self.record_to_index = {}

        for i, sample in enumerate(samples):
            # Create patient_to_index mapping
            patient_id = sample.get("patient_id")
            if patient_id is not None:
                if patient_id not in self.patient_to_index:
                    self.patient_to_index[patient_id] = []
                self.patient_to_index[patient_id].append(i)

            # Create record_to_index mapping (optional)
            record_id = sample.get("record_id", sample.get("visit_id"))
            if record_id is not None:
                if record_id not in self.record_to_index:
                    self.record_to_index[record_id] = []
                self.record_to_index[record_id].append(i)

        self.validate()
        self.build()

    def _get_processor_instance(self, processor_spec):
        """Get processor instance from either string alias or class reference.

        Args:
            processor_spec: Either a string alias or a processor class

        Returns:
            Instance of the processor
        """
        if isinstance(processor_spec, str):
            # Use existing registry system for string aliases
            return get_processor(processor_spec)()
        elif inspect.isclass(processor_spec) and issubclass(
            processor_spec, FeatureProcessor
        ):
            # Direct class reference
            return processor_spec()
        else:
            raise ValueError(
                f"Processor spec must be either a string alias or a "
                f"FeatureProcessor class, got {type(processor_spec)}"
            )

    def validate(self) -> None:
        """Validates that the samples match the input and output schemas."""
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for s in self.samples:
            assert input_keys.issubset(s.keys()), "Input schema does not match samples."
            assert output_keys.issubset(
                s.keys()
            ), "Output schema does not match samples."
        return

    def build(self) -> None:
        """Builds the processors for input and output data based on schemas."""
        # Only fit if processors weren't provided
        if not self.input_processors:
            for k, v in self.input_schema.items():
                self.input_processors[k] = self._get_processor_instance(v)
                self.input_processors[k].fit(self.samples, k)
        if not self.output_processors:
            for k, v in self.output_schema.items():
                self.output_processors[k] = self._get_processor_instance(v)
                self.output_processors[k].fit(self.samples, k)
        # Always process samples with the (fitted) processors
        for sample in tqdm(self.samples, desc="Processing samples"):
            for k, v in sample.items():
                if k in self.input_processors:
                    sample[k] = self.input_processors[k].process(v)
                elif k in self.output_processors:
                    sample[k] = self.output_processors[k].process(v)
        return

    def __getitem__(self, index: int) -> Dict:
        """Returns a sample by index.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            Dict: A dict with patient_id, visit_id/record_id, and other
            task-specific attributes as key. Conversion to index/tensor
            will be done in the model.
        """
        return self.samples[index]

    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            str: A string with the dataset and task names.
        """
        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def __len__(self) -> int:
        """Returns the number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.samples)


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
        """
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.dataset_name = dataset_name or ""
        self.task_name = task_name or ""
        self.dev = dev

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
        suffix = "_dev" if self.dev else ""

        self._sample_cache_path = (
            self.cache_dir
            / f"{self.dataset_name}_{self.task_name}_samples{suffix}.parquet"
        )

        # Track number of samples for __len__
        self._num_samples = 0
        self._samples_finalized = False

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
                "Cannot add more samples after finalize_samples() has been " "called"
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

        # Write to parquet (append mode)
        if self._num_samples == 0:
            # First write - create new file
            sample_df.write_parquet(
                self._sample_cache_path,
                compression="zstd",
            )
        else:
            # Append to existing file
            # Note: Polars doesn't have native append, so we read and rewrite
            existing = pl.read_parquet(self._sample_cache_path)
            combined = pl.concat([existing, sample_df], how="diagonal")
            combined.write_parquet(self._sample_cache_path, compression="zstd")

        self._num_samples += len(samples)
        logger.debug(f"Added {len(samples)} samples (total: {self._num_samples})")

    def finalize_samples(self) -> None:
        """Finalize sample writing and prepare for reading.

        Call this after all samples have been added via add_samples_streaming().
        """
        if self._samples_finalized:
            logger.warning("finalize_samples() called multiple times")
            return

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

        # Step 1: Create processor instances
        if not self.input_processors:
            for k, v in self.input_schema.items():
                self.input_processors[k] = self._get_processor_instance(v)
        if not self.output_processors:
            for k, v in self.output_schema.items():
                self.output_processors[k] = self._get_processor_instance(v)

        # Step 2: Fit processors by reading samples in batches
        logger.info("Fitting processors on samples...")
        batch_size = 1000
        lf = pl.scan_parquet(self._sample_cache_path)

        # Fit all processors
        all_processors = {**self.input_processors, **self.output_processors}
        for key, processor in all_processors.items():
            logger.debug(f"Fitting processor for key: {key}")
            num_batches = (self._num_samples + batch_size - 1) // batch_size
            for i in range(num_batches):
                batch = lf.slice(i * batch_size, batch_size).collect()
                batch_samples = batch.to_dicts()

                # Deserialize samples using helper function
                restored_samples = [
                    deserialize_sample_from_parquet(s) for s in batch_samples
                ]

                # Use fit method with batch
                processor.fit(restored_samples, key)

        logger.info(
            "Processors fitted! Processing will happen on-the-fly during iteration."
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
        lf = pl.scan_parquet(self._sample_cache_path)

        if worker_info is None:
            # Single worker - iterate all samples
            num_batches = (self._num_samples + batch_size - 1) // batch_size
            for batch_idx in range(num_batches):
                offset = batch_idx * batch_size
                length = min(batch_size, self._num_samples - offset)
                batch = lf.slice(offset, length).collect()
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
                    batch = lf.slice(offset, length).collect()
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
