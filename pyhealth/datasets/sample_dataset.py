from typing import Any, Dict, Iterator, List, Optional, Tuple, Union, Type
from collections.abc import Sequence
from bisect import bisect_right
import inspect

from torch.utils.data import IterableDataset
from litdata.streaming import StreamingDataset
from litdata.utilities.train_test_split import deepcopy_dataset
import pickle

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor


class SampleDataset(IterableDataset):
    """Sample dataset class for handling and processing data samples.

    Attributes:
        samples (List[Dict]): List of data samples.
        input_schema (Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
            Schema for input data. Values can be string aliases, processor classes, processor instances, or tuples of (spec, kwargs_dict).
        output_schema (Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
            Schema for output data. Values can be string aliases, processor classes, processor instances, or tuples of (spec, kwargs_dict).
        dataset_name (Optional[str]): Name of the dataset.
        task_name (Optional[str]): Name of the task.
    """

    def __init__(
        self,
        dataset: StreamingDataset,
        input_schema: Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]],
        output_schema: Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> None:
        """Initializes the SampleDataset with samples and schemas.

        Args:
            samples (List[Dict]): List of data samples.
            input_schema (Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
                Schema for input data. Values can be string aliases, processor classes, processor instances, or tuples of (spec, kwargs_dict) for instantiation.
            output_schema (Dict[str, Union[str, Type[FeatureProcessor], FeatureProcessor, Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
                Schema for output data. Values can be string aliases, processor classes, processor instances, or tuples of (spec, kwargs_dict) for instantiation.
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
        self.dataset = dataset
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

        for i, sample in enumerate(iter(self.dataset)):
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

        # Apply processors
        self.dataset = StreamingDataset(
            input_dir=self.dataset.input_dir,
            cache_dir=self.dataset.cache_dir,
            transform=self.transform,
        )

    def _get_processor_instance(self, processor_spec):
        """Get processor instance from either string alias, class reference, processor instance, or tuple with kwargs.

        Args:
            processor_spec: Either a string alias, a processor class, a processor instance, or a tuple (spec, kwargs_dict)

        Returns:
            Instance of the processor
        """
        if isinstance(processor_spec, tuple):
            spec, kwargs = processor_spec
            if isinstance(spec, str):
                return get_processor(spec)(**kwargs)
            elif inspect.isclass(spec) and issubclass(spec, FeatureProcessor):
                return spec(**kwargs)
            else:
                raise ValueError(
                    f"Processor spec in tuple must be either a string alias or a "
                    f"FeatureProcessor class, got {type(spec)}"
                )
        if isinstance(processor_spec, str):
            # Use existing registry system for string aliases
            return get_processor(processor_spec)()
        elif inspect.isclass(processor_spec) and issubclass(
            processor_spec, FeatureProcessor
        ):
            # Direct class reference
            return processor_spec()
        elif isinstance(processor_spec, FeatureProcessor):
            # Already an instance
            return processor_spec
        else:
            raise ValueError(
                f"Processor spec must be either a string alias, a "
                f"FeatureProcessor class, or a tuple (spec, kwargs_dict), got {type(processor_spec)}"
            )

    def validate(self) -> None:
        """Validates that the samples match the input and output schemas."""
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for s in iter(self.dataset):
            assert input_keys.issubset(s.keys()), "Input schema does not match samples."
            assert output_keys.issubset(s.keys()), (
                "Output schema does not match samples."
            )
        return

    def build(self) -> None:
        """Builds the processors for input and output data based on schemas."""
        # Only fit if processors weren't provided
        if not self.input_processors:
            for k, v in self.input_schema.items():
                self.input_processors[k] = self._get_processor_instance(v)
                self.input_processors[k].fit(iter(self.dataset), k)
        if not self.output_processors:
            for k, v in self.output_schema.items():
                self.output_processors[k] = self._get_processor_instance(v)
                self.output_processors[k].fit(iter(self.dataset), k)
        return

    def transform(self, sample) -> Dict:
        """Applies the input and output processors to a sample."""
        for k, v in sample.items():
            if k in self.input_processors:
                sample[k] = self.input_processors[k].process(pickle.loads(v))
            elif k in self.output_processors:
                sample[k] = self.output_processors[k].process(pickle.loads(v))
            else:
                sample[k] = pickle.loads(v)
        return sample

    def set_shuffle(self, shuffle: bool) -> None:
        """Sets whether to shuffle the dataset during iteration.

        Args:
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset.set_shuffle(shuffle)
        return

    def __iter__(self) -> Iterator:
        """Returns an iterator over the dataset samples."""
        return self.dataset.__iter__()

    def __getitem__(self, index: int) -> Dict:
        """Gets a sample by index. Try to use iterator for better performance."""
        return self.dataset.__getitem__(index)

    def __str__(self) -> str:
        """String representation of the SampleDataset."""
        return f"Sample dataset {self.dataset_name} {self.task_name}"

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self.dataset.__len__()

class SampleSubset(IterableDataset):
    """A subset of the SampleDataset.

    Args:
        sample_dataset (SampleDataset): The original SampleDataset.
        indices (List[int]): List of indices to include in the subset.
    """

    def __init__(self, dataset: SampleDataset, indices: Sequence[int]) -> None:
        self.dataset_name = dataset.dataset_name
        self.task_name = dataset.task_name

        base_dataset: StreamingDataset = deepcopy_dataset(dataset.dataset)
        base_dataset.set_shuffle(False) # Disable shuffling when creating subset
        self.dataset, self._length = self._build_subset_dataset(
            base_dataset, indices
        )

    def _build_subset_dataset(
        self, base_dataset: StreamingDataset, indices: Sequence[int]
    ) -> Tuple[StreamingDataset, int]:
        """Create a StreamingDataset restricted to the provided indices."""
        if len(base_dataset.subsampled_files) != len(base_dataset.region_of_interest):
            raise ValueError(
                "The provided dataset has mismatched subsampled_files and region_of_interest lengths."
            )

        dataset_length = sum(
            end - start for start, end in base_dataset.region_of_interest
        )
        if any(idx < 0 or idx >= dataset_length for idx in indices):
            raise ValueError(
                f"Subset indices must be in [0, {dataset_length - 1}] for the provided dataset."
            )

        # Build chunk boundaries so we can translate global indices into
        # chunk-local (start, end) pairs that litdata understands.
        chunk_starts: List[int] = []
        chunk_boundaries: List[Tuple[str, int, int, int, int]] = []
        cursor = 0
        for filename, (roi_start, roi_end) in zip(
            base_dataset.subsampled_files, base_dataset.region_of_interest
        ):
            chunk_len = roi_end - roi_start
            if chunk_len <= 0:
                continue
            chunk_starts.append(cursor)
            chunk_boundaries.append(
                (filename, roi_start, roi_end, cursor, cursor + chunk_len)
            )
            cursor += chunk_len

        new_subsampled_files: List[str] = []
        new_roi: List[Tuple[int, int]] = []
        prev_chunk_idx: Optional[int] = None

        for idx in indices:
            chunk_idx = bisect_right(chunk_starts, idx) - 1
            if chunk_idx < 0 or idx >= chunk_boundaries[chunk_idx][4]:
                raise ValueError(f"Index {idx} is out of bounds for the dataset.")

            filename, roi_start, _, global_start, _ = chunk_boundaries[chunk_idx]
            offset_in_chunk = roi_start + (idx - global_start)

            if (
                new_roi
                and prev_chunk_idx == chunk_idx
                and offset_in_chunk == new_roi[-1][1]
            ):
                new_roi[-1] = (new_roi[-1][0], new_roi[-1][1] + 1)
            else:
                new_subsampled_files.append(filename)
                new_roi.append((offset_in_chunk, offset_in_chunk + 1))

            prev_chunk_idx = chunk_idx

        base_dataset.subsampled_files = new_subsampled_files
        base_dataset.region_of_interest = new_roi
        base_dataset.reset()
        subset_length = sum(end - start for start, end in new_roi)

        return base_dataset, subset_length
    
    def set_shuffle(self, shuffle: bool) -> None:
        """Sets whether to shuffle the dataset during iteration.

        Args:
            shuffle (bool): Whether to shuffle the dataset.
        """
        self.dataset.set_shuffle(shuffle)
        return

    def __iter__(self) -> Iterator:
        """Returns an iterator over the dataset samples."""
        return self.dataset.__iter__()

    def __getitem__(self, index: int) -> Dict:
        """Gets a sample by index. Try to use iterator for better performance."""
        return self.dataset.__getitem__(index)

    def __str__(self) -> str:
        """String representation of the SampleDataset."""
        return f"Sample dataset {self.dataset_name} {self.task_name} subset"

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return self._length
