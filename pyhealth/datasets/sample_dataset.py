from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union, Type
import inspect

from torch.utils.data import Dataset
from tqdm import tqdm

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor


class SampleBuilder:
    """Utility to fit processors and transform samples without materializing a Dataset."""

    def __init__(
        self,
        input_schema: Dict[str, Any],
        output_schema: Dict[str, Any],
        input_processors: Optional[Dict[str, FeatureProcessor]] = None,
        output_processors: Optional[Dict[str, FeatureProcessor]] = None,
    ) -> None:
        self.input_schema = input_schema
        self.output_schema = output_schema
        self._input_processors = (
            input_processors if input_processors is not None else {}
        )
        self._output_processors = (
            output_processors if output_processors is not None else {}
        )
        self._patient_to_index: Dict[str, List[int]] = {}
        self._record_to_index: Dict[str, List[int]] = {}
        self._fitted = False

    @property
    def input_processors(self) -> Dict[str, FeatureProcessor]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing input_processors."
            )
        return self._input_processors

    @property
    def output_processors(self) -> Dict[str, FeatureProcessor]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing output_processors."
            )
        return self._output_processors

    @property
    def patient_to_index(self) -> Dict[str, List[int]]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing patient_to_index."
            )
        return self._patient_to_index

    @property
    def record_to_index(self) -> Dict[str, List[int]]:
        if not self._fitted:
            raise RuntimeError(
                "SampleBuilder.fit must be called before accessing record_to_index."
            )
        return self._record_to_index

    def _get_processor_instance(self, processor_spec):
        """Instantiate a processor using the same resolution logic as SampleDataset."""
        if isinstance(processor_spec, tuple):
            spec, kwargs = processor_spec
            if isinstance(spec, str):
                return get_processor(spec)(**kwargs)
            if inspect.isclass(spec) and issubclass(spec, FeatureProcessor):
                return spec(**kwargs)
            raise ValueError(
                "Processor spec in tuple must be either a string alias or a "
                f"FeatureProcessor class, got {type(spec)}"
            )
        if isinstance(processor_spec, str):
            return get_processor(processor_spec)()
        if inspect.isclass(processor_spec) and issubclass(
            processor_spec, FeatureProcessor
        ):
            return processor_spec()
        if isinstance(processor_spec, FeatureProcessor):
            return processor_spec
        raise ValueError(
            "Processor spec must be either a string alias, a FeatureProcessor "
            f"class, or a tuple (spec, kwargs_dict), got {type(processor_spec)}"
        )

    def fit(
        self,
        samples: Iterable[Dict[str, Any]],
    ) -> None:
        """Fit processors and build index mappings from an iterator of samples."""
        # Validate the samples
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for sample in samples:
            assert input_keys.issubset(
                sample.keys()
            ), "Input schema does not match samples."
            assert output_keys.issubset(
                sample.keys()
            ), "Output schema does not match samples."

        # Build index mappings
        self._patient_to_index = {}
        self._record_to_index = {}
        for i, sample in enumerate(samples):
            patient_id = sample.get("patient_id")
            if patient_id is not None:
                self._patient_to_index.setdefault(patient_id, []).append(i)
            record_id = sample.get("record_id", sample.get("visit_id"))
            if record_id is not None:
                self._record_to_index.setdefault(record_id, []).append(i)

        # Fit processors if they were not provided
        if not self._input_processors:
            for key, spec in self.input_schema.items():
                processor = self._get_processor_instance(spec)
                processor.fit(samples, key)
                self._input_processors[key] = processor
        if not self._output_processors:
            for key, spec in self.output_schema.items():
                processor = self._get_processor_instance(spec)
                processor.fit(samples, key)
                self._output_processors[key] = processor

        self._fitted = True

    def transform(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample dictionary using the fitted processors."""
        if not self._fitted:
            raise RuntimeError("SampleBuilder.fit must be called before transform().")

        transformed: Dict[str, Any] = {}
        for key, value in sample.items():
            if key in self._input_processors:
                transformed[key] = self._input_processors[key].process(value)
            elif key in self._output_processors:
                transformed[key] = self._output_processors[key].process(value)
            else:
                transformed[key] = value
        return transformed


class SampleDataset(Dataset):
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
        samples: List[Dict],
        input_schema: Dict[
            str,
            Union[
                str,
                Type[FeatureProcessor],
                FeatureProcessor,
                Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]],
            ],
        ],
        output_schema: Dict[
            str,
            Union[
                str,
                Type[FeatureProcessor],
                FeatureProcessor,
                Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]],
            ],
        ],
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
