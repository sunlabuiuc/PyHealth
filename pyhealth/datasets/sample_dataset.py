import pickle
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, Type, override
import inspect

from torch.utils.data import Dataset
from tqdm import tqdm
from litdata import StreamingDataset

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

    def transform(self, sample: dict[str, bytes]) -> Dict[str, Any]:
        """Transform a pickled sample using the fitted processors."""
        if not self._fitted:
            raise RuntimeError("SampleBuilder.fit must be called before transform().")

        transformed: Dict[str, Any] = {}
        for key, value in pickle.loads(sample["sample"]).items():
            if key in self._input_processors:
                transformed[key] = self._input_processors[key].process(value)
            elif key in self._output_processors:
                transformed[key] = self._output_processors[key].process(value)
            else:
                transformed[key] = value
        return transformed

    def save(self, path: str) -> None:
        """Saves the fitted metadata to the specified path."""
        if not self._fitted:
            raise RuntimeError("SampleBuilder.fit must be called before save().")
        metadata = {
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "input_processors": self._input_processors,
            "output_processors": self._output_processors,
            "patient_to_index": self._patient_to_index,
            "record_to_index": self._record_to_index,
        }
        with open(path, "wb") as f:
            pickle.dump(metadata, f)


class SampleDataset(StreamingDataset):
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
        path: str,
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
        **kwargs,
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
        super().__init__(path, **kwargs)

        self.dataset_name = "" if dataset_name is None else dataset_name
        self.task_name = "" if task_name is None else task_name

        with open(f"{path}/schema.pkl", "rb") as f:
            metadata = pickle.load(f)

        self.input_schema = metadata["input_schema"]
        self.output_schema = metadata["output_schema"]
        self.input_processors = metadata["input_processors"]
        self.output_processors = metadata["output_processors"]

        self.patient_to_index = metadata["patient_to_index"]
        self.record_to_index = metadata["record_to_index"]

    @override
    def __str__(self) -> str:
        """Returns a string representation of the dataset.

        Returns:
            str: A string with the dataset and task names.
        """
        return f"Sample dataset {self.dataset_name} {self.task_name}"
