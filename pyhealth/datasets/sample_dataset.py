from typing import Dict, List, Optional, Union, Type
import inspect

from torch.utils.data import Dataset
from tqdm import tqdm

from ..processors import get_processor
from ..processors.base_processor import FeatureProcessor


class SampleDataset(Dataset):
    """Sample dataset class for handling and processing data samples.

    Attributes:
        samples (List[Dict]): List of data samples.
        input_schema (Dict[str, Union[str, Type[FeatureProcessor], Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
            Schema for input data. Values can be string aliases, processor classes, or tuples of (spec, kwargs_dict).
        output_schema (Dict[str, Union[str, Type[FeatureProcessor], Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
            Schema for output data. Values can be string aliases, processor classes, or tuples of (spec, kwargs_dict).
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
            input_schema (Dict[str, Union[str, Type[FeatureProcessor], Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
                Schema for input data. Values can be string aliases, processor classes, or tuples of (spec, kwargs_dict) for instantiation.
            output_schema (Dict[str, Union[str, Type[FeatureProcessor], Tuple[Union[str, Type[FeatureProcessor]], Dict[str, Any]]]]):
                Schema for output data. Values can be string aliases, processor classes, or tuples of (spec, kwargs_dict) for instantiation.
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
        """Get processor instance from either string alias, class reference, or tuple with kwargs.

        Args:
            processor_spec: Either a string alias or a processor class

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
