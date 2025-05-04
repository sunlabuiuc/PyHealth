from typing import Dict, List, Optional

from torch.utils.data import Dataset
from tqdm import tqdm

from ..processors import get_processor


class SampleDataset(Dataset):
    """Sample dataset class for handling and processing data samples.

    Attributes:
        samples (List[Dict]): List of data samples.
        input_schema (Dict[str, str]): Schema for input data.
        output_schema (Dict[str, str]): Schema for output data.
        dataset_name (Optional[str]): Name of the dataset.
        task_name (Optional[str]): Name of the task.
    """

    def __init__(
        self,
        samples: List[Dict],
        input_schema: Dict[str, str],
        output_schema: Dict[str, str],
        dataset_name: Optional[str] = None,
        task_name: Optional[str] = None,
    ) -> None:
        """Initializes the SampleDataset with samples and schemas.

        Args:
            samples (List[Dict]): List of data samples.
            input_schema (Dict[str, str]): Schema for input data.
            output_schema (Dict[str, str]): Schema for output data.
            dataset_name (Optional[str], optional): Name of the dataset. Defaults to None.
            task_name (Optional[str], optional): Name of the task. Defaults to None.
        """
        if dataset_name is None:
            dataset_name = ""
        if task_name is None:
            task_name = ""
        self.samples = samples
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.input_processors = {}
        self.output_processors = {}
        self.dataset_name = dataset_name
        self.task_name = task_name
        # Create patient_to_index and record_to_index mappings
        self.patient_to_index = {}
        self.record_to_index = {}
        
        for i, sample in enumerate(samples):
            # Create patient_to_index mapping
            patient_id = sample.get('patient_id')
            if patient_id is not None:
                if patient_id not in self.patient_to_index:
                    self.patient_to_index[patient_id] = []
                self.patient_to_index[patient_id].append(i)
            
            # Create record_to_index mapping (optional)
            record_id = sample.get('record_id', sample.get('visit_id'))
            if record_id is not None:
                if record_id not in self.record_to_index:
                    self.record_to_index[record_id] = []
                self.record_to_index[record_id].append(i)
                
        self.validate()
        self.build()

    def validate(self) -> None:
        """Validates that the samples match the input and output schemas."""
        input_keys = set(self.input_schema.keys())
        output_keys = set(self.output_schema.keys())
        for s in self.samples:
            assert input_keys.issubset(s.keys()), \
                "Input schema does not match samples."
            assert output_keys.issubset(s.keys()), \
                "Output schema does not match samples."
        return

    def build(self) -> None:
        """Builds the processors for input and output data based on schemas."""
        for k, v in self.input_schema.items():
            self.input_processors[k] = get_processor(v)()
            self.input_processors[k].fit(self.samples, k)
        for k, v in self.output_schema.items():
            self.output_processors[k] = get_processor(v)()
            self.output_processors[k].fit(self.samples, k)
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
            Dict: A dict with patient_id, visit_id/record_id, and other task-specific
            attributes as key. Conversion to index/tensor will be done
            in the model.
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
