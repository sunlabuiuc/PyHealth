import logging
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image
from torch.utils.data import DataLoader

from pyhealth.datasets.nih_cxr8 import NIHChestXray8Dataset
from pyhealth.tasks.base_task import BaseTask


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChestXrayImageClassificationTask(BaseTask):
    """Task for classifying X-ray images from the NIH ChestXray8 dataset.

    This task processes a sample (from NIHChestXray8Dataset.process()) and returns
    a tuple containing the image and its binary classification label.

    Attributes:
        task_name (str): Name of the task.
        input_fields (Dict[str, str]): Dictionary mapping input field names to types.
        output_fields (Dict[str, str]): Dictionary mapping output field names to types.
    """
    task_name: str = "ChestXrayImageClassification"
    input_fields: Dict[str, str] = field(default_factory=lambda: {"image": "image"})
    output_fields: Dict[str, str] = field(default_factory=lambda: {"label": "int"})

    def __call__(self, data: Dict) -> List[Dict]:
        """Load an image and return it along with its associated label.

        Args:
            data (Dict): A dictionary representing a sample, containing:
                - "image_path" (str): The path to the image file.
                - "label" (int): Binary label indicating the presence of a finding.

        Returns:
            List[Dict]: A list with one dictionary, where:
                - "image": A PIL.Image.Image object of the loaded image.
                - "label": The integer label (0 or 1).

        Raises:
            ValueError: If "image_path" or "label" is missing, or the image cannot be loaded.

        Example:
            >>> sample = {"image_path": "/tmp/CXR8/images/00000001_000.png", "label": 1}
            >>> task = ChestXrayImageClassificationTask()
            >>> result = task(sample)
        """
        image_path = data.get("image_path")
        label = data.get("label")

        if image_path is None or label is None:
            raise ValueError("The sample must contain 'image_path' (str) and 'label' (int).")

        image = self._load_image(image_path)
        return [{"image": image, "label": label}]

    @staticmethod
    def _load_image(image_path: str) -> Image:
        """Helper function to load an image from the specified path.

        Args:
            image_path (str): The path to the image file.

        Returns:
            PIL.Image.Image: The loaded image.

        Raises:
            ValueError: If the image cannot be loaded from the provided path.
        """
        try:
            return Image.open(image_path).convert("RGB")
        except Exception as error:
            logger.error(f"Error loading image from {image_path}: {error}")
            raise ValueError(f"Error loading image from {image_path}: {error}")


def load_dataset(root_directory: str, split_type: str) -> NIHChestXray8Dataset:
    """Function to load the NIH Chest X-ray dataset.

    Args:
        root_directory (str): The root directory where the dataset is stored.
        split_type (str): The dataset split to load (e.g., "training", "test").

    Returns:
        NIHChestXray8Dataset: The loaded dataset object.

    Raises:
        ValueError: If the dataset cannot be loaded.
    """
    logger.info(f"Loading dataset from {root_directory} with split type '{split_type}'")
    return NIHChestXray8Dataset(dataset_dir=root_directory, split=split_type)


def process_single_sample(task: ChestXrayImageClassificationTask, sample_data: Dict) -> None:
    """Process and output a single sample.

    Args:
        task (ChestXrayImageClassificationTask): The task that will process the sample.
        sample_data (Dict): A dictionary representing a sample, containing keys like 'image_path' and 'label'.
    """
    output_result = task(sample_data)
    logger.info(f"Processed output: {output_result}")


def process_batch_samples(task: ChestXrayImageClassificationTask, dataset: NIHChestXray8Dataset) -> None:
    """Process a batch of samples and print the results.

    Args:
        task (ChestXrayImageClassificationTask): The task that will process the batch of samples.
        dataset (NIHChestXray8Dataset): The dataset containing the samples to process.
    """
    sample_batch = list(dataset.samples.values())
    data_loader = DataLoader(
        sample_batch,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: batch
    )
    batch_data = next(iter(data_loader))
    processed_batch = [task(item)[0] for item in batch_data]
    logger.info(f"Processed batch labels: {[item['label'] for item in processed_batch]}")
