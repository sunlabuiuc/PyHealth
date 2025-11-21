import logging
from typing import Dict, Optional

from pyhealth.datasets.base_dataset import BaseDataset
from pyhealth.processors.image_processor import ImageProcessor
from pyhealth.tasks.base_task import BaseTask
from pyhealth.datasets.sample_dataset import SampleDataset

logger = logging.getLogger(__name__)


class BaseImageDataset(BaseDataset):
    """Base class for image datasets in PyHealth.

    This class provides common functionality for loading and processing image data,
    including default image processing setup.

    Args:
        root: Root directory of the raw data containing the dataset files.
        dataset_name: Optional name of the dataset. Defaults to "base_image".
        config_path: Optional path to the configuration file.

    Attributes:
        root: Root directory of the raw data.
        dataset_name: Name of the dataset.
        config_path: Path to the configuration file.
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Expand user path (e.g., ~/Downloads -> /home/user/Downloads)
        root = os.path.expanduser(root)

        super().__init__(
            root=root,
            dataset_name=dataset_name or "base_image",
            config_path=config_path,
            **kwargs,
        )

    def set_task(
        self,
        task: BaseTask | None = None,
        num_workers: int = 1,
        cache_dir: str | None = None,
        cache_format: str = "parquet",
        input_processors: Dict[str, any] | None = None,
        output_processors: Dict[str, any] | None = None,
    ) -> SampleDataset:
        """Set the task for the dataset with default image processing.

        If no image processor is provided, defaults to ImageProcessor with
        image_size=299 and mode="L" (grayscale).

        Args:
            task: The task to set.
            num_workers: Number of workers for processing.
            cache_dir: Directory for caching.
            cache_format: Format for caching.
            input_processors: Input processors.
            output_processors: Output processors.

        Returns:
            SampleDataset: The processed sample dataset.
        """
        if input_processors is None or "image" not in input_processors:
            image_processor = ImageProcessor(
                image_size=299,  # Default image size
                mode="L",  # Grayscale by default
            )
            if input_processors is None:
                input_processors = {}
            input_processors["image"] = image_processor

        return super().set_task(
            task,
            num_workers,
            cache_dir,
            cache_format,
            input_processors,
            output_processors,
        )