from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image

# Import the TaskTemplate from the PyHealth package.
from pyhealth.tasks.task_template import TaskTemplate


@dataclass(frozen=True)
class ChestXrayClassificationTask(TaskTemplate):
    """Task for Chest X-ray Classification.

    This task processes a sample from the NIHChestXrayDataset by loading the image
    from the sample's "image_path" and assigning a dummy classification label. In
    practice, labels would be generated from provided annotations.

    Example:
        >>> sample = {"image_path": "/tmp/nih_chestxray/database/train/image00001.jpg"}
        >>> task = ChestXrayClassificationTask()
        >>> processed_samples = task(sample)
        >>> print(processed_samples)
    """
    task_name: str = "ChestXrayClassification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"image": "image"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "int"})

    def __call__(self, sample: Dict) -> List[Dict]:
        """Convert a sample from the dataset into a task-specific sample.

        This function performs the following:
          1. Loads an image from the provided file path in the sample.
          2. Assigns a dummy label (0) to the sample.
          3. Returns a list with a single dictionary, representing the processed sample.

        Args:
            sample (Dict): Dictionary representing one sample from the dataset.
                - Expected key: "image_path" (str) that points to a JPEG image file.

        Returns:
            List[Dict]: A list containing a single dictionary with two keys:
                - "image": A PIL.Image.Image object of the loaded image.
                - "label": An int representing the classification label, set to 0.

        Example:
            >>> sample = {"image_path": "/tmp/nih_chestxray/database/train/image00001.jpg"}
            >>> task = ChestXrayClassificationTask()
            >>> processed_sample = task(sample)
            >>> print(processed_sample)
        """
        image_path = sample.get("image_path")
        if not image_path:
            raise ValueError("The sample must contain an 'image_path' key.")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as err:
            raise ValueError(f"Error loading image from {image_path}: {err}")

        # For demonstration purposes, assign a dummy label (e.g., 0 represents 'normal')
        label = 0

        processed_sample = {"image": image, "label": label}
        return [processed_sample]


if __name__ == "__main__":
    # Example test case for the ChestXrayClassificationTask.
    example_sample = {
        "image_path": "/tmp/nih_chestxray/database/train/image00001.jpg"
    }
    task = ChestXrayClassificationTask()
    processed_samples = task(example_sample)
    print("Processed sample:", processed_samples)
