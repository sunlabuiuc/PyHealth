import argparse
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image

from pyhealth.tasks.base_task import BaseTask


@dataclass(frozen=True)
class ChestXrayClassificationTask(BaseTask):
    """Task for Chest X‑ray Classification.

    This task takes a sample dict (as produced by NIHChestXrayDataset.process()) with
    the key "image_path", loads that image, and assigns a dummy label of 0.

    Example:
        >>> from pyhealth.tasks.nih_cxr_classification import ChestXrayClassificationTask
        >>> sample = {"image_path": "/tmp/nih_chestxray/images_001/images/00000001_000.png"}
        >>> task = ChestXrayClassificationTask()
        >>> out = task(sample)
        >>> print(out)
        [{"image": <PIL.Image.Image image mode=RGB size=...>, "label": 0}]
    """
    task_name: str = "ChestXrayClassification"
    input_schema: Dict[str, str] = field(default_factory=lambda: {"image": "image"})
    output_schema: Dict[str, str] = field(default_factory=lambda: {"label": "int"})

    def __call__(self, sample: Dict) -> List[Dict]:
        """Load the image at `sample["image_path"]` and return it with a dummy label.

        Args:
            sample (Dict): Must contain:
                - "image_path" (str): full path to a .png image on disk

        Returns:
            List[Dict]: A single‐element list whose dict has:
                - "image": the loaded PIL.Image.Image
                - "label": int (0 for demonstration)

        Raises:
            ValueError: If "image_path" is missing or the file cannot be opened.
        """
        image_path = sample.get("image_path")
        if not image_path:
            raise ValueError("Sample must contain an 'image_path' key.")

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as err:
            raise ValueError(f"Error loading image from {image_path}: {err}")

        # Dummy label—replace with real logic if you have annotations
        label = 0
        return [{"image": img, "label": label}]


if __name__ == "__main__":
    # Smoke‐test this task against your NIHChestXrayDataset
    from pyhealth.datasets.nih_cxr import NIHChestXrayDataset

    parser = argparse.ArgumentParser(
        description="Smoke test for ChestXrayClassificationTask"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/tmp/nih_chestxray",
        help="Root directory where NIH CXR was downloaded & extracted",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="training",
        choices=["training", "test"],
        help="Which split to load",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="If set, will re-download & extract the dataset",
    )
    args = parser.parse_args()

    # Load the dataset
    ds = NIHChestXrayDataset(
        root=args.root, split=args.split, transform=None, download=args.download
    )

    # Grab the very first sample dict
    sample_dict = ds.patients[0]

    # Run it through our task
    task = ChestXrayClassificationTask()
    output = task(sample_dict)
    print("Processed output:", output)
