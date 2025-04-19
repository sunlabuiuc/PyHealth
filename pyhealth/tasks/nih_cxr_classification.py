import argparse
from dataclasses import dataclass, field
from typing import Dict, List
from PIL import Image
from torch.utils.data import DataLoader

from pyhealth.tasks.base_task import BaseTask


@dataclass(frozen=True)
class ChestXrayClassificationTask(BaseTask):
    """Classification task for NIH Chest X‑ray.

    This task processes a sample dictionary (from NIHChestXrayDataset.process()),
    loads the image at ``sample["image_path"]``, and returns it together with
    its true binary label.

    Example:
        >>> from pyhealth.datasets.nih_cxr import NIHChestXrayDataset
        >>> from pyhealth.tasks.nih_cxr_classification import ChestXrayClassificationTask
        >>> ds = NIHChestXrayDataset(root="/tmp/nih", split="training", download=False)
        >>> sample = ds.patients[0]
        >>> task = ChestXrayClassificationTask()
        >>> output = task(sample)
        >>> print(output)
        [{"image": <PIL.Image.Image ...>, "label": 0}]
    """
    task_name: str = "ChestXrayClassification"
    input_schema: Dict[str, str] = field(
        default_factory=lambda: {"image": "image"}
    )
    output_schema: Dict[str, str] = field(
        default_factory=lambda: {"label": "int"}
    )

    def __call__(self, sample: Dict) -> List[Dict]:
        """Load image and return it with its binary label.

        Args:
            sample (Dict): A dictionary representing one data point, with keys:
                - "image_path" (str): full path to the PNG image file.
                - "label" (int): binary label (0 = No Finding, 1 = Finding).

        Returns:
            List[Dict]: A single-item list where the dict has:
                - "image": a PIL.Image.Image object of the loaded image.
                - "label": an int (0 or 1) representing the ground-truth label.

        Raises:
            ValueError: If "image_path" or "label" is missing from sample, or
                if the image file cannot be opened.

        Example:
            >>> sample = {"image_path": "/tmp/nih/.../00000001_000.png", "label": 1}
            >>> task = ChestXrayClassificationTask()
            >>> result = task(sample)
        """
        image_path = sample.get("image_path")
        label = sample.get("label")
        if image_path is None or label is None:
            raise ValueError(
                "Sample must contain keys 'image_path' (str) and 'label' (int)."
            )

        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as err:
            raise ValueError(f"Error loading image from {image_path}: {err}")

        return [{"image": img, "label": label}]


if __name__ == "__main__":
    # Smoke‐test this task using the real dataset labels.
    from pyhealth.datasets.nih_cxr import NIHChestXrayDataset

    parser = argparse.ArgumentParser(
        description="Smoke test for ChestXrayClassificationTask"
    )
    parser.add_argument(
        "--root",
        type=str,
        default="/tmp/nih_chestxray",
        help="Root dir of NIH Chest X-ray data",
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
        help="Re-download & extract if set",
    )
    args = parser.parse_args()

    # Load dataset (with real labels)
    ds = NIHChestXrayDataset(
        root=args.root, split=args.split, transform=None, download=args.download
    )

    # Retrieve the first sample dict (contains 'image_path' and 'label')
    sample = ds.patients[0]

    # Process a single sample
    task = ChestXrayClassificationTask()
    output = task(sample)
    print("Processed output:", output)

    # Batch a few sample dicts via DataLoader and apply the task
    sample_list = list(ds.patients.values())
    loader = DataLoader(
        sample_list,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: x,  # return list of dicts un-collated
    )
    batch = next(iter(loader))
    processed_batch = [task(s)[0] for s in batch]
    print("Batch labels:", [p["label"] for p in processed_batch])
