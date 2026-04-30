from typing import Any, Dict, List, Tuple
import numpy as np

from pyhealth.tasks.base_task import BaseTask



class RetinaUNetDetectionTask(BaseTask):

    def __init__(self, min_area: int = 10) -> None:
        super().__init__()
        self.min_area = min_area

    def __call__(self, sample):
        """Required by BaseTask."""
        return self.process_sample(sample)

    def process_sample(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        image = sample["image"]
        mask = sample["mask"]

        boxes, labels = self._extract_instances(mask)

        # Preserve any upstream metadata (e.g. patient_id, slice_idx from
        # a dataset) so it travels with the detection targets.
        processed = {k: v for k, v in sample.items() if k not in {"image", "mask"}}
        processed.update({
            "image": image,
            "boxes": boxes,
            "labels": labels,
            "mask": mask,
        })
        return processed

    def _extract_instances(
        self, mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids != 0]  # remove background

        boxes: List[List[int]] = []
        labels: List[int] = []

        for instance_id in instance_ids:
            binary_mask = mask == instance_id

            if binary_mask.sum() < self.min_area:
                continue

            box = self._mask_to_bbox(binary_mask)

            boxes.append(box)
            labels.append(1)  # single-class (can extend later)

        if len(boxes) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
            )

        return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)

    def _mask_to_bbox(self, binary_mask: np.ndarray) -> List[int]:

        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)

        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]

        return [x_min, y_min, x_max, y_max]

    def collate_fn(
        self, batch: List[Dict[str, Any]]
    ) -> Dict[str, Any]:

        images = [item["image"] for item in batch]
        boxes = [item["boxes"] for item in batch]
        labels = [item["labels"] for item in batch]
        masks = [item["mask"] for item in batch]

        return {
            "images": images,
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
        }


