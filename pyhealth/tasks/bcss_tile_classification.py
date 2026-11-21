"""Tissue-class classification of BCSS pathology tiles."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .base_task import BaseTask

BCSS_DEFAULT_CLASSES: tuple[str, ...] = (
    "tumor",
    "stroma",
    "lymphocytic_infiltrate",
    "necrosis_or_debris",
)
"""The four main BCSS tissue classes, in label-index order."""


class BCSSTileClassification(BaseTask):
    """Classify the dominant tissue class of a BCSS pathology tile.

    Works with :class:`~pyhealth.datasets.BCSSDataset`.

    One sample per tile kept by the dataset's tiling recipe (tile size,
    stride, purity threshold; see :class:`~pyhealth.datasets.BCSSDataset`).
    The label is the tile's dominant mask class name. The class list is
    pinned, so every split shares the same label indices. Tiles whose class
    is not in ``classes`` are skipped, which lets a task use a subset of the
    dataset's classes.

    Use the dataset's :meth:`~pyhealth.datasets.BCSSDataset.official_split`
    with ``set_task(..., split=...)`` for the official site-held-out test
    set. Never split BCSS tiles at random: tiles of one ROI are highly
    correlated.

    Args:
        classes: Class names in label-index order. Defaults to tumor,
            stroma, lymphocytic infiltrate and necrosis/debris.
        image_size: Side the tile is resized to. The default 224 matches
            the dataset's default ``tile_size``, so no resizing happens.

    Raises:
        ValueError: If ``classes`` is empty or has duplicates.

    Examples:
        >>> from pyhealth.tasks import BCSSTileClassification
        >>> task = BCSSTileClassification(classes=["tumor", "stroma"])
        >>> task.output_schema["label"]
        ('multiclass', {'labels': ['tumor', 'stroma']})
        >>> from pyhealth.datasets import BCSSDataset
        >>> dataset = BCSSDataset(root="/data/bcss")  # doctest: +SKIP
        >>> train, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=dataset.official_split()
        ... )
    """

    task_name: str = "BCSSTileClassification"

    def __init__(
        self,
        classes: Sequence[str] = BCSS_DEFAULT_CLASSES,
        image_size: int = 224,
    ) -> None:
        classes = [str(c) for c in classes]
        if not classes or len(set(classes)) != len(classes):
            raise ValueError("classes must be non-empty and without duplicates.")
        super().__init__()
        self.classes = classes
        self.image_size = int(image_size)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {"label": ("multiclass", {"labels": list(classes)})}

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per tile of the patient's ROI.

        Args:
            patient: A PyHealth ``Patient`` with ``bcss`` events.

        Returns:
            Samples with ``patient_id``, ``roi_id``, ``tile_id``, ``split``,
            ``image`` (the tile path) and ``label`` (a class name).
        """
        samples = []
        for event in patient.get_events(event_type="bcss"):
            if event.label not in self.classes:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "roi_id": event.roi_id,
                    "tile_id": event.tile_id,
                    "split": event.split,
                    "image": event.path,
                    "label": event.label,
                }
            )
        return samples
