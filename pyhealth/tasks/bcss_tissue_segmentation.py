"""Semantic tissue segmentation of BCSS pathology tiles."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import torch

from .base_task import BaseTask

BCSS_SEGMENTATION_CLASSES: tuple[str, ...] = (
    "other",
    "tumor",
    "stroma",
    "inflammatory_infiltrate",
    "necrosis",
)
"""The five region classes of the BCSS baseline, in class-index order."""

BCSS_IGNORE_INDEX = 255
"""Mask value the loss ignores (matches the ``UNet`` default)."""

BCSS_SEGMENTATION_GROUPS: dict[int, int] = {
    0: BCSS_IGNORE_INDEX,  # outside_roi: don't care
    1: 1,  # tumor
    2: 2,  # stroma
    3: 3,  # lymphocytic_infiltrate -> inflammatory
    4: 4,  # necrosis_or_debris
    5: 0,  # glandular_secretions
    6: 0,  # blood
    7: BCSS_IGNORE_INDEX,  # exclude
    8: 0,  # metaplasia_NOS
    9: 0,  # fat
    10: 3,  # plasma_cells -> inflammatory
    11: 3,  # other_immune_infiltrate -> inflammatory
    12: 0,  # mucoid_material
    13: 0,  # normal_acinus_or_duct
    14: 0,  # lymphatics
    15: 0,  # undetermined
    16: 0,  # nerve
    17: 0,  # skin_adnexa
    18: 0,  # blood_vessel
    19: 1,  # angioinvasion -> tumor
    20: 1,  # dcis -> tumor
    21: 0,  # other
}
"""Raw BCSS mask code to class index, following the BCSS baseline grouping
(https://bcsegmentation.grand-challenge.org/Baseline/): angioinvasion and
DCIS join tumor; lymphocytes, plasma cells and other immune infiltrates form
inflammatory infiltrate; the remaining annotated codes are "other". Codes
``0`` (outside ROI) and ``7`` (exclude) map to :data:`BCSS_IGNORE_INDEX`."""


class BCSSTissueSegmentation(BaseTask):
    """Five-class semantic segmentation of BCSS tiles (the BCSS benchmark).

    Works with :class:`~pyhealth.datasets.BCSSDataset`.

    The BCSS paper and challenge train a network to map every pixel to one
    of five region classes: tumor, stroma, inflammatory infiltrate,
    necrosis and other (:data:`BCSS_SEGMENTATION_CLASSES`), grouping rare
    codes as in :data:`BCSS_SEGMENTATION_GROUPS`. Outside-ROI pixels (and
    the ``exclude`` code) get :data:`BCSS_IGNORE_INDEX`, which
    :class:`~pyhealth.models.UNet` ignores by default. Class 0 is "other",
    so ``UNet``'s default ``include_background=False`` leaves it out of the
    Dice term but keeps it in cross-entropy.

    One sample per tile of the dataset's recipe. ``image`` is the RGB tile
    (``(3, H, W)``, values in ``[0, 1]``) and ``mask`` an integer tensor of
    shape ``(1, H, W)``. Build the dataset with every class and a low
    purity threshold so tiles are not filtered by their dominant class, for
    example ``BCSSDataset(root, class_codes=BCSS_CLASS_CODES,
    min_purity=0.3)``. Use
    :meth:`~pyhealth.datasets.BCSSDataset.official_split` for the official
    site-held-out test set.

    This is a tile-level version of the benchmark: the paper trains on
    random crops and reports pixel accuracy on whole ROIs, so the numbers
    are not directly comparable.

    Args:
        image_size: Side the tile (and, with nearest-neighbour resampling,
            the mask) is resized to. Use the dataset's ``tile_size`` to
            avoid resizing. Default 224.
        min_labelled: Skip tiles whose share of non-ignored pixels is below
            this value. Default 0.5.
        groups: Optional raw-code to class-index mapping replacing
            :data:`BCSS_SEGMENTATION_GROUPS`. Unlisted codes are ignored.
        num_classes: Number of classes ``groups`` maps to. Default 5.

    Raises:
        ValueError: If an argument is out of range.

    Examples:
        >>> from pyhealth.tasks import BCSSTissueSegmentation
        >>> task = BCSSTissueSegmentation(image_size=224)
        >>> task.num_classes
        5
        >>> task.output_schema["mask"][0]
        'tensor'
        >>> from pyhealth.datasets import BCSSDataset
        >>> from pyhealth.datasets.bcss import BCSS_CLASS_CODES
        >>> dataset = BCSSDataset(  # doctest: +SKIP
        ...     root="/data/bcss", class_codes=BCSS_CLASS_CODES, min_purity=0.3
        ... )
        >>> train, test = dataset.set_task(  # doctest: +SKIP
        ...     task, split=dataset.official_split()
        ... )
    """

    task_name: str = "BCSSTissueSegmentation"

    def __init__(
        self,
        image_size: int = 224,
        min_labelled: float = 0.5,
        groups: Mapping[int, int] | None = None,
        num_classes: int = len(BCSS_SEGMENTATION_CLASSES),
    ) -> None:
        if int(image_size) <= 0:
            raise ValueError("image_size must be positive.")
        if not 0.0 <= float(min_labelled) <= 1.0:
            raise ValueError("min_labelled must be in [0, 1].")
        groups = dict(BCSS_SEGMENTATION_GROUPS if groups is None else groups)
        bad = [
            v
            for v in groups.values()
            if v != BCSS_IGNORE_INDEX and not 0 <= v < int(num_classes)
        ]
        if bad:
            raise ValueError(f"groups maps to class indices outside num_classes: {bad}")
        super().__init__()
        self.image_size = int(image_size)
        self.min_labelled = float(min_labelled)
        self.num_classes = int(num_classes)
        self.groups = groups
        self._lut = np.full(256, BCSS_IGNORE_INDEX, dtype=np.int64)
        for code, index in groups.items():
            self._lut[int(code)] = int(index)
        self.input_schema = {
            "image": ("image", {"image_size": self.image_size, "mode": "RGB"})
        }
        self.output_schema = {"mask": ("tensor", {"dtype": torch.long})}

    def load_mask(self, mask_path: str) -> np.ndarray | None:
        """Read one raw mask tile and map it to class indices.

        Args:
            mask_path: Path of a ``*_mask.png`` tile written by the dataset.

        Returns:
            An ``int64`` array of shape ``(1, image_size, image_size)``, or
            ``None`` if fewer than ``min_labelled`` of its pixels are
            labelled.
        """
        from PIL import Image

        with Image.open(mask_path) as img:
            if img.size != (self.image_size, self.image_size):
                img = img.resize((self.image_size, self.image_size), Image.NEAREST)
            raw = np.asarray(img)
        if raw.ndim == 3:
            raw = raw[..., 0]
        mask = self._lut[raw]
        if (mask != BCSS_IGNORE_INDEX).mean() < self.min_labelled:
            return None
        return mask[None]

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per sufficiently labelled tile of the patient.

        Args:
            patient: A PyHealth ``Patient`` with ``bcss`` events.

        Returns:
            Samples with ``patient_id``, ``roi_id``, ``tile_id``, ``split``,
            ``image`` (the tile path) and ``mask`` (class indices).
        """
        samples = []
        for event in patient.get_events(event_type="bcss"):
            mask = self.load_mask(str(event.mask_path))
            if mask is None:
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "roi_id": event.roi_id,
                    "tile_id": event.tile_id,
                    "split": event.split,
                    "image": event.path,
                    "mask": mask,
                }
            )
        return samples
