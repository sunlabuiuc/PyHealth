"""KiTS23 kidney, tumour and cyst segmentation (the challenge task)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from .base_task import BaseTask

#: Foreground class names of the KiTS23 masks, by label (0 is background).
KITS23_SEGMENTATION_CLASSES = ("kidney", "tumor", "cyst")


class KiTS23Segmentation(BaseTask):
    """Segment kidneys, kidney tumours and cysts in a KiTS23 CT (3D).

    Works with :class:`~pyhealth.datasets.KiTS23Dataset`.

    This is the KiTS23 challenge task. One sample per case that has both
    ``imaging.nii.gz`` and ``segmentation.nii.gz`` (the ``malignant`` label
    is not needed). The CT and the mask are read lazily by the ``nifti``
    processor, reoriented to RAS, and optionally resampled together to
    ``target_spacing`` or ``target_shape`` ``(D, H, W)``. The CT is clipped
    to ``hu_window`` and min-max scaled; the mask keeps the KiTS23 labels
    0 background, 1 kidney, 2 tumour, 3 cyst (``num_classes = 4``, nearest
    neighbour resampling). Pair it with :class:`~pyhealth.models.UNet3D`.

    The challenge ranks methods by Dice and surface Dice on three nested
    "hierarchical evaluation classes": kidney + tumour + cyst, tumour +
    cyst, and tumour. Per-label Dice from
    :func:`~pyhealth.models.evaluate_segmentation` is a simpler proxy; use
    the official ``kits23_compute_metrics`` for challenge numbers.

    Args:
        target_shape: Optional output shape ``(D, H, W)``. Full KiTS23 CTs
            are 512 x 512 in-plane with up to about 1,000 slices, so set this
            or ``target_spacing`` (or train on patches) to fit in memory.
        target_spacing: Optional output spacing in mm, ``(D, H, W)`` order.
        hu_window: ``(low, high)`` HU clipping range. Defaults to
            ``(-200, 300)``.

    Examples:
        >>> from pyhealth.tasks import KiTS23Segmentation
        >>> task = KiTS23Segmentation(target_shape=(64, 128, 128))
        >>> task.output_schema["mask"][1]["num_classes"]
        4
        >>> from pyhealth.datasets import KiTS23Dataset
        >>> dataset = KiTS23Dataset(root="/data/kits23/dataset")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "KiTS23Segmentation"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": (
            "nifti",
            {"lazy": True, "clip": (-200.0, 300.0), "normalize": "minmax"},
        )
    }
    output_schema: ClassVar[dict[str, Any]] = {
        "mask": (
            "nifti",
            {
                "lazy": True,
                "is_mask": True,
                "num_classes": len(KITS23_SEGMENTATION_CLASSES) + 1,
            },
        )
    }

    def __init__(
        self,
        target_shape: Sequence[int] | None = None,
        target_spacing: Sequence[float] | None = None,
        hu_window: tuple[float, float] = (-200.0, 300.0),
    ) -> None:
        super().__init__()
        self.hu_window = tuple(float(v) for v in hu_window)
        if self.hu_window[1] <= self.hu_window[0]:
            raise ValueError("hu_window must be (low, high) with high > low.")
        resample: dict[str, Any] = {}
        if target_shape is not None:
            shape = tuple(int(v) for v in target_shape)
            if len(shape) != 3:
                raise ValueError("target_shape must be (D, H, W).")
            resample["target_shape"] = shape
        if target_spacing is not None:
            spacing = tuple(float(v) for v in target_spacing)
            if len(spacing) != 3:
                raise ValueError("target_spacing must be (D, H, W).")
            resample["target_spacing"] = spacing
        self.input_schema = {
            "image": (
                "nifti",
                {
                    "lazy": True,
                    "clip": self.hu_window,
                    "normalize": "minmax",
                    **resample,
                },
            )
        }
        self.output_schema = {
            "mask": (
                "nifti",
                {
                    "lazy": True,
                    "is_mask": True,
                    "num_classes": len(KITS23_SEGMENTATION_CLASSES) + 1,
                    **resample,
                },
            )
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per case with a CT and a segmentation.

        Args:
            patient: A PyHealth ``Patient`` with ``cases`` events.

        Returns:
            Sample dicts with ``image`` and ``mask`` NIfTI paths.
        """
        samples = []
        for event in patient.get_events(event_type="cases"):
            image = event.image_path
            mask = event.segmentation_path
            if not (isinstance(image, str) and image):
                continue
            if not (isinstance(mask, str) and mask):
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": image,
                    "mask": mask,
                }
            )
        return samples
