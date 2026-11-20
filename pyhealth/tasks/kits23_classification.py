"""KiTS23 benign vs malignant kidney tumour classification."""

from __future__ import annotations

import logging
from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

logger = logging.getLogger(__name__)

#: Label of the tumour class in KiTS23 segmentations (1 kidney, 3 cyst).
KITS23_TUMOR_LABEL = 2


def _labelled_cases(patient: Any, need_segmentation: bool = False) -> list[Any]:
    """Return a patient's case events with imaging and a known label."""
    events = []
    for event in patient.get_events(event_type="cases"):
        if int(float(event.has_imaging)) != 1:
            continue
        if int(float(event.malignant)) not in (0, 1):
            continue
        if need_segmentation and not (
            isinstance(event.segmentation_path, str) and event.segmentation_path
        ):
            continue
        events.append(event)
    return events


class KiTS23MalignancyClassification(BaseTask):
    """Classify a KiTS23 CT as a benign or malignant kidney tumour (3D).

    Works with :class:`~pyhealth.datasets.KiTS23Dataset`.

    One sample per case that has imaging and a known ``malignant`` label.
    The CT is read lazily by the ``nifti`` processor, reoriented to RAS,
    resampled to ``target_shape`` ``(D, H, W)``, clipped to ``hu_window``
    and min-max scaled, giving a ``(1, D, H, W)`` tensor for 3D models.
    Only 9% of cases are benign; report AUROC / AUPRC. PyHealth's
    ``split_by_patient`` is not stratified, so on a small subset a split
    may contain one class only, which makes ROC AUC undefined. For 2D
    models, see
    :class:`KiTS23MalignancySliceClassification`.

    Args:
        target_shape: Output volume shape ``(D, H, W)``. Defaults to
            ``(64, 128, 128)``.
        hu_window: ``(low, high)`` HU clipping range. Defaults to
            ``(-200, 300)``, a soft-tissue window.

    Examples:
        >>> from pyhealth.datasets import KiTS23Dataset
        >>> from pyhealth.tasks import KiTS23MalignancyClassification
        >>> task = KiTS23MalignancyClassification(target_shape=(32, 64, 64))
        >>> task.input_schema["image"][1]["target_shape"]
        (32, 64, 64)
        >>> dataset = KiTS23Dataset(root="/data/kits23/dataset")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "KiTS23MalignancyClassification"
    input_schema: ClassVar[dict[str, Any]] = {
        "image": (
            "nifti",
            {
                "lazy": True,
                "target_shape": (64, 128, 128),
                "clip": (-200.0, 300.0),
                "normalize": "minmax",
            },
        )
    }
    output_schema: ClassVar[dict[str, Any]] = {"malignant": "binary"}

    def __init__(
        self,
        target_shape: tuple[int, int, int] = (64, 128, 128),
        hu_window: tuple[float, float] = (-200.0, 300.0),
    ) -> None:
        super().__init__()
        self.target_shape = tuple(int(v) for v in target_shape)
        self.hu_window = tuple(float(v) for v in hu_window)
        if len(self.target_shape) != 3:
            raise ValueError("target_shape must be (D, H, W).")
        if self.hu_window[1] <= self.hu_window[0]:
            raise ValueError("hu_window must be (low, high) with high > low.")
        self.input_schema = {
            "image": (
                "nifti",
                {
                    "lazy": True,
                    "target_shape": self.target_shape,
                    "clip": self.hu_window,
                    "normalize": "minmax",
                },
            )
        }

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per labelled case with imaging.

        Args:
            patient: A PyHealth ``Patient`` with ``cases`` events.

        Returns:
            Sample dicts with ``image`` (a NIfTI path) and ``malignant``.
        """
        return [
            {
                "patient_id": patient.patient_id,
                "record_id": event.record_id,
                "image": event.image_path,
                "malignant": int(float(event.malignant)),
            }
            for event in _labelled_cases(patient)
        ]


class KiTS23MalignancySliceClassification(BaseTask):
    """Classify KiTS23 tumours as benign or malignant from axial slices (2.5D).

    Works with :class:`~pyhealth.datasets.KiTS23Dataset`.

    For 2D models such as :class:`~pyhealth.models.TorchvisionModel`. The
    task reads the CT and its segmentation, finds the axial slice with the
    largest tumour cross-section (label 2; the largest kidney slice when a
    case has no tumour voxels), and stacks it with its two neighbours as a
    ``(3, image_size, image_size)`` float32 array, clipped to ``hu_window`` and
    scaled to [0, 1]. Reading full volumes makes ``set_task`` slow on the
    whole dataset (about 42 GB of CT), but the result is cached.

    The slice is chosen from the reference segmentation, so the input tells
    the model where the tumour is. That is fine for studying tumour
    characterisation, not for a detection pipeline.

    Args:
        image_size: Edge length of the output slices. Defaults to 224.
        hu_window: ``(low, high)`` HU clipping range. Defaults to
            ``(-200, 300)``.

    Examples:
        >>> from pyhealth.tasks import KiTS23MalignancySliceClassification
        >>> task = KiTS23MalignancySliceClassification(image_size=128)
        >>> task.output_schema
        {'malignant': 'binary'}
    """

    task_name: str = "KiTS23MalignancySliceClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {"malignant": "binary"}

    def __init__(
        self,
        image_size: int = 224,
        hu_window: tuple[float, float] = (-200.0, 300.0),
    ) -> None:
        super().__init__()
        self.image_size = int(image_size)
        self.hu_window = tuple(float(v) for v in hu_window)
        if self.hu_window[1] <= self.hu_window[0]:
            raise ValueError("hu_window must be (low, high) with high > low.")

    @staticmethod
    def best_slice(segmentation: np.ndarray) -> int:
        """Return the axial index with the most tumour (else kidney) voxels.

        Args:
            segmentation: Label volume ``(D, H, W)`` with ``D`` axial.

        Examples:
            >>> import numpy as np
            >>> seg = np.zeros((5, 4, 4), dtype=int)
            >>> seg[3, :2, :2] = 2
            >>> KiTS23MalignancySliceClassification.best_slice(seg)
            3
        """
        for label in (KITS23_TUMOR_LABEL, 1):
            area = (segmentation == label).reshape(segmentation.shape[0], -1).sum(1)
            if area.max() > 0:
                return int(area.argmax())
        return segmentation.shape[0] // 2

    def slices(self, volume: np.ndarray, segmentation: np.ndarray) -> np.ndarray:
        """Cut, window and resize the three slices around the tumour.

        Args:
            volume: CT volume ``(D, H, W)`` in HU, ``D`` axial.
            segmentation: Label volume of the same shape.

        Returns:
            A float32 array ``(3, image_size, image_size)`` in [0, 1].

        Examples:
            >>> import numpy as np
            >>> task = KiTS23MalignancySliceClassification(image_size=8)
            >>> volume = np.full((4, 16, 16), 300.0)
            >>> out = task.slices(volume, np.zeros((4, 16, 16), dtype=int))
            >>> out.shape, float(out.max())
            ((3, 8, 8), 1.0)
        """
        import torch
        import torch.nn.functional as F

        centre = self.best_slice(segmentation)
        last = volume.shape[0] - 1
        index = [min(max(centre + step, 0), last) for step in (-1, 0, 1)]
        low, high = self.hu_window
        stack = (np.clip(volume[index].astype(np.float32), low, high) - low) / (
            high - low
        )
        resized = F.interpolate(
            torch.from_numpy(stack)[None],
            size=(self.image_size, self.image_size),
            mode="bilinear",
            align_corners=False,
        )[0]
        return resized.numpy().astype(np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one 2.5D sample per labelled case with imaging and a mask.

        Args:
            patient: A PyHealth ``Patient`` with ``cases`` events.

        Returns:
            Sample dicts with ``image`` (a float32 numpy array of shape
            ``(3, image_size, image_size)``) and ``malignant``. Cases whose
            CT and segmentation shapes differ are logged and skipped.
        """
        from ..processors.image_io.readers.nifti import NiftiReader

        reader = NiftiReader()
        samples = []
        for event in _labelled_cases(patient, need_segmentation=True):
            volume = reader.load(event.image_path, canonical=True)[0]
            segmentation = reader.load(
                event.segmentation_path, canonical=True, is_mask=True
            )[0]
            if volume.shape != segmentation.shape:
                logger.warning(
                    "Skipping KiTS23 case %s: imaging shape %s differs from "
                    "segmentation shape %s.",
                    event.record_id,
                    volume.shape,
                    segmentation.shape,
                )
                continue
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "image": self.slices(volume, segmentation),
                    "malignant": int(float(event.malignant)),
                }
            )
        return samples
