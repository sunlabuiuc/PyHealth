# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
"""Benign vs. malignant classification of PH2 images under altered views.

Turns each PH2 image into a model input under one of the paper's 12 dataset
modes, optionally with a hand-drawn artifact added.
"""

import zlib
from typing import Any, ClassVar

import numpy as np
from PIL import Image

from .base_task import BaseTask
from .dermoscopy_artifact_views import ARTIFACTS, VIEWS, add_artifact, make_view

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class PH2ArtifactClassification(BaseTask):
    """Classify PH2 dermoscopy images as benign or malignant, under a chosen view.

    Works with :class:`~pyhealth.datasets.PH2Dataset`. Follows the setup of
    Jin and Ghassemi, "A Study of Artifacts on Melanoma Classification under
    Diffusion-Based Perturbations" (CHIL 2025,
    https://proceedings.mlr.press/v287/jin25b.html).

    One sample per image. The label ``malignant`` is 1 for melanoma and 0 for
    common or atypical nevus. The input is built in three steps, each of which
    can be switched on to ask a different question about what the classifier
    uses:

    1. The image and its lesion mask are resized to ``image_size`` pixels.
    2. If ``artifact`` is set, that artifact is hand-drawn on the image, never
       on a lesion pixel (see
       :func:`~pyhealth.tasks.dermoscopy_artifact_views.add_artifact`). Test a
       trained classifier on ``artifact="ruler"`` and compare with
       ``artifact=None`` to measure its sensitivity to rulers, without a
       diffusion model.
    3. The ``view`` is applied (see
       :func:`~pyhealth.tasks.dermoscopy_artifact_views.make_view`): one of the
       paper's twelve dataset modes ``whole``, ``lesion``, ``background``,
       ``bbox``, ``bbox70``, ``bbox90``, and ``whole``/``lesion``/
       ``background`` with ``_high`` or ``_low`` for a high- or low-pass filter.

    The result is normalised with the ImageNet mean and standard deviation
    (as in the paper) and stored as a ``(3, image_size, image_size)`` float32
    tensor, about 600 KB per image at 224 px, 120 MB for the 200 images.

    Args:
        view: Dataset mode, one of ``VIEWS``. Defaults to ``"whole"``.
        artifact: None, or an artifact to draw, one of ``ARTIFACTS``
            (``dark_corner``, ``gel_bubble``, ``ink``, ``patches``,
            ``ruler``). Defaults to None.
        image_size: Side length of the square input image. Defaults to 224.
        seed: Seed for the artifact placement. The same image and seed always
            give the same artifact. Defaults to 0.

    Raises:
        ValueError: If ``view``, ``artifact`` or ``image_size`` is invalid.

    Examples:
        >>> from pyhealth.datasets import PH2Dataset
        >>> from pyhealth.tasks import PH2ArtifactClassification
        >>> dataset = PH2Dataset(root="/path/to/PH2Dataset")  # doctest: +SKIP
        >>> task = PH2ArtifactClassification(view="bbox70", artifact="ink")
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
        >>> task.output_schema
        {'malignant': 'binary'}
    """

    task_name: str = "PH2ArtifactClassification"
    input_schema: ClassVar[dict[str, Any]] = {"image": "tensor"}
    output_schema: ClassVar[dict[str, str]] = {"malignant": "binary"}

    def __init__(
        self,
        view: str = "whole",
        artifact: str | None = None,
        image_size: int = 224,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if view not in VIEWS:
            raise ValueError(f"view must be one of {VIEWS}, got {view!r}.")
        if artifact is not None and artifact not in ARTIFACTS:
            raise ValueError(f"artifact must be None or one of {ARTIFACTS}.")
        if image_size < 8:
            raise ValueError(f"image_size must be at least 8, got {image_size}.")
        self.view = view
        self.artifact = artifact
        self.image_size = int(image_size)
        self.seed = int(seed)

    def _load(self, image_path: str, mask_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Load and resize an image to float ``(H, W, 3)`` and a bool mask.

        Called by :meth:`render`. The image is resized bilinearly and the mask
        with nearest neighbour, so the mask stays binary.

        Args:
            image_path: Path to the dermoscopic image.
            mask_path: Path to the binary lesion mask.

        Returns:
            The image as float32 in ``[0, 1]``, shape ``(image_size, image_size,
            3)``, and the lesion mask as a boolean array of shape
            ``(image_size, image_size)``.
        """
        size = (self.image_size, self.image_size)
        with Image.open(image_path) as img:
            image = np.asarray(img.convert("RGB").resize(size, Image.BILINEAR))
        with Image.open(mask_path) as msk:
            mask = np.asarray(msk.convert("L").resize(size, Image.NEAREST)) > 127
        return image.astype(np.float32) / 255.0, mask

    def render(self, image_path: str, mask_path: str, image_id: str) -> np.ndarray:
        """Build the model input for one image.

        Args:
            image_path: Path to the dermoscopic image.
            mask_path: Path to the binary lesion mask.
            image_id: Image name, which seeds the artifact placement.

        Returns:
            Float32 array of shape ``(3, image_size, image_size)``, ImageNet
            normalised.
        """
        image, mask = self._load(image_path, mask_path)
        if self.artifact is not None:
            seed = zlib.crc32(f"{self.seed}-{image_id}".encode())
            image = add_artifact(image, mask, self.artifact, seed)
        image = make_view(image, mask, self.view)
        image = (image - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.ascontiguousarray(image.transpose(2, 0, 1), dtype=np.float32)

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per ``ph2`` image event of a patient.

        Args:
            patient: A PyHealth ``Patient`` with ``ph2`` events.

        Returns:
            Samples with ``patient_id``, ``image_id``, ``image`` (the
            normalised view, shape ``(3, image_size, image_size)``) and
            ``malignant`` (1 for melanoma, else 0).
        """
        return [
            {
                "patient_id": patient.patient_id,
                "image_id": event.image_id,
                "image": self.render(event.image_path, event.mask_path, event.image_id),
                "malignant": int(event.diagnosis == "melanoma"),
            }
            for event in patient.get_events(event_type="ph2")
        ]
