"""LNDb lung finding classification on CT patches."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np

from .base_task import BaseTask

#: Class names of the three-class ``finding_class`` label, by index.
LNDB_FINDING_CLASSES = ("non_nodule", "nodule_lt_3mm", "nodule_ge_3mm")


class LNDbFindingClassification(BaseTask):
    """Classify an LNDb CT finding from a patch centred on it.

    Works with :class:`~pyhealth.datasets.LNDbDataset`.

    Each finding whose CT is on disk becomes one sample. The task crops a
    ``patch_size`` voxel cube centred on the finding's world coordinate (out
    of volume voxels are filled with air), clips it to ``hu_window`` and
    rescales it to [0, 1]. The crop is in voxels, so it is not isotropic:
    LNDb CTs have about 0.6-0.8 mm in-plane and 1 mm slice spacing.

    Labels:
        - ``binary=False`` (default): ``finding_class`` with the three classes
          of :data:`LNDB_FINDING_CLASSES`: 0 non-nodule, 1 nodule < 3 mm,
          2 nodule >= 3 mm. The class count is fixed at 3, so a subset that
          lacks a class keeps the same indices.
        - ``binary=True``: ``nodule_ge_3mm``, 1 for a nodule >= 3 mm and 0
          for everything else. This is the question CLIMB's QA asks (and the
          size above which the Fleischner guidelines act). The binary label
          processor needs both labels, so ``set_task`` raises ``ValueError``
          on a subset (e.g. a single CT, or a high ``min_agreement``) that
          holds only one of them.

    Inputs (``patch``, a ``tensor``):
        - ``patch_mode="3d"`` (default): shape ``(1, P, P, P)`` in
          ``(z, y, x)`` order, for 3D models.
        - ``patch_mode="2.5d"``: shape ``(3, P, P)``, the axial, coronal and
          sagittal planes through the centre stacked as channels, for 2D
          models such as :class:`~pyhealth.models.TorchvisionModel`.

    Args:
        patch_size: Edge length of the patch in voxels. Defaults to 32.
        hu_window: ``(low, high)`` Hounsfield range mapped to [0, 1].
            Defaults to ``(-1000, 400)``.
        patch_mode: ``"3d"`` or ``"2.5d"``.
        binary: Predict nodule >= 3 mm vs the rest instead of three classes.
        min_agreement: Keep findings marked by at least this many
            radiologists (``agreement``, 1-3). Defaults to 1, all findings.

    Examples:
        >>> from pyhealth.datasets import LNDbDataset
        >>> from pyhealth.tasks import LNDbFindingClassification
        >>> task = LNDbFindingClassification(patch_mode="2.5d", binary=True)
        >>> task.output_schema
        {'nodule_ge_3mm': 'binary'}
        >>> dataset = LNDbDataset(root="/data/lndb")  # doctest: +SKIP
        >>> samples = dataset.set_task(task)  # doctest: +SKIP
    """

    task_name: str = "LNDbFindingClassification"
    input_schema: ClassVar[dict[str, Any]] = {"patch": "tensor"}
    output_schema: ClassVar[dict[str, Any]] = {
        "finding_class": ("multiclass", {"labels": len(LNDB_FINDING_CLASSES)})
    }

    def __init__(
        self,
        patch_size: int = 32,
        hu_window: tuple[float, float] = (-1000.0, 400.0),
        patch_mode: str = "3d",
        binary: bool = False,
        min_agreement: int = 1,
    ) -> None:
        if patch_mode not in ("3d", "2.5d"):
            raise ValueError("patch_mode must be '3d' or '2.5d'.")
        if patch_size < 1:
            raise ValueError("patch_size must be positive.")
        low, high = (float(v) for v in hu_window)
        if high <= low:
            raise ValueError("hu_window must be (low, high) with high > low.")
        super().__init__()
        self.patch_size = int(patch_size)
        self.hu_window = (low, high)
        self.patch_mode = patch_mode
        self.binary = bool(binary)
        self.min_agreement = int(min_agreement)
        self.label_key = "nodule_ge_3mm" if self.binary else "finding_class"
        self.output_schema = (
            {"nodule_ge_3mm": "binary"}
            if self.binary
            else {
                "finding_class": (
                    "multiclass",
                    {"labels": len(LNDB_FINDING_CLASSES)},
                )
            }
        )

    def crop(self, volume: np.ndarray, center_zyx) -> np.ndarray:
        """Crop, window and rescale a patch around a voxel.

        Args:
            volume: CT volume in ``(z, y, x)`` order, in HU.
            center_zyx: Integer voxel index of the patch centre.

        Returns:
            A float32 array shaped as described by ``patch_mode``.

        Examples:
            >>> import numpy as np
            >>> task = LNDbFindingClassification(patch_size=4)
            >>> volume = np.full((8, 8, 8), 400, dtype=np.int16)
            >>> patch = task.crop(volume, (0, 4, 4))
            >>> patch.shape, float(patch[0, 0, 0, 0]), float(patch[0, 3, 0, 0])
            ((1, 4, 4, 4), 0.0, 1.0)
        """
        size = self.patch_size
        half = size // 2
        low, high = self.hu_window
        out = np.full((size, size, size), low, dtype=np.float32)
        src, dst = [], []
        for centre, length in zip(center_zyx, volume.shape):
            start = int(centre) - half
            lo, hi = max(start, 0), min(start + size, length)
            if hi <= lo:
                lo = hi = 0
            src.append(slice(lo, hi))
            dst.append(slice(lo - start, lo - start + (hi - lo)))
        out[tuple(dst)] = volume[tuple(src)]
        out = (np.clip(out, low, high) - low) / (high - low)
        if self.patch_mode == "2.5d":
            return np.stack(
                [out[half, :, :], out[:, half, :], out[:, :, half]]
            ).astype(np.float32)
        return out[None]

    def label(self, event: Any) -> int:
        """Return the label of a finding event for the configured scheme.

        Examples:
            >>> from types import SimpleNamespace
            >>> event = SimpleNamespace(finding_class="2")
            >>> LNDbFindingClassification(binary=True).label(event)
            1
        """
        finding_class = int(float(event.finding_class))
        if self.binary:
            return int(finding_class == 2)
        return finding_class

    def keep(self, event: Any) -> bool:
        """Return whether a finding with a CT on disk becomes a sample.

        Subclasses narrow the findings here; this task keeps all of them.

        Examples:
            >>> LNDbFindingClassification().keep(object())
            True
        """
        return True

    def __call__(self, patient: Any) -> list[dict[str, Any]]:
        """Build one sample per finding of a patient's CT.

        Args:
            patient: A PyHealth ``Patient`` with ``findings`` events.

        Returns:
            Sample dicts with ``patient_id``, ``record_id``, ``fold``,
            ``patch`` and the label.
        """
        from ..datasets.lndb import read_mhd

        events = [
            event
            for event in patient.get_events(event_type="findings")
            if isinstance(event.image_path, str)
            and event.image_path
            and int(float(event.agreement)) >= self.min_agreement
            and self.keep(event)
        ]
        if not events:
            return []
        image = read_mhd(events[0].image_path)
        samples = []
        for event in events:
            center = image.world_to_voxel(
                [float(event.x), float(event.y), float(event.z)]
            )
            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "record_id": event.record_id,
                    "fold": int(float(event.fold)),
                    "patch": self.crop(image.array, center),
                    self.label_key: self.label(event),
                }
            )
        return samples
