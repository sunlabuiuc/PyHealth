"""CT volume dataset for Retina U-Net detection.

This dataset loads per-patient 3D CT volumes and their instance-labeled
segmentation masks, slices them into 2D axial samples, and yields samples
that are directly consumable by
:class:`pyhealth.tasks.retina_unet_detection.RetinaUNetDetectionTask`.

Two input modes are supported:

1. **Disk mode** — pass a ``root`` directory laid out as::

       root/
         patient_<id>/
           volume.npy   # shape (D, H, W)
           mask.npy     # shape (D, H, W)

2. **In-memory mode** — pass ``volumes`` and ``masks`` dicts keyed by
   patient id. Intended for tests and toy experiments without I/O.

Masks are treated as instance label maps: 0 is background, every other
integer is a separate object instance. Binary masks still work (the task
treats the single non-zero label as one instance); users wanting connected
components on binary masks should apply :func:`scipy.ndimage.label` before
constructing the dataset.
"""

from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import numpy as np

from pyhealth.datasets.base_dataset import BaseDataset


class RetinaUNetCTDataset(BaseDataset):
    """CT volume dataset sliced into 2D detection samples.

    Each sample corresponds to one axial slice of one patient's volume
    and has the shape the Retina U-Net task expects: an ``image`` array
    of shape ``(H, W, 1)`` and a ``mask`` array of shape ``(H, W)``.

    Args:
        root: Path to a directory with one subdirectory per patient,
            each containing ``volume.npy`` and ``mask.npy``. Ignored
            when ``volumes`` / ``masks`` are supplied.
        volumes: Optional mapping ``{patient_id: volume_ndarray}``.
            When provided, disk I/O is skipped entirely.
        masks: Optional mapping ``{patient_id: mask_ndarray}``. Must
            share keys and shapes with ``volumes``.
        skip_empty_slices: If True, drop slices whose mask contains no
            non-zero pixels. Typical for lesion detection, where most
            CT slices are background-only.
        hu_window: Optional ``(low, high)`` HU window. Pixel values are
            clipped to this range and then linearly scaled to ``[0, 1]``.
            If None, the volume is passed through unchanged.
        axial_axis: Which axis of the volume to slice along. Defaults
            to 0 (standard NIfTI/DICOM convention for axial).
        config_path: Optional path to a YAML config (kept for
            forward-compatibility with upstream PyHealth).
    """

    def __init__(
        self,
        root: str = ".",
        volumes: Optional[Dict[str, np.ndarray]] = None,
        masks: Optional[Dict[str, np.ndarray]] = None,
        skip_empty_slices: bool = False,
        hu_window: Optional[Tuple[float, float]] = None,
        axial_axis: int = 0,
        config_path: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            root=root,
            dataset_name="RetinaUNetCTDataset",
            config_path=config_path,
            **kwargs,
        )

        self.skip_empty_slices = skip_empty_slices
        self.hu_window = hu_window
        self.axial_axis = axial_axis

        self._in_memory_volumes: Dict[str, np.ndarray] = {}
        self._in_memory_masks: Dict[str, np.ndarray] = {}

        if volumes is not None or masks is not None:
            if volumes is None or masks is None:
                raise ValueError(
                    "volumes and masks must be provided together"
                )
            if set(volumes.keys()) != set(masks.keys()):
                raise ValueError(
                    "volumes and masks must have the same patient ids"
                )
            for pid, vol in volumes.items():
                if vol.shape != masks[pid].shape:
                    raise ValueError(
                        f"volume/mask shape mismatch for patient {pid}: "
                        f"{vol.shape} vs {masks[pid].shape}"
                    )
            self._in_memory_volumes = dict(volumes)
            self._in_memory_masks = dict(masks)
            self._patient_ids = sorted(self._in_memory_volumes.keys())
        else:
            self._patient_ids = self._scan_root(root)

        self._index: List[Tuple[str, int]] = self._build_index()

    def _scan_root(self, root: str) -> List[str]:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(
                f"root directory does not exist: {root}"
            )

        patient_ids: List[str] = []
        for entry in sorted(root_path.iterdir()):
            if not entry.is_dir():
                continue
            if (entry / "volume.npy").exists() and (entry / "mask.npy").exists():
                patient_ids.append(entry.name)
        return patient_ids

    def _load_volume_and_mask(
        self, patient_id: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        if patient_id in self._in_memory_volumes:
            return (
                self._in_memory_volumes[patient_id],
                self._in_memory_masks[patient_id],
            )

        patient_dir = Path(self.root) / patient_id
        volume = np.load(patient_dir / "volume.npy")
        mask = np.load(patient_dir / "mask.npy")
        return volume, mask

    def _build_index(self) -> List[Tuple[str, int]]:
        index: List[Tuple[str, int]] = []
        for pid in self._patient_ids:
            volume, mask = self._load_volume_and_mask(pid)
            num_slices = volume.shape[self.axial_axis]

            for slice_idx in range(num_slices):
                if self.skip_empty_slices:
                    mask_slice = np.take(mask, slice_idx, axis=self.axial_axis)
                    if not np.any(mask_slice):
                        continue
                index.append((pid, slice_idx))
        return index

    def _apply_window(self, image: np.ndarray) -> np.ndarray:
        if self.hu_window is None:
            return image.astype(np.float32)
        low, high = self.hu_window
        clipped = np.clip(image, low, high).astype(np.float32)
        if high > low:
            clipped = (clipped - low) / (high - low)
        return clipped

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        patient_id, slice_idx = self._index[idx]
        volume, mask = self._load_volume_and_mask(patient_id)

        image_slice = np.take(volume, slice_idx, axis=self.axial_axis)
        mask_slice = np.take(mask, slice_idx, axis=self.axial_axis)

        image_slice = self._apply_window(image_slice)

        if image_slice.ndim == 2:
            image_slice = image_slice[..., np.newaxis]

        return {
            "patient_id": patient_id,
            "slice_idx": int(slice_idx),
            "image": image_slice,
            "mask": np.asarray(mask_slice),
        }

    def iter_patients(self) -> Iterator[str]:
        return iter(self._patient_ids)

    def get_patient(self, patient_id: str) -> List[Dict[str, Any]]:
        """Return all slice samples belonging to a given patient, in order."""
        return [
            self[i]
            for i, (pid, _) in enumerate(self._index)
            if pid == patient_id
        ]

    def set_task(
        self, task: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """Apply a task to every slice sample and return the processed list.

        If ``task`` is None, falls back to :attr:`default_task`.
        """
        if task is None:
            task = self.default_task
        return [task(self[i]) for i in range(len(self))]

    @property
    def default_task(self) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
        from pyhealth.tasks.retina_unet_detection import RetinaUNetDetectionTask

        return RetinaUNetDetectionTask()

    def stats(self) -> Dict[str, int]:
        """Return a small summary dict (patient count, slice count)."""
        return {
            "num_patients": len(self._patient_ids),
            "num_slices": len(self._index),
        }
