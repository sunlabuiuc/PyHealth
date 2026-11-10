"""Dataset view for patch-based medical image segmentation.

``PatchDataset`` wraps any ``SampleDataset`` whose samples contain an image and
a segmentation mask. The actual patch-location logic lives in
:class:`pyhealth.sampler.PatchSampler`, so foreground oversampling can be
reused without this dataset class.

Examples:
    >>> import torch
    >>> from pyhealth.datasets import create_sample_dataset
    >>> from pyhealth.datasets.imaging.patches import PatchDataset
    >>> from pyhealth.sampler import PatchSampler
    >>> samples = [
    ...     {
    ...         "patient_id": "p0",
    ...         "image": torch.zeros(1, 8, 8, 8),
    ...         "mask": torch.zeros(1, 8, 8, 8, dtype=torch.long),
    ...     }
    ... ]
    >>> dataset = create_sample_dataset(
    ...     samples, {"image": "raw"}, {"mask": "raw"}
    ... )
    >>> patches = PatchDataset(dataset, PatchSampler((4, 4, 4)), mode="center")
    >>> patches[0]["image"].shape
    torch.Size([1, 4, 4, 4])
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from pyhealth.sampler.patch_sampler import PatchLocation, PatchSampler


def _is_lazy_value(value: Any) -> bool:
    """Whether a processed field holds lazy references instead of pixels."""
    if isinstance(value, (str, Path, dict)):
        return True
    return isinstance(value, (list, tuple)) and all(
        isinstance(item, (str, Path)) for item in value
    )


def _as_tensor(value: Any, dtype: torch.dtype) -> torch.Tensor:
    """Convert a tensor or array-like value to the requested dtype."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value)
    return value.to(dtype)


def _pad_right(
    tensor: torch.Tensor,
    patch_size: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    """Pad a ``(C, *spatial)`` tensor on the right until it fits a patch."""
    spatial = tensor.shape[1:]
    if len(spatial) != len(patch_size):
        raise ValueError(
            f"Patch size {patch_size} does not match tensor shape {tuple(tensor.shape)}."
        )
    pads: list[int] = []
    for size, patch in reversed(list(zip(spatial, patch_size))):
        pads.extend([0, max(0, int(patch) - int(size))])
    if any(pads):
        tensor = F.pad(tensor, pads, value=value)
    return tensor


def _crop(
    tensor: torch.Tensor,
    origin: tuple[int, ...],
    patch_size: tuple[int, ...],
    value: float,
) -> torch.Tensor:
    """Crop one fixed-size patch from a ``(C, *spatial)`` tensor."""
    tensor = _pad_right(tensor, patch_size, value)
    slices = [slice(None)]
    for start, size in zip(origin, patch_size):
        slices.append(slice(int(start), int(start) + int(size)))
    return tensor[tuple(slices)]


class PatchDataset(torch.utils.data.Dataset):
    """Wrap a ``SampleDataset`` and return fixed-size image/mask patches.

    Args:
        dataset: A ``SampleDataset`` whose samples contain ``input_key`` and
            ``label_key`` tensors.
        sampler: A :class:`~pyhealth.sampler.PatchSampler` that chooses patch
            locations. Its ``patch_size`` must match the image spatial rank,
            or be 2D for a 3D volume.
        input_key: Name of the image field. Defaults to ``"image"``.
        label_key: Name of the mask field. Defaults to ``"mask"``.
        mode: ``"train"`` uses the sampler (including foreground
            oversampling); ``"center"`` uses a deterministic center patch.
        num_patches: Number of patch items exposed per underlying sample. The
            underlying volume is re-read for each patch.
        ignore_index: Value used to pad the mask. Defaults to 255.
        return_meta: Whether to include ``patch_origin``, ``patch_slice`` and
            ``volume_shape`` in each sample. Models ignore these fields.
            ``patch_slice`` is omitted for 3D patches because it is ``None``
            there and would break default collation.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets import create_sample_dataset
        >>> from pyhealth.datasets.imaging.patches import PatchDataset
        >>> from pyhealth.sampler import PatchSampler
        >>> samples = [
        ...     {
        ...         "patient_id": "p0",
        ...         "image": torch.zeros(1, 8, 8, 8),
        ...         "mask": torch.zeros(1, 8, 8, 8, dtype=torch.long),
        ...     }
        ... ]
        >>> dataset = create_sample_dataset(
        ...     samples, {"image": "raw"}, {"mask": "raw"}
        ... )
        >>> patches = PatchDataset(dataset, PatchSampler((4, 4, 4)), mode="center")
        >>> patches[0]["mask"].shape
        torch.Size([1, 4, 4, 4])
    """

    def __init__(
        self,
        dataset: Any,
        sampler: PatchSampler,
        input_key: str = "image",
        label_key: str = "mask",
        mode: str = "train",
        num_patches: int = 1,
        ignore_index: int = 255,
        return_meta: bool = True,
    ) -> None:
        if mode not in ("train", "center"):
            raise ValueError("mode must be 'train' or 'center'.")
        if int(num_patches) < 1:
            raise ValueError("num_patches must be at least 1.")
        self.dataset = dataset
        self.sampler = sampler
        self.input_key = input_key
        self.label_key = label_key
        self.mode = mode
        self.num_patches = int(num_patches)
        self.ignore_index = int(ignore_index)
        self.return_meta = bool(return_meta)
        # litdata keeps one open chunk handle per reader. A forked DataLoader
        # worker inherits that handle and therefore its file offset, so
        # concurrent workers can read each other's samples. Remember the
        # owning process and give each worker its own cache on first access.
        self._cache_pid = os.getpid()

    def __len__(self) -> int:
        """Return the number of patch items."""
        return len(self.dataset) * self.num_patches

    def _location(self, spatial_shape: tuple[int, ...], mask: torch.Tensor) -> PatchLocation:
        """Choose a patch location for one sample."""
        if self.mode == "center":
            return self.sampler.center_location(spatial_shape)
        return self.sampler.sample_location(spatial_shape, mask)

    def _materialize(self, sample: dict[str, Any], key: str) -> Any:
        """Materialise one lazy field using the wrapped dataset's processor."""
        value = sample[key]
        if not _is_lazy_value(value):
            return value
        processors = {
            **getattr(self.dataset, "input_processors", {}),
            **getattr(self.dataset, "output_processors", {}),
        }
        processor = processors.get(key)
        if processor is None or not hasattr(processor, "materialize"):
            raise TypeError(
                f"Cannot materialise the lazy value of {key!r}: the wrapped "
                "dataset has no processor with materialize()."
            )
        return processor.materialize(value)

    def _reset_inherited_cache(self) -> None:
        """Drop the litdata cache a forked worker inherited from its parent."""
        dataset = self.dataset
        if getattr(dataset, "cache", None) is None:
            return
        if not hasattr(dataset, "shuffler"):
            raise RuntimeError(
                "PatchDataset fork reset expects litdata's cache/shuffler "
                "attributes; the installed litdata version changed that API."
            )
        dataset.cache = None
        dataset.shuffler = None

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one fixed-size patch sample."""
        pid = os.getpid()
        if pid != self._cache_pid:
            self._reset_inherited_cache()
            self._cache_pid = pid

        sample_index = int(index) // self.num_patches
        sample = self.dataset[sample_index]
        if not isinstance(sample, dict):
            raise TypeError(
                "PatchDataset expects the wrapped dataset to return dictionaries."
            )
        image = _as_tensor(
            self._materialize(sample, self.input_key), torch.float32
        )
        mask = _as_tensor(
            self._materialize(sample, self.label_key), torch.long
        )
        if image.ndim < 3:
            raise ValueError(
                f"Expected an image with a channel axis and spatial axes, got "
                f"shape {tuple(image.shape)}."
            )
        spatial_shape = tuple(int(v) for v in image.shape[1:])
        if mask.ndim == image.ndim - 1:
            mask = mask[None]
        if mask.shape[1:] != spatial_shape:
            raise ValueError(
                f"Mask spatial shape {tuple(mask.shape[1:])} does not match "
                f"image spatial shape {spatial_shape}."
            )

        volume_shape = spatial_shape
        location = self._location(spatial_shape, mask)
        if location.slice_index is not None:
            if image.ndim != 4:
                raise ValueError(
                    "A slice index was returned for a non-3D image."
                )
            image = image[:, location.slice_index]
            mask = mask[:, location.slice_index]

        patch_size = tuple(int(v) for v in self.sampler.patch_size)
        if len(patch_size) != image.ndim - 1:
            raise ValueError(
                f"Patch size {patch_size} does not match image spatial rank "
                f"{image.ndim - 1}."
            )
        image = _crop(image, location.origin, patch_size, 0.0)
        mask = _crop(mask, location.origin, patch_size, self.ignore_index)

        result = dict(sample)
        result[self.input_key] = image
        result[self.label_key] = mask
        if self.return_meta:
            # Keep metadata as lists: the generic collator treats 2-tuples as
            # temporal (time, values) pairs, and a 2D patch origin is a 2-tuple.
            result["patch_origin"] = list(location.origin)
            if location.slice_index is not None:
                result["patch_slice"] = location.slice_index
            result["volume_shape"] = list(volume_shape)
        return result
