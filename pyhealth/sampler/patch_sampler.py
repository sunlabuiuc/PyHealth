"""Patch sampling for medical image segmentation.

The sampler decides *where* a patch comes from. It does not know about
PyHealth datasets or features, which keeps foreground oversampling reusable
across 2D and 3D segmentation tasks.

The foreground probability defaults to 1/3, matching nnU-Net's
``oversample_foreground_percent=0.33``.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PatchLocation:
    """Location of one patch inside a volume.

    Args:
        slice_index: Slice index when sampling a 2D patch from a 3D volume,
            otherwise None.
        origin: Patch origin for each spatial axis.

    Examples:
        >>> from pyhealth.sampler import PatchLocation
        >>> PatchLocation(slice_index=None, origin=(0, 0, 0)).origin
        (0, 0, 0)
    """

    slice_index: int | None
    origin: tuple[int, ...]


def _as_spatial_shape(value: Sequence[int] | int, dims: int) -> tuple[int, ...]:
    """Normalise a patch size to a tuple of ``dims`` positive integers."""
    if isinstance(value, int):
        return (value,) * dims
    result = tuple(int(v) for v in value)
    if len(result) != dims:
        raise ValueError(f"Expected {dims} patch dimensions, got {result}.")
    if any(v <= 0 for v in result):
        raise ValueError(f"Patch dimensions must be positive, got {result}.")
    return result


class PatchSampler:
    """Sample patch locations, optionally oversampling the foreground.

    Args:
        patch_size: Patch size as a tuple of 2 or 3 ints. A 2-tuple samples a
            2D patch from a 3D volume (one slice at a time); a 3-tuple samples
            a 3D patch.
        foreground_prob: Probability of forcing the patch to contain a
            foreground voxel when the mask has one. Defaults to 1/3.
        rng: Random source with ``random()`` and ``randrange()`` methods.
            Defaults to Python's global ``random`` module, which PyTorch
            reseeds in each DataLoader worker.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.sampler.patch_sampler import PatchSampler
        >>> sampler = PatchSampler(patch_size=(2, 2), foreground_prob=1.0)
        >>> mask = np.zeros((4, 4), dtype=np.int64)
        >>> mask[3, 3] = 1
        >>> location = sampler.sample_location((4, 4), mask)
        >>> len(location.origin)
        2
    """

    def __init__(
        self,
        patch_size: Sequence[int],
        foreground_prob: float = 1.0 / 3.0,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= float(foreground_prob) <= 1.0:
            raise ValueError("foreground_prob must be between 0 and 1.")
        if isinstance(patch_size, int):
            raise TypeError("patch_size must be a 2/3-tuple, not an int.")
        raw = tuple(int(v) for v in patch_size)
        if len(raw) not in (2, 3):
            raise ValueError("patch_size must be a 2/3-tuple.")
        if any(v <= 0 for v in raw):
            raise ValueError("patch_size values must be positive.")
        self.patch_size = raw
        self.foreground_prob = float(foreground_prob)
        self.rng = random if rng is None else rng

    def _spatial_mask(
        self,
        volume_shape: tuple[int, ...],
        mask: np.ndarray | None,
    ) -> np.ndarray | None:
        """Remove a leading channel axis and validate the mask shape."""
        if mask is None:
            return None
        array = np.asarray(mask)
        if array.ndim == len(volume_shape) + 1 and array.shape[0] == 1:
            array = array[0]
        if array.shape != tuple(volume_shape):
            raise ValueError(
                f"Mask shape {array.shape} does not match volume shape "
                f"{volume_shape}."
            )
        return array

    def _random_origin(self, volume_shape: tuple[int, ...]) -> tuple[int, ...]:
        """Return a uniformly random valid patch origin."""
        origins = []
        for size, patch in zip(volume_shape, self.patch_size):
            limit = max(1, size - patch + 1)
            origins.append(int(self.rng.randrange(limit)))
        return tuple(origins)

    def _foreground_origin(
        self,
        volume_shape: tuple[int, ...],
        mask: np.ndarray,
        slice_index: int | None,
    ) -> tuple[int, ...] | None:
        """Return a patch origin around a random foreground voxel."""
        if slice_index is not None:
            foreground = np.argwhere(mask[slice_index] > 0)
        else:
            foreground = np.argwhere(mask > 0)
        if foreground.size == 0:
            return None
        point = foreground[int(self.rng.randrange(len(foreground)))]
        origin = []
        for coord, size, patch in zip(point, volume_shape, self.patch_size):
            offset = int(self.rng.randrange(patch))
            value = int(coord) - offset
            value = min(max(value, 0), max(0, size - patch))
            origin.append(value)
        return tuple(origin)

    def sample_location(
        self,
        volume_shape: Sequence[int],
        mask: np.ndarray | None = None,
    ) -> PatchLocation:
        """Sample one patch location.

        Args:
            volume_shape: Spatial shape of the volume, without the channel
                axis, for example ``(D, H, W)`` or ``(H, W)``.
            mask: Optional integer segmentation mask with the same spatial
                shape, or a leading singleton channel axis.

        Returns:
            A :class:`PatchLocation` with an optional slice index and a patch
            origin.
        """
        shape = tuple(int(v) for v in volume_shape)
        if len(shape) not in (2, 3):
            raise ValueError(f"Expected a 2D or 3D volume, got {shape}.")
        patch = _as_spatial_shape(self.patch_size, len(self.patch_size))
        if len(patch) == 3 and len(shape) != 3:
            raise ValueError("A 3D patch needs a 3D volume.")
        if len(patch) == 2 and len(shape) not in (2, 3):
            raise ValueError("A 2D patch needs a 2D or 3D volume.")

        spatial_mask = self._spatial_mask(shape, mask)
        use_foreground = (
            spatial_mask is not None
            and self.foreground_prob > 0
            and self.rng.random() < self.foreground_prob
        )

        if len(patch) == 3:
            if use_foreground:
                origin = self._foreground_origin(shape, spatial_mask, None)
                if origin is not None:
                    return PatchLocation(slice_index=None, origin=origin)
            return PatchLocation(slice_index=None, origin=self._random_origin(shape))

        if len(shape) == 2:
            if use_foreground:
                origin = self._foreground_origin(shape, spatial_mask, None)
                if origin is not None:
                    return PatchLocation(slice_index=None, origin=origin)
            return PatchLocation(slice_index=None, origin=self._random_origin(shape))

        # 2D patch from a 3D volume: choose a slice first.
        depth, height, width = shape
        slice_index = int(self.rng.randrange(depth))
        if use_foreground:
            slices = np.flatnonzero(np.any(spatial_mask > 0, axis=(1, 2)))
            if slices.size:
                slice_index = int(slices[int(self.rng.randrange(len(slices)))])
                origin = self._foreground_origin(
                    (height, width), spatial_mask[slice_index], None
                )
                if origin is not None:
                    return PatchLocation(
                        slice_index=slice_index, origin=origin
                    )
        return PatchLocation(
            slice_index=slice_index,
            origin=self._random_origin((height, width)),
        )

    def center_location(self, volume_shape: Sequence[int]) -> PatchLocation:
        """Return a deterministic center patch location.

        Args:
            volume_shape: Spatial shape of the volume, without the channel
                axis.

        Returns:
            A :class:`PatchLocation` at the center of the volume.
        """
        shape = tuple(int(v) for v in volume_shape)
        if len(shape) not in (2, 3):
            raise ValueError(f"Expected a 2D or 3D volume, got {shape}.")
        patch = _as_spatial_shape(self.patch_size, len(self.patch_size))
        if len(patch) == 3:
            if len(shape) != 3:
                raise ValueError("A 3D patch needs a 3D volume.")
            origin = tuple(max(0, (s - p) // 2) for s, p in zip(shape, patch))
            return PatchLocation(slice_index=None, origin=origin)
        if len(shape) == 3:
            depth, height, width = shape
            origin = tuple(
                max(0, (s - p) // 2) for s, p in zip((height, width), patch)
            )
            return PatchLocation(slice_index=depth // 2, origin=origin)
        origin = tuple(max(0, (s - p) // 2) for s, p in zip(shape, patch))
        return PatchLocation(slice_index=None, origin=origin)
