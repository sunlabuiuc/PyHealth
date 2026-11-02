"""Array-level transforms shared by the medical imaging readers.

Examples:
    >>> import numpy as np
    >>> from pyhealth.processors.image_io.transforms import normalize_intensity
    >>> normalize_intensity(np.array([0.0, 5.0, 10.0]), mode="minmax")
    array([0. , 0.5, 1. ])
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np


def _require_nibabel():
    """Import nibabel or raise an actionable error."""
    try:
        import nibabel as nib
    except ImportError as exc:  # pragma: no cover - exercised without extra
        raise ImportError(
            "Medical imaging support requires nibabel. "
            "Install it with `pip install pyhealth[imaging]`."
        ) from exc
    return nib


def canonicalize_volume(
    data: np.ndarray,
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reorient a 3D volume to canonical RAS axis order.

    Args:
        data: 3D array whose axes match the columns of ``affine``.
        affine: 4x4 voxel-to-world (RAS) affine.

    Returns:
        A tuple ``(data, affine)`` in canonical RAS orientation.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.processors.image_io.transforms import canonicalize_volume
        >>> data = np.zeros((2, 3, 4), dtype=np.float32)
        >>> affine = np.eye(4)
        >>> canonical, _ = canonicalize_volume(data, affine)
        >>> canonical.shape
        (2, 3, 4)
    """
    nib = _require_nibabel()
    image = nib.Nifti1Image(np.asarray(data), np.asarray(affine, dtype=float))
    canonical = nib.as_closest_canonical(image)
    return np.asanyarray(canonical.dataobj), canonical.affine


def resample_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Sequence[float] | None = None,
    target_shape: Sequence[int] | None = None,
    order: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a 3D volume to a target spacing or shape.

    ``target_spacing`` and ``target_shape`` use the same axis order as
    ``data``. If both are given, ``target_shape`` wins. The affine is
    updated so the resampled volume keeps its world-space position.

    Args:
        data: 3D array whose axes match the columns of ``affine``.
        affine: 4x4 voxel-to-world affine.
        target_spacing: Desired voxel spacing.
        target_shape: Desired array shape.
        order: Interpolation order. Use 0 for masks.

    Returns:
        A tuple ``(data, affine)`` for the resampled volume.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.processors.image_io.transforms import resample_volume
        >>> data = np.zeros((4, 5, 6), dtype=np.float32)
        >>> resampled, _ = resample_volume(
        ...     data, np.eye(4), target_shape=(2, 3, 4)
        ... )
        >>> resampled.shape
        (2, 3, 4)
    """
    nib = _require_nibabel()
    from nibabel.processing import resample_from_to

    if target_spacing is None and target_shape is None:
        return np.asarray(data), np.asarray(affine, dtype=float)

    source_shape = np.asarray(data.shape, dtype=int)
    source_affine = np.asarray(affine, dtype=float)
    current_spacing = np.asarray(
        nib.affines.voxel_sizes(source_affine), dtype=float
    )

    if target_shape is not None:
        new_shape = tuple(int(x) for x in target_shape)
        if len(new_shape) != 3 or any(x <= 0 for x in new_shape):
            raise ValueError("target_shape must contain three positive values.")
        new_spacing = current_spacing * source_shape / np.asarray(new_shape)
    else:
        new_spacing = np.asarray(target_spacing, dtype=float)
        if len(new_spacing) != 3 or np.any(new_spacing <= 0):
            raise ValueError(
                "target_spacing must contain three positive values."
            )
        new_shape = tuple(
            max(1, round(source_shape[i] * current_spacing[i] / new_spacing[i]))
            for i in range(3)
        )

    new_affine = nib.affines.rescale_affine(
        source_affine,
        tuple(source_shape),
        tuple(new_spacing),
        new_shape,
    )
    source = nib.Nifti1Image(np.asarray(data), source_affine)
    resampled = resample_from_to(source, (new_shape, new_affine), order=order)
    return np.asanyarray(resampled.dataobj), resampled.affine


def apply_window(
    data: np.ndarray,
    center: float,
    width: float,
) -> np.ndarray:
    """Apply a DICOM-style linear window to an array.

    Args:
        data: Input pixel array.
        center: Window centre.
        width: Window width.

    Returns:
        An array scaled to ``[0, 1]``.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.processors.image_io.transforms import apply_window
        >>> apply_window(np.array([0.0, 50.0, 100.0]), 50.0, 100.0)
        array([0. , 0.5, 1. ])
    """
    array = np.asarray(data, dtype=np.float32)
    if width == 0:
        raise ValueError("Window width must be non-zero.")
    low = float(center) - float(width) / 2.0
    high = float(center) + float(width) / 2.0
    array = (array - low) / (high - low)
    return np.clip(array, 0.0, 1.0)


def normalize_intensity(
    data: np.ndarray,
    mode: str | None = None,
    clip: tuple[float, float] | None = None,
    percentile_range: Iterable[float] = (1.0, 99.0),
) -> np.ndarray:
    """Normalise image intensities.

    Args:
        data: Input array.
        mode: One of ``None``, ``"minmax"``, ``"zscore"``, ``"percentile"``
            or ``"hu"``.
        clip: Optional ``(low, high)`` clipping range applied before
            normalisation.
        percentile_range: Lower and upper percentiles for ``"percentile"``.

    Returns:
        The normalised float32 array.

    Raises:
        ValueError: If ``mode`` is unknown.

    Examples:
        >>> import numpy as np
        >>> from pyhealth.processors.image_io.transforms import normalize_intensity
        >>> normalize_intensity(np.array([0.0, 5.0, 10.0]), "minmax")
        array([0. , 0.5, 1. ])
    """
    array = np.asarray(data, dtype=np.float32)
    if mode is None:
        return array

    if clip is not None:
        array = np.clip(array, float(clip[0]), float(clip[1]))

    if mode == "minmax":
        low = float(np.min(array))
        high = float(np.max(array))
        return (array - low) / (high - low) if high > low else np.zeros_like(array)

    if mode == "zscore":
        mean = float(np.mean(array))
        std = float(np.std(array))
        return (array - mean) / std if std > 0 else np.zeros_like(array)

    if mode == "percentile":
        low, high = np.percentile(array, list(percentile_range))
        array = np.clip(array, low, high)
        return normalize_intensity(array, mode="minmax")

    if mode == "hu":
        low, high = clip if clip is not None else (-1000.0, 1000.0)
        array = np.clip(array, low, high)
        return normalize_intensity(array, mode="minmax")

    raise ValueError(
        "Unknown intensity mode "
        f"{mode!r}; expected minmax, zscore, percentile or hu."
    )
