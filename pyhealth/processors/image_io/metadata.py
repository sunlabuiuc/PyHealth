"""Shared image metadata helpers for tasks and processors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .readers.factory import get_reader, get_reader_by_name
from .types import PathLike


def _target_layout(
    metadata: dict[str, Any],
    target_shape: Sequence[int] | None,
    target_spacing: Sequence[float] | None,
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    """Return shape and spacing adjusted to the requested target grid."""
    shape = tuple(int(value) for value in metadata.get("shape", ()))
    spacing = tuple(float(value) for value in metadata.get("spacing", ()))
    if len(shape) != 4 or len(spacing) != 3:
        raise ValueError(
            "target_shape and target_spacing require 3D volume metadata "
            "with shape (C, D, H, W) and three spacing values."
        )
    spatial = shape[1:]
    if target_shape is not None:
        new_shape = tuple(int(value) for value in target_shape)
        new_spacing = tuple(
            spacing[index] * spatial[index] / new_shape[index]
            for index in range(3)
        )
    else:
        new_spacing = tuple(float(value) for value in target_spacing)
        new_shape = tuple(
            max(
                1,
                round(spatial[index] * spacing[index] / new_spacing[index]),
            )
            for index in range(3)
        )
    return (shape[0],) + new_shape, new_spacing


def read_image_metadata(
    path: PathLike,
    *,
    reader: str | None = None,
    target_shape: Sequence[int] | None = None,
    target_spacing: Sequence[float] | None = None,
    **reader_kwargs: Any,
) -> dict[str, Any]:
    """Read image metadata as a plain dictionary.

    Tasks can store the result as an ordinary sample key. For a spaced 3D
    volume, ``target_shape`` or ``target_spacing`` makes the returned shape
    and spacing describe the resampled grid; the affine remains the source
    affine. Target grids on other metadata raise ``ValueError``.

    Args:
        path: File path, directory, or ordered path sequence.
        reader: Optional reader name. Inferred from the path when omitted.
            Use ``"image_mask"`` for mask directories.
        target_shape: Optional target spatial shape ``(D, H, W)``. Only
            supported for 3D volume metadata with spacing.
        target_spacing: Optional target spacing ``(D, H, W)``. Only supported
            for 3D volume metadata with spacing.
        **reader_kwargs: Extra arguments forwarded to ``read_metadata``.

    Returns:
        A dictionary with keys such as ``format``, ``shape``, ``spacing``,
        ``affine``, ``orientation``, ``dtype``, ``modality`` and ``source``.
        Target shape/spacing are only reported for 3D volumes.

    Raises:
        ValueError: If a target grid is requested for metadata that is not a
            spaced 3D volume.

    Examples:
        >>> read_image_metadata("scan.nii.gz")  # doctest: +SKIP
        {'format': 'nifti', ...}
    """
    instance = (
        get_reader_by_name(reader) if reader is not None else get_reader(path)
    )
    metadata = instance.read_metadata(path, **reader_kwargs)
    if target_shape is not None or target_spacing is not None:
        shape, spacing = _target_layout(metadata, target_shape, target_spacing)
        metadata = {**metadata, "shape": shape, "spacing": spacing}
    return metadata
