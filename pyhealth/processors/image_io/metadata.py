"""Shared image metadata helpers for tasks and processors."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .readers.factory import get_reader, get_reader_by_name
from .readers.image_mask import (
    ImageMaskReader,
    _mask_files,
    _prepare_mask_files,
)
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
        if len(new_shape) != 3 or any(value <= 0 for value in new_shape):
            raise ValueError("target_shape must contain three positive values.")
        new_spacing = tuple(
            spacing[index] * spatial[index] / new_shape[index]
            for index in range(3)
        )
    else:
        new_spacing = tuple(float(value) for value in target_spacing)
        if len(new_spacing) != 3 or any(value <= 0 for value in new_spacing):
            raise ValueError(
                "target_spacing must contain three positive values."
            )
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
        Available keys vary by reader: NIfTI has no ``modality``, and a mask
        directory has no ``spacing``/``affine`` unless a reference DICOM
        series is supplied as ``(mask_dir, series_path)``. Target shape/spacing
        are only reported for 3D volumes.

    Raises:
        ValueError: If a target grid is requested for metadata that is not a
            spaced 3D volume.

    Examples:
        >>> read_image_metadata("scan.nii.gz")  # doctest: +SKIP
        {'format': 'nifti', ...}
    """
    if reader == "image_mask":
        canonical = bool(reader_kwargs.pop("canonical", True))
        if (
            isinstance(path, (tuple, list))
            and len(path) == 2
            and Path(path[0]).is_dir()
        ):
            files, series_metadata = _prepare_mask_files(
                _mask_files(Path(path[0])),
                path[1],
                canonical=canonical,
            )
            metadata = ImageMaskReader().read_metadata(tuple(files))
            if series_metadata is not None:
                source_shape = series_metadata.get("source_shape")
                if source_shape is not None:
                    mask_shape = tuple(
                        int(value) for value in metadata["shape"][1:]
                    )
                    if mask_shape != tuple(source_shape):
                        raise ValueError(
                            f"Mask volume shape {mask_shape} does not match "
                            f"the reference image shape {tuple(source_shape)}; "
                            "check the mask slice count and in-plane "
                            "rows/columns."
                        )
                if series_metadata.get("shape") is not None:
                    # Shape must follow the same canonical/source layout as
                    # affine and spacing from the reference series.
                    metadata["shape"] = series_metadata["shape"]
                for key in (
                    "affine",
                    "source_affine",
                    "spacing",
                    "orientation",
                    "modality",
                    "patient_id",
                    "study_id",
                    "series_id",
                    "n_frames",
                ):
                    if series_metadata.get(key) is not None:
                        metadata[key] = series_metadata[key]
        else:
            metadata = ImageMaskReader().read_metadata(path)
    else:
        instance = (
            get_reader_by_name(reader)
            if reader is not None
            else get_reader(path)
        )
        metadata = instance.read_metadata(path, **reader_kwargs)
    if target_shape is not None or target_spacing is not None:
        shape, spacing = _target_layout(metadata, target_shape, target_spacing)
        metadata = {**metadata, "shape": shape, "spacing": spacing}
    return metadata
