"""Shared lightweight types for medical imaging I/O."""

from __future__ import annotations

PathLike = str | tuple[str, ...]

# A lazy image value stored in a sample. NIfTI and single-frame DICOM store a
# path string, DICOM series store an ordered list of paths, and masks store a
# directory path or ``(mask_dir, series_path)``.
ImageValue = str | list[str] | tuple[str, ...]
