"""Reader selection helpers.

Examples:
    >>> from pyhealth.processors.image_io.readers.factory import get_reader_by_name
    >>> get_reader_by_name("nifti").name
    'nifti'
"""

from __future__ import annotations

from pathlib import Path

from ..types import PathLike
from .base import BaseImageReader
from .dicom import DicomReader, DicomSeriesReader
from .image_mask import ImageMaskReader
from .nifti import NiftiReader

_READERS = {
    "nifti": NiftiReader,
    "dicom": DicomReader,
    "dicom_series": DicomSeriesReader,
    "image_mask": ImageMaskReader,
}


def get_reader_by_name(name: str) -> BaseImageReader:
    """Return a reader instance by registry name.

    Args:
        name: One of ``"nifti"``, ``"dicom"``, ``"dicom_series"`` or
            ``"image_mask"``.

    Returns:
        A new reader instance.

    Raises:
        ValueError: If the name is unknown.

    Examples:
        >>> from pyhealth.processors.image_io.readers.factory import get_reader_by_name
        >>> get_reader_by_name("dicom").name
        'dicom'
    """
    try:
        reader_class = _READERS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown imaging reader {name!r}; expected one of "
            f"{sorted(_READERS)}."
        ) from exc
    return reader_class()


def get_reader(path: PathLike) -> BaseImageReader:
    """Return a reader inferred from a path.

    Directories and explicit file lists are treated as DICOM series.
    ``.nii``/``.nii.gz`` files use the NIfTI reader. Other files are treated
    as single DICOM images.

    Args:
        path: File path, directory, or ordered tuple of file paths.

    Returns:
        A reader instance.

    Examples:
        >>> from pyhealth.processors.image_io.readers.factory import get_reader
        >>> get_reader("scan.nii.gz").name
        'nifti'
    """
    if isinstance(path, (tuple, list)):
        return DicomSeriesReader()
    source = Path(path)
    name = source.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        return NiftiReader()
    if source.is_dir():
        return DicomSeriesReader()
    return DicomReader()
